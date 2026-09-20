"""Compositing, plus the instance-id mask that makes labels exact.

Sprites are pasted far-to-near so nearer buildings occlude farther ones, exactly as
in the game. A parallel uint32 id canvas is stamped in the same order, so after
rendering we can read off, per instance:

* its **tight pixel bbox**, measured from surviving alpha rather than approximated
  from the footprint diamond;
* its **visibility**, the fraction of its own pixels not painted over by something
  nearer.

Both come out of the mask for free, and both are things an approximation would get
subtly wrong -- an isometric sprite stands well above its footprint, so a box derived
from the diamond alone is simply the wrong box.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from random import Random

import numpy as np
from PIL import Image

from .assets import SpriteLibrary
from .catalog import Catalog
from .placement import Placement
from .project import Projection

#: Alpha at or below this counts as transparent when stamping the id mask.
ALPHA_CUTOFF = 8


@dataclass(frozen=True)
class RenderedInstance:
    """Per-instance pixel facts measured off the id mask."""

    instance_id: int
    bbox_px: tuple[int, int, int, int]
    visibility: float
    sprite_name: str
    polygon_px: list[tuple[int, int]]


@dataclass(frozen=True)
class RenderResult:
    image: Image.Image
    instances: dict[int, RenderedInstance]
    projection: Projection
    dropped: list[int]


class Renderer:
    """Renders a placement result to an image plus exact per-instance geometry."""

    def __init__(
        self,
        catalog: Catalog,
        library: SpriteLibrary,
        tile_w: int = 32,
        headroom_tiles: float = 1.6,
        min_visibility: float = 0.05,
        margin: int = 24,
        background: tuple[int, int, int] = (94, 137, 68),
    ):
        self.cat = catalog
        self.lib = library
        self.tile_w = tile_w
        self.headroom_tiles = headroom_tiles
        self.min_visibility = min_visibility
        self.margin = margin
        self.background = background

    # ---- public ----------------------------------------------------------

    def render(self, placements: list[Placement], rng: Random, pan: tuple[int, int] = (0, 0)) -> RenderResult:
        headroom = int(round(self.tile_w * self.headroom_tiles))
        proj, size = Projection.centred(
            self.cat.grid_tiles, self.tile_w, headroom=headroom, margin=self.margin, pan=pan
        )

        canvas = self._terrain(size, proj, rng)
        ids = np.zeros((size[1], size[0]), dtype=np.uint32)
        own_pixels: dict[int, int] = {}
        rects: dict[int, tuple[int, int, int, int]] = {}
        meta: dict[int, tuple[str, list[tuple[int, int]]]] = {}

        # Far to near: the painter's algorithm gives correct occlusion.
        for p in sorted(placements, key=lambda q: proj.depth(*q.tile, *q.footprint)):
            sprite = self.lib.get(self.cat[p.type_id], p.level, p.direction)
            ax, ay = proj.anchor(*p.tile, *p.footprint)
            ox, oy = ax - sprite.anchor[0], ay - sprite.anchor[1]

            canvas.alpha_composite(sprite.image, (ox, oy))
            own, rect = self._stamp(ids, sprite.image, (ox, oy), p.instance_id, size)
            own_pixels[p.instance_id] = own
            rects[p.instance_id] = rect
            meta[p.instance_id] = (sprite.name, proj.footprint_polygon(*p.tile, *p.footprint))

        instances, dropped = self._measure(ids, own_pixels, rects, meta)
        return RenderResult(canvas.convert("RGB"), instances, proj, dropped)

    # ---- internals -------------------------------------------------------

    def _terrain(self, size: tuple[int, int], proj: Projection, rng: Random) -> Image.Image:
        """Flat ground with deterministic mottling.

        A perfectly uniform background is something a model can key off instead of
        learning the buildings, so the ground gets low-frequency noise. Generated
        small and upscaled bilinearly, which is both cheap and looks like terrain.
        """
        w, h = size
        gen = np.random.default_rng(rng.getrandbits(32))
        coarse = gen.integers(128 - 16, 128 + 17, size=(max(2, h // 16), max(2, w // 16)), dtype=np.uint8)
        smooth = Image.fromarray(coarse, mode="L").resize((w, h), Image.Resampling.BILINEAR)
        offset = np.array(smooth, dtype=np.int16) - 128

        base = np.array(self.background, dtype=np.int16)[None, None, :]
        ground = np.clip(base + offset[:, :, None], 0, 255).astype(np.uint8)
        return Image.fromarray(ground, mode="RGB").convert("RGBA")

    @staticmethod
    def _stamp(
        ids: np.ndarray, sprite: Image.Image, at: tuple[int, int], iid: int, size: tuple[int, int]
    ) -> tuple[int, tuple[int, int, int, int]]:
        """Write `iid` into the mask wherever the sprite is opaque.

        Returns how many pixels this sprite owns *on canvas* -- the denominator for
        visibility, so a sprite clipped at the canvas edge is not reported as
        occluded by something that is not there -- and the canvas rect it touched.
        An instance can only ever survive inside that rect, which is what lets
        `_measure` avoid rescanning the whole canvas per instance.
        """
        ox, oy = at
        cw, ch = size
        alpha = np.array(sprite.getchannel("A"))
        sh, sw = alpha.shape

        # Intersect the sprite rect with the canvas.
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(cw, ox + sw), min(ch, oy + sh)
        if x0 >= x1 or y0 >= y1:
            return 0, (0, 0, 0, 0)
        sub = alpha[y0 - oy : y1 - oy, x0 - ox : x1 - ox] > ALPHA_CUTOFF
        ids[y0:y1, x0:x1][sub] = iid
        return int(sub.sum()), (x0, y0, x1, y1)

    def _measure(
        self,
        ids: np.ndarray,
        own_pixels: dict[int, int],
        rects: dict[int, tuple[int, int, int, int]],
        meta: dict[int, tuple[str, list[tuple[int, int]]]],
    ) -> tuple[dict[int, RenderedInstance], list[int]]:
        """Read tight boxes and visibility off the finished id mask.

        Only each instance's own paste rect is scanned. Scanning the whole canvas
        per instance is the obvious implementation and is ~50x slower on a dense
        base, because a Town-Hall-9 layout has 300+ instances on a 1.2M-pixel canvas.
        """
        instances: dict[int, RenderedInstance] = {}
        dropped: list[int] = []

        for iid, total in own_pixels.items():
            x0, y0, x1, y1 = rects[iid]
            if total == 0 or x0 >= x1:
                dropped.append(iid)
                continue

            mask = ids[y0:y1, x0:x1] == iid
            visible = int(mask.sum())
            if visible == 0 or visible / total < self.min_visibility:
                dropped.append(iid)
                continue

            # Collapse to per-axis extents instead of materialising every coordinate.
            rows = np.flatnonzero(mask.any(axis=1))
            cols = np.flatnonzero(mask.any(axis=0))
            name, poly = meta[iid]
            instances[iid] = RenderedInstance(
                instance_id=iid,
                bbox_px=(x0 + int(cols[0]), y0 + int(rows[0]),
                         int(cols[-1] - cols[0]) + 1, int(rows[-1] - rows[0]) + 1),
                visibility=round(min(1.0, visible / total), 4),
                sprite_name=name,
                polygon_px=poly,
            )
        return instances, dropped
