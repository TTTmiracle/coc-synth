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
from PIL import Image, ImageDraw

from .assets import SpriteLibrary
from .catalog import Catalog
from .placement import Placement
from .project import Projection

#: Alpha at or below this counts as transparent when stamping the id mask.
ALPHA_CUTOFF = 8

#: Shadow placement, as fractions of tile width. The game lights from the upper
#: left, so shadows fall slightly down and to the right of what casts them.
SHADOW_DX = 0.10
SHADOW_LIFT = 0.16


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
        shadows: bool = True,
        background: tuple[int, int, int] = (94, 137, 68),
    ):
        self.cat = catalog
        self.lib = library
        self.tile_w = tile_w
        self.headroom_tiles = headroom_tiles
        self.min_visibility = min_visibility
        self.margin = margin
        self.shadows = shadows
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

        order = sorted(placements, key=lambda q: proj.depth(*q.tile, *q.footprint))
        resolved = []
        for p in order:
            sprite = self.lib.get(self.cat[p.type_id], p.level, p.direction, p.frame, p.connections)
            ax, ay = proj.anchor(*p.tile, *p.footprint)
            resolved.append((p, sprite, ax - sprite.anchor[0], ay - sprite.anchor[1], ay))

        # Shadows all go down first, so no building ends up under another's shadow.
        # Deliberately not stamped into the id mask: a shadow is not the building,
        # and including it would inflate every bounding box.
        if self.shadows:
            for _, sprite, ox, _, ay in resolved:
                if sprite.shadow is None:
                    continue
                sy = ay - sprite.shadow.height + int(self.tile_w * SHADOW_LIFT)
                canvas.alpha_composite(sprite.shadow, (ox + int(self.tile_w * SHADOW_DX), sy))

        # Far to near: the painter's algorithm gives correct occlusion.
        for p, sprite, ox, oy, _ in resolved:
            canvas.alpha_composite(sprite.image, (ox, oy))
            own, rect = self._stamp(ids, sprite.image, (ox, oy), p.instance_id, size)
            own_pixels[p.instance_id] = own
            rects[p.instance_id] = rect
            meta[p.instance_id] = (sprite.name, proj.footprint_polygon(*p.tile, *p.footprint))

        instances, dropped = self._measure(ids, own_pixels, rects, meta)
        return RenderResult(canvas.convert("RGB"), instances, proj, dropped)

    # ---- internals -------------------------------------------------------

    def _terrain(self, size: tuple[int, int], proj: Projection, rng: Random) -> Image.Image:
        """The ground the base stands on.

        Three things separate real Clash ground from a green rectangle, and all
        three are here. Broad noise gives patches of lighter and darker grass.
        The grid itself is faintly visible -- every tile carries a small constant
        brightness offset, so the diamond lattice reads through the grass the way
        it does in game. And the buildable area ends somewhere: outside it the
        ground drops to a darker, cooler green behind a shadowed lip, which is
        the single strongest cue that this is a base sitting in a landscape
        rather than a texture filling a frame.
        """
        w, h = size
        gen = np.random.default_rng(rng.getrandbits(32))

        # --- broad patchiness -------------------------------------------------
        field = np.zeros((h, w), dtype=np.float32)
        weight = 0.0
        for cells, amplitude in ((7, 0.75), (19, 0.85), (47, 0.5), (115, 0.28)):
            rows = max(2, int(cells * h / max(w, 1)))
            coarse = gen.random((max(2, rows), cells), dtype=np.float32)
            up = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
            field += amplitude * np.asarray(
                up.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
            weight += amplitude
        field /= weight

        # --- the grid, read through the grass ---------------------------------
        ys, xs = np.mgrid[0:h, 0:w]
        tx, ty = proj.px_to_tile(xs.astype(np.float32), ys.astype(np.float32))
        ix, iy = np.floor(tx).astype(np.int64), np.floor(ty).astype(np.int64)
        # Cheap spatial hash: a fixed per-tile value, stable across the image.
        tile_noise = (((ix * 73856093) ^ (iy * 19349663)) & 0x3FF) / 1023.0
        field += (tile_noise.astype(np.float32) - 0.5) * 0.05

        # Blade-scale grain. Added after the upscales so it stays pixel-crisp
        # instead of being smeared into the soft look of resized noise.
        field += (gen.random((h, w), dtype=np.float32) - 0.5) * 0.07
        field = np.clip(field, 0.0, 1.0)

        dark = np.array((58, 101, 41), dtype=np.float32)
        light = np.array((129, 172, 80), dtype=np.float32)
        t = np.clip((field - 0.34) * 2.1, 0.0, 1.0)[:, :, None]
        ground = dark * (1.0 - t) + light * t

        # --- edge of the buildable area ---------------------------------------
        n = float(proj.tiles)
        inside = (tx >= 0) & (tx <= n) & (ty >= 0) & (ty <= n)
        # Distance to the nearest grid edge, in tiles, negative outside.
        edge = np.minimum.reduce([tx, n - tx, ty, n - ty]).astype(np.float32)
        if (~inside).any():
            outer = np.array((46, 80, 39), dtype=np.float32)
            # Ramp over a couple of tiles either side of the boundary: in game the
            # ground beyond the base darkens gradually, it is not a drawn border.
            k = np.clip((0.6 - edge) / 2.6, 0.0, 1.0)[:, :, None]
            ground = ground * (1.0 - k) + (ground * 0.62 + outer * 0.38) * k

        img = Image.fromarray(np.clip(ground, 0, 255).astype(np.uint8), mode="RGB")
        self._scatter_tufts(img, proj, rng)
        return img.convert("RGBA")

    def _scatter_tufts(self, img: Image.Image, proj: Projection, rng: Random) -> None:
        """Flick short blade marks over the grass.

        Noise alone varies colour but has no shape to it. A sparse layer of tiny
        two-stroke tufts gives the ground something with structure at the scale a
        detector actually looks at, and keeps flat regions from reading as fill.
        """
        d = ImageDraw.Draw(img, "RGBA")
        w, h = img.size
        count = max(1, (w * h) // max(1, self.tile_w * 3))
        blade = max(2, round(self.tile_w * 0.07))
        for _ in range(count):
            x, y = rng.randrange(w), rng.randrange(h)
            if rng.random() < 0.55:
                c = (50, 90, 35, rng.randint(45, 90))
            else:
                c = (150, 190, 100, rng.randint(35, 70))
            for _ in range(rng.randint(1, 2)):
                lean = rng.uniform(-0.55, 0.55)
                d.line([(x, y), (x + lean * blade, y - blade * rng.uniform(0.5, 1.0))],
                       fill=c, width=1)

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
