"""Sprite loading, with fallback on two axes.

The folder contract (see assets/README.md):

    assets/sprites/<type>/<type>_lvl<NN>.png          fixed-orientation buildings
    assets/sprites/<type>/<type>_lvl<NN>_dir<D>.png   directional buildings
    assets/sprites/manifest.json                      anchor overrides, mirror pairs

Resolution falls back so a partially populated folder still renders:

1. **Level** -- a request for level 14 uses the highest present sprite <= 14. Real
   Clash art only changes at a handful of levels, so this means one image per *visual
   variant*, not one per level.
2. **Direction** -- a missing dir<D> tries its `mirror_of` partner (flipped
   horizontally, anchor flipped with it), then dir0.
3. **Nothing at all** -- a procedural placeholder, with a warning.

The `sprite` field in the label records what was actually used, so a dataset can
always be audited for how much real art went into it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageEnhance

from . import placeholders
from .catalog import BuildingDef

log = logging.getLogger(__name__)

DEFAULT_SPRITE_DIR = Path(__file__).resolve().parent.parent / "assets" / "sprites"
SPRITE_RE = re.compile(
    r"^(?P<type>.+)_lvl(?P<level>\d+)(?:_dir(?P<dir>\d+))?(?:_f(?P<frame>\d+))?\.png$", re.I
)

#: Distinct per-instance appearance variants cached per sprite. Buildings in Clash
#: are animated, so two Cannons in one base should not be pixel-identical. When real
#: animation frames exist they are used; otherwise each instance gets a subtle
#: brightness/scale variation so the model cannot memorise one exact pose.
AUG_BUCKETS = 8

PLACEHOLDER_PREFIX = "placeholder:"


@dataclass(frozen=True)
class Sprite:
    """A sprite ready to composite."""

    image: Image.Image
    anchor: tuple[int, int]
    name: str

    @property
    def is_placeholder(self) -> bool:
        return self.name.startswith(PLACEHOLDER_PREFIX)


def _vary(img: Image.Image, anchor: tuple[int, int], bucket: int) -> tuple[Image.Image, tuple[int, int]]:
    """Apply a small per-instance appearance change, keeping the anchor correct.

    This is a stand-in for animation, not a replacement for it: it stops a model
    memorising one exact pose per building, but real captured frames are better where
    you have them. Bucket 0 is the untouched sprite.
    """
    if bucket == 0:
        return img, anchor
    brightness, scale = SpriteLibrary._bucket_params(bucket)

    if scale != 1.0:
        w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        ax, ay = anchor
        anchor = (round(ax * w / img.width), round(ay * h / img.height))
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(img.convert("RGB")).enhance(brightness)
        rgb.putalpha(img.getchannel("A"))
        img = rgb
    return img, anchor


def trim_alpha(img: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Crop fully transparent margins. Returns the crop and the offset removed."""
    img = img.convert("RGBA")
    box = img.getbbox()
    if box is None:
        return img, (0, 0)
    return img.crop(box), (box[0], box[1])


class SpriteLibrary:
    """Resolves (type, level, direction) to a concrete sprite."""

    def __init__(self, sprite_dir: Path | str = DEFAULT_SPRITE_DIR, tile_w: int = 32):
        self.dir = Path(sprite_dir)
        self.tile_w = tile_w
        self.manifest = self._load_manifest()
        self._index = self._build_index()
        self._cache: dict[tuple, Sprite] = {}
        self._warned: set[str] = set()

    # ---- discovery -------------------------------------------------------

    def _load_manifest(self) -> dict:
        path = self.dir / "manifest.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8")).get("sprites", {})

    def _build_index(self) -> dict[str, dict[int, dict[int, dict[int, Path]]]]:
        """type -> level -> direction -> frame -> path, from whatever is on disk."""
        index: dict[str, dict[int, dict[int, dict[int, Path]]]] = {}
        if not self.dir.exists():
            return index
        for path in sorted(self.dir.rglob("*.png")):
            m = SPRITE_RE.match(path.name)
            if not m:
                log.debug("ignoring unrecognised sprite filename: %s", path.name)
                continue
            level = int(m.group("level"))
            direction = int(m.group("dir") or 0)
            frame = int(m.group("frame") or 0)
            index.setdefault(m.group("type"), {}).setdefault(level, {}).setdefault(direction, {})[frame] = path
        return index

    def coverage(self) -> dict[str, int]:
        """Sprite files present per type -- useful for reporting real-art coverage."""
        return {t: sum(len(f) for d in lv.values() for f in d.values())
                for t, lv in self._index.items()}

    def frame_counts(self) -> dict[str, int]:
        """Most animation frames available for any one level/direction, per type."""
        out: dict[str, int] = {}
        for t, lv in self._index.items():
            out[t] = max((len(f) for d in lv.values() for f in d.values()), default=0)
        return out

    def directional_gaps(self, catalog) -> dict[str, list[int]]:
        """Facings that would silently collapse onto another sprite.

        This is the quiet corruption worth guarding against: the label records the
        facing the placer *chose*, but if the art for that facing is missing the
        renderer falls back to dir0 and draws the wrong thing. The image then
        disagrees with its own label and nothing errors.

        Reported per directional type that has *some* art. A type with no art at all
        is excluded -- it uses placeholders, which render every facing correctly.
        """
        gaps: dict[str, list[int]] = {}
        for bdef in catalog.buildings.values():
            if not bdef.is_directional or bdef.id not in self._index:
                continue
            missing = []
            for d in range(bdef.directions):
                for level in sorted(self._index[bdef.id]):
                    by_dir = self._index[bdef.id][level]
                    if d not in by_dir and self._mirror_partner(bdef.id, d) not in by_dir:
                        missing.append((level, d))
                        break
            if missing:
                gaps[bdef.id] = sorted({d for _, d in missing})
        return gaps

    # ---- resolution ------------------------------------------------------

    def get(self, bdef: BuildingDef, level: int, direction: int, frame: int = 0) -> Sprite:
        """Resolve a sprite. `frame` selects an animation frame when real ones exist,
        and always drives a subtle per-instance variation so identical buildings in
        one base are not pixel-identical copies."""
        bucket = frame % AUG_BUCKETS
        key = (bdef.id, level, direction, frame, self.tile_w)
        if key not in self._cache:
            self._cache[key] = self._resolve(bdef, level, direction, frame, bucket)
        return self._cache[key]

    def _resolve(self, bdef: BuildingDef, level: int, direction: int,
                 frame: int, bucket: int) -> Sprite:
        found = self._lookup(bdef.id, level, direction, frame)
        if found is None:
            return self._placeholder(bdef, level, direction)
        path, mirrored = found
        img, _ = trim_alpha(Image.open(path))
        anchor = self._anchor_for(path, img, mirrored)
        if mirrored:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img, anchor = _vary(img, anchor, bucket)
        return Sprite(img, anchor, path.name + (" (mirrored)" if mirrored else ""))

    def _lookup(self, type_id: str, level: int, direction: int, frame: int) -> tuple[Path, bool] | None:
        """Find the best sprite file, returning (path, needs_mirroring)."""
        by_level = self._index.get(type_id)
        if not by_level:
            return None
        # Level axis: highest available at or below the requested level.
        usable = [lv for lv in by_level if lv <= level] or [min(by_level)]
        by_dir = by_level[max(usable)]

        if direction in by_dir:
            return self._pick_frame(by_dir[direction], frame), False
        mirror = self._mirror_partner(type_id, direction)
        if mirror is not None and mirror in by_dir:
            return self._pick_frame(by_dir[mirror], frame), True
        if 0 in by_dir:
            return self._pick_frame(by_dir[0], frame), False
        return self._pick_frame(next(iter(by_dir.values())), frame), False

    @staticmethod
    def _pick_frame(frames: dict[int, Path], frame: int) -> Path:
        """Map a rolled frame value onto however many frames actually exist.

        The placer rolls from a fixed range without knowing what art is on disk, so
        placement stays reproducible and identical whether one frame is present or
        twenty. One frame means every instance uses it -- no error, just less variety.
        """
        order = sorted(frames)
        return frames[order[frame % len(order)]]

    def _mirror_partner(self, type_id: str, direction: int) -> int | None:
        """The direction this one is a horizontal flip of, per the manifest."""
        entry = self.manifest.get(type_id, {})
        pairs = entry.get("mirror_of", {})
        value = pairs.get(str(direction), pairs.get(direction))
        return int(value) if value is not None else None

    @staticmethod
    def _bucket_params(bucket: int) -> tuple[float, float]:
        """Brightness and scale for one variation bucket."""
        brightness = 0.94 + (bucket % 5) * 0.03      # 0.94 .. 1.06
        scale = 0.985 + ((bucket // 5) % 3) * 0.015  # 0.985 .. 1.015
        return brightness, scale

    def _anchor_for(self, path: Path, img: Image.Image, mirrored: bool) -> tuple[int, int]:
        """Anchor point inside the sprite, from the manifest or defaulted.

        Default is bottom-centre, which is correct for art drawn standing on its
        footprint. Directional sprites usually need overrides -- a sweeper facing
        away does not sit the same way in its tile as one facing the camera.
        """
        entry = self.manifest.get(path.parent.name, {})
        anchors = entry.get("anchors", {})
        raw = anchors.get(path.name) or entry.get("anchor")
        ax, ay = (int(raw[0]), int(raw[1])) if raw else (img.width // 2, img.height - 1)
        if mirrored:
            ax = img.width - 1 - ax
        return ax, ay

    def _placeholder(self, bdef: BuildingDef, level: int, direction: int) -> Sprite:
        if bdef.id not in self._warned:
            self._warned.add(bdef.id)
            log.warning("no sprite for %s; using procedural placeholder", bdef.id)
        img, anchor = placeholders.make_sprite(
            bdef.id, bdef.category, bdef.footprint, level,
            tile_w=self.tile_w, direction=direction, directions=bdef.directions,
        )
        suffix = f"_dir{direction}" if bdef.is_directional else ""
        return Sprite(img, anchor, f"{PLACEHOLDER_PREFIX}{bdef.id}_lvl{level:02d}{suffix}")
