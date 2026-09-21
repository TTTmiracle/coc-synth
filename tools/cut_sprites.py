"""Cut building art straight out of a screenshot of the game.

Better than scaling a published icon, because the icon's framing was never the
building's footprint and had to be corrected for by a measured factor. What the
game draws is already the right shape at the right size; lifting it out needs no
factor at all.

The ground is a two-tone checkerboard with grain, which makes it separable: a
pixel far from both tile shades belongs to the building. Holes are filled from
the crop border so green parts *inside* a building survive, and the edge is
feathered by a pixel so the cut does not leave a grass fringe.

What it cannot do is invent the levels or types that are not in the picture, and
anything with a UI marker over it has to be given a box that excludes the marker.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent


def grass_mask(rgb: np.ndarray) -> np.ndarray:
    """Ground, including the parts a building's own shadow has darkened.

    A tight brightness threshold keeps the shadowed grass beside a building,
    which then rides along in the cut as a green wedge under one corner.
    Shadow dims grass without changing that it is green, so the test leans on
    the colour relationships and stays loose about level.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (g > 80) & (g - r > 8) & (g - b > 38)


def _shift_and(mask: np.ndarray, op) -> np.ndarray:
    out = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rolled = np.roll(np.roll(mask, dy, 0), dx, 1)
        if dy == 1:
            rolled[0, :] = op is np.logical_and
        elif dy == -1:
            rolled[-1, :] = op is np.logical_and
        if dx == 1:
            rolled[:, 0] = op is np.logical_and
        elif dx == -1:
            rolled[:, -1] = op is np.logical_and
        out = op(out, rolled)
    return out


def erode(mask: np.ndarray, n: int = 1) -> np.ndarray:
    for _ in range(n):
        mask = _shift_and(mask, np.logical_and)
    return mask


def dilate(mask: np.ndarray, n: int = 1) -> np.ndarray:
    for _ in range(n):
        mask = _shift_and(mask, np.logical_or)
    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Just the building: drop villagers, UI labels and anything else nearby.

    Everything not-grass inside the crop would otherwise be kept, and at this
    zoom that means the odd villager standing beside a building and the dashed
    label boxes the game floats above them.
    """
    seen = np.zeros(mask.shape, bool)
    best: list[tuple[int, int]] = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        q = deque([(sy, sx)])
        seen[sy, sx] = True
        pts = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if len(pts) > len(best):
            best = pts
    out = np.zeros(mask.shape, bool)
    for y, x in best:
        out[y, x] = True
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Everything not reachable from the border is inside the building."""
    h, w = mask.shape
    outside = np.zeros_like(mask)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    return ~outside


def cut(img: Image.Image, box: tuple[int, int, int, int], pad: int = 6) -> Image.Image:
    x, y, w, h = box
    crop = img.crop((x - pad, y - pad, x + w + pad, y + h + pad)).convert("RGB")
    arr = np.asarray(crop).astype(int)
    raw = ~grass_mask(arr)
    # Open before choosing the component: a villager standing against a wall is
    # joined to it by a few pixels, and without breaking that thread it is part of
    # the building. Eroding severs the thread, the building survives as the larger
    # body, and dilating back restores its true edge.
    core = largest_component(erode(raw, 2))
    solid = fill_holes(raw & dilate(core, 4))

    alpha = Image.fromarray((solid * 255).astype(np.uint8), mode="L")
    # Feather by a pixel: a hard cut against grass leaves a green rim, because the
    # screenshot's own edge pixels are already a blend of building and ground.
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))

    out = crop.convert("RGBA")
    out.putalpha(alpha)
    return out.crop(out.getbbox())


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    img = Image.open(ROOT / spec["screenshot"])
    tile_w = float(spec["tile_w"])
    manifest_path = ROOT / "assets" / "sprites" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sprites = manifest["sprites"]

    for type_id, entry in spec["buildings"].items():
        box = tuple(entry["box"])
        level = entry.get("level", 1)
        sprite = cut(img, box)
        d = ROOT / "assets" / "sprites" / type_id
        d.mkdir(parents=True, exist_ok=True)
        name = f"{type_id}_lvl{level:02d}.png"
        sprite.save(d / name)
        # The cut is already at game scale, so its width in tiles is what it should
        # render at; record that against the footprint it stands on.
        fp = entry["footprint"]
        diamond = (fp[0] + fp[1]) / 2.0
        scale = round((sprite.width / tile_w) / diamond, 4)
        e = sprites.setdefault(type_id, {})
        e.setdefault("scales", {})[name] = scale
        e.setdefault("cut_from", spec["screenshot"])
        print(f"{type_id:18s} {sprite.width:3d}x{sprite.height:3d}px  "
              f"{sprite.width / tile_w:.2f} tiles  scale {scale}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest updated for {len(spec['buildings'])} types")


if __name__ == "__main__":
    main(sys.argv[1])
