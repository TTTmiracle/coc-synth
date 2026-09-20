"""Derive wall segment art from the published wall icons.

The published collection ships one image per wall level, and it is the shop
icon: a single tapered post, drawn at icon proportions (its top sits more than
a tile-width above the ground). Dropped onto a grid it reads as a fence stake,
and no uniform scale fixes that -- a wall segment is wider than it is tall and
the icon is the opposite.

What the icon *is* good for is colour. Clash walls are simple extruded blocks,
so this builds the sixteen connection states as real geometry -- full tile
extent on connected faces, inset on free ones -- and paints each face with the
palette sampled from that level's own icon. Level 1 comes out wood brown,
level 5 gold, level 6 pink crystal, because those are the icon's own colours.

Writes assets/sprites/wall/wall_lvl<NN>_c<MASK>.png plus the anchor and
per-file scale entries the renderer needs to place them exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
WALL_DIR = ROOT / "assets" / "sprites" / "wall"
MANIFEST = ROOT / "assets" / "sprites" / "manifest.json"

CONNECT_N, CONNECT_E, CONNECT_S, CONNECT_W = 1, 2, 4, 8

TW = 256           # working tile width; stored art is rendered at this scale
TH = TW // 2
SS = 4             # supersample factor for clean diamond edges
INSET = 0.075      # how far a free face pulls back from the tile edge, in tiles

#: Above-ground height per wall level, in tile widths. Clash walls grow with
#: level but stay well under half a tile -- they are barriers, not towers.
HEIGHTS = {1: 0.30, 2: 0.32, 3: 0.34, 4: 0.35, 5: 0.37,
           6: 0.39, 7: 0.41, 8: 0.43, 9: 0.45, 10: 0.47}


def sample_palette(path: Path) -> dict[str, tuple[int, int, int]]:
    """Pull top / down-left / down-right face colours out of a wall icon."""
    arr = np.asarray(Image.open(path).convert("RGBA")).astype(int)
    alpha = arr[..., 3]
    rgb = arr[..., :3]
    ys, xs = np.nonzero(alpha > 200)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    grid_y, grid_x = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2

    def face(sel: np.ndarray) -> tuple[int, int, int]:
        mask = (alpha > 200) & sel
        px = rgb[mask]
        # Trim the outline (darkest) and any specular sparkle (brightest) so the
        # average lands on the material, not on its edges.
        order = np.argsort(px.sum(1))
        keep = order[int(len(order) * 0.25):int(len(order) * 0.85)]
        return tuple(px[keep].mean(0).round().astype(int))

    return {
        "top": face(grid_y < y0 + (y1 - y0) * 0.35),
        "left": face((grid_y > mid_y) & (grid_x < mid_x)),
        "right": face((grid_y > mid_y) & (grid_x >= mid_x)),
    }


def shade(c: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(v * f)))) for v in c)


def extent(mask: int) -> tuple[float, float, float, float]:
    """Footprint of one segment in tile coordinates, given its neighbours.

    A connected face runs out to the tile boundary so it meets its neighbour
    with no seam; a free face pulls back, which is what turns the last segment
    of a run into an end cap and a lone segment into a post.
    """
    return (
        0.0 if mask & CONNECT_W else INSET,          # x0
        1.0 if mask & CONNECT_E else 1.0 - INSET,    # x1
        0.0 if mask & CONNECT_N else INSET,          # y0
        1.0 if mask & CONNECT_S else 1.0 - INSET,    # y1
    )


def build(mask: int, height: float, pal: dict) -> tuple[Image.Image, tuple[int, int], float]:
    """Render one connection state; returns (image, anchor, scale)."""
    x0, x1, y0, y1 = extent(mask)
    tw, th = TW * SS, TH * SS

    def proj(px: float, py: float, z: float) -> tuple[float, float]:
        return ((px - py) * tw / 2, (px + py) * th / 2 - z * tw)

    corners = [proj(px, py, z)
               for px in (x0, x1) for py in (y0, y1) for z in (0.0, height)]
    anchor_pt = proj(1.0, 1.0, 0.0)
    pad = SS * 3
    min_x = min(p[0] for p in corners) - pad
    min_y = min(p[1] for p in corners) - pad
    w = int(max(p[0] for p in corners) - min_x + pad)
    h = int(max(p[1] for p in corners) - min_y + pad)

    def pt(px: float, py: float, z: float) -> tuple[float, float]:
        sx, sy = proj(px, py, z)
        return (sx - min_x, sy - min_y)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    outline = shade(pal["left"], 0.45)

    # Down-left face (the y = y1 plane) and down-right face (x = x1). Both face away
    # from the camera's shoulder, so the down-right one catches more light --
    # matching how the icons themselves are lit.
    d.polygon([pt(x0, y1, height), pt(x1, y1, height), pt(x1, y1, 0), pt(x0, y1, 0)],
              fill=pal["left"] + (255,), outline=outline + (255,), width=SS)
    d.polygon([pt(x1, y0, height), pt(x1, y1, height), pt(x1, y1, 0), pt(x1, y0, 0)],
              fill=pal["right"] + (255,), outline=outline + (255,), width=SS)

    # Top slab, then a brighter inset capstone: Clash walls read as a coping
    # stone sitting on the block rather than one flat-topped solid.
    top = [pt(x0, y0, height), pt(x1, y0, height), pt(x1, y1, height), pt(x0, y1, height)]
    d.polygon(top, fill=pal["top"] + (255,), outline=outline + (255,), width=SS)
    cap = 0.055
    inner = [pt(x0 + cap, y0 + cap, height), pt(x1 - cap, y0 + cap, height),
             pt(x1 - cap, y1 - cap, height), pt(x0 + cap, y1 - cap, height)]
    if x1 - cap > x0 + cap and y1 - cap > y0 + cap:
        d.polygon(inner, fill=shade(pal["top"], 1.10) + (255,))

    # Ridge highlight where the top meets each visible side.
    hi = shade(pal["top"], 1.22) + (255,)
    d.line([pt(x0, y1, height), pt(x1, y1, height)], fill=hi, width=SS)
    d.line([pt(x1, y0, height), pt(x1, y1, height)], fill=hi, width=SS)

    img = img.resize((max(1, w // SS), max(1, h // SS)), Image.Resampling.LANCZOS)
    ax = (anchor_pt[0] - min_x) / SS
    ay = (anchor_pt[1] - min_y) / SS

    box = img.getbbox()
    img = img.crop(box)
    anchor = (round(ax - box[0]), round(ay - box[1]))
    # The renderer scales every sprite's width to the footprint diamond, so a
    # segment narrower than a full tile needs that ratio recorded per file.
    return img, anchor, round(img.width / TW, 5)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    entry = manifest["sprites"].setdefault("wall", {})
    anchors = entry.setdefault("anchors", {})
    scales = entry.setdefault("scales", {})

    made = 0
    for level, height in sorted(HEIGHTS.items()):
        src = WALL_DIR / f"wall_lvl{level:02d}.png"
        if not src.exists():
            print(f"  skip level {level}: no icon to sample")
            continue
        pal = sample_palette(src)
        for mask in range(16):
            img, anchor, scale = build(mask, height, pal)
            name = f"wall_lvl{level:02d}_c{mask}.png"
            img.save(WALL_DIR / name)
            anchors[name] = list(anchor)
            scales[name] = scale
            made += 1
        print(f"  level {level:2d}  h={height:.2f}  top={pal['top']}  16 states")

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{made} wall segments written to {WALL_DIR}")


if __name__ == "__main__":
    main()
