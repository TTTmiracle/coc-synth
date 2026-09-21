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
TH = round(TW * 0.75)   # the game's measured tile aspect, see project.TILE_ASPECT
SS = 4             # supersample factor for clean diamond edges
INSET = 0.290      # how far the BODY's free face pulls back from the tile edge
#
# Measured, not chosen. In reference/th1_spread.jpg a two-segment run spans 0.98
# tiles and a 2x2 block spans 1.48; a run of n segments spans (n-1)/2 + w and a
# block spans 1 + w, so both give an art width w of 0.48 tiles, hence an inset of
# (1 - 0.48) / 2. The same two shapes give a height above ground of 0.34 and 0.33
# tiles independently. Two arrangements agreeing is what makes it a measurement
# rather than a reading.
CAP_OVERHANG = 0.03  # how far the coping stone oversails the body, in tiles
CAP_THICK = 0.075  # coping stone thickness, in tile widths

#: Above-ground height per wall level, in tile widths. Level 1 is measured at
#: 0.33 (see INSET); the rest are stepped up from it and remain unmeasured.
HEIGHTS = {1: 0.33, 2: 0.35, 3: 0.37, 4: 0.39, 5: 0.41,
           6: 0.43, 7: 0.45, 8: 0.47, 9: 0.49, 10: 0.51}


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
    # The coping stone on top overhangs the body. That overhang is most of what
    # gives a Clash wall its silhouette -- without it a run is a plain kerb -- but
    # it can only overhang on a free face, because a connected face has to stay
    # flush with the tile edge or the seam with the neighbour opens up.
    cap_in = max(0.0, INSET - CAP_OVERHANG)
    c0 = 0.0 if mask & CONNECT_W else cap_in
    c1 = 1.0 if mask & CONNECT_E else 1.0 - cap_in
    r0 = 0.0 if mask & CONNECT_N else cap_in
    r1 = 1.0 if mask & CONNECT_S else 1.0 - cap_in
    body_h = height - CAP_THICK
    tw, th = TW * SS, TH * SS

    def proj(px: float, py: float, z: float) -> tuple[float, float]:
        return ((px - py) * tw / 2, (px + py) * th / 2 - z * tw)

    corners = [proj(px, py, z)
               for px, py in ((c0, r0), (c1, r0), (c0, r1), (c1, r1))
               for z in (0.0, height)]
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

    def box(bx0, bx1, by0, by1, z_lo, z_hi, lit):
        """The two camera-facing sides and the top of one slab.

        Only the y = by1 and x = bx1 faces are ever visible: screen x runs with
        (px - py), so growing either axis moves toward the viewer. `lit` brightens
        the coping stone, which catches the sun the body does not.
        """
        d.polygon([pt(bx0, by1, z_hi), pt(bx1, by1, z_hi),
                   pt(bx1, by1, z_lo), pt(bx0, by1, z_lo)],
                  fill=shade(pal["left"], lit) + (255,), outline=outline + (255,), width=SS)
        d.polygon([pt(bx1, by0, z_hi), pt(bx1, by1, z_hi),
                   pt(bx1, by1, z_lo), pt(bx1, by0, z_lo)],
                  fill=shade(pal["right"], lit) + (255,), outline=outline + (255,), width=SS)

    box(x0, x1, y0, y1, 0.0, body_h, 0.74)                    # body
    box(c0, c1, r0, r1, body_h, height, 0.86)                 # coping stone

    top = [pt(c0, r0, height), pt(c1, r0, height),
           pt(c1, r1, height), pt(c0, r1, height)]
    d.polygon(top, fill=shade(pal["top"], 0.80) + (255,),
              outline=outline + (255,), width=SS)

    # A stud on each segment. A Clash wall run is visibly made of blocks; a smooth
    # unbroken bar is the one thing it never looks like.
    sc, sz = 0.5, 0.15
    if (c1 - c0) > sz * 2.2 and (r1 - r0) > sz * 2.2:
        mx, my = (c0 + c1) / 2, (r0 + r1) / 2
        s0x, s1x, s0y, s1y = mx - sz, mx + sz, my - sz, my + sz
        top_z = height + CAP_THICK * 0.55
        d.polygon([pt(s0x, s1y, top_z), pt(s1x, s1y, top_z),
                   pt(s1x, s1y, height), pt(s0x, s1y, height)],
                  fill=shade(pal["left"], 0.95) + (255,), outline=outline + (255,), width=SS)
        d.polygon([pt(s1x, s0y, top_z), pt(s1x, s1y, top_z),
                   pt(s1x, s1y, height), pt(s1x, s0y, height)],
                  fill=shade(pal["right"], 0.95) + (255,), outline=outline + (255,), width=SS)
        d.polygon([pt(s0x, s0y, top_z), pt(s1x, s0y, top_z),
                   pt(s1x, s1y, top_z), pt(s0x, s1y, top_z)],
                  fill=shade(pal["top"], 1.02) + (255,), outline=outline + (255,), width=SS)
        _ = sc

    hi = shade(pal["top"], 1.20) + (255,)
    d.line([pt(c0, r1, height), pt(c1, r1, height)], fill=hi, width=SS)
    d.line([pt(c1, r0, height), pt(c1, r1, height)], fill=hi, width=SS)

    img = img.resize((max(1, w // SS), max(1, h // SS)), Image.Resampling.LANCZOS)
    ax = (anchor_pt[0] - min_x) / SS
    ay = (anchor_pt[1] - min_y) / SS

    box_ = img.getbbox()
    img = img.crop(box_)
    anchor = (round(ax - box_[0]), round(ay - box_[1]))
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
