"""Procedural isometric sprites.

These exist so the geometry, placement, labelling and training loop are all provable
before any real art lands. A placeholder is an extruded diamond prism sitting on its
footprint: colour and height are derived deterministically from the type name, level
is shown as roof pips, and directional buildings get an arrow on the roof pointing
along their grid heading.

They are schematic on purpose. A model trained on these learns this geometry, not
Clash of Clans -- swap in real sprites (see assets/README.md) for a dataset that
transfers. What they *do* guarantee is that a label error is visible to the eye.
"""
from __future__ import annotations

import colorsys
import hashlib

from PIL import Image, ImageDraw

from .schema import heading_vector
from .project import TILE_ASPECT

#: Body height as a multiple of tile_w, by category. Towers stand tall so that
#: occlusion actually happens and the visibility field gets exercised.
CATEGORY_HEIGHT: dict[str, float] = {
    "town-hall": 1.05,
    "defense": 0.85,
    "wall": 0.34,
    "trap": 0.10,
    "resource": 0.55,
    "army": 0.50,
    "research": 0.62,
    "other": 0.45,
}


def _type_hash(type_id: str) -> int:
    """Stable hash, independent of PYTHONHASHSEED (unlike builtin hash)."""
    return int.from_bytes(hashlib.sha256(type_id.encode()).digest()[:4], "big")


def type_colour(type_id: str) -> tuple[int, int, int]:
    """A distinct, readable colour per building type."""
    h = _type_hash(type_id)
    hue = (h % 997) / 997.0
    sat = 0.45 + ((h >> 10) % 100) / 100.0 * 0.30
    val = 0.62 + ((h >> 20) % 100) / 100.0 * 0.26
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def body_height_px(type_id: str, category: str, tile_w: int) -> int:
    """Pixel height of the prism body, with small per-type variation."""
    base = CATEGORY_HEIGHT.get(category, 0.5)
    jitter = ((_type_hash(type_id) >> 4) % 25 - 12) / 100.0  # +/- 12%
    return max(2, int(round(tile_w * base * (1.0 + jitter))))


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)  # type: ignore[return-value]


def make_sprite(
    type_id: str,
    category: str,
    footprint: tuple[int, int],
    level: int,
    tile_w: int,
    direction: int = 0,
    directions: int = 1,
) -> tuple[Image.Image, tuple[int, int]]:
    """Render one placeholder sprite.

    Returns the RGBA image and its anchor -- the pixel inside the image that must be
    placed on the footprint diamond's bottom (south) vertex.
    """
    w, h = footprint
    tile_h = round(tile_w * TILE_ASPECT)
    body = body_height_px(type_id, category, tile_w)

    # Footprint diamond in sprite-local coordinates, shifted so x starts at 0.
    dia_w = (w + h) * tile_w // 2
    dia_h = (w + h) * tile_h // 2
    n = (h * tile_w // 2, 0)
    e = ((h + w) * tile_w // 2, w * tile_h // 2)
    s = (w * tile_w // 2, dia_h)
    ww = (0, h * tile_h // 2)

    img = Image.new("RGBA", (dia_w + 1, dia_h + body + 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    ground = [(x, y + body) for x, y in (n, e, s, ww)]
    roof = [(x, y) for x, y in (n, e, s, ww)]
    gn, ge, gs, gw = ground
    rn, re, rs, rw = roof

    base = type_colour(type_id)
    outline = _shade(base, 0.45)

    # Two camera-facing vertical walls, then the roof on top.
    d.polygon([gw, gs, rs, rw], fill=_shade(base, 0.62), outline=outline)
    d.polygon([gs, ge, re, rs], fill=_shade(base, 0.80), outline=outline)
    d.polygon(roof, fill=base, outline=outline)

    _draw_level_pips(d, roof, level, base)
    if directions > 1:
        _draw_facing_arrow(d, roof, direction, directions, tile_w)

    anchor = (s[0], s[1] + body)
    return img, anchor


def _draw_level_pips(draw: ImageDraw.ImageDraw, roof: list[tuple[int, int]], level: int, base) -> None:
    """Mark the level as dots on the roof. Skipped when the roof is too small."""
    rn, re, rs, rw = roof
    cx = (rn[0] + rs[0]) // 2
    cy = (rn[1] + rs[1]) // 2
    span_x = (re[0] - rw[0]) // 2
    span_y = (rs[1] - rn[1]) // 2
    if span_x < 8 or span_y < 4:
        return  # 1x1 walls and traps have no room; level is still in the JSON

    pip = max(1, span_x // 14)
    cols = min(5, max(1, span_x // (pip * 4)))
    colour = _shade(base, 0.30)
    for i in range(min(level, 20)):
        row, col = divmod(i, cols)
        px = cx + (col - (cols - 1) / 2) * pip * 3
        py = cy + (row - 1) * pip * 3
        # Keep pips inside the diamond: |dx|/span_x + |dy|/span_y <= 1
        if abs(px - cx) / span_x + abs(py - cy) / span_y > 0.82:
            continue
        draw.ellipse([px - pip, py - pip, px + pip, py + pip], fill=colour)


def _draw_facing_arrow(
    draw: ImageDraw.ImageDraw,
    roof: list[tuple[int, int]],
    direction: int,
    directions: int,
    tile_w: int,
) -> None:
    """Draw an arrow on the roof pointing along the building's grid heading.

    The grid vector is projected the same way the renderer projects tiles, so the
    arrow on screen genuinely points where the `heading` field says it does.
    """
    rn, re, rs, rw = roof
    cx = (rn[0] + rs[0]) / 2
    cy = (rn[1] + rs[1]) / 2
    gx, gy = heading_vector(direction, directions)

    # Project the grid vector into screen space and normalise its length.
    sx = (gx - gy) * tile_w / 2
    sy = (gx + gy) * round(tile_w * TILE_ASPECT) / 2
    mag = (sx * sx + sy * sy) ** 0.5 or 1.0
    reach = min(re[0] - cx, cy - rn[1] + (rs[1] - cy)) * 0.72
    ux, uy = sx / mag * reach, sy / mag * reach

    tip = (cx + ux, cy + uy)
    tail = (cx - ux * 0.45, cy - uy * 0.45)
    # Perpendicular, for the arrowhead barbs.
    pxv, pyv = -uy * 0.32, ux * 0.32
    draw.line([tail, tip], fill=(255, 255, 255, 235), width=max(1, tile_w // 16))
    draw.polygon(
        [tip, (cx + ux * 0.45 + pxv, cy + uy * 0.45 + pyv),
         (cx + ux * 0.45 - pxv, cy + uy * 0.45 - pyv)],
        fill=(255, 255, 255, 235),
    )
