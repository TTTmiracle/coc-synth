"""Debug overlay -- the tool that actually catches label bugs.

Anchor and projection mistakes look perfectly fine in the JSON: the numbers are
self-consistent, they are just consistently wrong. Drawing the boxes, footprint
diamonds and facing arrows back onto the image makes them obvious instantly, so
`--overlay` is the first thing to reach for when something looks off.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .placeholders import type_colour
from .project import Projection
from .schema import HEADING_VECTORS, HEADINGS, BaseLabel

BOX_COLOUR = (255, 64, 64)
DIAMOND_COLOUR = (64, 220, 255)
ARROW_COLOUR = (255, 235, 60)


def _font(size: int = 11) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(
    image: Image.Image,
    label: BaseLabel,
    show_boxes: bool = True,
    show_diamonds: bool = True,
    show_text: bool = True,
    only_types: set[str] | None = None,
) -> Image.Image:
    """Return a copy of `image` with ground truth drawn on top."""
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    font = _font()
    proj = Projection(label.grid.tiles, label.grid.tile_w, tuple(label.grid.origin_px))

    for b in label.buildings:
        if only_types and b.type not in only_types:
            continue

        if show_diamonds:
            d.polygon([tuple(p) for p in b.footprint_polygon_px],
                      outline=DIAMOND_COLOUR + (255,), width=1)
        if show_boxes:
            x, y, w, h = b.bbox_px
            d.rectangle([x, y, x + w - 1, y + h - 1], outline=BOX_COLOUR + (255,), width=1)

        # Facing arrow, drawn from the label's own direction field so a mismatch
        # between the sprite and the recorded heading is immediately visible.
        if b.rotation_deg or b.heading != "N":
            _facing_arrow(d, proj, b)

        if show_text:
            x, y, _, _ = b.bbox_px
            caption = f"{b.type} {b.level}"
            if b.heading != "N" or b.rotation_deg:
                caption += f" {b.heading}"
            d.text((x + 1, y - 10), caption, fill=type_colour(b.type), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def _facing_arrow(d: ImageDraw.ImageDraw, proj: Projection, b) -> None:
    """Arrow from the footprint centre along the building's recorded heading."""
    poly = b.footprint_polygon_px
    cx = sum(p[0] for p in poly) / 4
    cy = sum(p[1] for p in poly) / 4

    # Read the vector straight off the recorded heading -- no need to know how many
    # facings the type has, and it cross-checks the heading string against the sprite.
    gx, gy = HEADING_VECTORS[HEADINGS.index(b.heading)]
    sx = (gx - gy) * proj.tile_w / 2
    sy = (gx + gy) * proj.tile_h / 2
    mag = (sx * sx + sy * sy) ** 0.5 or 1.0
    reach = max(proj.tile_w, 18)
    tip = (cx + sx / mag * reach, cy + sy / mag * reach)
    d.line([(cx, cy), tip], fill=ARROW_COLOUR + (255,), width=2)
    d.ellipse([tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3], fill=ARROW_COLOUR + (255,))


def legend(label: BaseLabel) -> str:
    """One-line summary for logs and filenames."""
    counts = label.counts()
    top = ", ".join(f"{k}x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6])
    return f"TH{label.town_hall_level} seed={label.seed} {len(label.buildings)} buildings: {top}"
