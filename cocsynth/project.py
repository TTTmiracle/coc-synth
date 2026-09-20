"""Isometric tile <-> pixel projection.

Standard 2:1 isometric: a tile is a diamond twice as wide as it is tall. Tile (0, 0)
sits at the top of the screen and the grid fans out downwards, so screen y grows with
(tx + ty) -- which is also the painter's-algorithm depth key.

    sx = origin_x + (tx - ty) * tile_w / 2
    sy = origin_y + (tx + ty) * tile_h / 2

Per-image scale and pan jitter is applied here, by varying `tile_w` and `origin`,
rather than by warping the finished image. Labels are computed from the same
projection, so they stay exact for free instead of needing to be transformed too.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Projection:
    """Maps tile coordinates to pixels for one image."""

    tiles: int
    tile_w: int
    origin: tuple[int, int]

    @property
    def tile_h(self) -> int:
        """2:1 isometric -- height is always half the width."""
        return self.tile_w // 2

    def tile_to_px(self, tx: float, ty: float) -> tuple[int, int]:
        """Project a tile *corner* (grid vertex) to pixels.

        Note this takes vertex coordinates, not tile indices: tile (3, 4)'s four
        corners are vertices (3,4), (4,4), (4,5), (3,5).
        """
        ox, oy = self.origin
        return (
            int(round(ox + (tx - ty) * self.tile_w / 2)),
            int(round(oy + (tx + ty) * self.tile_h / 2)),
        )

    def footprint_polygon(self, tx: int, ty: int, w: int, h: int) -> list[tuple[int, int]]:
        """The four pixel corners of a w x h footprint at tile (tx, ty).

        Order is top, right, bottom, left as drawn on screen -- a closed quad ready
        for `ImageDraw.polygon`.
        """
        return [
            self.tile_to_px(tx, ty),          # top (north corner)
            self.tile_to_px(tx + w, ty),      # right (east)
            self.tile_to_px(tx + w, ty + h),  # bottom (south) -- the sprite anchor
            self.tile_to_px(tx, ty + h),      # left (west)
        ]

    def anchor(self, tx: int, ty: int, w: int, h: int) -> tuple[int, int]:
        """Where a sprite's anchor point must land: the footprint's bottom vertex.

        Sprites are drawn standing on their footprint, so the bottom (nearest) corner
        of the diamond is the reference point everything else is measured from.
        """
        return self.tile_to_px(tx + w, ty + h)

    def depth(self, tx: int, ty: int, w: int, h: int) -> int:
        """Painter's-algorithm sort key: larger means nearer the camera.

        Using the footprint's far edge (tx+w)+(ty+h) rather than its origin means a
        big building correctly draws in front of a small one it overlaps.
        """
        return (tx + w) + (ty + h)

    def grid_bounds_px(self) -> tuple[int, int, int, int]:
        """Bounding box (x0, y0, x1, y1) of the whole grid diamond."""
        corners = [
            self.tile_to_px(0, 0),
            self.tile_to_px(self.tiles, 0),
            self.tile_to_px(self.tiles, self.tiles),
            self.tile_to_px(0, self.tiles),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def centred(
        cls,
        tiles: int,
        tile_w: int,
        headroom: int = 0,
        margin: int = 0,
        pan: tuple[int, int] = (0, 0),
    ) -> tuple[Projection, tuple[int, int]]:
        """Build a projection whose grid fits a canvas, plus that canvas size.

        `headroom` is extra space above the grid for sprites that stand taller than
        their footprint. `margin` pads every side, which is what gives `pan` room to
        shift the grid without pushing it off-canvas -- pan is clamped to the margin.
        """
        probe = cls(tiles, tile_w, (0, 0))
        x0, y0, x1, y1 = probe.grid_bounds_px()
        px = max(-margin, min(margin, pan[0]))
        py = max(-margin, min(margin, pan[1]))
        width = (x1 - x0) + 1 + 2 * margin
        height = (y1 - y0) + 1 + headroom + 2 * margin
        origin = (-x0 + margin + px, -y0 + headroom + margin + py)
        return cls(tiles, tile_w, origin), (width, height)
