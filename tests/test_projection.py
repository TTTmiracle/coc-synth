"""Isometric geometry. Anchor and projection bugs are invisible in the JSON."""
import pytest

from cocsynth.project import Projection


def test_tile_matches_the_measured_game_aspect():
    """Not 2:1. Measured off an empty home village, the lattice is 32.90px across
    and 24.74px down -- 4:3. Rendering at 2:1 squashes the whole image vertically
    by a third, which reads as a wrong camera rather than as a bug."""
    p = Projection(44, 32, (0, 0))
    assert p.tile_h == 24
    poly = p.footprint_polygon(0, 0, 1, 1)
    assert poly[1][0] - poly[3][0] == 32  # width, west to east
    assert poly[2][1] - poly[0][1] == 24  # height, north to south
    assert abs((poly[1][0] - poly[3][0]) / (poly[2][1] - poly[0][1]) - 4 / 3) < 0.01


def test_polygon_corners_are_in_screen_order():
    p = Projection(44, 32, (500, 0))
    north, east, south, west = p.footprint_polygon(10, 10, 3, 3)
    assert north[1] < east[1] < south[1]
    assert west[0] < north[0] < east[0]
    assert west[0] < south[0] < east[0]


def test_anchor_is_the_south_vertex():
    """Sprites stand on the near corner of their footprint."""
    p = Projection(44, 32, (500, 20))
    for tx, ty, w, h in [(0, 0, 1, 1), (10, 4, 3, 3), (20, 20, 4, 4)]:
        assert p.anchor(tx, ty, w, h) == p.footprint_polygon(tx, ty, w, h)[2]


def test_square_footprints_sit_on_the_vertical_centreline():
    p = Projection(44, 32, (704, 0))
    for n in (1, 2, 3, 4):
        assert p.anchor(20, 20, n, n)[0] == 704


def test_depth_orders_near_over_far():
    p = Projection(44, 32, (0, 0))
    assert p.depth(0, 0, 1, 1) < p.depth(10, 10, 1, 1)
    # A bigger building at the same origin reaches nearer the camera.
    assert p.depth(5, 5, 4, 4) > p.depth(5, 5, 1, 1)


def test_centred_canvas_contains_the_whole_grid():
    for tile_w in (16, 32, 64):
        p, size = Projection.centred(44, tile_w, headroom=40, margin=24)
        x0, y0, x1, y1 = p.grid_bounds_px()
        assert x0 >= 0 and y0 >= 0
        assert x1 < size[0] and y1 < size[1]


@pytest.mark.parametrize("pan", [(0, 0), (24, 24), (-24, -24), (9999, -9999)])
def test_pan_is_clamped_to_the_margin(pan):
    """Pan jitter must never push the grid off the canvas."""
    p, size = Projection.centred(44, 32, headroom=40, margin=24, pan=pan)
    x0, y0, x1, y1 = p.grid_bounds_px()
    assert 0 <= x0 and 0 <= y0
    assert x1 < size[0] and y1 < size[1]


def test_grid_bounds_span_the_full_diamond():
    p, _ = Projection.centred(44, 32, margin=0)
    x0, y0, x1, y1 = p.grid_bounds_px()
    assert x1 - x0 == 44 * 32          # widest point, west to east
    assert y1 - y0 == 44 * 24          # tallest point, north to south
