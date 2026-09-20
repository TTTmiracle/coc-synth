"""Rendering, measured labels, and reproducibility."""
import json
from random import Random

import pytest

from cocsynth.labels import build_label, read_label, write_label
from cocsynth.schema import HEADINGS, heading_of, rotation_deg_of


def test_label_validates_and_round_trips(sample_label, tmp_path):
    label, _, _ = sample_label
    path = tmp_path / "label.json"
    write_label(label, path)
    assert read_label(path).model_dump() == label.model_dump()


def test_every_box_lies_inside_the_canvas(sample_label):
    label, _, _ = sample_label
    iw, ih = label.image_size
    for b in label.buildings:
        x, y, w, h = b.bbox_px
        assert w > 0 and h > 0
        assert 0 <= x and 0 <= y and x + w <= iw and y + h <= ih


def test_box_and_footprint_agree(sample_label):
    """An anchor bug moves the sprite off its tile while the JSON stays consistent.

    The two label fields are measured independently -- the box from rendered alpha,
    the polygon from the projection -- so requiring them to line up catches it.

    Only unoccluded instances qualify. When a nearer building hides most of a wall,
    the few visible pixels are its roof, which genuinely sits *above* its own
    footprint; that is correct output, not a misplaced sprite.
    """
    label, _, _ = sample_label
    checked = 0
    for b in label.buildings:
        if b.visibility < 0.95:
            continue
        checked += 1
        bx, by, bw, bh = b.bbox_px
        xs = [p[0] for p in b.footprint_polygon_px]
        ys = [p[1] for p in b.footprint_polygon_px]
        assert bx < max(xs) and bx + bw > min(xs), f"{b.type} box and footprint miss horizontally"
        assert by < max(ys) and by + bh > min(ys), f"{b.type} box and footprint miss vertically"
        # A sprite standing on its footprint is horizontally centred over it.
        assert min(xs) <= bx + bw / 2 <= max(xs), f"{b.type} is not centred on its tile"
    assert checked > 20, f"only {checked} unoccluded instances -- test is not proving much"


def test_occluded_boxes_still_sit_within_the_sprite_column(sample_label):
    """Partly hidden buildings get a looser but still meaningful check: whatever is
    visible must fall inside the horizontal span the whole sprite could occupy."""
    label, _, _ = sample_label
    iw, _ = label.image_size
    for b in label.buildings:
        if b.visibility >= 0.95:
            continue
        bx, by, bw, bh = b.bbox_px
        xs = [p[0] for p in b.footprint_polygon_px]
        span = max(xs) - min(xs)
        # Allow a full footprint width of overhang either side for tall art.
        assert bx + bw > min(xs) - span and bx < max(xs) + span, \
            f"{b.type} visible pixels are nowhere near its tile"
        assert 0 <= bx and bx + bw <= iw


def test_visibility_is_a_sane_fraction(sample_label):
    label, _, _ = sample_label
    for b in label.buildings:
        assert 0.0 < b.visibility <= 1.0
    assert any(b.visibility > 0.99 for b in label.buildings), "nothing is fully visible"


def test_hidden_buildings_are_left_out_of_the_label(sample_label):
    """Occluded instances are dropped rather than labelled -- asking a model to
    detect what it cannot see teaches it to hallucinate."""
    label, placement, rendered = sample_label
    assert len(label.buildings) == len(placement.placements) - len(rendered.dropped)
    labelled = {b.id for b in label.buildings}
    assert labelled.isdisjoint(set(rendered.dropped))


def test_labels_only_contain_legal_buildings(sample_label, catalog):
    label, _, _ = sample_label
    th = label.town_hall_level
    for type_id, n in label.counts().items():
        rule = catalog.rule(type_id, th)
        assert rule is not None
        assert n <= rule.count


def test_rendering_is_reproducible(catalog, placer, renderer):
    seed = 4242
    out = []
    for _ in range(2):
        placement = placer.generate(9, Random(seed))
        rendered = renderer.render(placement.placements, Random(seed))
        label = build_label("x.png", seed, placement, rendered, catalog)
        out.append((rendered.image.tobytes(), json.dumps(label.model_dump(mode="json"), sort_keys=True)))
    assert out[0][0] == out[1][0], "same seed produced different pixels"
    assert out[0][1] == out[1][1], "same seed produced a different label"


# ---- rotation in the label -------------------------------------------------

def test_fixed_buildings_report_north(sample_label, catalog):
    label, _, _ = sample_label
    for b in label.buildings:
        if not catalog[b.type].is_directional:
            assert (b.direction, b.heading, b.rotation_deg) == (0, "N", 0)


def test_heading_and_degrees_agree_with_direction(sample_label, catalog):
    label, _, _ = sample_label
    for b in label.buildings:
        directions = catalog[b.type].directions
        assert b.heading == heading_of(b.direction, directions)
        assert b.rotation_deg == rotation_deg_of(b.direction, directions)
        assert b.heading in HEADINGS


@pytest.mark.parametrize("direction", range(8))
def test_each_facing_renders_and_labels_consistently(catalog, placer, renderer, direction):
    """Force one facing and check it survives all the way into the label."""
    from cocsynth.placement import Placement

    p = Placement(1, "air_sweeper", 2, (20, 20), (2, 2), direction, 8)
    rendered = renderer.render([p], Random(1))
    label = build_label("x.png", 1, _fake_result(p), rendered, catalog)
    assert len(label.buildings) == 1
    b = label.buildings[0]
    assert b.direction == direction
    assert b.heading == HEADINGS[direction]
    assert b.rotation_deg == direction * 45


def _fake_result(*placements):
    from cocsynth.placement import PlacementResult
    import numpy as np
    return PlacementResult(9, list(placements), np.zeros((44, 44), dtype=np.uint16))
