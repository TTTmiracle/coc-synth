"""YOLO export: stable class ids and a lossless box round trip."""
import pytest

from cocsynth.export_yolo import class_names, class_of, label_to_lines, line_to_bbox


def test_class_ids_follow_the_catalog_order(catalog):
    names = class_names(catalog, "ignore")
    assert names == catalog.type_names
    assert names == sorted(names)


def test_split_mode_expands_only_directional_types(catalog):
    plain = class_names(catalog, "ignore")
    split = class_names(catalog, "split")
    # air_sweeper (8 facings) replaces 1 class with 8, so +7.
    assert len(split) == len(plain) + 7
    assert [n for n in split if n.startswith("air_sweeper")] == [f"air_sweeper_dir{d}" for d in range(8)]
    assert "cannon" in split and "cannon_dir0" not in split


def test_boxes_round_trip_within_rounding(sample_label, catalog):
    label, _, _ = sample_label
    index = {n: i for i, n in enumerate(class_names(catalog, "ignore"))}
    lines = label_to_lines(label, index)
    assert len(lines) == len(label.buildings)

    for b, line in zip(label.buildings, lines):
        cls, (x, y, w, h) = line_to_bbox(line, label.image_size)
        assert cls == index[b.type]
        ox, oy, ow, oh = b.bbox_px
        assert abs(x - ox) < 1.0 and abs(y - oy) < 1.0
        assert abs(w - ow) < 1.0 and abs(h - oh) < 1.0


def test_split_mode_encodes_the_facing(sample_label, catalog):
    label, _, _ = sample_label
    names = class_names(catalog, "split")
    index = {n: i for i, n in enumerate(names)}
    for b in label.buildings:
        cls = class_of(b.type, b.direction, index, "split")
        if catalog[b.type].is_directional:
            assert names[cls] == f"{b.type}_dir{b.direction}"
        else:
            assert names[cls] == b.type


def test_visibility_filter_drops_occluded_boxes(sample_label, catalog):
    label, _, _ = sample_label
    index = {n: i for i, n in enumerate(class_names(catalog, "ignore"))}
    assert len(label_to_lines(label, index, min_visibility=0.9)) < len(label_to_lines(label, index))


def test_normalised_coordinates_stay_in_range(sample_label, catalog):
    label, _, _ = sample_label
    index = {n: i for i, n in enumerate(class_names(catalog, "ignore"))}
    for line in label_to_lines(label, index):
        cx, cy, w, h = (float(v) for v in line.split()[1:5])
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_unknown_direction_mode_is_rejected(catalog):
    with pytest.raises(ValueError, match="direction_mode"):
        class_names(catalog, "sideways")
