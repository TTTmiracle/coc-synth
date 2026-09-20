"""End-to-end: the CLI produces a usable dataset on disk."""
import json

import pytest

from cocsynth.generate import derive_seed, main, parse_th_range


def test_seed_derivation_is_stable_and_independent():
    assert derive_seed(42, 7) == derive_seed(42, 7)
    assert derive_seed(42, 7) != derive_seed(42, 8)
    assert derive_seed(43, 7) != derive_seed(42, 7)


def test_town_hall_range_parsing(catalog):
    assert parse_th_range("9", catalog) == [9]
    assert parse_th_range("1-9", catalog) == list(range(1, 10))
    assert parse_th_range("3,5,8", catalog) == [3, 5, 8]
    with pytest.raises(SystemExit):
        parse_th_range("99", catalog)


def test_generates_images_labels_and_yolo(tmp_path):
    out = tmp_path / "ds"
    assert main(["--th", "9", "--n", "2", "--out", str(out), "--seed", "1",
                 "--formats", "json,yolo", "--overlay", "-q"]) == 0

    images = sorted((out / "images").glob("*.png"))
    labels = sorted((out / "labels").glob("*.json"))
    overlays = sorted((out / "overlays").glob("*.png"))
    yolo = sorted((out / "yolo" / "labels").glob("*.txt"))
    assert len(images) == len(labels) == len(overlays) == len(yolo) == 2
    assert (out / "yolo" / "data.yaml").exists()

    manifest = json.loads((out / "dataset.json").read_text())
    assert manifest["images"] == 2
    assert manifest["boxes"] > 0
    assert manifest["yolo"]["files"] == 2


def test_mixed_town_hall_range_covers_every_level(tmp_path):
    out = tmp_path / "mixed"
    assert main(["--th", "1-9", "--n", "9", "--out", str(out), "--seed", "5", "-q"]) == 0
    stems = {p.stem.split("_")[0] for p in (out / "labels").glob("*.json")}
    assert stems == {f"th{i}" for i in range(1, 10)}


def test_only_flag_reproduces_one_image_byte_for_byte(tmp_path):
    full = tmp_path / "full"
    main(["--th", "8", "--n", "3", "--out", str(full), "--seed", "77", "-q"])
    one = tmp_path / "one"
    main(["--th", "8", "--n", "3", "--only", "1", "--out", str(one), "--seed", "77", "-q"])

    for sub, name in (("images", "th8_000001.png"), ("labels", "th8_000001.json")):
        assert (full / sub / name).read_bytes() == (one / sub / name).read_bytes()


def test_odd_tile_width_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="even"):
        main(["--th", "9", "--n", "1", "--out", str(tmp_path / "x"), "--tile-w", "33", "-q"])


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown format"):
        main(["--th", "9", "--n", "1", "--out", str(tmp_path / "x"), "--formats", "coco", "-q"])


def test_scale_jitter_varies_image_size(tmp_path):
    out = tmp_path / "jit"
    main(["--th", "5", "--n", "6", "--out", str(out), "--seed", "9", "--scale-jitter", "-q"])
    sizes = {tuple(json.loads(p.read_text())["image_size"]) for p in (out / "labels").glob("*.json")}
    assert len(sizes) > 1, "scale jitter produced a single image size"
