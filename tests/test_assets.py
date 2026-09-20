"""Sprite resolution: the fallback chain, and the facing-collapse trap."""
from pathlib import Path

import pytest
from PIL import Image

from cocsynth.assets import SpriteLibrary, trim_alpha
from cocsynth.placeholders import make_sprite


@pytest.fixture
def sprite_dir(tmp_path, catalog):
    """A sprite folder with full Cannon art at two levels and all 8 sweeper facings."""
    for level in (1, 7):
        d = tmp_path / "cannon"
        d.mkdir(exist_ok=True)
        make_sprite("cannon", "defense", (3, 3), level, 32)[0].save(d / f"cannon_lvl{level:02d}.png")
    d = tmp_path / "air_sweeper"
    d.mkdir(exist_ok=True)
    for direction in range(8):
        make_sprite("air_sweeper", "defense", (2, 2), 2, 32, direction, 8)[0].save(
            d / f"air_sweeper_lvl02_dir{direction}.png")
    return tmp_path


def test_trim_removes_transparent_margins():
    inner, _ = make_sprite("cannon", "defense", (3, 3), 5, 32)
    padded = Image.new("RGBA", (inner.width + 40, inner.height + 40), (0, 0, 0, 0))
    padded.paste(inner, (20, 20))
    trimmed, offset = trim_alpha(padded)
    assert trimmed.size == inner.size
    assert offset == (20, 20)


def test_level_falls_back_to_the_highest_available_below(sprite_dir, catalog):
    lib = SpriteLibrary(sprite_dir, tile_w=32)
    assert lib.get(catalog["cannon"], 1, 0).name == "cannon_lvl01.png"
    assert lib.get(catalog["cannon"], 6, 0).name == "cannon_lvl01.png"
    assert lib.get(catalog["cannon"], 7, 0).name == "cannon_lvl07.png"
    assert lib.get(catalog["cannon"], 11, 0).name == "cannon_lvl07.png"


def test_full_directional_art_resolves_to_the_requested_facing(sprite_dir, catalog):
    """The corruption this guards against: label says dir5, image draws dir0."""
    lib = SpriteLibrary(sprite_dir, tile_w=32)
    for direction in range(8):
        assert lib.get(catalog["air_sweeper"], 2, direction).name == \
            f"air_sweeper_lvl02_dir{direction}.png"
    assert lib.directional_gaps(catalog) == {}


def test_partial_directional_art_is_reported_as_a_gap(tmp_path, catalog):
    d = tmp_path / "air_sweeper"
    d.mkdir(parents=True)
    for direction in (0, 1):
        make_sprite("air_sweeper", "defense", (2, 2), 2, 32, direction, 8)[0].save(
            d / f"air_sweeper_lvl02_dir{direction}.png")
    gaps = SpriteLibrary(tmp_path, tile_w=32).directional_gaps(catalog)
    assert gaps == {"air_sweeper": [2, 3, 4, 5, 6, 7]}


def test_mirror_entry_satisfies_a_missing_facing(tmp_path, catalog):
    import json
    d = tmp_path / "air_sweeper"
    d.mkdir(parents=True)
    make_sprite("air_sweeper", "defense", (2, 2), 2, 32, 3, 8)[0].save(d / "air_sweeper_lvl02_dir3.png")
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"sprites": {"air_sweeper": {"mirror_of": {str(d): 3 for d in (0, 1, 2, 4, 5, 6, 7)}}}}))
    lib = SpriteLibrary(tmp_path, tile_w=32)
    assert "mirrored" in lib.get(catalog["air_sweeper"], 2, 5).name
    assert lib.directional_gaps(catalog) == {}


def test_missing_art_falls_back_to_a_placeholder(catalog):
    lib = SpriteLibrary(Path("/nonexistent"), tile_w=32)
    sprite = lib.get(catalog["mortar"], 4, 0)
    assert sprite.is_placeholder
    assert sprite.name == "placeholder:mortar_lvl04"
    # Placeholders render every facing, so they are never a gap.
    assert lib.directional_gaps(catalog) == {}


def test_anchor_defaults_to_bottom_centre(sprite_dir, catalog):
    sprite = SpriteLibrary(sprite_dir, tile_w=32).get(catalog["cannon"], 7, 0)
    assert sprite.anchor == (sprite.image.width // 2, sprite.image.height - 1)


def test_manifest_anchor_overrides_the_default(sprite_dir, catalog):
    import json
    (sprite_dir / "manifest.json").write_text(json.dumps(
        {"sprites": {"cannon": {"anchors": {"cannon_lvl07.png": [5, 9]}}}}))
    assert SpriteLibrary(sprite_dir, tile_w=32).get(catalog["cannon"], 7, 0).anchor == (5, 9)
