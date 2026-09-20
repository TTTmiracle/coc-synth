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


# ---- animation frames ------------------------------------------------------

def _write_frames(root, n_frames, level=7):
    d = root / "cannon"
    d.mkdir(parents=True, exist_ok=True)
    for f in range(n_frames):
        make_sprite("cannon", "defense", (3, 3), level + f, 32)[0].save(
            d / f"cannon_lvl{level:02d}_f{f:02d}.png")
    return root


def test_frames_are_selected_round_robin(tmp_path, catalog):
    """A rolled frame maps onto however many frames actually exist."""
    lib = SpriteLibrary(_write_frames(tmp_path, 3), tile_w=32)
    picked = {lib.get(catalog["cannon"], 7, 0, f).name.split(" ")[0] for f in range(24)}
    assert picked == {f"cannon_lvl07_f{f:02d}.png" for f in range(3)}


def test_a_single_frame_is_reused_without_error(tmp_path, catalog):
    """One frame on disk must not break anything -- just less variety."""
    lib = SpriteLibrary(_write_frames(tmp_path, 1), tile_w=32)
    names = {lib.get(catalog["cannon"], 7, 0, f).name.split(" ")[0] for f in range(16)}
    assert names == {"cannon_lvl07_f00.png"}


def test_frame_counts_reported(tmp_path, catalog):
    assert SpriteLibrary(_write_frames(tmp_path, 4), tile_w=32).frame_counts()["cannon"] == 4


def test_instances_vary_even_with_one_sprite(tmp_path, catalog):
    """Two identical buildings in one base must not be pixel-identical."""
    lib = SpriteLibrary(_write_frames(tmp_path, 1), tile_w=32)
    variants = {lib.get(catalog["cannon"], 7, 0, f).image.tobytes() for f in range(8)}
    assert len(variants) > 1, "per-instance variation produced identical sprites"


def test_variation_keeps_the_anchor_proportional(tmp_path, catalog):
    lib = SpriteLibrary(_write_frames(tmp_path, 1), tile_w=32)
    for f in range(8):
        s = lib.get(catalog["cannon"], 7, 0, f)
        ax, ay = s.anchor
        assert 0 <= ax < s.image.width and 0 <= ay < s.image.height
        # Anchor stays at the bottom-centre of whatever size the sprite became.
        assert abs(ax - s.image.width // 2) <= 1
        assert s.image.height - ay <= 2
