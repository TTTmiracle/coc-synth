"""Filing a downloaded sprite collection into the folder contract."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.import_sprites import dedupe, discover  # noqa: E402

from cocsynth.placeholders import make_sprite  # noqa: E402


def _tree(root: Path, group: str, building: str, n: int, art_every: int = 1):
    d = root / group / building / "normal"
    d.mkdir(parents=True, exist_ok=True)
    for lv in range(1, n + 1):
        art = ((lv - 1) // art_every) + 1
        make_sprite(building.replace("-", "_"), "defense", (3, 3), art, 32)[0].save(d / f"level-{lv}.png")
    return d


def test_discovers_grouped_and_ungrouped_layouts(tmp_path):
    _tree(tmp_path, "defenses", "cannon", 3)
    # Town Hall sits one level shallower in the published tree.
    d = tmp_path / "town-hall" / "normal"
    d.mkdir(parents=True)
    make_sprite("town_hall", "town-hall", (4, 4), 1, 32)[0].save(d / "level-1.png")

    found = discover(tmp_path, "normal")
    assert set(found) == {"cannon", "town_hall"}
    assert sorted(found["cannon"]) == [1, 2, 3]


def test_hyphens_become_underscores(tmp_path):
    _tree(tmp_path, "defenses", "air-sweeper", 2)
    assert "air_sweeper" in discover(tmp_path, "normal")


def test_other_variants_are_ignored(tmp_path):
    _tree(tmp_path, "defenses", "cannon", 2)
    geared = tmp_path / "defenses" / "cannon" / "geared-up-burst"
    geared.mkdir(parents=True)
    make_sprite("cannon", "defense", (3, 3), 9, 32)[0].save(geared / "level-1.png")
    assert sorted(discover(tmp_path, "normal")["cannon"]) == [1, 2]


def test_dedupe_keeps_only_levels_where_the_art_changes(tmp_path):
    d = _tree(tmp_path, "defenses", "cannon", 12, art_every=3)
    levels = {lv: d / f"level-{lv}.png" for lv in range(1, 13)}
    assert sorted(dedupe(levels)) == [1, 4, 7, 10]


def test_dedupe_keeps_everything_when_every_level_differs(tmp_path):
    d = _tree(tmp_path, "defenses", "cannon", 5, art_every=1)
    levels = {lv: d / f"level-{lv}.png" for lv in range(1, 6)}
    assert sorted(dedupe(levels)) == [1, 2, 3, 4, 5]


def test_deduped_set_still_covers_every_level_via_fallback(tmp_path, catalog):
    """Dropping duplicates is only safe because lookup resolves downward."""
    from cocsynth.assets import SpriteLibrary
    d = _tree(tmp_path, "defenses", "cannon", 12, art_every=3)
    kept = dedupe({lv: d / f"level-{lv}.png" for lv in range(1, 13)})

    out = tmp_path / "sprites" / "cannon"
    out.mkdir(parents=True)
    for lv, p in kept.items():
        (out / f"cannon_lvl{lv:02d}.png").write_bytes(p.read_bytes())

    lib = SpriteLibrary(tmp_path / "sprites", tile_w=32)
    for lv in range(1, 13):
        assert not lib.get(catalog["cannon"], lv, 0).is_placeholder, f"level {lv} lost its art"
