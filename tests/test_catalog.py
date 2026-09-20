"""The rules layer. If these fail, every generated image is wrong."""
import pytest

from cocsynth.catalog import Catalog, CatalogError


def test_loads_and_validates(catalog):
    assert catalog.grid_tiles == 44
    assert catalog.max_town_hall >= 9
    assert len(catalog.buildings) > 20


def test_town_hall_range_is_contiguous(catalog):
    """Once a building unlocks it never disappears at a higher Town Hall."""
    for b in catalog.buildings.values():
        ths = sorted(b.per_th)
        assert ths == list(range(ths[0], ths[-1] + 1)), f"{b.id} has gaps: {ths}"
        assert ths[-1] == catalog.max_town_hall, f"{b.id} vanishes before the top TH"


def test_counts_and_levels_never_regress(catalog):
    for b in catalog.buildings.values():
        ths = sorted(b.per_th)
        for prev, cur in zip(ths, ths[1:]):
            assert b.per_th[cur].count >= b.per_th[prev].count, f"{b.id} count drops at TH{cur}"
            assert b.per_th[cur].max_level >= b.per_th[prev].max_level, f"{b.id} max_level drops at TH{cur}"


def test_late_game_buildings_absent_from_low_town_halls(catalog):
    """The Eagle Artillery case, asserted directly.

    Eagle Artillery is TH11+ and Inferno Tower TH10+, so in a TH1-9 catalog they
    must not exist at all -- not merely be capped at zero.
    """
    for late in ("eagle_artillery", "inferno_tower", "scattershot", "monolith", "firespitter"):
        assert late not in catalog, f"{late} must not appear in a TH1-9 catalog"

    if catalog.max_town_hall == 9:
        for th in range(1, 10):
            ids = {b.id for b in catalog.roster(th)}
            assert "eagle_artillery" not in ids
            assert "inferno_tower" not in ids


def test_x_bow_only_at_town_hall_nine(catalog):
    """A building that unlocks at the very top of the range."""
    xbow = catalog["x_bow"]
    assert xbow.min_th == 9
    assert catalog.rule("x_bow", 8) is None
    assert catalog.rule("x_bow", 9).count == 2


def test_singletons_stay_singular(catalog):
    for type_id in ("town_hall", "clan_castle", "laboratory", "spell_factory"):
        for th in sorted(catalog[type_id].per_th):
            assert catalog.rule(type_id, th).count == 1, f"{type_id} is not a singleton at TH{th}"


def test_air_sweeper_is_the_directional_building(catalog):
    """Rotation is real, and driven by the catalog rather than hard-coded."""
    assert catalog["air_sweeper"].directions == 8
    assert catalog["air_sweeper"].is_directional
    directional = {b.id for b in catalog.buildings.values() if b.is_directional}
    assert directional == {"air_sweeper"}, f"unexpected directional set: {directional}"
    for b in catalog.buildings.values():
        assert 8 % b.directions == 0, f"{b.id}: directions must divide the 8 compass headings"


def test_every_roster_physically_fits(catalog):
    for th in range(1, catalog.max_town_hall + 1):
        assert catalog.roster_area(th) < catalog.grid_tiles**2 * 0.8


def test_class_ids_are_stable_and_sorted(catalog):
    names = catalog.type_names
    assert names == sorted(names)
    assert catalog.class_index()[names[0]] == 0
    assert len(set(names)) == len(names)


def test_validation_rejects_a_regressing_count(catalog, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "max_town_hall: 2\ngrid_tiles: 44\nbuildings:\n"
        "  cannon:\n    display_name: Cannon\n    category: defense\n"
        "    footprint: [3, 3]\n    directions: 1\n    per_th:\n"
        "      1: { count: 5, max_level: 1 }\n      2: { count: 2, max_level: 1 }\n"
    )
    with pytest.raises(CatalogError, match="count drops"):
        Catalog.load(bad)


def test_validation_rejects_a_gap_in_the_town_hall_range(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "max_town_hall: 3\ngrid_tiles: 44\nbuildings:\n"
        "  cannon:\n    display_name: Cannon\n    category: defense\n"
        "    footprint: [3, 3]\n    directions: 1\n    per_th:\n"
        "      1: { count: 1, max_level: 1 }\n      3: { count: 1, max_level: 1 }\n"
    )
    with pytest.raises(CatalogError, match="gaps"):
        Catalog.load(bad)
