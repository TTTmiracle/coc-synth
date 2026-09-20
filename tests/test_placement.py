"""Placement legality -- the guarantee the whole project rests on."""
from collections import Counter
from random import Random

import numpy as np
import pytest

from cocsynth.placement import Placer

BASES_PER_TH = 25  # 9 Town Halls x 25 = 225 random bases


def _bases(placer, th, n=BASES_PER_TH):
    for s in range(n):
        yield placer.generate(th, Random(th * 100_000 + s))


@pytest.mark.parametrize("th", range(1, 10))
def test_no_overlaps_and_everything_in_bounds(placer, catalog, th):
    for result in _bases(placer, th):
        occ = result.occupancy
        assert occ.shape == (catalog.grid_tiles, catalog.grid_tiles)
        # Each instance owns exactly its own footprint: total occupied cells must
        # equal the sum of areas, which is only true if nothing overlapped.
        assert int((occ > 0).sum()) == sum(p.area for p in result.placements)
        for p in result.placements:
            tx, ty = p.tile
            w, h = p.footprint
            assert 0 <= tx and 0 <= ty
            assert tx + w <= catalog.grid_tiles and ty + h <= catalog.grid_tiles
            assert np.all(occ[ty : ty + h, tx : tx + w] == p.instance_id)


@pytest.mark.parametrize("th", range(1, 10))
def test_counts_never_exceed_the_cap(placer, catalog, th):
    """No Eagle Artillery at TH8, and never two of a singleton."""
    for result in _bases(placer, th):
        for type_id, n in result.counts().items():
            rule = catalog.rule(type_id, th)
            assert rule is not None, f"{type_id} is not unlocked at TH{th} but was placed"
            assert n <= rule.count, f"TH{th}: {n} x {type_id} exceeds cap of {rule.count}"


@pytest.mark.parametrize("th", range(1, 10))
def test_levels_stay_within_the_town_hall_cap(placer, catalog, th):
    for result in _bases(placer, th):
        for p in result.placements:
            cap = catalog.rule(p.type_id, th).max_level
            assert 1 <= p.level <= cap, f"{p.type_id} level {p.level} over TH{th} cap {cap}"


@pytest.mark.parametrize("th", range(1, 10))
def test_exactly_one_town_hall(placer, th):
    for result in _bases(placer, th, n=8):
        assert result.counts().get("town_hall") == 1


def test_instance_ids_are_unique(placer):
    for result in _bases(placer, 9, n=8):
        ids = [p.instance_id for p in result.placements]
        assert len(set(ids)) == len(ids)
        assert min(ids) >= 1  # 0 means "free" in the occupancy grid


def test_exact_count_mode_hits_the_cap(catalog):
    placer = Placer(catalog, count_mode="exact")
    result = placer.generate(9, Random(3))
    for type_id, n in result.counts().items():
        assert n == catalog.rule(type_id, 9).count


def test_jitter_mode_stays_within_its_floor(catalog):
    placer = Placer(catalog, count_mode="jitter")
    seen = Counter()
    for s in range(20):
        for type_id, n in placer.generate(9, Random(s)).counts().items():
            cap = catalog.rule(type_id, 9).count
            assert 0 < n <= cap
            seen[type_id] = max(seen[type_id], n)
    # Jitter must still be able to reach the cap, or the dataset is permanently thin.
    assert seen["cannon"] == catalog.rule("cannon", 9).count


def test_maxed_level_policy(catalog):
    placer = Placer(catalog, level_policy="maxed")
    for p in placer.generate(9, Random(1)).placements:
        assert p.level == catalog.rule(p.type_id, 9).max_level


@pytest.mark.parametrize("style", ["scatter", "clustered", "compartment"])
def test_every_layout_style_produces_legal_bases(catalog, style):
    placer = Placer(catalog, layout_style=style)
    result = placer.generate(9, Random(11))
    assert int((result.occupancy > 0).sum()) == sum(p.area for p in result.placements)
    assert result.counts().get("town_hall") == 1


def test_seed_determines_the_base(placer):
    a = placer.generate(8, Random(99))
    b = placer.generate(8, Random(99))
    assert [(p.type_id, p.tile, p.level, p.direction) for p in a.placements] == \
           [(p.type_id, p.tile, p.level, p.direction) for p in b.placements]


# ---- rotation --------------------------------------------------------------

@pytest.mark.parametrize("th", range(1, 10))
def test_fixed_buildings_never_rotate(placer, catalog, th):
    for result in _bases(placer, th, n=6):
        for p in result.placements:
            if not catalog[p.type_id].is_directional:
                assert p.direction == 0, f"{p.type_id} is fixed but got direction {p.direction}"


def test_directional_buildings_stay_in_range(placer, catalog):
    for result in _bases(placer, 9, n=20):
        for p in result.placements:
            assert 0 <= p.direction < catalog[p.type_id].directions


def test_facings_are_uniformly_distributed(placer, catalog):
    """A collapsed direction sampler would show up here as a lopsided histogram."""
    hist = Counter()
    for s in range(200):
        for p in placer.generate(9, Random(s)).placements:
            if p.type_id == "air_sweeper":
                hist[p.direction] += 1
    assert set(hist) == set(range(8)), f"some facings never occurred: {sorted(hist)}"
    expected = sum(hist.values()) / 8
    assert min(hist.values()) > expected * 0.5
    assert max(hist.values()) < expected * 1.5


def test_rotation_does_not_change_the_footprint(placer, catalog):
    """Rotation changes the sprite and firing arc in Clash, not the tiles used."""
    for result in _bases(placer, 9, n=10):
        for p in result.placements:
            assert p.footprint == catalog[p.type_id].footprint


def test_wall_connection_masks_match_neighbours(placer, catalog):
    """Each wall's mask must name exactly the sides that have a wall next to them."""
    from cocsynth.schema import CONNECT_E, CONNECT_N, CONNECT_S, CONNECT_W

    result = placer.generate(9, Random(21))
    tiles = {p.tile for p in result.placements if p.type_id == "wall"}
    for p in result.placements:
        if p.type_id != "wall":
            continue
        x, y = p.tile
        expected = 0
        for bit, (dx, dy) in ((CONNECT_N, (0, -1)), (CONNECT_E, (1, 0)),
                              (CONNECT_S, (0, 1)), (CONNECT_W, (-1, 0))):
            if (x + dx, y + dy) in tiles:
                expected |= bit
        assert p.connections == expected, f"wall at {p.tile}: {p.connections} != {expected}"


def test_non_walls_have_no_connections(placer, catalog):
    for p in placer.generate(9, Random(22)).placements:
        if p.type_id != "wall":
            assert p.connections == 0


def test_long_wall_runs_are_detected(placer):
    """Perimeters should mostly be straight runs, not isolated posts -- if nearly
    everything came back as mask 0 the wall router is scattering, not building."""
    result = placer.generate(9, Random(23))
    masks = [p.connections for p in result.placements if p.type_id == "wall"]
    straight = sum(1 for m in masks if m in (5, 10))
    assert straight > len(masks) * 0.5, f"only {straight}/{len(masks)} walls are in runs"


def test_defences_end_up_inside_the_walls(catalog):
    """At TH8 and above the wall budget can enclose the whole defensive roster, so
    it must. Defences left out in the grass is the single most obvious way for a
    generated layout to read as fake -- nobody builds a base like that."""
    from cocsynth.placement import wall_priority

    for th in (8, 9):
        for seed in range(4):
            result = Placer(catalog).generate(th, Random(seed))
            walls = [p.tile for p in result.placements if p.type_id == "wall"]
            x0, x1 = min(t[0] for t in walls), max(t[0] for t in walls)
            y0, y1 = min(t[1] for t in walls), max(t[1] for t in walls)
            for p in result.placements:
                if p.type_id == "wall":
                    continue
                if wall_priority(catalog.buildings[p.type_id]) != 0:
                    continue
                w, h = p.footprint
                assert x0 <= p.tile[0] and p.tile[0] + w - 1 <= x1, \
                    f"TH{th} seed {seed}: {p.type_id} is outside the walls"
                assert y0 <= p.tile[1] and p.tile[1] + h - 1 <= y1, \
                    f"TH{th} seed {seed}: {p.type_id} is outside the walls"


def test_collectors_are_the_ones_left_outside(catalog):
    """The flip side: mines, collectors and camps should mostly sit in the grass.
    If everything ends up inside, the priority ordering has stopped doing anything
    and low Town Halls will overflow their walls at random instead."""
    from cocsynth.placement import wall_priority

    outside = total = 0
    for seed in range(6):
        result = Placer(catalog).generate(9, Random(seed))
        walls = [p.tile for p in result.placements if p.type_id == "wall"]
        x0, x1 = min(t[0] for t in walls), max(t[0] for t in walls)
        y0, y1 = min(t[1] for t in walls), max(t[1] for t in walls)
        for p in result.placements:
            if p.type_id == "wall" or wall_priority(catalog.buildings[p.type_id]) != 2:
                continue
            total += 1
            w, h = p.footprint
            if not (x0 <= p.tile[0] and p.tile[0] + w - 1 <= x1
                    and y0 <= p.tile[1] and p.tile[1] + h - 1 <= y1):
                outside += 1
    assert total > 20
    assert outside / total > 0.15, "nothing is being left outside the walls any more"


def test_wall_levels_track_the_town_hall(catalog):
    """Wooden level 1 walls around a maxed TH9 is the sort of thing a player spots
    instantly. Every segment should sit within two levels of the cap."""
    for th in (5, 7, 9):
        cap = catalog.rule("wall", th).max_level
        for seed in range(4):
            result = Placer(catalog).generate(th, Random(seed))
            levels = {p.level for p in result.placements if p.type_id == "wall"}
            assert levels, f"TH{th} produced no walls"
            assert min(levels) >= max(1, cap - 2), \
                f"TH{th} seed {seed}: wall levels {sorted(levels)} against cap {cap}"
            assert max(levels) <= cap


def test_walls_form_runs_rather_than_scattered_posts(catalog):
    """Clash walls are laid as compartments. If most segments have no neighbour the
    autotile mask is meaningless and the render is back to a field of fence posts."""
    result = Placer(catalog).generate(9, Random(4))
    walls = [p for p in result.placements if p.type_id == "wall"]
    lonely = [p for p in walls if p.connections == 0]
    assert len(lonely) / len(walls) < 0.05


def test_the_exhaustive_fallback_finds_a_gap_random_probing_would_miss(catalog):
    """A full grid with one hole left: random probing can miss it, the scan cannot."""
    placer = Placer(catalog)
    occ = np.ones((placer.tiles, placer.tiles), dtype=np.uint16)
    occ[7:10, 20:23] = 0
    assert placer._find_spot((3, 3), occ, Random(1)) == (20, 7)
    occ[7:10, 20:23] = 1
    assert placer._find_spot((3, 3), occ, Random(1)) is None


def test_the_town_hall_is_always_at_its_town_hall_level(catalog):
    """A TH9 base has a level 9 Town Hall -- that is what the number means. Rolling
    it like any other building produced TH9 bases wearing level 4 Town Hall art,
    which makes every TH9-only building beside it look illegal."""
    for th in range(1, catalog.max_town_hall + 1):
        for seed in range(5):
            result = Placer(catalog).generate(th, Random(seed))
            halls = [p for p in result.placements if p.type_id == "town_hall"]
            assert len(halls) == 1
            assert halls[0].level == th, \
                f"TH{th} seed {seed} rendered a level {halls[0].level} Town Hall"
