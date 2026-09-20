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
