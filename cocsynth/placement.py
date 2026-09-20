"""Legal random placement of a Town Hall roster onto the 44x44 grid.

Two separate guarantees, and it is worth keeping them distinct:

* **Legality** is non-negotiable. Counts and levels come from the catalog, so a base
  can never hold an Eagle Artillery at TH8 or two of a singleton. Nothing here can
  exceed a cap -- the work list is built *from* the caps.
* **Realism** is a knob. Positions are random within a chosen layout style. Positional
  diversity is good for a detector, so this is deliberately looser than legality.

Occupancy is a tiles x tiles array of instance ids (0 = free), which doubles as the
overlap check and as a debugging view of the base.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from random import Random

import numpy as np

from .schema import CONNECT_E, CONNECT_N, CONNECT_S, CONNECT_W

from .catalog import BuildingDef, Catalog

LayoutStyle = str  # "scatter" | "clustered" | "compartment"

#: What a player protects first when there are not enough walls to cover
#: everything. Defences, heroes and the storages holding the loot go inside;
#: collectors, camps and builder huts live in the grass outside, where losing
#: them costs least. Below TH7 the wall budget cannot enclose the whole roster,
#: so this ordering is the difference between a base and a scattered pile.
_ALWAYS_INSIDE = {"town_hall", "clan_castle", "gold_storage", "elixir_storage",
                  "dark_elixir_storage"}
_HAPPY_OUTSIDE = {"army_camp", "builders_hut", "barracks", "dark_barracks",
                  "gold_mine", "elixir_collector", "dark_elixir_drill"}
#: Chance a building of each tier is placed outside the walls even when there is
#: room inside, so the split never looks mechanical.
_OUTSIDE_CHANCE = (0.03, 0.12, 0.55)


def wall_priority(bdef: BuildingDef) -> int:
    """0 goes inside the walls first, 2 is content to sit outside."""
    if bdef.id in _ALWAYS_INSIDE or bdef.category in ("defense", "hero"):
        return 0
    if bdef.id in _HAPPY_OUTSIDE:
        return 2
    return 1
CountMode = str    # "exact" | "jitter"
LevelPolicy = str  # "maxed" | "clustered" | "uniform"

#: Fraction of a building's cap that `jitter` mode may drop to.
JITTER_FLOOR = 0.70

#: Animation frames are rolled from this range and mapped onto however many frames
#: actually exist on disk. Decoupling the two means placement stays reproducible and
#: identical whether or not the art has been added yet.
FRAME_SPACE = 256


@dataclass(frozen=True)
class Placement:
    """One building committed to a tile position."""

    instance_id: int
    type_id: str
    level: int
    tile: tuple[int, int]
    footprint: tuple[int, int]
    direction: int
    directions: int
    frame: int = 0
    connections: int = 0

    @property
    def area(self) -> int:
        return self.footprint[0] * self.footprint[1]


@dataclass
class PlacementResult:
    town_hall_level: int
    placements: list[Placement]
    occupancy: np.ndarray
    shortfall: dict[str, int] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.placements:
            out[p.type_id] = out.get(p.type_id, 0) + 1
        return out


class PlacementError(RuntimeError):
    """Raised when a base could not be filled anywhere near its roster."""


class Placer:
    """Builds legal random bases for a given catalog."""

    def __init__(
        self,
        catalog: Catalog,
        layout_style: LayoutStyle = "clustered",
        count_mode: CountMode = "jitter",
        level_policy: LevelPolicy = "clustered",
        max_attempts: int = 160,
        max_shortfall: float = 0.10,
        outside_wall_rate: float = 0.12,
        spacing_effort: float = 0.55,
    ):
        self.cat = catalog
        self.layout_style = layout_style
        self.count_mode = count_mode
        self.level_policy = level_policy
        self.max_attempts = max_attempts
        self.max_shortfall = max_shortfall
        self.outside_wall_rate = outside_wall_rate
        self.spacing_effort = spacing_effort
        self.tiles = catalog.grid_tiles

    # ---- public ----------------------------------------------------------

    def generate(self, th: int, rng: Random) -> PlacementResult:
        """Place a full legal roster for Town Hall level `th`."""
        occ = np.zeros((self.tiles, self.tiles), dtype=np.uint16)
        placements: list[Placement] = []
        shortfall: dict[str, int] = {}
        next_id = 1

        roster = [b for b in self.cat.roster(th) if b.category != "wall"]
        walls = [b for b in self.cat.roster(th) if b.category == "wall"]

        town_hall = next((b for b in roster if b.id == "town_hall"), None)
        if town_hall is not None:
            roster.remove(town_hall)

        # Biggest footprint first so large pieces get space before it fragments.
        # Ties are shuffled for variety.
        work = self._work_list(roster, th, rng)
        work.sort(key=lambda item: (wall_priority(item[0]), -item[0].area))

        # Walls go down before the buildings do. A player lays out compartments and
        # then fills them; walling in whatever happens to be left at the end gives
        # one outer ring and no interior, because the interior is already occupied.
        # Knowing the roster's total area up front is what lets the compartments be
        # sized to hold it.
        # Room needed, not footprint area: buildings prefer a clear tile around
        # them, so a 3x3 occupies closer to 4x4 of compartment. Sizing the ring off
        # the raw area makes it too small, the compartments fill, and the overflow
        # ends up outside the walls -- a base with a third of its defences in the
        # grass is the most obvious thing a generator can get wrong.
        spread = [(b.footprint[0] + 1) * (b.footprint[1] + 1) for b, _ in work]
        if town_hall is not None:
            spread.append((town_hall.footprint[0] + 1) * (town_hall.footprint[1] + 1))
        demand = sum(spread)
        cells: list[tuple[int, int, int, int]] = []
        for bdef in walls:
            next_id, cells = self._place_walls(
                bdef, th, occ, placements, next_id, rng, shortfall, demand)

        # Town Hall next: it anchors the base, as it does in a real layout.
        if town_hall is not None:
            next_id = self._place_town_hall(town_hall, th, occ, placements, next_id, rng)

        for bdef, level in work:
            pos = self._find_spot(bdef.footprint, occ, rng, cells, wall_priority(bdef))
            if pos is None:
                shortfall[bdef.id] = shortfall.get(bdef.id, 0) + 1
                continue
            next_id = self._commit(bdef, level, pos, occ, placements, next_id, rng)

        placements = self._link_walls(placements)
        self._check_shortfall(th, placements, shortfall)
        return PlacementResult(th, placements, occ, shortfall)

    @staticmethod
    def _link_walls(placements: list[Placement]) -> list[Placement]:
        """Work out which walls touch, so each can render its connection variant.

        Clash walls autotile: a segment in a straight run is a continuous bar, a
        corner turns, an endpoint caps off. Rendering every wall as the same isolated
        post turns a wall into a row of fence stakes, which is exactly what it looks
        like. The mask is computed once here, from final positions.
        """
        wall_tiles = {p.tile for p in placements if p.type_id == "wall"}
        if not wall_tiles:
            return placements

        out = []
        for p in placements:
            if p.type_id != "wall":
                out.append(p)
                continue
            x, y = p.tile
            mask = 0
            for bit, (dx, dy) in ((CONNECT_N, (0, -1)), (CONNECT_E, (1, 0)),
                                  (CONNECT_S, (0, 1)), (CONNECT_W, (-1, 0))):
                if (x + dx, y + dy) in wall_tiles:
                    mask |= bit
            out.append(replace(p, connections=mask))
        return out

    # ---- roster expansion ------------------------------------------------

    def _work_list(self, roster: list[BuildingDef], th: int, rng: Random) -> list[tuple[BuildingDef, int]]:
        """Expand per-type caps into one (building, level) item per instance."""
        items: list[tuple[BuildingDef, int]] = []
        for bdef in roster:
            for _ in range(self._instance_count(bdef, th, rng)):
                items.append((bdef, self._roll_level(bdef, th, rng)))
        rng.shuffle(items)
        return items

    def _instance_count(self, bdef: BuildingDef, th: int, rng: Random) -> int:
        """How many to build. Never above the cap, under either mode."""
        cap = bdef.per_th[th].count
        if self.count_mode == "exact":
            return cap
        if self.count_mode == "jitter":
            return rng.randint(max(1, math.ceil(cap * JITTER_FLOOR)), cap)
        raise ValueError(f"unknown count_mode {self.count_mode!r}")

    def _roll_level(self, bdef: BuildingDef, th: int, rng: Random) -> int:
        """Pick a level within this TH's cap."""
        cap = bdef.per_th[th].max_level
        if self.level_policy == "maxed" or cap == 1:
            return cap
        if self.level_policy == "uniform":
            return rng.randint(1, cap)
        if self.level_policy == "clustered":
            # Real bases sit near the cap, not spread evenly below it.
            return max(1, min(cap, int(round(rng.triangular(1, cap, cap)))))
        raise ValueError(f"unknown level_policy {self.level_policy!r}")

    def _roll_direction(self, bdef: BuildingDef, rng: Random) -> int:
        """Facing is independent of placement -- rotation changes the sprite and
        firing arc in Clash, not the tiles consumed."""
        return rng.randrange(bdef.directions) if bdef.is_directional else 0

    # ---- position search -------------------------------------------------

    def _fits(self, tx: int, ty: int, w: int, h: int, occ: np.ndarray,
              margin: int = 0) -> bool:
        """Is this footprint free? `margin` also requires a clear ring around it."""
        if tx < 0 or ty < 0 or tx + w > self.tiles or ty + h > self.tiles:
            return False
        if margin:
            mx0, my0 = max(0, tx - margin), max(0, ty - margin)
            mx1, my1 = min(self.tiles, tx + w + margin), min(self.tiles, ty + h + margin)
            return not occ[my0:my1, mx0:mx1].any()
        return not occ[ty : ty + h, tx : tx + w].any()

    def _find_spot(self, footprint: tuple[int, int], occ: np.ndarray, rng: Random,
                   cells: list[tuple[int, int, int, int]] | None = None,
                   priority: int = 1) -> tuple[int, int] | None:
        """Random legal origin for a footprint, or None after max_attempts.

        The first pass insists on a clear tile around the building. Players leave
        those gaps deliberately -- they break up Wall Breaker chains and leave room
        for traps -- and packing every building flush against its neighbour instead
        means each one buries the sprite behind it. Once the easy room is gone the
        pass is dropped and buildings are allowed to touch, which is also what a
        crowded corner of a real base looks like.
        """
        w, h = footprint
        for attempt in range(self.max_attempts):
            frac = attempt / self.max_attempts
            # Late attempts give up on both niceties in turn: first the clear ring,
            # then the compartments themselves. Without that second release a
            # building whose compartments are full never tries the open grid and is
            # dropped from the base entirely, which shows up as a missing defence.
            target = cells if frac < 0.7 else None
            margin = 1 if frac < self.spacing_effort else 0
            if target is None and frac >= 0.9:
                # Last resort: anywhere legal at all. The centre-weighted sampler
                # keeps aiming at the middle of the grid, which by now is solid
                # base, so a large building can burn every attempt there and be
                # dropped from the roster while the outskirts sit empty.
                tx = rng.randint(0, self.tiles - w)
                ty = rng.randint(0, self.tiles - h)
            else:
                tx, ty = self._candidate(w, h, rng, target, priority)
            if self._fits(tx, ty, w, h, occ, margin):
                return tx, ty
        return self._scan_spot(w, h, occ, rng)

    def _scan_spot(self, w: int, h: int, occ: np.ndarray,
                   rng: Random) -> tuple[int, int] | None:
        """Exhaustive fallback: every legal origin at once, pick one at random.

        Random probing is fine while the grid is empty and hopeless once it is not
        -- a 4x4 Army Camp placed last has to find one of a handful of surviving
        gaps, and a few hundred dice rolls usually will not. Missing it silently
        drops a building the roster says exists, so the label and the image
        disagree with the catalog. A summed-area table answers 'which origins are
        free' for the whole grid in one pass, which turns that into a certainty.
        """
        blocked = (occ != 0).astype(np.int32)
        table = np.pad(blocked.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        # Window sum over every h x w origin via the four corners of the table.
        sums = (table[h:, w:] - table[:-h, w:] - table[h:, :-w] + table[:-h, :-w])
        ys, xs = np.nonzero(sums == 0)
        if len(xs) == 0:
            return None
        i = rng.randrange(len(xs))
        return int(xs[i]), int(ys[i])

    def _compartment_candidate(
        self, w: int, h: int, rng: Random, cells: list[tuple[int, int, int, int]]
    ) -> tuple[int, int] | None:
        """Drop the building inside one of the walled compartments.

        Sampling the grid with a centre bias instead piles everything into the
        middle and leaves outer compartments as empty boxes, which is the one thing
        no real base has -- players fill a compartment before opening another.
        Choosing the compartment first, weighted by how much room it has, spreads
        the roster the way a player would.
        """
        room = [max(0, (c[2] - c[0] - 1) - w + 1) * max(0, (c[3] - c[1] - 1) - h + 1)
                for c in cells]
        if not any(room):
            return None
        cx0, cy0, cx1, cy1 = rng.choices(cells, weights=room, k=1)[0]
        return (rng.randint(cx0 + 1, cx1 - w), rng.randint(cy0 + 1, cy1 - h))

    def _outside_candidate(
        self, w: int, h: int, rng: Random, cells: list[tuple[int, int, int, int]]
    ) -> tuple[int, int] | None:
        """A spot in the grass just beyond the walls.

        Collectors and camps belong outside, but the centre-weighted sampler aims
        at the middle of the base, so "outside" has to be asked for explicitly or
        it never happens -- and a building that keeps aiming into solid base gets
        dropped from the roster entirely.
        """
        x0 = min(c[0] for c in cells)
        x1 = max(c[2] for c in cells)
        y0 = min(c[1] for c in cells)
        y1 = max(c[3] for c in cells)
        band = 6
        for _ in range(8):
            tx = rng.randint(max(0, x0 - band), min(self.tiles - w, x1 + band))
            ty = rng.randint(max(0, y0 - band), min(self.tiles - h, y1 + band))
            if tx + w - 1 < x0 or tx > x1 or ty + h - 1 < y0 or ty > y1:
                return tx, ty
        return None

    def _candidate(self, w: int, h: int, rng: Random,
                   cells: list[tuple[int, int, int, int]] | None = None,
                   priority: int = 1) -> tuple[int, int]:
        """Sample a candidate origin according to the layout style."""
        hi_x, hi_y = self.tiles - w, self.tiles - h
        # A handful of buildings always sit outside the walls in a real base --
        # collectors and army camps mostly -- so this is not an unconditional rule.
        if cells:
            if rng.random() > _OUTSIDE_CHANCE[priority] * self.outside_wall_rate / 0.12:
                spot = self._compartment_candidate(w, h, rng, cells)
                if spot is not None:
                    return spot
            else:
                spot = self._outside_candidate(w, h, rng, cells)
                if spot is not None:
                    return spot
        if self.layout_style == "scatter":
            return rng.randint(0, hi_x), rng.randint(0, hi_y)
        if self.layout_style in ("clustered", "compartment"):
            # Centre-weighted but spread across the walled area. Too tight and every
            # base becomes one dense clump in the middle, which no real layout is.
            sigma = self.tiles / (5.5 if self.layout_style == "compartment" else 4.0)
            c = self.tiles / 2
            tx = int(round(rng.gauss(c - w / 2, sigma)))
            ty = int(round(rng.gauss(c - h / 2, sigma)))
            return max(0, min(hi_x, tx)), max(0, min(hi_y, ty))
        raise ValueError(f"unknown layout_style {self.layout_style!r}")

    def _commit(
        self,
        bdef: BuildingDef,
        level: int,
        pos: tuple[int, int],
        occ: np.ndarray,
        out: list[Placement],
        next_id: int,
        rng: Random,
    ) -> int:
        tx, ty = pos
        w, h = bdef.footprint
        occ[ty : ty + h, tx : tx + w] = next_id
        out.append(
            Placement(
                instance_id=next_id,
                type_id=bdef.id,
                level=level,
                tile=(tx, ty),
                footprint=bdef.footprint,
                direction=self._roll_direction(bdef, rng),
                directions=bdef.directions,
                frame=rng.randrange(FRAME_SPACE),
            )
        )
        return next_id + 1

    def _place_town_hall(
        self, bdef: BuildingDef, th: int, occ: np.ndarray, out: list[Placement],
        next_id: int, rng: Random,
    ) -> int:
        w, h = bdef.footprint
        c = self.tiles // 2
        jitter = max(1, self.tiles // 10)
        for _ in range(self.max_attempts):
            tx = c - w // 2 + rng.randint(-jitter, jitter)
            ty = c - h // 2 + rng.randint(-jitter, jitter)
            if self._fits(tx, ty, w, h, occ):
                return self._commit(bdef, self._roll_level(bdef, th, rng), (tx, ty), occ, out, next_id, rng)
        # An empty grid always has room for a 4x4 at the centre.
        raise PlacementError("could not place the Town Hall on an empty grid")

    # ---- walls -----------------------------------------------------------

    def _place_walls(
        self, bdef: BuildingDef, th: int, occ: np.ndarray, out: list[Placement],
        next_id: int, rng: Random, shortfall: dict[str, int], demand: int,
    ) -> tuple[int, list[tuple[int, int, int, int]]]:
        """Lay the wall skeleton: an outer ring cut into compartments.

        `demand` is the total tile area the roster needs, which decides how big the
        ring has to be. Undersize it and the buildings spill outside the walls;
        oversize it and the base is a ring around empty grass. Neither looks like a
        base somebody actually built.
        """
        budget = self._instance_count(bdef, th, rng)
        placed = 0

        # Walls are the one thing a player upgrades as a block, and they track the
        # Town Hall closely -- wooden level 1 walls around a maxed TH9 is the sort
        # of thing anyone who plays would spot immediately. A base part way through
        # an upgrade has two adjacent levels at once, and the unfinished ones are
        # the inner runs, because the outer ring gets done first.
        cap = bdef.per_th[th].max_level
        if self.level_policy == "maxed":
            front, behind, share = cap, cap, 0.0
        else:
            front = max(1, cap - rng.choices((0, 1, 2), weights=(6, 3, 1))[0])
            behind = max(1, front - 1)
            share = rng.uniform(0.0, 0.45) if front > behind else 0.0

        skeleton, cells = self._wall_skeleton(budget, demand, rng)
        finished = int(len(skeleton) * (1.0 - share))
        for index, (tx, ty) in enumerate(skeleton):
            if placed >= budget:
                break
            if not self._fits(tx, ty, 1, 1, occ):
                continue
            level = front if index < finished else behind
            next_id = self._commit(bdef, level, (tx, ty), occ, out, next_id, rng)
            placed += 1

        if placed < budget:
            shortfall[bdef.id] = budget - placed
        return next_id, cells

    def _wall_skeleton(
        self, budget: int, demand: int, rng: Random
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
        """Compartment walls, as a list of tiles in the order they should be spent.

        The ring comes first so a partial budget still closes it, then dividers,
        largest compartment first. Splitting the biggest cell each time is what
        produces the mix of sizes a real base has -- a roomy Town Hall box next to
        a run of small single-building cells -- rather than a uniform lattice.
        """
        if budget < 12:
            return [], []

        # Interior big enough to hold the roster at a realistic packing density,
        # but never bigger than the budget can enclose.
        need = int(math.ceil(math.sqrt(demand / 0.78))) + 2
        afford = (budget + 4) // 4
        side = max(6, min(need, afford, self.tiles - 2))

        slack = self.tiles - side
        x0 = max(0, slack // 2 + rng.randint(-slack // 4, slack // 4) if slack >= 4 else 0)
        y0 = max(0, slack // 2 + rng.randint(-slack // 4, slack // 4) if slack >= 4 else 0)
        x0 = min(x0, self.tiles - side - 1)
        y0 = min(y0, self.tiles - side - 1)
        x1, y1 = x0 + side - 1, y0 + side - 1

        tiles = self._rect_perimeter(x0, y0, x1, y1)
        seen = set(tiles)
        remaining = budget - len(tiles)

        cells = [(x0, y0, x1, y1)]
        # 4 is the largest footprint in the TH1-9 range, so a compartment narrower
        # than that on either side can never be filled and is not worth walling.
        min_interior = 4
        while remaining > min_interior:
            cells.sort(key=lambda c: -((c[2] - c[0]) * (c[3] - c[1])))
            for i, (cx0, cy0, cx1, cy1) in enumerate(cells):
                wide = (cx1 - cx0 - 1) >= min_interior * 2 + 1
                tall = (cy1 - cy0 - 1) >= min_interior * 2 + 1
                if not (wide or tall):
                    continue
                vertical = wide if not tall else (wide and rng.random() < 0.5)
                if vertical:
                    cut = rng.randint(cx0 + min_interior + 1, cx1 - min_interior - 1)
                    line = [(cut, y) for y in range(cy0, cy1 + 1)]
                    sub = [(cx0, cy0, cut, cy1), (cut, cy0, cx1, cy1)]
                else:
                    cut = rng.randint(cy0 + min_interior + 1, cy1 - min_interior - 1)
                    line = [(x, cut) for x in range(cx0, cx1 + 1)]
                    sub = [(cx0, cy0, cx1, cut), (cx0, cut, cx1, cy1)]
                fresh = [t for t in line if t not in seen]
                if len(fresh) > remaining:
                    continue
                tiles += fresh
                seen.update(fresh)
                remaining -= len(fresh)
                cells[i:i + 1] = sub
                break
            else:
                break

        # A surplus goes on rings just inside the outer one, which is what players
        # do with spare walls. Keep offering candidates until the budget is covered:
        # some will land on tiles a divider already took, and an unspent wall is a
        # wall the roster said the base has.
        inset = 2
        while len(tiles) < budget and side - 2 * inset >= 6:
            for t in self._rect_perimeter(x0 + inset, y0 + inset, x1 - inset, y1 - inset):
                if t not in seen:
                    tiles.append(t)
                    seen.add(t)
            inset += 2
        return tiles, cells

    @staticmethod
    def _rect_perimeter(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        top = [(x, y0) for x in range(x0, x1 + 1)]
        bottom = [(x, y1) for x in range(x0, x1 + 1)]
        left = [(x0, y) for y in range(y0 + 1, y1)]
        right = [(x1, y) for y in range(y0 + 1, y1)]
        return top + bottom + left + right

    # ---- sanity ----------------------------------------------------------

    def _check_shortfall(self, th: int, placements: list[Placement], shortfall: dict[str, int]) -> None:
        """Fail loudly if the grid could not hold anything like the roster."""
        missed = sum(shortfall.values())
        total = len(placements) + missed
        if total and missed / total > self.max_shortfall:
            raise PlacementError(
                f"TH{th}: only placed {len(placements)} of {total} buildings "
                f"({missed / total:.1%} short, limit {self.max_shortfall:.0%}): {shortfall}"
            )
