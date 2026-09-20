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
    ):
        self.cat = catalog
        self.layout_style = layout_style
        self.count_mode = count_mode
        self.level_policy = level_policy
        self.max_attempts = max_attempts
        self.max_shortfall = max_shortfall
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

        # Town Hall first: it anchors the base, as it does in a real layout.
        town_hall = next((b for b in roster if b.id == "town_hall"), None)
        if town_hall is not None:
            roster.remove(town_hall)
            next_id = self._place_town_hall(town_hall, th, occ, placements, next_id, rng)

        # Then everything else, biggest footprint first so large pieces get space
        # before it fragments. Ties are shuffled for variety.
        work = self._work_list(roster, th, rng)
        work.sort(key=lambda item: -item[0].area)
        for bdef, level in work:
            pos = self._find_spot(bdef.footprint, occ, rng)
            if pos is None:
                shortfall[bdef.id] = shortfall.get(bdef.id, 0) + 1
                continue
            next_id = self._commit(bdef, level, pos, occ, placements, next_id, rng)

        for bdef in walls:
            next_id = self._place_walls(bdef, th, occ, placements, next_id, rng, shortfall)

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

    def _fits(self, tx: int, ty: int, w: int, h: int, occ: np.ndarray) -> bool:
        if tx < 0 or ty < 0 or tx + w > self.tiles or ty + h > self.tiles:
            return False
        return not occ[ty : ty + h, tx : tx + w].any()

    def _find_spot(self, footprint: tuple[int, int], occ: np.ndarray, rng: Random) -> tuple[int, int] | None:
        """Random legal origin for a footprint, or None after max_attempts."""
        w, h = footprint
        for _ in range(self.max_attempts):
            tx, ty = self._candidate(w, h, rng)
            if self._fits(tx, ty, w, h, occ):
                return tx, ty
        return None

    def _candidate(self, w: int, h: int, rng: Random) -> tuple[int, int]:
        """Sample a candidate origin according to the layout style."""
        hi_x, hi_y = self.tiles - w, self.tiles - h
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
        next_id: int, rng: Random, shortfall: dict[str, int],
    ) -> int:
        """Spend the wall budget on compartment perimeters rather than scattering.

        Scattered single tiles do not read as a base. This traces the ring around
        what has already been placed, then internal dividers, consuming free tiles
        until the budget runs out.
        """
        budget = self._instance_count(bdef, th, rng)
        level = self._roll_level(bdef, th, rng)
        placed = 0

        for tx, ty in self._wall_tiles(occ, rng):
            if placed >= budget:
                break
            if not self._fits(tx, ty, 1, 1, occ):
                continue
            next_id = self._commit(bdef, level, (tx, ty), occ, out, next_id, rng)
            placed += 1

        if placed < budget:
            shortfall[bdef.id] = budget - placed
        return next_id

    def _wall_tiles(self, occ: np.ndarray, rng: Random) -> list[tuple[int, int]]:
        """Candidate wall tiles: outer ring first, then internal dividers."""
        ys, xs = np.nonzero(occ)
        if len(xs) == 0:
            return []
        pad = 1
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(self.tiles - 1, int(xs.max()) + pad)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(self.tiles - 1, int(ys.max()) + pad)

        tiles = self._rect_perimeter(x0, y0, x1, y1)

        # Internal dividers split the interior into compartments.
        n_div = rng.randint(2, 4)
        for _ in range(n_div):
            if rng.random() < 0.5 and x1 - x0 > 6:
                x = rng.randint(x0 + 3, x1 - 3)
                tiles += [(x, y) for y in range(y0, y1 + 1)]
            elif y1 - y0 > 6:
                y = rng.randint(y0 + 3, y1 - 3)
                tiles += [(x, y) for x in range(x0, x1 + 1)]

        # Concentric inner rings mop up whatever budget is left after the outer ring
        # and dividers, so a large wall allowance is actually spent.
        for inset in (3, 6, 9, 12):
            ix0, iy0 = x0 + inset, y0 + inset
            ix1, iy1 = x1 - inset, y1 - inset
            if ix1 - ix0 < 4 or iy1 - iy0 < 4:
                break
            tiles += self._rect_perimeter(ix0, iy0, ix1, iy1)
        return tiles

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
