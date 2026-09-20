#!/usr/bin/env python3
"""Generate config/buildings.yaml from published Clash of Clans game data.

Source: https://github.com/chiefpansancolt/clash-of-clans-data (MIT), pinned to a
commit so regeneration is deterministic. Only *numeric* data is used -- footprints,
per-Town-Hall counts and level/TH requirements. No art is downloaded.

Why generated rather than hand-written: transcribing per-TH count tables by hand is
how you end up with 2 Eagle Artilleries at TH8. The catalog is the rule layer the
whole generator obeys, so it comes from the upstream data verbatim.

Usage:
    python tools/import_game_data.py            # regenerate config/buildings.yaml
    python tools/import_game_data.py --max-th 12
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = "chiefpansancolt/clash-of-clans-data"
# Pinned for reproducibility. Bump deliberately, then re-run the test suite.
COMMIT = "55d9bb09914f3a14420734074a68a02ac50a76ef"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"

# Home-village directories that contain placeable structures.
SUBDIRS = ("defenses", "walls", "traps", "resource-buildings", "army-buildings", "town-hall")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "config" / "_gamedata_cache"
OUT = ROOT / "config" / "buildings.yaml"

# A building is directional when the game text says so. Detected rather than listed,
# so buildings added upstream later are picked up automatically.
# Confirmed: Air Sweeper and Firespitter both state they face a single direction.
DIRECTION_RE = re.compile(
    r"can only face|face(?:s)? (?:only )?(?:a )?single direction|only face one direction", re.I
)
# Air Sweeper rotates in 45-degree steps -> 8 selectable headings.
DIRECTION_COUNT = 8

# Buildings with no `availablePerTownHall` block that still need an entry.
# The Town Hall is implicit: exactly one, and its level *is* the TH level.
SPECIAL = {"town-hall": {"count": 1}}


def fetch(path: str) -> dict:
    """Read a data file from cache, downloading it on first use."""
    dest = CACHE / path.replace("/", "__")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{RAW}/{path}"
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - pinned raw URL
            dest.write_bytes(r.read())
    return json.loads(dest.read_text(encoding="utf-8"))


def list_data_files() -> list[str]:
    """List home-village data files in the pinned tree."""
    url = f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
        tree = json.load(r)
    want = re.compile(rf"^data/home/({'|'.join(SUBDIRS)})/[^/]+\.json$")
    return sorted(p["path"] for p in tree["tree"] if want.match(p["path"]))


def parse_size(size: str | None) -> tuple[int, int] | None:
    """'3x3' -> (3, 3)."""
    if not isinstance(size, str):
        return None
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)", size.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def max_level_at(levels: list[dict], th: int) -> int | None:
    """Highest level whose townHallRequired is <= th."""
    ok = [lv["level"] for lv in levels if (lv.get("townHallRequired") or 1) <= th]
    return max(ok) if ok else None


def build_entry(d: dict, max_th: int) -> dict | None:
    """Turn one upstream building record into a catalog entry, or None if out of range."""
    bid = d["id"]
    footprint = parse_size(d.get("size"))
    if footprint is None:
        return None  # heroes and other non-placeable records

    counts = {e["townHallLevel"]: e.get("count") or 0 for e in (d.get("availablePerTownHall") or [])}
    special = SPECIAL.get(bid)
    levels = d.get("levels") or []

    per_th: dict[int, dict[str, int]] = {}
    for th in range(1, max_th + 1):
        if special:
            count = special["count"]
            # The Town Hall's level is the Town Hall level.
            mx = th if bid == "town-hall" else (max_level_at(levels, th) or 1)
        else:
            count = counts.get(th, 0)
            mx = max_level_at(levels, th) or 1
        if count > 0:
            per_th[th] = {"count": int(count), "max_level": int(mx)}

    if not per_th:
        return None  # not unlocked anywhere in range (Eagle Artillery, Inferno Tower, ...)

    directional = bool(DIRECTION_RE.search(d.get("description") or ""))
    return {
        "id": bid,
        "display_name": d.get("name") or bid,
        "category": "wall" if bid == "wall" else (d.get("category") or "other"),
        "footprint": list(footprint),
        "directions": DIRECTION_COUNT if directional else 1,
        "per_th": per_th,
    }


def emit_yaml(entries: list[dict], max_th: int) -> str:
    """Hand-rolled YAML so per_th stays compact and diff-friendly."""
    order = {"town-hall": 0, "wall": 1, "defense": 2, "trap": 3, "resource": 4,
             "army": 5, "research": 6, "other": 7}
    entries = sorted(entries, key=lambda e: (order.get(e["category"], 9), e["id"]))

    out = [
        "# GENERATED FILE -- do not edit by hand.",
        f"# Regenerate with: python tools/import_game_data.py --max-th {max_th}",
        f"# Source: https://github.com/{REPO}",
        f"# Pinned commit: {COMMIT}",
        "#",
        "# per_th holds only Town Hall levels where the building is UNLOCKED.",
        "# A missing TH means the building cannot exist at that TH at all.",
        "# count = hard maximum number allowed. max_level = highest level buildable.",
        "# directions = selectable facings (1 = fixed orientation).",
        f"max_town_hall: {max_th}",
        "grid_tiles: 44",
        "buildings:",
    ]
    for e in entries:
        out += [
            f"  {e['id'].replace('-', '_')}:",
            f"    display_name: {e['display_name']}",
            f"    category: {e['category']}",
            f"    footprint: [{e['footprint'][0]}, {e['footprint'][1]}]",
            f"    directions: {e['directions']}",
            "    per_th:",
        ]
        for th, v in sorted(e["per_th"].items()):
            out.append(f"      {th}: {{ count: {v['count']}, max_level: {v['max_level']} }}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-th", type=int, default=9, help="highest Town Hall level to include")
    args = ap.parse_args()

    entries, skipped = [], []
    for path in list_data_files():
        entry = build_entry(fetch(path), args.max_th)
        (entries if entry else skipped).append(entry or path.rsplit("/", 1)[-1][:-5])

    OUT.write_text(emit_yaml(entries, args.max_th), encoding="utf-8")

    directional = [e["id"] for e in entries if e["directions"] > 1]
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} buildings for TH1-{args.max_th}")
    print(f"  directional: {', '.join(directional) or 'none'}")
    print(f"  excluded (not unlocked in range / not placeable): {len(skipped)}")
    print(f"    {', '.join(sorted(skipped))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
