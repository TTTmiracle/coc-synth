#!/usr/bin/env python3
"""File a downloaded sprite collection into the generator's folder contract.

Handles trees laid out like the published Clash data sets:

    images/home/defenses/cannon/normal/level-7.png
    images/home/town-hall/normal/level-9.png

and turns them into:

    assets/sprites/cannon/cannon_lvl07.png
    assets/sprites/town_hall/town_lvl09.png   (plus a manifest anchor entry)

    python tools/import_sprites.py --src ~/Downloads/clash-data/images/home
    python tools/import_sprites.py --src ~/Downloads/img --dry-run

This does not download anything. Point it at files you already have.

Levels whose art is byte-identical to the level below are skipped: the generator
falls back to the highest sprite at or below the level it wants, so duplicates are
dead weight. That usually cuts the file count by more than half.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cocsynth.assets import trim_alpha  # noqa: E402
from cocsynth.catalog import Catalog  # noqa: E402

SPRITES = ROOT / "assets" / "sprites"
MANIFEST = SPRITES / "manifest.json"
LEVEL_RE = re.compile(r"^level[-_]?(\d+)\.png$", re.I)


def discover(src: Path, variant: str) -> dict[str, dict[int, Path]]:
    """Find building sprites in a downloaded tree: type -> level -> path.

    The building name is the directory above the variant directory, which covers
    both `<group>/<building>/<variant>/level-N.png` and `<building>/<variant>/level-N.png`.
    """
    found: dict[str, dict[int, Path]] = {}
    for path in sorted(src.rglob("*.png")):
        m = LEVEL_RE.match(path.name)
        if not m or path.parent.name.lower() != variant:
            continue
        building = path.parent.parent.name.replace("-", "_")
        found.setdefault(building, {})[int(m.group(1))] = path
    return found


def dedupe(levels: dict[int, Path]) -> dict[int, Path]:
    """Drop levels whose art repeats the level below.

    Clash art only changes at a handful of levels, and the sprite library resolves a
    request downward to the nearest available level -- so keeping an identical copy
    at every level adds files without adding information.
    """
    kept: dict[int, Path] = {}
    previous: str | None = None
    for level in sorted(levels):
        digest = hashlib.sha256(levels[level].read_bytes()).hexdigest()
        if digest != previous:
            kept[level] = levels[level]
            previous = digest
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="directory to scan")
    ap.add_argument("--variant", default="normal", help="variant subdirectory to use")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="file every level even when the art is unchanged")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"{args.src} does not exist")

    catalog = Catalog.load()
    found = discover(args.src, args.variant.lower())
    if not found:
        raise SystemExit(
            f"no '{args.variant}/level-N.png' sprites under {args.src}; "
            "check --src points at the image tree and try --variant"
        )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"sprites": {}}
    max_th = catalog.max_town_hall
    filed = skipped_dupes = 0
    unknown: list[str] = []

    for building, levels in sorted(found.items()):
        if building not in catalog:
            unknown.append(building)
            continue
        # Nothing above the catalog's top Town Hall can ever be requested.
        cap = catalog[building].per_th[max_th].max_level
        usable = {lv: p for lv, p in levels.items() if lv <= cap}
        chosen = usable if args.keep_duplicates else dedupe(usable)
        skipped_dupes += len(usable) - len(chosen)

        for level, path in sorted(chosen.items()):
            name = f"{building}_lvl{level:02d}.png"
            if not args.dry_run:
                img, _ = trim_alpha(Image.open(path))
                dest = SPRITES / building / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                img.save(dest)
                manifest["sprites"].setdefault(building, {}).setdefault("anchors", {})[name] = [
                    img.width // 2, img.height - 1
                ]
            filed += 1
        print(f"  {building:22} {len(chosen):2d} sprites  (levels {sorted(chosen)})")

    if not args.dry_run:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    verb = "would file" if args.dry_run else "filed"
    print(f"\n{verb} {filed} sprites, skipped {skipped_dupes} unchanged duplicates")
    missing = sorted(set(catalog.type_names) - set(found) - set(unknown))
    if missing:
        print(f"still missing ({len(missing)}): {', '.join(missing)}")
    if unknown:
        print(f"not in catalog, ignored: {', '.join(sorted(unknown)[:10])}")
    print("\nNext: python -m cocsynth.generate --th 9 --n 4 --out data/debug --overlay")
    print("Open data/debug/overlays/ and check the boxes sit on the buildings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
