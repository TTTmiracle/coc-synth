#!/usr/bin/env python3
"""Normalise raw building images into the sprite folder contract.

Takes whatever crops you have, cleans them up, and files them where the generator
expects. Does NOT obtain art for you -- point it at images you already have.

    # one image, explicit type/level
    python tools/ingest_sprites.py raw/cannon5.png --type cannon --level 5

    # a directional sprite
    python tools/ingest_sprites.py raw/sweeper_sw.png --type air_sweeper --level 2 --dir 5

    # a whole folder named "<type>_lvl<NN>[_dir<D>].png"
    python tools/ingest_sprites.py raw/ --batch

What it does: optional background removal via rembg, crops fully transparent
margins, verifies the type exists in the catalog, and records the anchor point in
assets/sprites/manifest.json (bottom-centre unless you pass --anchor).

Anchors matter more than they look. The anchor is the pixel placed on the footprint
diamond's bottom vertex; if it is wrong the building renders visibly off its tile
while the label JSON still looks perfectly consistent. Generate a few images with
--overlay after ingesting and check the boxes actually sit on the buildings.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cocsynth.assets import SPRITE_RE, trim_alpha  # noqa: E402
from cocsynth.catalog import Catalog  # noqa: E402

SPRITES = ROOT / "assets" / "sprites"
MANIFEST = SPRITES / "manifest.json"


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"sprites": {}}


def save_manifest(data: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")


def remove_background(img: Image.Image) -> Image.Image:
    """Strip an opaque background with rembg, if it is installed."""
    try:
        from rembg import remove
    except ImportError:
        print("  ! rembg not available; skipping background removal", file=sys.stderr)
        return img
    return remove(img)


def ingest_one(
    src: Path, type_id: str, level: int, direction: int | None,
    catalog: Catalog, manifest: dict, cutout: bool, anchor: tuple[int, int] | None,
    dry_run: bool,
) -> str:
    if type_id not in catalog:
        known = ", ".join(sorted(catalog.buildings)[:8])
        raise SystemExit(f"unknown building type {type_id!r}; catalog has e.g. {known}, ...")
    bdef = catalog[type_id]
    if direction is not None and not bdef.is_directional:
        print(f"  ! {type_id} has directions=1; ignoring --dir {direction}", file=sys.stderr)
        direction = None
    if direction is not None and direction >= bdef.directions:
        raise SystemExit(f"{type_id} has {bdef.directions} facings; --dir {direction} is out of range")

    img = Image.open(src).convert("RGBA")
    if cutout:
        img = remove_background(img)
    img, _ = trim_alpha(img)
    if img.getbbox() is None:
        raise SystemExit(f"{src}: image is fully transparent after processing")

    suffix = f"_dir{direction}" if direction is not None else ""
    name = f"{type_id}_lvl{level:02d}{suffix}.png"
    dest = SPRITES / type_id / name

    point = anchor or (img.width // 2, img.height - 1)
    if not (0 <= point[0] < img.width and 0 <= point[1] < img.height):
        raise SystemExit(f"anchor {point} is outside the {img.width}x{img.height} sprite")

    if dry_run:
        return f"would write {dest.relative_to(ROOT)} ({img.width}x{img.height}, anchor {point})"

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    entry = manifest["sprites"].setdefault(type_id, {})
    entry.setdefault("anchors", {})[name] = list(point)
    return f"wrote {dest.relative_to(ROOT)} ({img.width}x{img.height}, anchor {point})"


def parse_batch_name(path: Path) -> tuple[str, int, int | None] | None:
    m = SPRITE_RE.match(path.name)
    if not m:
        return None
    return m.group("type"), int(m.group("level")), (int(m.group("dir")) if m.group("dir") else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="source image, or folder with --batch")
    ap.add_argument("--batch", action="store_true",
                    help="treat src as a folder of '<type>_lvl<NN>[_dir<D>].png' files")
    ap.add_argument("--type", help="building type id, e.g. cannon")
    ap.add_argument("--level", type=int, help="building level")
    ap.add_argument("--dir", type=int, default=None, dest="direction", help="facing index")
    ap.add_argument("--anchor", type=int, nargs=2, metavar=("X", "Y"),
                    help="anchor pixel; defaults to bottom-centre")
    ap.add_argument("--cutout", action="store_true", help="run rembg to strip the background")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    catalog = Catalog.load()
    manifest = load_manifest()

    if args.batch:
        files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not files:
            raise SystemExit(f"no images found in {args.src}")
        skipped = 0
        for path in files:
            parsed = parse_batch_name(path)
            if parsed is None:
                print(f"  - skip {path.name} (expected '<type>_lvl<NN>[_dir<D>].png')")
                skipped += 1
                continue
            t, lv, d = parsed
            print("  " + ingest_one(path, t, lv, d, catalog, manifest,
                                    args.cutout, tuple(args.anchor) if args.anchor else None,
                                    args.dry_run))
        print(f"{len(files) - skipped} ingested, {skipped} skipped")
    else:
        if not args.type or args.level is None:
            raise SystemExit("--type and --level are required without --batch")
        print("  " + ingest_one(args.src, args.type, args.level, args.direction, catalog,
                                manifest, args.cutout,
                                tuple(args.anchor) if args.anchor else None, args.dry_run))

    if not args.dry_run:
        save_manifest(manifest)
        print(f"manifest: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
