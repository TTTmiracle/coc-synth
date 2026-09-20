#!/usr/bin/env python3
"""Normalise raw building images into the sprite folder contract.

Takes whatever crops you have, cleans them up, and files them where the generator
expects. Does NOT obtain art for you -- point it at images you already have.

    # one image, explicit type/level
    python tools/ingest_sprites.py raw/cannon5.png --type cannon --level 5

    # a directional sprite
    python tools/ingest_sprites.py raw/sweeper_sw.png --type air_sweeper --level 2 --dir 5

    # a whole folder named "<type>_lvl<NN>[_dir<D>].png"
    # an animated source -> one file per frame
    python tools/ingest_sprites.py raw/cannon.gif --type cannon --level 7 --frames 4
    python tools/ingest_sprites.py raw/clip.mp4 --type mortar --level 5 --frames 6 --cutout

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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageSequence

#: Containers ffmpeg is used for. Everything else Pillow opens directly.
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

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


def load_frames(src: Path, max_frames: int) -> list[Image.Image]:
    """Read an image, animation or video as a list of RGBA frames.

    Animated GIF/APNG/WebP go through Pillow; video goes through ffmpeg. Frames are
    sampled evenly across the whole clip rather than taken from the start, so a
    recording of an idle building covers its full loop.
    """
    if src.suffix.lower() in VIDEO_SUFFIXES:
        frames = _video_frames(src)
    else:
        img = Image.open(src)
        frames = [f.convert("RGBA") for f in ImageSequence.Iterator(img)]

    if not frames:
        raise SystemExit(f"{src}: no frames could be read")
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    return [frames[min(len(frames) - 1, int(i * step))] for i in range(max_frames)]


def _video_frames(src: Path) -> list[Image.Image]:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed to read video; install it or export frames yourself")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(src), "-vsync", "0",
             str(Path(tmp) / "f_%05d.png")],
            check=True,
        )
        return [Image.open(f).convert("RGBA").copy() for f in sorted(Path(tmp).glob("f_*.png"))]


def align_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """Crop every frame to one shared box.

    Trimming frames independently is the obvious thing and it is wrong: the alpha
    bounds shift as the building animates, so each frame gets cropped differently and
    the building visibly jitters between frames. The union box keeps them registered.
    """
    boxes = [f.getbbox() for f in frames]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        raise SystemExit("every frame is fully transparent after processing")
    union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
             max(b[2] for b in boxes), max(b[3] for b in boxes))
    return [f.crop(union) for f in frames]


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
    dry_run: bool, max_frames: int = 1,
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

    frames = load_frames(src, max_frames)
    if cutout:
        frames = [remove_background(f) for f in frames]
    frames = align_frames(frames)

    dir_part = f"_dir{direction}" if direction is not None else ""
    entry = manifest["sprites"].setdefault(type_id, {}) if not dry_run else {}
    written = []

    for i, frame in enumerate(frames):
        frame_part = f"_f{i:02d}" if len(frames) > 1 else ""
        name = f"{type_id}_lvl{level:02d}{dir_part}{frame_part}.png"
        dest = SPRITES / type_id / name

        point = anchor or (frame.width // 2, frame.height - 1)
        if not (0 <= point[0] < frame.width and 0 <= point[1] < frame.height):
            raise SystemExit(f"anchor {point} is outside the {frame.width}x{frame.height} sprite")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            frame.save(dest)
            entry.setdefault("anchors", {})[name] = list(point)
        written.append(name)

    verb = "would write" if dry_run else "wrote"
    size = f"{frames[0].width}x{frames[0].height}"
    if len(written) == 1:
        return f"{verb} {type_id}/{written[0]} ({size})"
    return f"{verb} {type_id}/ {len(written)} frames {written[0]}..{written[-1]} ({size}, aligned)"


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
    ap.add_argument("--frames", type=int, default=1, metavar="N",
                    help="extract up to N animation frames from a GIF/APNG/WebP/video, "
                         "sampled evenly across the clip (default 1)")
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
                                    args.dry_run, args.frames))
        print(f"{len(files) - skipped} ingested, {skipped} skipped")
    else:
        if not args.type or args.level is None:
            raise SystemExit("--type and --level are required without --batch")
        print("  " + ingest_one(args.src, args.type, args.level, args.direction, catalog,
                                manifest, args.cutout,
                                tuple(args.anchor) if args.anchor else None,
                                args.dry_run, args.frames))

    if not args.dry_run:
        save_manifest(manifest)
        print(f"manifest: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
