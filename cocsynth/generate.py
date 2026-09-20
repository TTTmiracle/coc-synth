"""CLI: generate a synthetic base-layout dataset.

    python -m cocsynth.generate --th 9 --n 500 --out data/th9 --seed 42
    python -m cocsynth.generate --th 1-9 --n 5000 --out data/mixed --formats json,yolo
    python -m cocsynth.generate --th 9 --n 4 --out data/debug --overlay

Every image gets its own seed, derived from the master seed and the image index, so
any single image can be reproduced on its own without regenerating the whole run --
`--only 123` re-renders image 123 byte-identically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from random import Random

from .assets import SpriteLibrary
from .catalog import Catalog
from .export_yolo import export as export_yolo
from .labels import build_label, write_label
from .overlay import draw_overlay
from .placement import Placer
from .render import Renderer

log = logging.getLogger("cocsynth")

#: Scale jitter picks from these tile widths. All even so tile_h = tile_w/2 is exact.
SCALE_CHOICES = (28, 32, 36, 40)


def derive_seed(master: int, index: int) -> int:
    """A stable, independent seed per image, addressable by index."""
    digest = hashlib.sha256(f"{master}:{index}".encode()).digest()
    return int.from_bytes(digest[:7], "big")


def parse_th_range(spec: str, catalog: Catalog) -> list[int]:
    """'9' -> [9];  '1-9' -> [1..9];  '3,5,8' -> [3,5,8]."""
    levels: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            levels += list(range(lo, hi + 1))
        else:
            levels.append(int(part))
    for th in levels:
        if not 1 <= th <= catalog.max_town_hall:
            raise SystemExit(f"Town Hall {th} outside catalog range 1..{catalog.max_town_hall}")
    return levels


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cocsynth.generate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--th", default="9", help="Town Hall level, range ('1-9') or list ('3,5,8')")
    p.add_argument("--n", type=int, default=10, help="number of images to generate")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--seed", type=int, default=0, help="master seed")
    p.add_argument("--only", type=int, default=None, help="regenerate just this image index")
    p.add_argument("--formats", default="json", help="comma-separated: json, yolo")
    p.add_argument("--overlay", action="store_true", help="also write ground-truth overlay images")

    p.add_argument("--tile-w", type=int, default=32, help="tile diamond width in pixels (even)")
    p.add_argument("--scale-jitter", action="store_true",
                   help=f"vary tile width per image among {SCALE_CHOICES}")
    p.add_argument("--margin", type=int, default=24, help="padding around the grid, and pan range")
    p.add_argument("--layout", default="clustered", choices=("scatter", "clustered", "compartment"))
    p.add_argument("--count-mode", default="jitter", choices=("exact", "jitter"))
    p.add_argument("--level-policy", default="clustered", choices=("maxed", "clustered", "uniform"))
    p.add_argument("--min-visibility", type=float, default=0.05,
                   help="drop instances occluded below this fraction")
    p.add_argument("--yolo-direction-mode", default="ignore", choices=("ignore", "split"),
                   help="'split' gives each facing its own YOLO class")
    p.add_argument("--sprites", type=Path, default=None, help="sprite directory override")
    p.add_argument("--config", type=Path, default=None, help="buildings.yaml override")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s", stream=sys.stderr)
    if args.tile_w % 2:
        raise SystemExit(f"--tile-w must be even (got {args.tile_w}); tile height is half of it")

    catalog = Catalog.load(args.config) if args.config else Catalog.load()
    levels = parse_th_range(args.th, catalog)
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    unknown = formats - {"json", "yolo"}
    if unknown:
        raise SystemExit(f"unknown format(s): {', '.join(sorted(unknown))}")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.overlay:
        (out / "overlays").mkdir(parents=True, exist_ok=True)

    placer = Placer(catalog, layout_style=args.layout, count_mode=args.count_mode,
                    level_policy=args.level_policy)
    libraries: dict[int, SpriteLibrary] = {}

    # A directional type with partial art renders the wrong facing while the label
    # still claims the right one. Surface it before generating thousands of images.
    probe = SpriteLibrary(args.sprites, tile_w=args.tile_w) if args.sprites else SpriteLibrary(tile_w=args.tile_w)
    for type_id, missing in probe.directional_gaps(catalog).items():
        log.warning("WARNING: %s has art but is missing facings %s; those will render as "
                    "another direction while the label still says otherwise. Add the "
                    "sprites, or a mirror_of entry in manifest.json.", type_id, missing)

    indices = [args.only] if args.only is not None else list(range(args.n))
    started = time.perf_counter()
    total_boxes = 0
    placeholder_types: set[str] = set()

    for i in indices:
        seed = derive_seed(args.seed, i)
        rng = Random(seed)
        th = levels[i % len(levels)] if len(levels) > 1 else levels[0]

        tile_w = rng.choice(SCALE_CHOICES) if args.scale_jitter else args.tile_w
        pan = (rng.randint(-args.margin, args.margin), rng.randint(-args.margin, args.margin))

        library = libraries.get(tile_w)
        if library is None:
            library = libraries[tile_w] = SpriteLibrary(
                args.sprites, tile_w=tile_w) if args.sprites else SpriteLibrary(tile_w=tile_w)

        placement = placer.generate(th, rng)
        renderer = Renderer(catalog, library, tile_w=tile_w, margin=args.margin,
                            min_visibility=args.min_visibility)
        rendered = renderer.render(placement.placements, rng, pan=pan)

        stem = f"th{th}_{i:06d}"
        rendered.image.save(out / "images" / f"{stem}.png")
        label = build_label(f"{stem}.png", seed, placement, rendered, catalog)
        write_label(label, out / "labels" / f"{stem}.json")
        total_boxes += len(label.buildings)
        placeholder_types |= {b.type for b in label.buildings if b.sprite.startswith("placeholder:")}

        if args.overlay:
            draw_overlay(rendered.image, label).save(out / "overlays" / f"{stem}.png")
        if not args.quiet and (i % 25 == 0 or i == indices[-1]):
            log.info("  [%d/%d] %s: %d buildings, %d hidden",
                     indices.index(i) + 1, len(indices), stem,
                     len(label.buildings), len(rendered.dropped))

    elapsed = time.perf_counter() - started
    manifest = {
        "images": len(indices),
        "boxes": total_boxes,
        "town_hall_levels": levels,
        "master_seed": args.seed,
        "settings": {
            "tile_w": args.tile_w, "scale_jitter": args.scale_jitter, "margin": args.margin,
            "layout": args.layout, "count_mode": args.count_mode,
            "level_policy": args.level_policy, "min_visibility": args.min_visibility,
        },
        "placeholder_sprite_types": sorted(placeholder_types),
    }
    if "yolo" in formats:
        manifest["yolo"] = export_yolo(out, catalog, args.yolo_direction_mode, args.min_visibility)
    (out / "dataset.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    rate = len(indices) / elapsed if elapsed else 0.0
    log.info("%d images, %d labelled buildings in %.1fs (%.1f img/s) -> %s",
             len(indices), total_boxes, elapsed, rate, out)
    if placeholder_types:
        log.warning("NOTE: %d/%d types used placeholder art (see assets/README.md); "
                    "this dataset will not transfer to real screenshots yet",
                    len(placeholder_types), len(catalog.buildings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
