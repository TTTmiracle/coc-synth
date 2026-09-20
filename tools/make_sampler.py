"""Render one of every building type, spaced out and labelled.

Per-type art sizes cannot all be measured from the screenshots available: the
colour method only works where a type owns a colour nothing else on the field
does, which is two of them. This puts the question to someone who can see the
game instead -- every type at its current size, far enough apart to judge, with
the size printed next to it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import ImageDraw  # noqa: E402

from cocsynth.assets import SpriteLibrary  # noqa: E402
from cocsynth.catalog import Catalog  # noqa: E402
from cocsynth.placement import Placement  # noqa: E402
from cocsynth.render import Renderer  # noqa: E402
from cocsynth.viewport import crop, grid_window  # noqa: E402

TILE_W = 44


def main(out: str = "/tmp/sampler.png") -> None:
    cat = Catalog.load()
    lib = SpriteLibrary(tile_w=TILE_W)
    types = [b for b in cat.buildings.values()
             if b.category != "wall" and b.id != "hero_banner"]
    types.sort(key=lambda b: (-b.footprint[0], b.id))

    placements, spots = [], []
    x = y = 2
    row_h = 0
    for i, b in enumerate(types):
        w, h = b.footprint
        if x + w + 6 > 42:
            x = 2
            y += row_h + 5
            row_h = 0
        level = b.per_th[max(b.per_th)].max_level
        placements.append(Placement(instance_id=i + 1, type_id=b.id, level=level,
                                    tile=(x, y), footprint=(w, h), direction=0,
                                    directions=b.directions))
        spots.append((b, (x, y)))
        row_h = max(row_h, h)
        x += w + 6

    r = Renderer(cat, lib, tile_w=TILE_W, margin=120)
    res = r.render(placements, Random(0))
    res = crop(res, grid_window(res, pad_tiles=0.5), min_visible=0.0)

    img = res.image.convert("RGB")
    d = ImageDraw.Draw(img)
    for p_, (b, (tx, ty)) in zip(placements, spots):
        inst = res.instances.get(p_.instance_id)
        if inst is None:
            continue
        bx, by, bw, bh = inst.bbox_px
        label = f"{b.id} {b.footprint[0]}x{b.footprint[1]}  art {bw / TILE_W:.2f}w"
        d.rectangle((bx, by - 15, bx + 8 * len(label), by - 2), fill=(0, 0, 0))
        d.text((bx + 2, by - 14), label, fill=(255, 255, 120))
        d.rectangle((bx, by, bx + bw, by + bh), outline=(255, 90, 90))
    img.save(out)
    print(f"{len(spots)} types -> {out}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main(*sys.argv[1:])
