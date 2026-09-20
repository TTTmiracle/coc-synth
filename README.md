# coc-synth

Generates synthetic top-down Clash of Clans base layouts with exact ground-truth
labels, for training a vision model that identifies which buildings are where.

```bash
python -m cocsynth.generate --th 9 --n 500 --out data/th9 --seed 42 --formats json,yolo
python -m cocsynth.generate --th 1-9 --n 5000 --out data/mixed --seed 1
python -m cocsynth.generate --th 9 --n 4 --out data/debug --overlay   # eyeball the labels
```

Output:

```
data/th9/
  images/th9_000000.png       rendered base
  labels/th9_000000.json      ground truth
  overlays/th9_000000.png     boxes + footprints + facings drawn on  (--overlay)
  yolo/labels/*.txt, data.yaml                                        (--formats yolo)
  dataset.json                settings, counts, placeholder coverage
```

## How it fits together

| Piece | File | Role |
|---|---|---|
| Rules | `config/buildings.yaml` | per-TH unlock / count / max-level, generated from upstream game data |
| Catalog | `cocsynth/catalog.py` | loads and re-validates the rules |
| Placement | `cocsynth/placement.py` | legal random layout on the 44×44 grid |
| Projection | `cocsynth/project.py` | 2:1 isometric tile ↔ pixel |
| Sprites | `cocsynth/assets.py` | real art with level/direction fallback |
| Renderer | `cocsynth/render.py` | compositing + instance-id mask |
| Labels | `cocsynth/labels.py` | the ground-truth JSON |
| Export | `cocsynth/export_yolo.py` | YOLO txt + `data.yaml` |
| Overlay | `cocsynth/overlay.py` | draws ground truth back onto the image |

## The rules are the point

`config/buildings.yaml` is the single source of truth and is **generated**, not
hand-written — transcribing per-TH count tables by hand is how you end up with two
Eagle Artilleries at TH8:

```bash
python tools/import_game_data.py --max-th 9
```

It pulls numeric data (footprints, per-TH counts, level requirements) from
[chiefpansancolt/clash-of-clans-data](https://github.com/chiefpansancolt/clash-of-clans-data)
at a pinned commit. No art is downloaded. A building with no entry at a Town Hall
level cannot be placed there at all, and the caps are what the work list is built
*from*, so they cannot be exceeded. `tests/test_placement.py` asserts this over 225
random bases.

Extending past TH9 is `--max-th 12` plus a re-run of the suite. Note TH16+ introduces
building merges (`countAfterMerges`), which the importer currently ignores.

## Rotation

Most buildings have a fixed orientation. Some do not — the **Air Sweeper** (TH6+) has
a player-set facing, 8 headings 45° apart, and the Firespitter (TH17) works the same
way. This is a real label dimension, driven by `directions:` in the catalog and
auto-detected from the upstream game description text.

Facing changes the sprite and firing arc, not the tiles consumed, so it is sampled
after collision and never interacts with placement.

Plain YOLO boxes have no orientation slot, hence `--yolo-direction-mode`:
`ignore` (default; facing stays in the JSON) or `split` (one class per facing).

## Labels

```json
{
  "id": 7, "type": "air_sweeper", "level": 2,
  "tile": [14, 26], "footprint": [2, 2],
  "direction": 5, "heading": "SW", "rotation_deg": 225,
  "bbox_px": [420, 512, 64, 72],
  "footprint_polygon_px": [[...], [...], [...], [...]],
  "visibility": 0.88,
  "sprite": "air_sweeper_lvl02_dir5.png"
}
```

* `bbox_px` is measured from rendered alpha via an instance-id mask, not approximated
  from the footprint — an isometric sprite stands well above its tile, so a box
  derived from the diamond alone is simply the wrong box.
* `visibility` is the fraction not occluded by a nearer building. Instances below
  `--min-visibility` are dropped from the label entirely: asking a model to detect
  something invisible teaches it to hallucinate.
* `sprite` records what was actually used, so a dataset can be audited for how much
  real art went into it (`placeholder:` prefix means procedural).
* `seed` + `grid` reproduce any image exactly: `--only <index>` re-renders it
  byte-for-byte.

## Art

Buildings render as procedural placeholders until you supply sprites. That makes the
geometry, labels and training loop provable today, but a model trained on placeholders
learns *this* geometry, not Clash of Clans. See [assets/README.md](assets/README.md)
for the folder contract, the level/mirror fallbacks that cut how many images you need,
and the anchor trap.

## Tests

```bash
python -m pytest tests/ -q
```

Requires Pillow, numpy, pydantic, PyYAML, pytest. `rembg` is optional (only
`tools/ingest_sprites.py --cutout`). Deliberately no opencv or ultralytics — those
belong to the training half, and their Python 3.14 wheels are unreliable.
