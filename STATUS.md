# Where this is up to

## Background

Earlier hand-made attempts at this (around 15 September 2026) were not
preserved. This repository starts from the current generator, and its commit
history begins on 20 September 2026.

## What works

The generator places buildings under real game rules and renders them with the
game's own geometry, measured rather than assumed:

- **Projection is 4:3, not 2:1.** `tile_h = 0.75 * tile_w`. Measured four ways
  across two different clients (lattice period by sub-pixel shift correlation,
  FFT, diamond edge fit) and agreeing to 0.75 within 0.25%. The camera is
  orthographic -- the lattice period is identical top to bottom of the grid, so
  there is no perspective to model.
- **Ground is a tile checkerboard**, coloured from real screenshots. The shade
  step is ~15 levels of luminance; measuring it by light/dark quantiles reads
  grain as well as tiles and gives ~40, which is visibly wrong.
- **Buildings stand at their footprint's centre**, not its south vertex. The
  old anchoring only makes sense if art fills its whole diamond, which it does
  not, and it hung every building over a tile-width low.
- **Nine building types and the walls are cut from a real screenshot** rather
  than scaled from published icons. Icons are framed to fill their own canvas,
  so they carry no footprint information and no per-type scale can be recovered
  from them -- an Army Camp icon is just its campfire.

## Trained model

`models/best.pt` -- YOLO11n, 40 epochs on 400 synthetic TH1-2 images, CPU.
Reached mAP50 0.987 / mAP50-95 0.95 on a *synthetic* validation split, which
is same-distribution and proves only that the labels are consistent.

**It has never been tested on a real screenshot.** That was the next step and
is the only number that matters. Note when doing it that the cut sprites came
from `reference/th1_spread.jpg`, so testing on that same image leaks the art
(though not the layout, background, lighting or UI). A second TH1-2 screenshot
with a different base would be a clean test.

Reproduce the dataset:

    python3 -m cocsynth.generate --th 1-2 --n 400 --out /tmp/ds --seed 99 \
        --tile-w 56 --view 800 800 --level-policy fresh --formats json,yolo \
        --skip-missing-art

## What is still wrong

- **Only levels 1-2 are measured.** Art size varies by level as well as type --
  a level 1 Elixir Storage is 0.41 of its footprint, the maxed one 0.93 -- so
  everything above TH2 still uses published icons at a global fill factor.
- **No obstacles at all.** No trees, bushes, stumps or rocks, inside the grid
  or out. Every real screenshot is full of them.
- **Outside the grid is flat dark green**, where the game has forest, water,
  shore rocks and cliffs.
- **Elixir Collector and Gold Mine are not cut** -- both had a UI marker over
  them in the reference shot.
- **Real bases pack tighter** than the placer does.

## What not to repeat

Template matching was tried three times to measure per-type art scale and gave
three different answers for the same building (0.47, 0.22, 0.75) depending on
which statistic scored it. Peak correlation rises as a template shrinks, so a
scale sweep walks downhill to the smallest size tried. Its numbers were written
into the manifest and later deleted. What worked instead: cut the art, or
segment a feature the type owns and measure it directly, or -- for walls --
measure two different arrangements and check they agree.

Crop boxes are picked by eye (`reference/cut_th1.json`) because no colour test
separates a villager standing against a wall from the building. Two size
"measurements" had a person in them before that was noticed.
