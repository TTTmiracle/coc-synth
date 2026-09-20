# Sprite folder contract

The generator reads building art from this directory. Anything missing falls back to
a procedural placeholder, so the pipeline always runs — but a dataset built on
placeholders teaches a model *this* geometry, not Clash of Clans. Real art is what
makes it transfer.

## Layout

```
assets/sprites/
  cannon/
    cannon_lvl01.png
    cannon_lvl07.png
  air_sweeper/
    air_sweeper_lvl02_dir0.png
    air_sweeper_lvl02_dir1.png
    ...
  manifest.json
```

* `<type>_lvl<NN>.png` — fixed-orientation buildings.
* `<type>_lvl<NN>_dir<D>.png` — directional buildings, `D` is `0..directions-1`.
* `<type>_lvl<NN>_f<FF>.png` — animation frames, `FF` is `00, 01, 02, …`.
  Combine freely: `air_sweeper_lvl02_dir3_f01.png`.
* `<type>` must match an id in `config/buildings.yaml` (`cannon`, `air_sweeper`,
  `town_hall`, …). Run `python -c "from cocsynth.catalog import Catalog;
  print(Catalog.load().type_names)"` for the list.
* Transparent PNG. Margins are trimmed automatically.

## You need far fewer images than it looks

**Levels fall back downward.** A request for level 11 uses the highest sprite at or
below 11. Clash art only changes at a handful of levels, so a Cannon needs roughly
five images (levels 1, 3, 7, 9, 12…), not twenty. Add a file only where the art
actually changes.

**Directions can be mirrored.** A missing `dir<D>` will use its `mirror_of` partner
flipped horizontally. Declaring the four mirror pairs halves the directional art:

```json
{
  "sprites": {
    "air_sweeper": {
      "mirror_of": { "7": 1, "6": 2, "5": 3 }
    }
  }
}
```

Only the Air Sweeper is directional in the TH1–9 catalog (8 facings, 45° apart).

## Animation

Clash buildings are animated, so one static sprite per level means the model only
ever sees one frozen pose. Add frames with `_f<FF>` and the generator picks one at
random per instance — two Cannons in the same base then sit at different points in
their animation.

You do **not** need frames for everything:

* One frame is fine. Nothing errors; you just get less variety.
* Every instance also gets a small automatic brightness/scale variation regardless,
  so identical buildings are never pixel-identical copies. That is a stand-in for
  animation, not a replacement — real frames are better where you have them.
* Idle state is what matters most. Defenses only animate while firing, which does
  not happen in a static base layout, so 2–3 idle frames goes a long way.

The frame drawn is recorded in each label as `frame`.

## Anchors

The **anchor** is the pixel placed on the footprint diamond's bottom (south) vertex.
It defaults to bottom-centre, which is right for art drawn standing on its footprint.

Directional sprites usually need overrides — a sweeper facing away does not sit in
its tile the same way as one facing the camera:

```json
{
  "sprites": {
    "air_sweeper": {
      "anchors": { "air_sweeper_lvl02_dir3.png": [31, 54] }
    }
  }
}
```

A wrong anchor renders the building visibly off its tile **while the label JSON still
looks perfectly consistent**. It is the single most likely thing to go wrong here, and
the only reliable way to catch it is to look:

```bash
python -m cocsynth.generate --th 9 --n 4 --out data/debug --overlay
```

Then open `data/debug/overlays/` and check the boxes sit on the buildings.

## Ingesting art

`tools/ingest_sprites.py` trims, optionally background-removes via `rembg`, validates
the type against the catalog, and writes the manifest entry:

```bash
python tools/ingest_sprites.py raw/cannon7.png --type cannon --level 7
python tools/ingest_sprites.py raw/sweeper_se.png --type air_sweeper --level 2 --dir 3
python tools/ingest_sprites.py raw/ --batch          # files already named correctly
python tools/ingest_sprites.py raw/photo.png --type mortar --level 4 --cutout
```

## Partial directional art is a silent trap

If a directional type has *some* art but not every facing, the renderer falls back to
another direction while the label still records the facing that was chosen. The image
and its label then disagree and nothing errors.

`cocsynth.generate` checks for this at startup and warns. A type with *no* art is
fine — placeholders render every facing correctly.

## Sourcing

This repo does not include or download building art. Whatever you put here is your
call; the generator only cares that the files match the contract above.
