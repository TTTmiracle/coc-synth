"""Cut wall art out of a screenshot and fill all sixteen connection states.

A Clash wall is a row of posts, not a continuous bar: consecutive segments sit
half a tile apart on screen and are barely wider than that gap, so a run is
posts standing shoulder to shoulder with the rope binding meeting across the
joins. That means one clean post per axis is enough to rebuild every state --
the ropes of neighbouring segments line up on their own.

Two posts are cut: one from a run along grid x and one from a run along grid y,
so the rope lies along the right axis in each case. Masks are then served by
whichever axis they run on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cut_sprites import cut  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONNECT_N, CONNECT_E, CONNECT_S, CONNECT_W = 1, 2, 4, 8


def axis_for(mask: int) -> str:
    """Which cut post serves this connection state."""
    ew = bool(mask & CONNECT_E) + bool(mask & CONNECT_W)
    ns = bool(mask & CONNECT_N) + bool(mask & CONNECT_S)
    if ew > ns:
        return "x"
    if ns > ew:
        return "y"
    # Ties -- corners, crosses and the lone post. Either reads correctly, because
    # a post is near enough symmetric; x keeps the rope on the wider screen axis.
    return "x"


def main(screenshot: str, tile_w: float, x_box, y_box, level: int = 1) -> None:
    from PIL import Image

    img = Image.open(ROOT / screenshot)
    posts = {"x": cut(img, tuple(x_box), pad=3), "y": cut(img, tuple(y_box), pad=3)}
    for k, v in posts.items():
        print(f"  {k}-axis post {v.width}x{v.height}px = {v.width / tile_w:.2f} tiles")

    d = ROOT / "assets" / "sprites" / "wall"
    manifest_path = ROOT / "assets" / "sprites" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["sprites"].setdefault("wall", {})
    scales = entry.setdefault("scales", {})
    anchors = entry.setdefault("anchors", {})

    for mask in range(16):
        post = posts[axis_for(mask)]
        name = f"wall_lvl{level:02d}_c{mask}.png"
        post.save(d / name)
        scales[name] = round(post.width / tile_w, 4)
        anchors[name] = [post.width // 2, post.height - 1]
    entry["cut_from"] = screenshot
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"16 states written from 2 cut posts, scale {scales[f'wall_lvl{level:02d}_c0.png']}")


if __name__ == "__main__":
    main("reference/th1_spread.jpg", 58.17, (1688, 580, 34, 44), (1572, 494, 38, 46))
