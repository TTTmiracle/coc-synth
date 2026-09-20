"""Measure how big the game draws a building, by its own distinctive colour.

Template matching failed here: its score depends on template size, so sweeping
it walks to whatever scale the statistic happens to favour, and it gave three
different answers for one building depending on which statistic was used.

This measures instead of matching. Many Clash buildings carry a colour that
nothing else on the field does -- the Elixir Storage's magenta, the Gold
Storage's gold. Segment that colour in a real screenshot and in a render, take
the median width of the blobs it forms, and the ratio is how far off the render
is. No scale sweep, no correlation, and it can be checked by eye.

It only works for types with a colour of their own, so it reports which ones it
could measure and stays quiet about the rest rather than inventing numbers.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def components(mask: np.ndarray, min_px: int = 250) -> list[tuple[int, int]]:
    """Widths and heights of connected blobs, largest-first by area."""
    seen = np.zeros(mask.shape, bool)
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        q = deque([(y, x)])
        seen[y, x] = True
        xs, ys = [], []
        while q:
            cy, cx = q.popleft()
            xs.append(cx)
            ys.append(cy)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if len(xs) >= min_px:
            out.append((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    return out


#: Colour tests that isolate one building type from everything else on the map.
#: Hand-written rather than derived, because "distinctive" has to mean distinctive
#: from the other 32 types and the grass, which a sprite cannot tell you alone.
SIGNATURES = {
    "elixir_storage": lambda r, g, b: (r > 150) & (b > 150) & (r - g > 60) & (b - g > 60),
    "gold_storage": lambda r, g, b: (r > 190) & (g > 150) & (b < 110) & (r - b > 95) & (g - b > 70),
    "dark_elixir_storage": lambda r, g, b: (r < 70) & (g < 62) & (b > 55) & (b > g + 8) & (b < 130),
}


def measure(path: str, type_id: str, tile_w: float, min_px: int = 250):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blobs = components(SIGNATURES[type_id](r, g, b), min_px)
    if not blobs:
        return None
    widths = sorted(w for w, _ in blobs)
    return {"n": len(widths), "median_px": widths[len(widths) // 2],
            "tiles": widths[len(widths) // 2] / tile_w}
