"""Cropping a rendered base to a camera view.

A full 44x44 grid rendered edge to edge, centred, with grass margin all round, does
not look like a screenshot -- it looks like a render, because no player ever sees
their whole base framed that way. The game camera sits closer and shows part of the
base with terrain running off every edge.

Cropping to a window does two things at once: it removes the give-away framing, and
it makes the task harder in a useful way, since the model now sees buildings cut off
by the frame exactly as it would in a real screenshot.

Labels are clipped to the window rather than recomputed. A building partly outside
keeps a box covering only the visible part, and its `visibility` is reduced by how
much was cut, so the same threshold that drops occluded buildings also drops ones
that are barely in frame.
"""
from __future__ import annotations

from dataclasses import replace
from random import Random

from .render import RenderResult, RenderedInstance


def pick_window(
    size: tuple[int, int], view: tuple[int, int], focus: tuple[int, int], rng: Random
) -> tuple[int, int, int, int]:
    """Choose a crop box of `view` size, biased towards `focus`, inside `size`."""
    iw, ih = size
    vw, vh = min(view[0], iw), min(view[1], ih)
    jitter_x, jitter_y = vw // 3, vh // 3

    x = focus[0] - vw // 2 + rng.randint(-jitter_x, jitter_x)
    y = focus[1] - vh // 2 + rng.randint(-jitter_y, jitter_y)
    x = max(0, min(iw - vw, x))
    y = max(0, min(ih - vh, y))
    return x, y, x + vw, y + vh


def content_centre(result: RenderResult, fallback: tuple[int, int]) -> tuple[int, int]:
    """Middle of everything actually drawn, so the camera looks at the base."""
    if not result.instances:
        return fallback
    xs, ys = [], []
    for inst in result.instances.values():
        x, y, w, h = inst.bbox_px
        xs.append(x + w / 2)
        ys.append(y + h / 2)
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def crop(result: RenderResult, box: tuple[int, int, int, int], min_visible: float = 0.35) -> RenderResult:
    """Crop the image and clip every label to the new frame.

    `min_visible` is the fraction of a building's box that must survive the crop for
    it to stay labelled. A sliver of a building at the frame edge is not something a
    detector should be asked to name.
    """
    x0, y0, x1, y1 = box
    image = result.image.crop(box)

    kept: dict[int, RenderedInstance] = {}
    dropped = list(result.dropped)

    for iid, inst in result.instances.items():
        bx, by, bw, bh = inst.bbox_px
        cx0, cy0 = max(bx, x0), max(by, y0)
        cx1, cy1 = min(bx + bw, x1), min(by + bh, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            dropped.append(iid)
            continue

        retained = ((cx1 - cx0) * (cy1 - cy0)) / (bw * bh)
        if retained < min_visible:
            dropped.append(iid)
            continue

        kept[iid] = replace(
            inst,
            bbox_px=(cx0 - x0, cy0 - y0, cx1 - cx0, cy1 - cy0),
            # Clipping hides part of the building just as an occluder would.
            visibility=round(min(1.0, inst.visibility * retained), 4),
            polygon_px=[(px - x0, py - y0) for px, py in inst.polygon_px],
        )

    return RenderResult(image, kept, result.projection, dropped)
