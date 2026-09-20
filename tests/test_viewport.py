"""Camera cropping, and clipping labels to the frame."""
from random import Random

from cocsynth.viewport import content_centre, crop, pick_window


def _rendered(catalog, library, placements, tile_w=32):
    from cocsynth.render import Renderer
    return Renderer(catalog, library, tile_w=tile_w).render(placements, Random(1))


def test_window_stays_inside_the_image():
    for _ in range(50):
        box = pick_window((1400, 800), (640, 480), (700, 400), Random(_))
        assert box[0] >= 0 and box[1] >= 0
        assert box[2] <= 1400 and box[3] <= 800
        assert box[2] - box[0] == 640 and box[3] - box[1] == 480


def test_window_clamps_a_view_larger_than_the_image():
    box = pick_window((300, 200), (640, 480), (150, 100), Random(1))
    assert (box[2] - box[0], box[3] - box[1]) == (300, 200)


def test_crop_clips_boxes_and_drops_what_falls_outside(catalog, library, placer):
    result = _rendered(catalog, library, placer.generate(9, Random(2)).placements)
    focus = content_centre(result, (result.image.width // 2, result.image.height // 2))
    box = pick_window(result.image.size, (600, 400), focus, Random(3))
    cropped = crop(result, box)

    assert cropped.image.size == (600, 400)
    assert len(cropped.instances) < len(result.instances), "nothing was cut -- view too big"
    for inst in cropped.instances.values():
        x, y, w, h = inst.bbox_px
        assert 0 <= x and 0 <= y and x + w <= 600 and y + h <= 400
        assert w > 0 and h > 0
        assert 0.0 < inst.visibility <= 1.0


def test_clipped_buildings_lose_visibility(catalog, library, placer):
    """A building half out of frame is half visible, so the same threshold that
    drops occluded buildings also drops ones barely in shot."""
    result = _rendered(catalog, library, placer.generate(9, Random(4)).placements)
    focus = content_centre(result, (result.image.width // 2, result.image.height // 2))
    cropped = crop(result, pick_window(result.image.size, (500, 350), focus, Random(5)))
    before = {i: v.visibility for i, v in result.instances.items()}
    assert any(v.visibility < before[i] for i, v in cropped.instances.items())


def test_polygons_move_with_the_crop(catalog, library, placer):
    result = _rendered(catalog, library, placer.generate(9, Random(6)).placements)
    box = pick_window(result.image.size, (700, 500), content_centre(result, (0, 0)), Random(7))
    cropped = crop(result, box)
    for iid, inst in cropped.instances.items():
        ox, oy = result.instances[iid].polygon_px[0]
        assert inst.polygon_px[0] == (ox - box[0], oy - box[1])
