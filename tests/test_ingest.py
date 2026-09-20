"""Ingesting animated sources into aligned frames."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.ingest_sprites import align_frames, load_frames  # noqa: E402

from cocsynth.placeholders import make_sprite  # noqa: E402


def _offset_gif(path: Path, n=6, step=3):
    """An animation whose subject drifts, so misalignment would be visible."""
    frames = []
    for i in range(n):
        base, _ = make_sprite("cannon", "defense", (3, 3), 7, 32)
        canvas = Image.new("RGBA", (base.width + 30, base.height + 30), (0, 0, 0, 0))
        canvas.paste(base, (i * step, 10))
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0, disposal=2)
    return frames


def test_reads_every_frame_of_an_animation(tmp_path):
    _offset_gif(tmp_path / "a.gif", n=6)
    assert len(load_frames(tmp_path / "a.gif", max_frames=99)) == 6


def test_samples_evenly_across_the_whole_clip(tmp_path):
    """Taking the first N frames would miss the rest of the loop."""
    _offset_gif(tmp_path / "a.gif", n=12)
    picked = load_frames(tmp_path / "a.gif", max_frames=4)
    assert len(picked) == 4
    widths = [f.getbbox()[0] for f in picked]
    assert widths == sorted(widths) and widths[-1] > widths[0], "frames came from the start only"


def test_a_static_image_reads_as_one_frame(tmp_path):
    make_sprite("cannon", "defense", (3, 3), 7, 32)[0].save(tmp_path / "s.png")
    assert len(load_frames(tmp_path / "s.png", max_frames=8)) == 1


def test_alignment_gives_every_frame_the_same_box(tmp_path):
    """Trimming frames independently makes the building jitter between them."""
    frames = _offset_gif(tmp_path / "a.gif", n=6, step=4)
    aligned = align_frames(frames)
    assert len({f.size for f in aligned}) == 1

    # Independently trimmed frames would NOT agree -- that is the bug being prevented.
    assert len({f.crop(f.getbbox()).size for f in frames}) == 1  # same subject...
    offsets = {f.getbbox()[0] for f in frames}
    assert len(offsets) > 1, "test fixture is not actually offset"


def test_alignment_preserves_relative_motion(tmp_path):
    """The subject must still move within the shared box, or we cropped the motion out."""
    frames = _offset_gif(tmp_path / "a.gif", n=6, step=4)
    aligned = align_frames(frames)
    assert len({f.tobytes() for f in aligned}) == len(aligned)


def test_all_transparent_input_is_rejected():
    with pytest.raises(SystemExit, match="transparent"):
        align_frames([Image.new("RGBA", (10, 10), (0, 0, 0, 0))])
