"""Builds a side-by-side comparison video from two rendered mp4s (e.g. the
addicted-vs-sober trained-fly renders from scripts/render_trained.sh):
each source video is resized to a PANEL_W x PANEL_H panel and horizontally
stacked into one PANEL_W*2 x PANEL_H frame; the shorter clip is padded by
freezing its last frame so both panels run for the full comparison's
length.

Run under `.venv-body` (needs imageio + Pillow, both installed there;
`.venv` has neither).

Usage
-----
    .venv-body/bin/python scripts/make_comparison_video.py A.mp4 B.mp4 OUT.mp4

Deliberately avoids `list(imageio.get_reader(path))` / relying on the
reader's reported frame count or length: imageio-ffmpeg's metadata for a
freshly-encoded h264 mp4 is not always a reliable *actual* frame count
(observed to make `list()` over-allocate and raise MemoryError even for a
short clip), so frame counts here are discovered by walking
`reader.get_data(i)` until it raises, and frames are streamed one pair at
a time rather than materialized as full-length lists.
"""

from __future__ import annotations

import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image

PANEL_W, PANEL_H = 640, 720


def _resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR))


def _count_frames(path: str) -> int:
    reader = imageio.get_reader(path)
    n = 0
    try:
        while True:
            reader.get_data(n)
            n += 1
    except (IndexError, EOFError):
        pass
    finally:
        reader.close()
    if n == 0:
        raise SystemExit(f"make_comparison_video: {path!r} has no readable frames")
    return n


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        raise SystemExit("usage: make_comparison_video.py A.mp4 B.mp4 OUT.mp4")
    path_a, path_b, out_path = argv

    n_a = _count_frames(path_a)
    n_b = _count_frames(path_b)
    n = max(n_a, n_b)
    if n_a != n_b:
        print(f"padding shorter clip ({min(n_a, n_b)} frames) to {n} frames "
              f"by freezing its last frame", file=sys.stderr)

    reader_a = imageio.get_reader(path_a)
    reader_b = imageio.get_reader(path_b)
    fps = reader_a.get_meta_data().get("fps", 30)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=None, bitrate=None, macro_block_size=1,
                                output_params=["-pix_fmt", "yuv420p", "-crf", "20"])
    try:
        for i in range(n):
            fa = reader_a.get_data(min(i, n_a - 1))
            fb = reader_b.get_data(min(i, n_b - 1))
            panel_a = _resize(fa, PANEL_W, PANEL_H)
            panel_b = _resize(fb, PANEL_W, PANEL_H)
            writer.append_data(np.concatenate([panel_a, panel_b], axis=1))
    finally:
        writer.close()
        reader_a.close()
        reader_b.close()
    print(f"wrote {out_path}: {n} frames, panels {PANEL_W}x{PANEL_H} -> {PANEL_W * 2}x{PANEL_H}")


if __name__ == "__main__":
    main()
