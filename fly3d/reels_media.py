"""Loads and preprocesses short video clips for the REELS phone screen.

Clips live in renders/assets/reels/*.mp4 (see SOURCES.md for where the
bundled ones came from). Any extra user-supplied *.mp4 dropped into that
same folder is picked up automatically on the next (cache-invalidated)
load.

Each clip is center-cropped to 9:16, resized to 270x480, trimmed to at
most REELS_MAX_SECONDS at REELS_FPS, and decoded to a plain uint8
(T, 480, 270, 3) numpy array. All clips are cached together in one
`reels_cache.npz` next to the source files so that after the first run,
scene start-up just loads the npz (no ffmpeg decode) -- the cache is
keyed on each source file's name/size/mtime and rebuilt automatically if
that signature changes (e.g. a clip is added/replaced).
"""

from __future__ import annotations

import glob
import math
import os
from typing import Dict, List, Tuple

import numpy as np

REELS_W, REELS_H = 270, 480
REELS_FPS = 15
REELS_MAX_SECONDS = 5.0
REELS_MAX_FRAMES = int(round(REELS_FPS * REELS_MAX_SECONDS))
CACHE_NAME = "reels_cache.npz"


def _source_signature(paths: List[str]) -> str:
    parts = []
    for p in sorted(paths):
        st = os.stat(p)
        parts.append(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts)


def _decode_clip(path: str, size=(REELS_W, REELS_H), fps: int = REELS_FPS,
                 max_frames: int = REELS_MAX_FRAMES) -> np.ndarray:
    import imageio
    from PIL import Image

    target_w, target_h = size
    target_aspect = target_w / target_h

    reader = imageio.get_reader(path, "ffmpeg")
    try:
        meta = reader.get_meta_data()
        src_fps = float(meta.get("fps") or 24.0)
        step = max(1, round(src_fps / fps))
        # Don't decode more of the source than we could possibly need.
        max_src_frames = int(math.ceil(REELS_MAX_SECONDS * src_fps)) + step

        frames = []
        for i, frame in enumerate(reader):
            if i >= max_src_frames or len(frames) >= max_frames:
                break
            if i % step != 0:
                continue
            h, w = frame.shape[:2]
            src_aspect = w / h
            if src_aspect > target_aspect:
                new_w = max(1, int(round(h * target_aspect)))
                x0 = (w - new_w) // 2
                crop = frame[:, x0:x0 + new_w]
            else:
                new_h = max(1, int(round(w / target_aspect)))
                y0 = (h - new_h) // 2
                crop = frame[y0:y0 + new_h, :]
            img = Image.fromarray(crop).convert("RGB").resize(size, Image.LANCZOS)
            frames.append(np.asarray(img, dtype=np.uint8))
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"could not decode any frames from {path}")
    return np.stack(frames, axis=0)  # (T, H, W, 3) uint8


def load_reels_clips(assets_dir: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Returns (clip_names, {name: frames (T,480,270,3) uint8}) for every
    *.mp4 in `assets_dir`, using/populating a cached .npz alongside them.
    """
    paths = sorted(glob.glob(os.path.join(assets_dir, "*.mp4")))
    if not paths:
        return [], {}

    cache_path = os.path.join(assets_dir, CACHE_NAME)
    sig = _source_signature(paths)

    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as npz:
                if str(npz["signature"][0]) == sig:
                    names = [str(n) for n in npz["names"]]
                    frames = {name: npz[f"clip_{i}"] for i, name in enumerate(names)}
                    return names, frames
        except Exception:
            pass  # fall through and rebuild

    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    frames = {}
    for p, name in zip(paths, names):
        frames[name] = _decode_clip(p)

    save_kwargs = {"signature": np.array([sig]), "names": np.array(names)}
    for i, name in enumerate(names):
        save_kwargs[f"clip_{i}"] = frames[name]
    try:
        np.savez_compressed(cache_path, **save_kwargs)
    except Exception:
        pass  # cache is a pure speedup; a failure to write it isn't fatal

    return names, frames
