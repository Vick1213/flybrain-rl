"""PIL compositing of the "reels" phone-screen image: the current video
clip cropped by the scroll-slide offset, plus a minimal reels-style UI
overlay (progress bar, like/comment/share column, profile + captions,
and the big center "jackpot" heart pop).

Pure numpy/PIL, no MuJoCo here -- `fly3d.scene` calls `compose_frame` and
uploads the result into a MuJoCo 2D texture.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

W, H = 270, 480


def _draw_heart(draw: ImageDraw.ImageDraw, center, size, color):
    """Draws a filled heart shape (two lobes + a point) of roughly `size`
    px centred at `center`, in `color` (RGBA). Hand-drawn rather than a
    font glyph -- not every system font ships a heart character, and a
    missing glyph renders as a hollow "tofu" box instead.
    """
    x, y = center
    r = size / 4.0
    draw.ellipse([x - 2 * r, y - 1.3 * r, x, y + 0.7 * r], fill=color)
    draw.ellipse([x, y - 1.3 * r, x + 2 * r, y + 0.7 * r], fill=color)
    draw.polygon([(x - 2 * r, y - 0.1 * r), (x + 2 * r, y - 0.1 * r), (x, y + 2.3 * r)],
                fill=color)


def _draw_comment_icon(draw: ImageDraw.ImageDraw, center, r, color):
    x, y = center
    draw.rounded_rectangle([x - r, y - r * 0.75, x + r, y + r * 0.55], radius=r * 0.4,
                           outline=color, width=2)
    draw.polygon([(x - r * 0.3, y + r * 0.5), (x + r * 0.1, y + r * 0.5),
                 (x - r * 0.15, y + r * 1.05)], fill=color)


def _draw_share_icon(draw: ImageDraw.ImageDraw, center, r, color):
    x, y = center
    draw.line([(x - r, y + r * 0.6), (x, y - r * 0.8)], fill=color, width=3)
    draw.line([(x, y - r * 0.8), (x + r, y + r * 0.6)], fill=color, width=3)
    draw.line([(x - r * 0.65, y + r * 0.15), (x, y - r * 0.8), (x + r * 0.65, y + r * 0.15)],
              fill=color, width=3)


def compose_frame(cur_frame: np.ndarray, next_frame: np.ndarray, slide_frac: float,
                  clip_progress: float, heart_active: bool, jackpot_glow: float,
                  dimmed: bool = False) -> np.ndarray:
    """Compose one 270x480 RGB reels-screen frame.

    cur_frame/next_frame: (480, 270, 3) uint8 decoded video frames.
    slide_frac: 0 -> show cur_frame fully; 1 -> show next_frame fully
      (the feed has slid up by a full screen height); values in between
      show the transition mid-slide.
    clip_progress: 0..1, fraction through the current clip's loop, drawn
      as the thin top progress bar.
    heart_active: True while a jackpot like-pop is active -> right-column
      heart icon renders filled red instead of a plain outline heart.
    jackpot_glow: 0..1 fade-envelope for the big centre heart pop (0 = not
      shown).
    dimmed: caller already applies the "not at phone" 50% dim via the
      MuJoCo material tint; this flag is accepted for completeness but
      intentionally unused here (dimming stays a render-time tint, not
      baked into the cached pixels, so the un-dimmed frame stays cached).
    """
    slide_frac = float(np.clip(slide_frac, 0.0, 1.0))
    if slide_frac <= 0.0:
        visible = cur_frame
    elif slide_frac >= 1.0:
        visible = next_frame
    else:
        strip = np.concatenate([cur_frame, next_frame], axis=0)  # (2H, W, 3)
        y0 = int(round(slide_frac * H))
        y0 = min(max(y0, 0), H)
        visible = strip[y0:y0 + H]
        if visible.shape[0] < H:
            pad = np.repeat(visible[-1:], H - visible.shape[0], axis=0)
            visible = np.concatenate([visible, pad], axis=0)

    im = Image.fromarray(visible).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # subtle bottom gradient so white UI stays legible over bright video
    grad = Image.new("L", (1, H), 0)
    for y in range(H):
        grad.putpixel((0, y), int(140 * max(0.0, (y - H * 0.55) / (H * 0.45))))
    shade = Image.merge("RGBA", (Image.new("L", (1, H), 0), Image.new("L", (1, H), 0),
                                 Image.new("L", (1, H), 0), grad)).resize((W, H))
    overlay = Image.alpha_composite(overlay, shade)
    draw = ImageDraw.Draw(overlay)

    # thin progress bar, top
    pad = 6
    bar_y = 8
    draw.rounded_rectangle([pad, bar_y, W - pad, bar_y + 3], radius=2, fill=(255, 255, 255, 70))
    fill_w = pad + (W - 2 * pad) * float(np.clip(clip_progress, 0.0, 1.0))
    draw.rounded_rectangle([pad, bar_y, fill_w, bar_y + 3], radius=2, fill=(255, 255, 255, 235))

    # right-side icon column: heart / comment / share
    icon_x = W - 26
    heart_color = (255, 45, 85, 255) if heart_active else (255, 255, 255, 220)
    _draw_heart(draw, (icon_x, H - 168), 22, heart_color)
    _draw_comment_icon(draw, (icon_x, H - 122), 11, (255, 255, 255, 220))
    _draw_share_icon(draw, (icon_x, H - 80), 11, (255, 255, 255, 220))

    # profile circle + two caption bars, bottom-left
    draw.ellipse([12, H - 66, 38, H - 40], fill=(130, 170, 250, 255),
                 outline=(255, 255, 255, 230), width=2)
    draw.rounded_rectangle([14, H - 34, 150, H - 25], radius=4, fill=(255, 255, 255, 190))
    draw.rounded_rectangle([14, H - 22, 108, H - 13], radius=4, fill=(255, 255, 255, 130))

    # big centre heart pop on jackpot
    if jackpot_glow > 0.01:
        grow = float(np.clip(jackpot_glow, 0.0, 1.0))
        size = int(70 + 40 * grow)
        alpha = int(255 * grow)
        _draw_heart(draw, (W / 2, H / 2), size, (255, 35, 70, alpha))

    composed = Image.alpha_composite(im, overlay).convert("RGB")
    return np.asarray(composed, dtype=np.uint8)
