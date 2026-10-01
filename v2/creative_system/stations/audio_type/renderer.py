"""Paint transcribed speech onto the three Audio Type layouts.

Type waveform (bars) matches the neon reference:
each vertical line is one spoken word (letters stacked), and line height
follows the audio envelope — the wave is built from the user's words.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .plates import _font

Color = Tuple[int, int, int]
Point = Tuple[float, float]

HORIZONTAL_STOPS: List[Tuple[float, Color]] = [
    (0.00, (0, 210, 255)),
    (0.16, (40, 90, 255)),
    (0.32, (150, 40, 230)),
    (0.50, (255, 45, 160)),
    (0.66, (255, 50, 55)),
    (0.82, (255, 150, 30)),
    (1.00, (255, 220, 55)),
]

RADIAL_STOPS: List[Tuple[float, Color]] = [
    (0.00, (40, 175, 255)),
    (0.16, (120, 70, 255)),
    (0.30, (255, 55, 185)),
    (0.44, (255, 40, 90)),
    (0.58, (255, 125, 35)),
    (0.72, (255, 215, 50)),
    (0.86, (255, 70, 145)),
    (1.00, (40, 175, 255)),
]


def render_style(style: str, envelope: Sequence[float], text: str) -> Image.Image:
    if style == "heart":
        return render_heart(envelope, text)
    if style == "bars":
        return render_bars(envelope, text)
    return render_rings(envelope, text)


def render_rings(envelope: Sequence[float], text: str) -> Image.Image:
    size = 4096
    cx = cy = size / 2
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    env = _resample(envelope, 360)
    words = _spoken_words(text)
    phrase = _ring_text(words)

    sharp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    _draw_circular_waveform(sharp, env, cx, cy, radius=1680, bar_len=340, width=6)
    _draw_circular_waveform(sharp, _resample(envelope, 96), cx, cy, radius=380, bar_len=90, width=8)
    canvas = _neon_composite(canvas, sharp, blur=26, glow=1.4)

    type_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # Concentric arc text bands matching reference structure.
    ring_specs = (
        (560, 36),
        (700, 38),
        (840, 40),
        (980, 42),
        (1120, 44),
        (1260, 46),
        (1400, 48),
    )
    for radius, font_size in ring_specs:
        _draw_circular_text(
            type_layer,
            phrase,
            cx,
            cy,
            radius,
            _font(font_size),
            letter_tracking=0.98,
            outline_px=2,
            outline_alpha=175,
            max_tilt=90.0,
        )
    canvas = _neon_composite(canvas, type_layer, blur=3, glow=1.05)

    mic = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    _draw_microphone(mic, cx, cy, scale=2.8, color=(255, 90, 185))
    return _neon_composite(canvas, mic, blur=20, glow=1.5)


def render_heart(envelope: Sequence[float], text: str) -> Image.Image:
    """Wave heart: spoken words ride smooth ribbons that open into a heart.

    Side ribbons are a symmetric sine band. In the middle they peel into the
    heart silhouette (lobes, cleft, and point). Audio is only the bright
    oscilloscope on the horizontal center — it does not bend the type.
    """
    width, height = 4096, 2522
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    words = _spoken_words(text)
    env = _resample(envelope, width)
    tops, bots = _heart_top_bottom(width, height)

    # Eleven lanes above the axis and eleven below, matching the reference density.
    n_side = 11
    lanes: List[float] = []
    for i in range(n_side):
        mag = 0.20 + 0.80 * (i / (n_side - 1))
        lanes.append(mag)
        lanes.append(-mag)

    font_px = max(18, int(round(height * 0.0138)))
    outer_px = max(font_px + 2, int(round(height * 0.0160)))
    font = _font(font_px)
    outer_font = _font(outer_px)

    type_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for i, lane in enumerate(lanes):
        points = _heart_ribbon_path(width, height, lane, tops, bots)
        _draw_words_along_path(
            type_layer,
            words,
            points,
            outer_font if abs(lane) > 0.93 else font,
            start_index=(i * 3) % len(words),
            max_angle=88.0,
            letter_tracking=0.98,
            word_gap_scale=1.12,
        )

    canvas = _neon_composite(canvas, type_layer, blur=max(3, height // 240), glow=1.14)

    wave = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    _draw_center_waveform(wave, env, width, height, thickness=max(2, width // 1600), amp_scale=0.17)
    return _neon_composite(canvas, wave, blur=max(6, height // 130), glow=1.55)



def render_bars(envelope: Sequence[float], text: str) -> Image.Image:
    """Neon type-waveform: each vertical line spells one spoken word.

    Matches the reference art — audio sets line height; each column spells
    one user word once (readable), cycled across the wave.
    """
    width, height = 4096, 1536
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    words = _spoken_words(text)
    columns = 110
    env = _resample(envelope, columns)
    cy = height / 2
    max_half = height * 0.46
    col_w = width / columns

    type_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    for i, amp in enumerate(env):
        color = _gradient_at(i / max(1, columns - 1), HORIZONTAL_STOPS)
        x = (i + 0.5) * col_w
        half = max(22.0, amp * max_half)
        bar_w = max(5, int(round(col_w * 0.52)))

        glow_draw.rectangle(
            [x - bar_w / 2, cy - half, x + bar_w / 2, cy + half],
            fill=color + (85,),
        )

        _draw_spoken_word_line(type_layer, words[i % len(words)], x, cy, half, color, max_w=max(28, int(col_w * 0.7)))

    axis = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(axis).line([(0, cy), (width, cy)], fill=(255, 230, 245, 220), width=4)

    canvas = _neon_composite(canvas, glow_layer, blur=12, glow=1.18)
    canvas = _neon_composite(canvas, type_layer, blur=2, glow=1.04)
    return _neon_composite(canvas, axis, blur=8, glow=1.2)


def _draw_spoken_word_line(
    img: Image.Image,
    word: str,
    x: float,
    cy: float,
    half: float,
    color: Color,
    max_w: int = 48,
) -> None:
    """Draw one spoken word as a single vertical text line (readable).

    Word is rendered horizontally, rotated into a vertical line, then scaled
    to the local wave height — same structure as the neon reference.
    """
    label = "".join(ch for ch in word if not ch.isspace()) or "·"
    if half < 20:
        draw = ImageDraw.Draw(img)
        draw.rectangle([x - 1.5, cy - half, x + 1.5, cy + half], fill=color + (255,))
        return

    # Start large; we'll scale the rotated word to fit `half`.
    font = _font(64)
    bbox = font.getbbox(label)
    pad = 6
    tw = max(1, bbox[2] - bbox[0] + pad * 2)
    th = max(1, bbox[3] - bbox[1] + pad * 2)
    plate = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(plate).text(
        (pad - bbox[0], pad - bbox[1]),
        label,
        font=font,
        fill=(255, 255, 255, 255),
    )
    # Vertical word line (reads away from the center axis).
    vertical = plate.rotate(90, resample=Image.BICUBIC, expand=True)

    target_h = max(12, int(half * 0.96))
    scale = target_h / max(1, vertical.height)
    target_w = max(2, int(round(vertical.width * scale)))
    # Keep columns from becoming too wide for the wave density.
    if target_w > max_w:
        shrink = max_w / target_w
        target_w = max_w
        target_h = max(12, int(round(target_h * shrink)))
    vertical = vertical.resize((max(2, target_w), max(12, target_h)), resample=Image.LANCZOS)

    tinted = Image.new("RGBA", vertical.size, color + (0,))
    tinted.putalpha(vertical.split()[-1])

    px = int(round(x - tinted.width / 2))
    # Above center
    img.alpha_composite(tinted, (px, int(round(cy - tinted.height))))
    # Below center (flip so letters still read away from axis)
    img.alpha_composite(tinted.transpose(Image.FLIP_TOP_BOTTOM), (px, int(round(cy))))


def _spoken_words(text: str) -> List[str]:
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", str(text or ""), flags=re.UNICODE)
    return words or ["voice"]


def _ring_text(words: Sequence[str], max_words: int = 18, max_chars: int = 220) -> str:
    """Build ring text from exact input words only (no invented tokens)."""
    selected = [w for w in words[:max_words] if w]
    separator = " · "
    line = separator.join(selected).strip() or "voice"
    if len(line) > max_chars:
        line = line[:max_chars].rsplit(" ", 1)[0].strip() or line[:max_chars]
    return line + separator


def _measure_text(font, text: str) -> float:
    if hasattr(font, "getlength"):
        try:
            return float(font.getlength(text))
        except Exception:
            pass
    bbox = font.getbbox(text or " ")
    return float(max(1, bbox[2] - bbox[0]))


def _draw_circular_word_ring(
    img: Image.Image,
    words: Sequence[str],
    cx: float,
    cy: float,
    radius: float,
    font,
    word_gap_scale: float = 1.15,
    letter_tracking: float = 1.0,
    outline_px: int = 0,
    outline_alpha: int = 0,
    max_tilt: float = 85.0,
) -> None:
    """Draw exact words around a ring with readable orientation."""
    if radius <= 0 or not words:
        return
    clean_words = [w for w in words if w and w.strip()]
    if not clean_words:
        clean_words = ["voice"]

    circumference = 2.0 * math.pi * radius
    word_gap = _measure_text(font, " ") * max(0.2, word_gap_scale)
    sep_token = "·"
    sep_gap = _measure_text(font, " ") * 0.55
    safety_gap = _font_height(font) * 0.10
    baseline_nudge = _font_height(font) * 0.03

    tokens: List[str] = []
    for word in clean_words:
        tokens.append(word)
        tokens.append(sep_token)
    if not tokens:
        return

    widths = [_token_width(font, t, letter_tracking=letter_tracking) for t in tokens]
    gaps = [(sep_gap if t == sep_token else word_gap) + safety_gap for t in tokens]
    cycle_len = sum(w + g for w, g in zip(widths, gaps))
    if cycle_len <= 1.0:
        return

    pos = 0.0
    repeats = max(1, int(math.ceil(circumference / cycle_len)) + 1)
    for _ in range(repeats):
        for token, token_w, gap in zip(tokens, widths, gaps):
            if pos + token_w / 2.0 > circumference:
                return
            center_angle = -math.pi / 2 + ((pos + token_w / 2.0) / radius)
            x = cx + math.cos(center_angle) * (radius + baseline_nudge)
            y = cy + math.sin(center_angle) * (radius + baseline_nudge)
            _paste_rotated_token(
                img,
                token,
                (x, y),
                0.0,
                font,
                letter_tracking=letter_tracking,
                outline_px=outline_px,
                outline_alpha=outline_alpha,
                color=_radial_color(center_angle),
            )
            pos += token_w + gap


def _token_width(font, token: str, letter_tracking: float = 1.0) -> float:
    return sum(_advance(font, ch) * letter_tracking for ch in token) or 1.0


def _upright_tangent_rotation(angle_rad: float, max_tilt: float = 90.0) -> float:
    """Return tangent rotation while keeping tokens upright and non-mirrored."""
    rot = math.degrees(angle_rad) + 90.0
    normalized = rot % 360.0
    # Flip whole token when the tangent would make it upside down.
    if 90.0 < normalized < 270.0:
        rot += 180.0
    # Normalize only; avoid extra clamping that causes jitter/offset.
    return ((rot + 180.0) % 360.0) - 180.0


def _paste_rotated_token(
    img: Image.Image,
    token: str,
    xy: Point,
    angle_deg: float,
    font,
    color: Color,
    letter_tracking: float = 1.0,
    outline_px: int = 0,
    outline_alpha: int = 0,
) -> None:
    if not token:
        return
    size = int(getattr(font, "size", 12) or 12)
    tracking_key = max(600, min(2000, int(round(letter_tracking * 1000))))
    glyph = _rotated_token(token, int(round(angle_deg)) % 360, size, tracking_key)
    if glyph is None:
        return
    x = int(round(xy[0] - glyph.width / 2))
    y = int(round(xy[1] - glyph.height / 2))
    if outline_px > 0 and outline_alpha > 0:
        outline = _alpha_outline(glyph.split()[-1], outline_px, outline_alpha)
        if outline is not None:
            img.alpha_composite(outline, (x, y))
    tinted = Image.new("RGBA", glyph.size, color + (0,))
    tinted.putalpha(glyph.split()[-1])
    img.alpha_composite(tinted, (x, y))


@lru_cache(maxsize=16384)
def _rotated_token(token: str, angle: int, size: int, tracking_key: int) -> Image.Image | None:
    if not token:
        return None
    plate = _token_plate(token, size, tracking_key)
    if angle % 360 == 0:
        return plate
    return plate.rotate(angle, resample=Image.BICUBIC, expand=True)


@lru_cache(maxsize=8192)
def _token_plate(token: str, size: int, tracking_key: int) -> Image.Image:
    tracking = tracking_key / 1000.0
    font = _font(size)
    advances = [_advance(font, ch) * tracking for ch in token]
    bbox = font.getbbox("Hg")
    text_h = max(1, bbox[3] - bbox[1])
    pad = 6
    w = max(1, int(round(sum(advances) + pad * 2 + 2)))
    h = max(1, int(round(text_h + pad * 2)))
    plate = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(plate)
    x = float(pad)
    for ch, adv in zip(token, advances):
        cb = font.getbbox(ch)
        draw.text((x - cb[0], pad - bbox[1]), ch, font=font, fill=(255, 255, 255, 255))
        x += adv
    return plate


def _resample(values: Sequence[float], count: int) -> List[float]:
    if count <= 0:
        return []
    if not values:
        return [0.2] * count
    if len(values) == count:
        return list(values)
    out: List[float] = []
    last = len(values) - 1
    for i in range(count):
        pos = i * last / max(1, count - 1)
        lo = int(math.floor(pos))
        hi = min(last, lo + 1)
        t = pos - lo
        out.append(values[lo] * (1.0 - t) + values[hi] * t)
    return out


def _lerp_color(a: Color, b: Color, t: float) -> Color:
    t = min(1.0, max(0.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _gradient_at(t: float, stops: Sequence[Tuple[float, Color]]) -> Color:
    t = min(1.0, max(0.0, t))
    for i in range(1, len(stops)):
        t1, c1 = stops[i]
        t0, c0 = stops[i - 1]
        if t <= t1:
            span = (t1 - t0) or 1e-6
            return _lerp_color(c0, c1, (t - t0) / span)
    return stops[-1][1]


def _radial_color(angle: float) -> Color:
    t = ((math.pi - angle) / (2 * math.pi)) % 1.0
    return _gradient_at(t, RADIAL_STOPS)


def _font_height(font) -> float:
    if hasattr(font, "size") and font.size:
        return float(font.size)
    bbox = font.getbbox("Hg")
    return float(max(8, bbox[3] - bbox[1]))


def _advance(font, ch: str) -> float:
    if hasattr(font, "getlength"):
        try:
            return max(1.0, float(font.getlength(ch)))
        except Exception:
            pass
    bbox = font.getbbox(ch or " ")
    return float(max(1, bbox[2] - bbox[0] + 1))


def _neon_composite(base: Image.Image, layer: Image.Image, blur: int = 12, glow: float = 1.3) -> Image.Image:
    if layer.mode != "RGBA":
        layer = layer.convert("RGBA")
    if blur <= 0:
        return Image.alpha_composite(base.convert("RGBA"), layer)

    arr = np.asarray(layer, dtype=np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    premul = np.clip(arr[:, :, :3] * alpha, 0, 255).astype(np.uint8)
    blur_rgb = Image.fromarray(premul, "RGB").filter(ImageFilter.GaussianBlur(blur))
    blur_a = layer.getchannel("A").filter(ImageFilter.GaussianBlur(blur))
    br = np.asarray(blur_rgb, dtype=np.float32)
    ba = np.asarray(blur_a, dtype=np.float32)
    out = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.float32)
    mask = ba > 0.5
    ba_norm = np.maximum(ba / 255.0, 1e-6)
    out[mask, 0] = np.clip(br[mask, 0] / ba_norm[mask], 0, 255)
    out[mask, 1] = np.clip(br[mask, 1] / ba_norm[mask], 0, 255)
    out[mask, 2] = np.clip(br[mask, 2] / ba_norm[mask], 0, 255)
    out[:, :, 3] = np.clip(ba * min(1.0, 0.55 + 0.35 * glow), 0, 255)
    if glow != 1.0:
        out[:, :, :3] = np.clip(out[:, :, :3] * glow, 0, 255)
    glow_img = Image.fromarray(out.astype(np.uint8), "RGBA")
    composed = Image.alpha_composite(base.convert("RGBA"), glow_img)
    return Image.alpha_composite(composed, layer)


def _draw_circular_waveform(
    img: Image.Image,
    env: Sequence[float],
    cx: float,
    cy: float,
    radius: float,
    bar_len: float,
    width: int,
) -> None:
    draw = ImageDraw.Draw(img)
    n = len(env)
    if not n:
        return
    for i, amp in enumerate(env):
        angle = -math.pi / 2 + (2 * math.pi * i / n)
        length = bar_len * (0.18 + 0.82 * amp)
        x0 = cx + math.cos(angle) * radius
        y0 = cy + math.sin(angle) * radius
        x1 = cx + math.cos(angle) * (radius + length)
        y1 = cy + math.sin(angle) * (radius + length)
        draw.line([(x0, y0), (x1, y1)], fill=_radial_color(angle) + (255,), width=width)


def _draw_circular_text(
    img: Image.Image,
    text: str,
    cx: float,
    cy: float,
    radius: float,
    font,
    letter_tracking: float = 1.04,
    outline_px: int = 0,
    outline_alpha: int = 0,
    max_tilt: float = 90.0,
) -> None:
    """Draw two readable arcs (top + bottom) for one circular text ring."""
    if not text or radius <= 0:
        return
    pad = 0.015
    # Top: left -> top -> right (clockwise), outward-facing letters.
    _fill_text_arc(
        img,
        text,
        cx,
        cy,
        radius,
        font,
        start_angle=math.pi + pad,
        end_angle=2 * math.pi - pad,
        tops_outward=True,
        letter_tracking=letter_tracking,
        outline_px=outline_px,
        outline_alpha=outline_alpha,
        max_tilt=max_tilt,
    )
    # Bottom: left -> bottom -> right (counter-clockwise), inward-facing letters.
    _fill_text_arc(
        img,
        text,
        cx,
        cy,
        radius,
        font,
        start_angle=math.pi - pad,
        end_angle=pad,
        tops_outward=False,
        letter_tracking=letter_tracking,
        outline_px=outline_px,
        outline_alpha=outline_alpha,
        max_tilt=max_tilt,
    )


def _fill_text_arc(
    img: Image.Image,
    text: str,
    cx: float,
    cy: float,
    radius: float,
    font,
    start_angle: float,
    end_angle: float,
    tops_outward: bool,
    letter_tracking: float,
    outline_px: int,
    outline_alpha: int,
    max_tilt: float,
) -> None:
    span = end_angle - start_angle
    if abs(span) < 1e-6:
        return
    arc_len = abs(span) * radius
    direction = 1.0 if span > 0 else -1.0
    unit = text if text.endswith(" ") else text + " "
    unit_w = sum(_advance(font, ch) * letter_tracking for ch in unit) or 1.0
    stream = unit * (max(1, int(math.ceil(arc_len / unit_w)) + 1))
    baseline_nudge = _font_height(font) * (0.18 if tops_outward else -0.18)

    pos = 0.0
    for ch in stream:
        adv = _advance(font, ch) * letter_tracking
        if pos + adv / 2 > arc_len:
            break
        if ch != " ":
            mid = pos + adv / 2
            angle = start_angle + direction * (mid / radius)
            rr = radius + baseline_nudge
            x = cx + math.cos(angle) * rr
            y = cy + math.sin(angle) * rr
            rot = math.degrees(angle) + (90.0 if tops_outward else -90.0)
            # Keep natural arc orientation; avoid additional flips/jitter.
            rot = ((rot + 180.0) % 360.0) - 180.0
            _paste_rotated_char(
                img,
                ch,
                (x, y),
                rot,
                font,
                _radial_color(angle),
                outline_px=outline_px,
                outline_alpha=outline_alpha,
            )
        pos += adv


def _paste_rotated_char(
    img: Image.Image,
    ch: str,
    xy: Point,
    angle_deg: float,
    font,
    color: Color,
    outline_px: int = 0,
    outline_alpha: int = 0,
) -> None:
    glyph = _rotated_glyph(ch, int(round(angle_deg)) % 360, int(getattr(font, "size", 12) or 12))
    if glyph is None:
        return
    x = int(round(xy[0] - glyph.width / 2))
    y = int(round(xy[1] - glyph.height / 2))
    if outline_px > 0 and outline_alpha > 0:
        outline = _rotated_glyph_outline(
            ch,
            int(round(angle_deg)) % 360,
            int(getattr(font, "size", 12) or 12),
            outline_px,
            min(255, max(0, int(outline_alpha))),
        )
        if outline is not None:
            img.alpha_composite(outline, (x, y))
    tinted = Image.new("RGBA", glyph.size, color + (0,))
    tinted.putalpha(glyph.split()[-1])
    img.alpha_composite(tinted, (x, y))


def _paste_rotated_word(
    img: Image.Image,
    word: str,
    xy: Point,
    angle_deg: float,
    font,
    color: Color,
    outline_px: int = 0,
    outline_alpha: int = 0,
) -> None:
    glyph = _rotated_word(word, int(round(angle_deg)) % 360, int(getattr(font, "size", 12) or 12))
    if glyph is None:
        return
    x = int(round(xy[0] - glyph.width / 2))
    y = int(round(xy[1] - glyph.height / 2))
    if outline_px > 0 and outline_alpha > 0:
        outline = _alpha_outline(glyph.split()[-1], outline_px, outline_alpha)
        if outline is not None:
            img.alpha_composite(outline, (x, y))
    tinted = Image.new("RGBA", glyph.size, color + (0,))
    tinted.putalpha(glyph.split()[-1])
    img.alpha_composite(tinted, (x, y))


def _upright_rotation(angle_deg: float, max_tilt: float = 90.0) -> float:
    """Map any angle to an upright, readability-friendly tilt."""
    # Normalize to [-180, 180].
    normalized = ((angle_deg + 180.0) % 360.0) - 180.0
    # Fold into [-90, 90] so letters never appear upside down.
    if normalized > 90.0:
        normalized -= 180.0
    elif normalized < -90.0:
        normalized += 180.0
    limit = max(5.0, min(90.0, float(max_tilt)))
    return max(-limit, min(limit, normalized))


@lru_cache(maxsize=8192)
def _rotated_word(word: str, angle: int, size: int) -> Image.Image | None:
    if not word:
        return None
    font = _font(size)
    bbox = font.getbbox(word)
    pad = 6
    w = max(1, bbox[2] - bbox[0] + pad * 2)
    h = max(1, bbox[3] - bbox[1] + pad * 2)
    plate = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(plate).text(
        (pad - bbox[0], pad - bbox[1]),
        word,
        font=font,
        fill=(255, 255, 255, 255),
    )
    return plate.rotate(angle, resample=Image.BICUBIC, expand=True)


def _alpha_outline(alpha: Image.Image, outline_px: int, outline_alpha: int) -> Image.Image | None:
    if outline_px <= 0 or outline_alpha <= 0:
        return None
    kernel = max(3, outline_px * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    expanded = alpha.filter(ImageFilter.MaxFilter(kernel))
    oa = min(255, max(0, int(outline_alpha)))
    if oa < 255:
        expanded = expanded.point(lambda a: (a * oa) // 255)
    outline = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    outline.putalpha(expanded)
    return outline


@lru_cache(maxsize=16384)
def _rotated_glyph(ch: str, angle: int, size: int) -> Image.Image | None:
    if ch == " ":
        return None
    scale = 2 if size <= 40 else 1
    font = _font(size * scale)
    bbox = font.getbbox(ch)
    pad = 5 * scale
    w = max(1, bbox[2] - bbox[0] + pad * 2)
    h = max(1, bbox[3] - bbox[1] + pad * 2)
    glyph = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glyph).text(
        (pad - bbox[0], pad - bbox[1]),
        ch,
        font=font,
        fill=(255, 255, 255, 255),
    )
    rotated = glyph.rotate(angle, resample=Image.BICUBIC, expand=True)
    if scale == 1:
        return rotated
    target = (
        max(1, int(round(rotated.width / scale))),
        max(1, int(round(rotated.height / scale))),
    )
    return rotated.resize(target, resample=Image.LANCZOS)


@lru_cache(maxsize=16384)
def _rotated_glyph_outline(
    ch: str,
    angle: int,
    size: int,
    outline_px: int,
    outline_alpha: int,
) -> Image.Image | None:
    glyph = _rotated_glyph(ch, angle, size)
    if glyph is None or outline_px <= 0 or outline_alpha <= 0:
        return None
    return _alpha_outline(glyph.split()[-1], outline_px, outline_alpha)


def _draw_microphone(img: Image.Image, cx: float, cy: float, scale: float, color: Color) -> None:
    draw = ImageDraw.Draw(img)
    s = 28 * scale
    fill = color + (255,)
    draw.rounded_rectangle(
        [cx - s * 0.42, cy - s * 1.15, cx + s * 0.42, cy + s * 0.15],
        radius=s * 0.42,
        outline=fill,
        width=max(3, int(3 * scale)),
    )
    for k in (-0.55, -0.25, 0.05):
        y = cy + s * k
        draw.line([(cx - s * 0.28, y), (cx + s * 0.28, y)], fill=fill, width=max(2, int(2 * scale)))
    draw.arc(
        [cx - s * 0.72, cy - s * 0.55, cx + s * 0.72, cy + s * 0.55],
        start=20,
        end=160,
        fill=fill,
        width=max(3, int(3 * scale)),
    )
    draw.line([(cx, cy + s * 0.55), (cx, cy + s * 0.95)], fill=fill, width=max(3, int(3 * scale)))
    draw.line(
        [(cx - s * 0.38, cy + s * 0.95), (cx + s * 0.38, cy + s * 0.95)],
        fill=fill,
        width=max(3, int(3 * scale)),
    )


def _side_half_amp(u: float) -> float:
    """Smooth side-band half-height, as a fraction of the canvas height.

    One broad swell on the left and one on the right. The middle stays
    narrow so the heart can open out of it.
    """
    edge = _smoothstep(0.0, 0.05, u) * (1.0 - _smoothstep(0.95, 1.0, u))
    left = math.exp(-((u - 0.16) ** 2) / 0.016)
    right = math.exp(-((u - 0.84) ** 2) / 0.014)
    # Keep a real ribbon at the ends (don't pinch to a point) and stay
    # shorter than the heart lobes.
    amp = 0.11 + 0.11 * left + 0.13 * right
    return amp * (0.78 + 0.22 * edge)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    span = edge1 - edge0
    if span == 0:
        return 1.0 if x >= edge1 else 0.0
    t = (x - edge0) / span
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _heart_top_bottom(width: int, height: int) -> Tuple[List[float | None], List[float | None]]:
    """Rounded heart: ellipse lobes on top, cubic cheeks into a soft point.

    Two circles meet at a sharp cleft, so the top is smoothed after sampling.
    The bottom stays wide through the cheeks, then tails into the point.
    """
    cx = width * 0.50
    lobe_y = height * 0.355
    rx = width * 0.122
    ry = height * 0.275
    sep = rx * 0.88
    tip_y = height * 0.935
    half_w = sep + rx
    drop = tip_y - lobe_y

    tops: List[float | None] = [None] * width
    bots: List[float | None] = [None] * width
    x0 = max(0, int(math.floor(cx - half_w)))
    x1 = min(width - 1, int(math.ceil(cx + half_w)))
    for x in range(x0, x1 + 1):
        upper: List[float] = []
        for lobe_cx in (cx - sep, cx + sep):
            nx = (x - lobe_cx) / rx
            if abs(nx) <= 1.0:
                upper.append(lobe_y - ry * math.sqrt(1.0 - nx * nx))
        if upper:
            tops[x] = min(upper)

    # Bottom: cubic from each side point. The first control drops almost
    # straight down (round cheek); the second pulls in toward a soft tip.
    def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
        u = 1.0 - t
        return (
            u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
            u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
        )

    buckets: List[List[float]] = [[] for _ in range(width)]
    steps = 2400
    for side in (-1.0, 1.0):
        p0 = (cx + side * half_w, lobe_y)
        p1 = (cx + side * half_w * 0.96, lobe_y + drop * 0.42)
        p2 = (cx + side * half_w * 0.34, tip_y - drop * 0.08)
        p3 = (cx, tip_y)
        for i in range(steps + 1):
            x, y = _cubic(p0, p1, p2, p3, i / steps)
            ix = int(round(x))
            if 0 <= ix < width:
                buckets[ix].append(y)
    for x in range(width):
        if buckets[x]:
            bots[x] = max(buckets[x])
    filled = [i for i, y in enumerate(bots) if y is not None]
    for a, b in zip(filled, filled[1:]):
        if b - a <= 1 or bots[a] is None or bots[b] is None:
            continue
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            bots[i] = bots[a] * (1.0 - t) + bots[b] * t

    _smooth_optional(tops, radius=max(8, width // 70))
    _smooth_optional(bots, radius=max(6, width // 110))
    return tops, bots


def _smooth_optional(values: List[float | None], radius: int) -> None:
    idx = [i for i, v in enumerate(values) if v is not None]
    if len(idx) < 3 or radius <= 0:
        return
    smoothed = _smooth_series([float(values[i]) for i in idx], radius)
    for i, y in zip(idx, smoothed):
        values[i] = y


def _heart_ribbon_path(
    width: int,
    height: int,
    lane: float,
    tops: Sequence[float | None],
    bots: Sequence[float | None],
) -> List[Point]:
    """One left-to-right ribbon.

    `lane` > 0 is above the center axis. Outside the heart every lane follows
    the same sine envelope. Inside, upper lanes track the lobes and cleft,
    lower lanes track the point, and the axis itself stays straight.
    """
    cy = height * 0.5
    step = 2
    xs: List[float] = []
    raw: List[float] = []
    for x in range(0, width, step):
        u = x / max(1, width - 1)
        amp = _side_half_amp(u) * height
        y_side = cy - lane * amp
        top = tops[x] if x < len(tops) else None
        bot = bots[x] if x < len(bots) else None
        if top is None or bot is None:
            xs.append(float(x))
            raw.append(y_side)
            continue
        span = bot - top
        influence = _smoothstep(0.27, 0.36, u) * (1.0 - _smoothstep(0.64, 0.73, u))
        influence *= _smoothstep(height * 0.035, height * 0.16, span)
        if lane >= 0.0:
            y_heart = (1.0 - lane) * cy + lane * top
        else:
            mag = -lane
            y_heart = (1.0 - mag) * cy + mag * bot
        xs.append(float(x))
        raw.append(y_side * (1.0 - influence) + y_heart * influence)

    smoothed = _smooth_series(raw, radius=max(4, width // 320))
    return list(zip(xs, smoothed))


def _smooth_series(values: Sequence[float], radius: int = 5) -> List[float]:
    if radius <= 0 or len(values) < 3:
        return list(values)
    sigma = max(1.0, radius / 2.4)
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(values, dtype=np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def _draw_words_along_path(
    img: Image.Image,
    words: Sequence[str],
    points: Sequence[Point],
    font,
    start_index: int = 0,
    max_angle: float = 48.0,
    letter_tracking: float = 0.94,
    word_gap_scale: float = 1.35,
) -> int:
    """Place spoken words along a wireframe path (readable, curve-following).

    Unique words run left→right; when the list ends they repeat. Returns the
    next word index so ribbons can continue the same sequence across the art.
    """
    if len(points) < 2 or not words:
        return start_index
    dists = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dists.append(dists[-1] + math.hypot(dx, dy))
    total = dists[-1]
    if total < 16:
        return start_index

    width = img.size[0]
    space_w = _advance(font, " ") * word_gap_scale
    pos = 0.0
    wi = start_index
    n = len(words)

    while pos < total - 4:
        word = words[wi % n]
        wi += 1
        # Measure word so we can bail early near the path end.
        if pos + 8 > total:
            break
        for ch in word:
            adv = _advance(font, ch) * letter_tracking
            if pos + adv > total:
                return wi
            pt, tangent = _point_at(points, dists, min(total, pos + adv / 2))
            # Paths run left→right, so the tangent stays in (-90, 90) and
            # letters stay upright. Steep heart sides need the real tangent;
            # clamping it is what turned the type into scribbles.
            angle = math.degrees(math.atan2(tangent[1], tangent[0]))
            if angle > max_angle:
                angle = max_angle
            elif angle < -max_angle:
                angle = -max_angle
            color = _gradient_at(pt[0] / max(1, width - 1), HORIZONTAL_STOPS)
            _paste_rotated_char(img, ch, pt, angle, font, color)
            pos += adv
        pos += space_w

    return wi


def _draw_text_along_path(
    img: Image.Image,
    text: str,
    points: Sequence[Point],
    font,
    max_angle: float = 28.0,
    tracking: float = 0.94,
) -> None:
    if len(points) < 2 or not text:
        return
    dists = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dists.append(dists[-1] + math.hypot(dx, dy))
    total = dists[-1]
    if total < 8:
        return
    width = img.size[0]
    pos = 0.0
    idx = 0
    n = len(text)
    while pos < total - 1:
        ch = text[idx % n]
        idx += 1
        adv = _advance(font, ch) * tracking
        if ch != " ":
            pt, tangent = _point_at(points, dists, min(total, pos + adv / 2))
            angle = max(-max_angle, min(max_angle, math.degrees(math.atan2(tangent[1], tangent[0]))))
            color = _gradient_at(pt[0] / max(1, width - 1), HORIZONTAL_STOPS)
            _paste_rotated_char(img, ch, pt, angle, font, color)
        pos += adv


def _point_at(points: Sequence[Point], dists: Sequence[float], s: float) -> Tuple[Point, Point]:
    n = len(points)
    if s <= 0:
        return points[0], (points[1][0] - points[0][0], points[1][1] - points[0][1])
    if s >= dists[-1]:
        return points[-1], (points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])
    lo, hi = 0, n - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if dists[mid] <= s:
            lo = mid
        else:
            hi = mid
    span = dists[hi] - dists[lo] or 1e-6
    t = (s - dists[lo]) / span
    x = points[lo][0] + (points[hi][0] - points[lo][0]) * t
    y = points[lo][1] + (points[hi][1] - points[lo][1]) * t
    return (x, y), (points[hi][0] - points[lo][0], points[hi][1] - points[lo][1])


def _draw_center_waveform(
    img: Image.Image,
    env: Sequence[float],
    width: int,
    height: int,
    thickness: int,
    amp_scale: float = 0.22,
) -> None:
    """Thin mirrored spikes along the center axis, taller through the heart."""
    draw = ImageDraw.Draw(img)
    cy = height / 2
    amp = height * amp_scale
    step = max(8, width // 340)
    spike_w = max(2, thickness)
    cap = height * 0.20
    for x in range(0, width, step):
        u = x / max(1, width - 1)
        ei = min(len(env) - 1, int(round(u * (len(env) - 1))))
        value = env[ei]
        detail = 0.40 + 0.60 * abs(math.sin(x * 0.105))
        heart = max(0.0, 1.0 - abs(u - 0.5) / 0.16)
        boost = 1.0 + 0.22 * heart
        h = min(cap, max(2.0, value * amp * detail * boost))
        color = _gradient_at(u, HORIZONTAL_STOPS) + (255,)
        draw.line([(x, cy - h), (x, cy + h)], fill=color, width=spike_w)

