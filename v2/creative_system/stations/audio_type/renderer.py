"""Paint transcribed speech onto the three Audio Type layouts."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

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
    size = 2048
    cx = cy = size / 2
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    phrase = _phrase(text)
    env = _resample(envelope, 360)

    sharp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    _draw_circular_waveform(sharp, env, cx, cy, radius=840, bar_len=170, width=3)
    _draw_circular_waveform(sharp, _resample(envelope, 96), cx, cy, radius=175, bar_len=46, width=4)
    canvas = _neon_composite(canvas, sharp, blur=16, glow=1.4)

    type_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for radius, font_size in ((250, 18), (340, 19), (440, 20), (545, 21), (655, 22)):
        _draw_circular_text(type_layer, phrase, cx, cy, radius, _font(font_size))
    canvas = _neon_composite(canvas, type_layer, blur=8, glow=1.12)

    mic = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    _draw_microphone(mic, cx, cy, scale=1.4, color=(255, 90, 185))
    return _neon_composite(canvas, mic, blur=14, glow=1.5).convert("RGB")


def render_heart(envelope: Sequence[float], text: str) -> Image.Image:
    width, height = 2048, 1152
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    phrase = _phrase(text)
    env = _resample(envelope, width)
    font = _font(14)
    n_rows = 36
    tops, bots = _heart_top_bottom(width, height)
    cy = height / 2
    band_top = cy - height * 0.18
    band_bot = cy + height * 0.18

    type_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for row in range(n_rows):
        t = row / (n_rows - 1)
        points: List[Point] = []
        for x in range(0, width, 2):
            e_top, e_bot = band_top, band_bot
            top, bot = tops[x], bots[x]
            if top is not None and bot is not None:
                e_top = min(band_top, top)
                e_bot = max(band_bot, bot)
            y = e_top + (e_bot - e_top) * t
            y += (env[x] - 0.35) * 14.0 * (1.0 - abs(t - 0.5) * 0.45)
            y += math.sin(x * 0.014 + row * 0.28) * 2.2
            points.append((float(x), y))
        _draw_text_along_path(type_layer, phrase, points, font)
    canvas = _neon_composite(canvas, type_layer, blur=6, glow=1.12)

    wave = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    _draw_center_waveform(wave, env, width, height, thickness=3)
    return _neon_composite(canvas, wave, blur=11, glow=1.5).convert("RGB")


def render_bars(envelope: Sequence[float], text: str) -> Image.Image:
    width, height = 2048, 768
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    phrase = _phrase(text)
    columns = 280
    env = _resample(envelope, columns)
    font = _font(9)
    cy = height / 2
    max_half = height * 0.44
    col_w = width / columns
    glyphs = [ch for ch in phrase if not ch.isspace()] or list(phrase)
    char_h = max(7, int(_font_height(font) * 0.82))

    type_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    line_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    type_draw = ImageDraw.Draw(type_layer)
    line_draw = ImageDraw.Draw(line_layer)
    for i, amp in enumerate(env):
        color = _gradient_at(i / max(1, columns - 1), HORIZONTAL_STOPS)
        x = (i + 0.5) * col_w
        half = max(8.0, amp * max_half)
        bar_w = max(1, int(round(col_w * 0.38)))
        line_draw.rectangle(
            [x - bar_w / 2, cy - half, x + bar_w / 2, cy + half],
            fill=color + (110,),
        )
        _draw_stacked_column(type_draw, type_layer, glyphs, font, char_h, x, cy, half, color)

    axis = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(axis).line([(0, cy), (width, cy)], fill=(255, 230, 245, 210), width=2)
    canvas = _neon_composite(canvas, line_layer, blur=8, glow=1.25)
    canvas = _neon_composite(canvas, type_layer, blur=4, glow=1.05)
    return _neon_composite(canvas, axis, blur=7, glow=1.3).convert("RGB")


def _phrase(text: str) -> str:
    cleaned = " ".join((text or "").split()).strip() or "voice"
    return cleaned if cleaned.endswith(" ") else cleaned + " "


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
    glow_layer = layer.filter(ImageFilter.GaussianBlur(blur))
    if glow != 1.0:
        glow_layer = ImageEnhance.Brightness(glow_layer).enhance(glow)
    out = Image.alpha_composite(base.convert("RGBA"), glow_layer)
    return Image.alpha_composite(out, layer)


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


def _draw_circular_text(img: Image.Image, text: str, cx: float, cy: float, radius: float, font) -> None:
    sample = text * 4
    avg = sum(_advance(font, ch) for ch in sample) / max(1, len(sample))
    n_chars = max(8, int((2 * math.pi * radius) / max(4.0, avg * 0.98)))
    stream = (text * ((n_chars // max(1, len(text))) + 3))[:n_chars]
    for i, ch in enumerate(stream):
        if ch == " ":
            continue
        angle = -math.pi / 2 + (2 * math.pi * i / n_chars)
        x = cx + math.cos(angle) * radius
        y = cy + math.sin(angle) * radius
        rot = math.degrees(angle) + 90
        _paste_rotated_char(img, ch, (x, y), rot, font, _radial_color(angle))


def _paste_rotated_char(img: Image.Image, ch: str, xy: Point, angle_deg: float, font, color: Color) -> None:
    glyph = _rotated_glyph(ch, int(round(angle_deg)) % 360, int(getattr(font, "size", 12) or 12))
    if glyph is None:
        return
    tinted = Image.new("RGBA", glyph.size, color + (0,))
    tinted.putalpha(glyph.split()[-1])
    x = int(round(xy[0] - tinted.width / 2))
    y = int(round(xy[1] - tinted.height / 2))
    img.alpha_composite(tinted, (x, y))


@lru_cache(maxsize=12288)
def _rotated_glyph(ch: str, angle: int, size: int) -> Image.Image | None:
    if ch == " ":
        return None
    font = _font(size)
    bbox = font.getbbox(ch)
    w = max(1, bbox[2] - bbox[0] + 6)
    h = max(1, bbox[3] - bbox[1] + 6)
    glyph = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glyph).text((3 - bbox[0], 3 - bbox[1]), ch, font=font, fill=(255, 255, 255, 255))
    return glyph.rotate(angle, resample=Image.BICUBIC, expand=True)


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


def _heart_outline() -> List[Point]:
    pts: List[Point] = []
    steps = 900
    for i in range(steps):
        t = 2 * math.pi * i / steps
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((x, y))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return [((x - minx) / (maxx - minx), (y - miny) / (maxy - miny)) for x, y in pts]


def _heart_top_bottom(width: int, height: int) -> Tuple[List[float | None], List[float | None]]:
    outline = _heart_outline()
    left, right = 0.22, 0.78
    top, bottom = 0.10, 0.90
    buckets: List[List[float]] = [[] for _ in range(width)]
    for nx, ny in outline:
        x = left + nx * (right - left)
        y = top + ny * (bottom - top)
        ix = int(round(x * (width - 1)))
        if 0 <= ix < width:
            buckets[ix].append(y * height)
    tops: List[float | None] = [None] * width
    bots: List[float | None] = [None] * width
    filled: List[int] = []
    for i, ys in enumerate(buckets):
        if ys:
            tops[i] = min(ys)
            bots[i] = max(ys)
            filled.append(i)
    for a, b in zip(filled, filled[1:]):
        gap = b - a
        if gap <= 1:
            continue
        ta, tb = tops[a], tops[b]
        ba, bb = bots[a], bots[b]
        if ta is None or tb is None or ba is None or bb is None:
            continue
        for i in range(a + 1, b):
            t = (i - a) / gap
            tops[i] = ta * (1.0 - t) + tb * t
            bots[i] = ba * (1.0 - t) + bb * t
    return tops, bots


def _draw_text_along_path(img: Image.Image, text: str, points: Sequence[Point], font) -> None:
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
        adv = _advance(font, ch) * 0.94
        if ch != " ":
            pt, tangent = _point_at(points, dists, min(total, pos + adv / 2))
            angle = max(-28.0, min(28.0, math.degrees(math.atan2(tangent[1], tangent[0]))))
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


def _draw_center_waveform(img: Image.Image, env: Sequence[float], width: int, height: int, thickness: int) -> None:
    draw = ImageDraw.Draw(img)
    cy = height / 2
    amp = height * 0.22
    pts_hi: List[Point] = []
    pts_lo: List[Point] = []
    step = max(1, width // max(2, len(env)))
    for i, value in enumerate(env):
        x = i if len(env) >= width else i * step
        if x >= width:
            break
        h = value * amp
        pts_hi.append((float(x), cy - h))
        pts_lo.append((float(x), cy + h))
    if len(pts_hi) < 2:
        return
    for i in range(1, len(pts_hi)):
        color = _gradient_at(pts_hi[i][0] / max(1, width - 1), HORIZONTAL_STOPS) + (255,)
        draw.line([pts_hi[i - 1], pts_hi[i]], fill=color, width=thickness)
        draw.line([pts_lo[i - 1], pts_lo[i]], fill=color, width=thickness)
        if i % 2 == 0:
            draw.line([pts_hi[i], pts_lo[i]], fill=color, width=1)


def _draw_stacked_column(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    glyphs: Sequence[str],
    font,
    char_h: float,
    x: float,
    cy: float,
    half: float,
    color: Color,
) -> None:
    if not glyphs or half < char_h:
        draw.rectangle([x - 1, cy - half, x + 1, cy + half], fill=color + (255,))
        return
    n = max(1, int(half / char_h))
    for direction in (-1, 1):
        for i in range(n):
            ch = glyphs[(i + int(abs(x))) % len(glyphs)]
            y = cy + direction * (i + 0.55) * char_h
            if abs(y - cy) > half:
                break
            _paste_rotated_char(img, ch, (x, y), 0, font, color)
