"""Overlay typography with precise spacing onto the Gemini heart-map plate."""
from __future__ import annotations

import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from .prompts import ART_COLOR, TEXT_COLOR

SCRIPT_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoesc.ttf",
    "C:/Windows/Fonts/segoescb.ttf",
    "C:/Windows/Fonts/BRUSHSCI.TTF",
    "C:/Windows/Fonts/ITCBLKAD.TTF",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

SANS_FONT_CANDIDATES = [
    "C:/Windows/Fonts/Montserrat-Regular.ttf",
    "C:/Windows/Fonts/montserrat.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

SANS_SEMIBOLD_CANDIDATES = [
    "C:/Windows/Fonts/Montserrat-SemiBold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


ART_RGB = _hex_rgb(ART_COLOR)
TEXT_RGB = _hex_rgb(TEXT_COLOR)


@lru_cache(maxsize=16)
def _font(candidates: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _sample_cream(img: Image.Image) -> tuple[int, int, int]:
    w, h = img.size
    pixels = img.convert("RGB").load()
    samples: list[tuple[int, int, int]] = []
    for x in range(0, min(40, w), 4):
        for y in range(0, min(40, h), 4):
            samples.append(pixels[x, y])
    if not samples:
        return (248, 246, 242)
    r = sum(c[0] for c in samples) // len(samples)
    g = sum(c[1] for c in samples) // len(samples)
    b = sum(c[2] for c in samples) // len(samples)
    return (r, g, b)


def _text_width_spaced(text: str, font, tracking_px: float) -> float:
    if not text:
        return 0.0
    width = 0.0
    for idx, ch in enumerate(text):
        width += font.getlength(ch) if hasattr(font, "getlength") else font.getbbox(ch)[2]
        if idx < len(text) - 1:
            width += tracking_px
    return width


def _draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    center_x: float,
    y: float,
    font,
    fill: tuple[int, int, int],
    tracking_em: float,
) -> float:
    if not text:
        return y
    size = getattr(font, "size", 32)
    tracking_px = size * tracking_em
    total_w = _text_width_spaced(text, font, tracking_px)
    x = center_x - total_w / 2
    for idx, ch in enumerate(text):
        draw.text((x, y), ch, font=font, fill=fill)
        advance = font.getlength(ch) if hasattr(font, "getlength") else font.getbbox(ch)[2]
        x += advance + (tracking_px if idx < len(text) - 1 else 0)
    bbox = font.getbbox(text)
    return y + (bbox[3] - bbox[1])


def _draw_brush_stroke(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    y: float,
    width: float,
    scale: float,
) -> None:
    left = center_x - width / 2
    right = center_x + width / 2
    mid_y = y
    steps = max(24, int(width / (6 * scale)))
    for i in range(steps):
        t = i / max(steps - 1, 1)
        x = left + t * width
        taper = 0.35 + 0.65 * (1 - abs(2 * t - 1) ** 1.4)
        h = max(2, int(3 * scale * taper))
        w = max(2, int(6 * scale * taper))
        draw.ellipse((x - w / 2, mid_y - h / 2, x + w / 2, mid_y + h / 2), fill=ART_RGB)
    draw.line((left + width * 0.08, mid_y, right - width * 0.08, mid_y), fill=ART_RGB, width=max(1, int(2 * scale)))


def _draw_small_heart(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    top_y: float,
    size: float,
) -> None:
    w = size
    h = size
    cx = center_x
    cy = top_y + h * 0.35
    left = (cx - w * 0.5, cy - h * 0.15, cx, cy + h * 0.2)
    right = (cx, cy - h * 0.15, cx + w * 0.5, cy + h * 0.2)
    draw.pieslice(left, 200, 340, fill=ART_RGB)
    draw.pieslice(right, 200, 340, fill=ART_RGB)
    draw.polygon(
        [(cx - w * 0.52, cy), (cx + w * 0.52, cy), (cx, cy + h * 0.78)],
        fill=ART_RGB,
    )


def _find_heart_bottom(img: Image.Image, *, scale: float) -> int:
    """Find the last row with significant map-red pixels so we don't clip the heart."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    pixels = rgb.load()
    start_y = int(h * 0.35)
    last_red_y = int(h * 0.58)
    for y in range(start_y, h):
        red_count = 0
        for x in range(0, w, 4):
            r, g, b = pixels[x, y]
            if r > 130 and g < 95 and b < 95 and r > g + 25:
                red_count += 1
        if red_count > 2:
            last_red_y = y
    return min(h - 1, last_red_y + int(10 * scale))


def composite_typography(
    base: Image.Image,
    message: str,
    location: str,
    coords: str | None,
) -> Image.Image:
    """Paint over Gemini's text band and render the full typography stack programmatically."""
    base = base.convert("RGB")
    w, h = base.size
    scale = w / 1024.0
    cream = _sample_cream(base)

    text_zone_start = _find_heart_bottom(base, scale=scale)
    text_zone_start = max(text_zone_start, int(h * 0.54))
    text_zone_start = min(text_zone_start, int(h * 0.72))
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, text_zone_start, w, h), fill=cream)

    cx = w / 2
    title_font = _font(tuple(SCRIPT_FONT_CANDIDATES), max(34, int(54 * scale)))
    location_font = _font(tuple(SANS_SEMIBOLD_CANDIDATES), max(16, int(26 * scale)))
    coords_font = _font(tuple(SANS_FONT_CANDIDATES), max(14, int(22 * scale)))

    gap_title_stroke = int(30 * scale)
    gap_stroke_location = int(34 * scale)
    gap_location_coords = int(18 * scale)
    gap_coords_icon = int(26 * scale)

    y = text_zone_start + int(18 * scale)
    title_bbox = title_font.getbbox(message or " ")
    title_w = title_font.getlength(message) if hasattr(title_font, "getlength") else title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]
    draw.text((cx - title_w / 2, y - title_bbox[1]), message, font=title_font, fill=TEXT_RGB)
    y += title_h + gap_title_stroke

    stroke_w = w * 0.22
    _draw_brush_stroke(draw, center_x=cx, y=y, width=stroke_w, scale=scale)
    y += gap_stroke_location

    loc = (location or "WYNWOOD, FLORIDA").strip().upper()
    loc_bbox = location_font.getbbox(loc)
    loc_h = loc_bbox[3] - loc_bbox[1]
    _draw_spaced_text(
        draw,
        loc,
        center_x=cx,
        y=y,
        font=location_font,
        fill=TEXT_RGB,
        tracking_em=0.28,
    )
    y += loc_h + gap_location_coords

    if coords:
        coords_bbox = coords_font.getbbox(coords)
        coords_h = coords_bbox[3] - coords_bbox[1]
        _draw_spaced_text(
            draw,
            coords,
            center_x=cx,
            y=y,
            font=coords_font,
            fill=ART_RGB,
            tracking_em=0.12,
        )
        y += coords_h + gap_coords_icon
    else:
        y += gap_coords_icon

    _draw_small_heart(draw, center_x=cx, top_y=y, size=16 * scale)
    return canvas
