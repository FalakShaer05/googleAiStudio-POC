"""Readable phrase plate so Gemini copies the spoken line, not the style sample."""
from __future__ import annotations

import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FONT_CANDIDATES = [
    os.path.join(PACKAGE_DIR, "static", "fonts", "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


@lru_cache(maxsize=4)
def _font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        width = font.getlength(trial) if hasattr(font, "getlength") else font.getbbox(trial)[2]
        if width <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_phrase_card(phrase: str) -> Image.Image:
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), (6, 6, 10))
    draw = ImageDraw.Draw(img)
    title_font = _font(36)
    body_font = _font(64)
    margin = 80
    draw.text((margin, 48), "SPOKEN LINE FROM THE AUDIO", font=title_font, fill=(180, 180, 200))
    lines = _wrap(phrase, body_font, width - margin * 2)
    y = 160
    line_h = 86
    for line in lines[:8]:
        draw.text((margin, y), line, font=body_font, fill=(255, 255, 255))
        y += line_h
    return img
