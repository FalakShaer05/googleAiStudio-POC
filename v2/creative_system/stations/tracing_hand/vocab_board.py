"""Render a plain vocabulary checklist image for Gemini role context."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

from .prompts import _normalized_words

_FONTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "static", "fonts")
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Poppins-Regular.ttf", "Poppins-Bold.ttf"):
        path = os.path.join(_FONTS_DIR, name)
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def render_vocabulary_board(words: list[str]) -> Image.Image:
    """
    Clean white board listing every required word once.
    Helps the model keep exact vocabulary coverage (no digit prefixes).
    """
    cleaned = [w.upper() for w in _normalized_words(words)]
    count = len(cleaned)
    title = f"REQUIRED WORDS — PLACE ALL {count} EXACTLY ONCE"
    lines = [f"[ ] {word}" for word in cleaned]

    title_font = _load_font(28)
    body_font = _load_font(26)

    pad_x, pad_y = 36, 28
    line_gap = 10
    measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def _size(text: str, font) -> tuple[int, int]:
        box = measure.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0]), int(box[3] - box[1])

    title_w, title_h = _size(title, title_font)
    line_sizes = [_size(line, body_font) for line in lines]
    content_w = max([title_w, *(w for w, _ in line_sizes)], default=200)
    content_h = title_h + 18 + sum(h + line_gap for _, h in line_sizes)
    width = max(420, content_w + pad_x * 2)
    height = max(240, content_h + pad_y * 2)

    board = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    draw.rectangle((2, 2, width - 3, height - 3), outline=(255, 32, 96), width=3)

    y = pad_y
    draw.text((pad_x, y), title, font=title_font, fill=(20, 20, 20))
    y += title_h + 18
    for line, (_, line_h) in zip(lines, line_sizes):
        draw.text((pad_x, y), line, font=body_font, fill=(20, 20, 20))
        y += line_h + line_gap
    return board
