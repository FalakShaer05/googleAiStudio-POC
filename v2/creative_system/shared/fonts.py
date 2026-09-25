"""Script fonts users can pick for artwork lettering."""
from __future__ import annotations

import os
from typing import Optional

FONTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "fonts"))

FONTS = [
    {
        "id": "parisienne",
        "label": "Parisienne",
        "file": "Parisienne-Regular.ttf",
        "description": (
            "Parisienne: a flowing French-style connected script with even medium strokes, "
            "moderate slant, rounded loops, and restrained flourishes."
        ),
    },
    {
        "id": "alex-brush",
        "label": "Alex Brush",
        "file": "AlexBrush-Regular.ttf",
        "description": (
            "Alex Brush: a connected brush-pen script with strong thick-thin contrast, "
            "a steep slant, and large swooping capital letters."
        ),
    },
    {
        "id": "ballet",
        "label": "Ballet",
        "file": "Ballet.ttf",
        "description": (
            "Ballet: an ornate high-contrast copperplate script with hairline upstrokes, "
            "heavy downstrokes, and elaborate looping capital letters."
        ),
    },
    {
        "id": "allura",
        "label": "Allura",
        "file": "Allura-Regular.ttf",
        "description": (
            "Allura: a light, elegant connected script with thin strokes, soft slant, "
            "tall ascenders, and simple graceful capitals."
        ),
    },
    {
        "id": "great-vibes",
        "label": "Great Vibes",
        "file": "GreatVibes-Regular.ttf",
        "description": (
            "Great Vibes: a formal connected calligraphy script with bold contrast, "
            "a lively slant, and large flourished capital letters."
        ),
    },
]

FONT_IDS = {item["id"] for item in FONTS}
DEFAULT_FONT_ID = "parisienne"


def get_font(font_id: Optional[str]) -> dict:
    key = (font_id or DEFAULT_FONT_ID).strip().lower()
    for item in FONTS:
        if item["id"] == key:
            return item
    raise ValueError(f"Choose a font: {', '.join(item['label'] for item in FONTS)}")


def font_path(font_id: Optional[str]) -> str:
    return os.path.join(FONTS_DIR, get_font(font_id)["file"])
