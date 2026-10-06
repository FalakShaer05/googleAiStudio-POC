"""Merch product catalog and on-disk template paths."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MERCH_DIR = os.path.join(PACKAGE_DIR, "static", "images", "merch")

# engine:
#   composite — fast PIL (stickers / paper products)
#   gemini    — photoreal mockups for apparel using blank templates
MERCH_PRODUCTS: List[Dict[str, Any]] = [
    {
        "id": "sticker",
        "label": "1 Sticker",
        "tags": ["Die-Cut"],
        "mode": "sticker_single",
        "engine": "composite",
        "aspect_ratio": "1:1",
        "style_ref": "style_sticker_single.png",
    },
    {
        "id": "stickers-5",
        "label": "5 Stickers",
        "tags": ["Pack"],
        "mode": "sticker_fan",
        "engine": "composite",
        "aspect_ratio": "3:2",
        "style_ref": "style_sticker_fan.png",
    },
    {
        "id": "sticker-sheet",
        "label": "Sticker Sheet",
        "tags": ["Sheet"],
        "mode": "sticker_sheet",
        "engine": "composite",
        "aspect_ratio": "3:2",
        "style_ref": "style_sticker_sheet.jpg",
    },
    {
        "id": "cap",
        "label": "Cap",
        "tags": ["White", "One Size"],
        "mode": "mockup",
        "engine": "gemini",
        "template": "cap.jpg",
        "aspect_ratio": "1:1",
    },
    {
        "id": "tshirt",
        "label": "T-Shirt",
        "tags": ["Medium", "Black"],
        "mode": "mockup",
        "engine": "gemini",
        "template": "tshirt.png",
        "aspect_ratio": "3:4",
    },
    {
        "id": "hoodie",
        "label": "Hoodie",
        "tags": ["Medium", "Black"],
        "mode": "mockup",
        "engine": "gemini",
        "template": "hoodie.png",
        "aspect_ratio": "3:4",
    },
    {
        "id": "tote",
        "label": "Tote Bag",
        "tags": ["One Size"],
        "mode": "mockup",
        "engine": "gemini",
        "template": "tote.png",
        "aspect_ratio": "3:4",
    },
    {
        "id": "photo-print",
        "label": "Photo Print",
        "tags": ["4×6 inches"],
        "mode": "photo_print",
        "engine": "composite",
        "template": "photo_print.png",
        "fit": "cover",
        "aspect_ratio": "3:2",
    },
    {
        "id": "canvas",
        "label": "Canvas",
        "tags": ["Square"],
        "mode": "canvas",
        "engine": "composite",
        "fit": "cover",
        "aspect_ratio": "1:1",
    },
]


def merch_templates_dir() -> str:
    return MERCH_DIR


def product_path(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    path = os.path.join(MERCH_DIR, filename)
    return path if os.path.isfile(path) else None


def list_products() -> List[Dict[str, Any]]:
    """Public catalog for the UI (no filesystem paths)."""
    out = []
    for item in MERCH_PRODUCTS:
        out.append(
            {
                "id": item["id"],
                "label": item["label"],
                "tags": list(item.get("tags") or []),
                "mode": item["mode"],
                "has_template": bool(product_path(item.get("template"))),
            }
        )
    return out
