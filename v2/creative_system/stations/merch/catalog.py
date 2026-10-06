"""Merch product catalog and on-disk template paths."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MERCH_DIR = os.path.join(PACKAGE_DIR, "static", "images", "merch")

# All products use fast PIL overlays (place art on blank templates).
# print_box = max zone; match_art_aspect=True keeps artwork proportions
# and centers the largest fitting rectangle inside that zone.
MERCH_PRODUCTS: List[Dict[str, Any]] = [
    {
        "id": "sticker",
        "label": "1 Sticker",
        "tags": ["Die-Cut"],
        "mode": "sticker_single",
        "engine": "composite",
    },
    {
        "id": "stickers-5",
        "label": "5 Stickers",
        "tags": ["Pack"],
        "mode": "sticker_fan",
        "engine": "composite",
    },
    {
        "id": "sticker-sheet",
        "label": "Sticker Sheet",
        "tags": ["Sheet"],
        "mode": "sticker_sheet",
        "engine": "composite",
    },
    {
        "id": "cap",
        "label": "Cap",
        "tags": ["White", "One Size"],
        "mode": "mockup",
        "engine": "composite",
        "template": "cap.jpg",
        # Front crown only — visor left blank; lower on the panel.
        "print_box": (0.30, 0.28, 0.70, 0.58),
        "fit": "cover",
        "match_art_aspect": True,
        "knockout": False,
        "shade": False,
    },
    {
        "id": "tshirt",
        "label": "T-Shirt",
        "tags": ["Medium", "Black"],
        "mode": "mockup",
        "engine": "composite",
        "template": "tshirt.png",
        # Centered chest print — slightly smaller, not over shoulders/sleeves.
        "print_box": (0.32, 0.28, 0.68, 0.53),
        "fit": "cover",
        "match_art_aspect": True,
        "knockout": False,
        "shade": False,
    },
    {
        "id": "hoodie",
        "label": "Hoodie",
        "tags": ["Medium", "Black"],
        "mode": "mockup",
        "engine": "composite",
        "template": "hoodie.png",
        # Chest print ~10% smaller, lower on torso, above pocket.
        "print_box": (0.30, 0.36, 0.70, 0.61),
        "fit": "cover",
        "match_art_aspect": True,
        "knockout": False,
        "shade": False,
    },
    {
        "id": "tote",
        "label": "Tote Bag",
        "tags": ["One Size"],
        "mode": "mockup",
        "engine": "composite",
        "template": "tote.png",
        # Larger centered panel with a bit of margin.
        "print_box": (0.14, 0.18, 0.86, 0.82),
        "fit": "cover",
        "match_art_aspect": True,
        "knockout": False,
        "shade": False,
    },
    {
        "id": "photo-print",
        "label": "Photo Print",
        "tags": ["4×6 inches"],
        "mode": "photo_print",
        "engine": "composite",
        "template": "photo_print.png",
        "fit": "contain",
    },
    {
        "id": "canvas",
        "label": "Canvas",
        "tags": ["Poster"],
        "mode": "canvas",
        "engine": "composite",
        "fit": "contain",
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
