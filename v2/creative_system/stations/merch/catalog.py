"""Merch product catalog and on-disk template paths."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MERCH_DIR = os.path.join(PACKAGE_DIR, "static", "images", "merch")

# Apparel colors available for tee / hoodie.
# tint = recolor light blank (white template) while keeping fabric folds.
APPAREL_COLORS: List[Dict[str, Any]] = [
    {"id": "white", "label": "White"},
    {"id": "red", "label": "Red", "tint": (190, 28, 36)},
    {"id": "green", "label": "Green", "tint": (34, 120, 58)},
    {"id": "black", "label": "Black"},
    {"id": "gray", "label": "Gray", "tint": (128, 128, 132)},
]

_APPAREL_COLOR_IDS = {c["id"] for c in APPAREL_COLORS}

# All products use fast PIL overlays (place art on blank templates).
# print_box = max zone; match_art_aspect=True keeps artwork proportions
# and centers the largest fitting rectangle inside that zone.
# auto_orientation=True → portrait/landscape is detected from artwork size.
MERCH_PRODUCTS: List[Dict[str, Any]] = [
    {
        "id": "sticker",
        "label": "1 Sticker",
        "tags": ["Die-Cut"],
        "mode": "sticker_single",
        "engine": "composite",
        "auto_orientation": True,
    },
    {
        "id": "stickers-3",
        "label": "3 Stickers",
        "tags": ["Pack"],
        "mode": "sticker_fan",
        "engine": "composite",
        "sticker_count": 3,
        "auto_orientation": True,
    },
    {
        "id": "stickers-4",
        "label": "4 Stickers",
        "tags": ["Pack"],
        "mode": "sticker_fan",
        "engine": "composite",
        "sticker_count": 4,
        "auto_orientation": True,
        "hidden": True,  # Temporarily hidden from UI / default runs
    },
    {
        "id": "stickers-5",
        "label": "5 Stickers",
        "tags": ["Pack"],
        "mode": "sticker_fan",
        "engine": "composite",
        "sticker_count": 5,
        "auto_orientation": True,
        "hidden": True,  # Temporarily hidden from UI / default runs
    },
    {
        "id": "sticker-sheet",
        "label": "Sticker Sheet",
        "tags": ["Sheet"],
        "mode": "sticker_sheet",
        "engine": "composite",
        "auto_orientation": True,
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
        "auto_orientation": True,
    },
    {
        "id": "tshirt",
        "label": "T-Shirt",
        "tags": ["Medium"],
        "mode": "mockup",
        "engine": "composite",
        "colors": {
            "white": {"label": "White", "template": "tshirt_white.png"},
            "red": {
                "label": "Red",
                "template": "tshirt_white.png",
                "tint": (190, 28, 36),
            },
            "green": {
                "label": "Green",
                "template": "tshirt_white.png",
                "tint": (34, 120, 58),
            },
            "black": {"label": "Black", "template": "tshirt.png"},
            "gray": {
                "label": "Gray",
                "template": "tshirt_white.png",
                "tint": (128, 128, 132),
            },
        },
        "default_colors": ["white", "black"],
        # Landscape / wide art — compact chest band.
        "print_box": (0.32, 0.28, 0.68, 0.53),
        # Portrait / square art — larger than landscape band, not full-chest.
        "portrait_print_box": (0.28, 0.22, 0.72, 0.58),
        "fit": "cover",
        "match_art_aspect": True,
        "knockout": False,
        "shade": False,
    },
    {
        "id": "hoodie",
        "label": "Hoodie",
        "tags": ["Medium"],
        "mode": "mockup",
        "engine": "composite",
        "colors": {
            "white": {"label": "White", "template": "hoodie_white.png"},
            "red": {
                "label": "Red",
                "template": "hoodie_white.png",
                "tint": (190, 28, 36),
            },
            "green": {
                "label": "Green",
                "template": "hoodie_white.png",
                "tint": (34, 120, 58),
            },
            "black": {"label": "Black", "template": "hoodie.png"},
            "gray": {
                "label": "Gray",
                "template": "hoodie_white.png",
                "tint": (128, 128, 132),
            },
        },
        "default_colors": ["white", "black"],
        # Landscape — compact band above pocket.
        "print_box": (0.30, 0.36, 0.70, 0.61),
        # Portrait / square — medium chest print, still above pocket.
        "portrait_print_box": (0.28, 0.30, 0.72, 0.62),
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
        "auto_orientation": True,
    },
    {
        "id": "photo-print",
        "label": "Photo Print",
        "tags": ["4×6 inches"],
        "mode": "photo_print",
        "engine": "composite",
        "template": "photo_print.png",
        "fit": "contain",
        "auto_orientation": True,
    },
    {
        "id": "canvas",
        "label": "Poster",
        "tags": ["Large"],
        "mode": "canvas",
        "engine": "composite",
        "fit": "contain",
        "auto_orientation": True,
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
        if item.get("hidden"):
            continue
        colors = item.get("colors") or {}
        color_opts = [
            {
                "id": cid,
                "label": meta.get("label") or cid.title(),
                "has_template": bool(product_path(meta.get("template"))),
            }
            for cid, meta in colors.items()
        ]
        templates = []
        if item.get("template"):
            templates.append(item["template"])
        templates.extend(meta.get("template") for meta in colors.values() if meta.get("template"))
        out.append(
            {
                "id": item["id"],
                "label": item["label"],
                "tags": list(item.get("tags") or []),
                "mode": item["mode"],
                "has_template": any(product_path(t) for t in templates) if templates else True,
                "colors": color_opts,
                "default_colors": list(item.get("default_colors") or [c["id"] for c in color_opts]),
                # Orientation is auto-detected from artwork — no UI picker.
                "orientations": [],
                "default_orientation": None,
                "auto_orientation": bool(item.get("auto_orientation")),
            }
        )
    return out


def expand_products_for_colors(
    products: List[Dict[str, Any]],
    color_choices: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Expand apparel products into one job per selected color.
    color_choices: {"tshirt": ["black","white"], "hoodie": ["black"]}
    """
    choices = color_choices or {}
    expanded: List[Dict[str, Any]] = []
    for product in products:
        colors = product.get("colors") or {}
        if not colors:
            expanded.append(dict(product))
            continue

        wanted = choices.get(product["id"])
        if wanted is None:
            wanted = list(product.get("default_colors") or colors.keys())
        # Preserve catalog order; drop unknowns / missing templates.
        for cid, meta in colors.items():
            if cid not in wanted:
                continue
            template = meta.get("template")
            if not product_path(template):
                continue
            color_label = meta.get("label") or cid.title()
            job = dict(product)
            job["template"] = template
            job["color"] = cid
            job["color_label"] = color_label
            if meta.get("tint"):
                job["tint"] = tuple(meta["tint"])
            job["variant_id"] = f"{product['id']}-{cid}"
            base_tags = [
                t
                for t in (product.get("tags") or [])
                if str(t).lower() not in _APPAREL_COLOR_IDS
            ]
            job["tags"] = base_tags + [color_label]
            expanded.append(job)
    return expanded
