"""Merch product catalog and on-disk template paths."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MERCH_DIR = os.path.join(PACKAGE_DIR, "static", "images", "merch")

# Suggested swatches for the UI. Backend accepts these names OR any #RRGGBB.
# rgb=None → use light blank as-is (no tint).
COLOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "white": {"label": "White", "rgb": None, "hex": "#FFFFFF"},
    "charcoal": {"label": "Charcoal", "rgb": (61, 61, 64), "hex": "#3D3D40"},
    "black": {"label": "Black", "rgb": (0, 0, 0), "hex": "#000000"},
}

_DEFAULT_SUGGESTED = ["white", "charcoal", "black"]
_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_RGB_RE = re.compile(
    r"^rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$",
    re.IGNORECASE,
)


def _clamp_byte(v: int) -> int:
    return max(0, min(255, int(v)))


def _luma(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def parse_color_token(raw: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a frontend color token into a generation spec.

    Accepts:
      - preset names: white, charcoal, black
      - hex: #RGB, #RRGGBB, RGB, RRGGBB
      - rgb(): rgb(61, 61, 64)

    Returns dict with: id, label, hex, tint (optional RGB), prefer_dark (bool)
    """
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None

    lower = token.lower()
    preset = COLOR_PRESETS.get(lower)
    if preset:
        rgb = preset.get("rgb")
        hex_val = str(preset.get("hex") or "#FFFFFF")
        return {
            "id": lower,
            "label": preset["label"],
            "hex": hex_val,
            "tint": None if rgb is None else tuple(int(x) for x in rgb),
            "prefer_dark": lower == "black",
        }

    m_rgb = _RGB_RE.match(token)
    if m_rgb:
        rgb = (
            _clamp_byte(int(m_rgb.group(1))),
            _clamp_byte(int(m_rgb.group(2))),
            _clamp_byte(int(m_rgb.group(3))),
        )
        slug = f"{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        return {
            "id": slug,
            "label": f"#{slug.upper()}",
            "hex": f"#{slug.upper()}",
            "tint": rgb,
            "prefer_dark": _luma(rgb) <= 28,
        }

    m_hex = _HEX_RE.match(token)
    if m_hex:
        hx = m_hex.group(1)
        if len(hx) == 3:
            hx = "".join(ch * 2 for ch in hx)
        slug = hx.lower()
        hex_val = f"#{slug.upper()}"
        # Map known swatch hexes back to named presets for clean labels.
        for name, preset in COLOR_PRESETS.items():
            if str(preset.get("hex", "")).upper() == hex_val:
                return parse_color_token(name)
        rgb = (
            int(hx[0:2], 16),
            int(hx[2:4], 16),
            int(hx[4:6], 16),
        )
        # Near-white → leave blank untinted.
        if _luma(rgb) >= 245 and max(rgb) - min(rgb) <= 12:
            return {
                "id": "white" if slug in {"ffffff", "fefefe"} else slug,
                "label": "White" if slug in {"ffffff", "fefefe"} else hex_val,
                "hex": hex_val,
                "tint": None,
                "prefer_dark": False,
            }
        return {
            "id": slug,
            "label": hex_val,
            "hex": hex_val,
            "tint": rgb,
            "prefer_dark": _luma(rgb) <= 28,
        }

    return None


# All products use fast PIL overlays (place art on blank templates).
# colorable=True → frontend may send any colors; light blank is tinted dynamically.
# dark_template (optional) used when the requested color is near-black.
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
        "tags": ["One Size"],
        "mode": "mockup",
        "engine": "composite",
        "colorable": True,
        "template": "cap.jpg",
        "suggested_colors": list(_DEFAULT_SUGGESTED),
        "default_colors": ["black"],
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
        "colorable": True,
        "template": "tshirt_white.png",
        "dark_template": "tshirt.png",
        "suggested_colors": list(_DEFAULT_SUGGESTED),
        "default_colors": list(_DEFAULT_SUGGESTED),
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
        "colorable": True,
        "template": "hoodie_white.png",
        "dark_template": "hoodie.png",
        "suggested_colors": list(_DEFAULT_SUGGESTED),
        "default_colors": list(_DEFAULT_SUGGESTED),
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
        "colorable": True,
        "template": "tote.png",
        "suggested_colors": list(_DEFAULT_SUGGESTED),
        "default_colors": list(_DEFAULT_SUGGESTED),
        # Transparent / keyed — large centered panel.
        "print_box": (0.14, 0.18, 0.86, 0.82),
        # Opaque — inset so the white print plate reads on cream canvas.
        "opaque_print_box": (0.20, 0.24, 0.80, 0.76),
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

COLORABLE_PRODUCT_IDS = {p["id"] for p in MERCH_PRODUCTS if p.get("colorable")}


def merch_templates_dir() -> str:
    return MERCH_DIR


def product_path(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    path = os.path.join(MERCH_DIR, filename)
    return path if os.path.isfile(path) else None


def _suggested_color_opts(product: Dict[str, Any]) -> List[Dict[str, Any]]:
    suggested = product.get("suggested_colors") or _DEFAULT_SUGGESTED
    opts = []
    seen = set()
    for token in suggested:
        parsed = parse_color_token(str(token))
        if not parsed or parsed["id"] in seen:
            continue
        seen.add(parsed["id"])
        opts.append(
            {
                "id": parsed["id"],
                "label": parsed["label"],
                "hex": parsed["hex"],
            }
        )
    return opts


def list_products() -> List[Dict[str, Any]]:
    """Public catalog for the UI (no filesystem paths)."""
    out = []
    for item in MERCH_PRODUCTS:
        if item.get("hidden"):
            continue
        colorable = bool(item.get("colorable"))
        color_opts = _suggested_color_opts(item) if colorable else []
        templates = []
        if item.get("template"):
            templates.append(item["template"])
        if item.get("dark_template"):
            templates.append(item["dark_template"])
        defaults = list(item.get("default_colors") or [c["id"] for c in color_opts[:2]])
        out.append(
            {
                "id": item["id"],
                "label": item["label"],
                "tags": list(item.get("tags") or []),
                "mode": item["mode"],
                "has_template": (
                    any(product_path(t) for t in templates) if templates else True
                ),
                "colorable": colorable,
                "colors": color_opts,
                "default_colors": defaults,
                # Orientation is auto-detected from artwork — no UI picker.
                "orientations": [],
                "default_orientation": None,
                "auto_orientation": bool(item.get("auto_orientation")),
            }
        )
    return out


def _resolve_job_template(
    product: Dict[str, Any],
    parsed: Dict[str, Any],
) -> Optional[str]:
    """Pick light or dark blank; require the file to exist on disk."""
    dark = product.get("dark_template")
    if parsed.get("prefer_dark") and dark and product_path(dark):
        return dark
    light = product.get("template")
    if product_path(light):
        return light
    if dark and product_path(dark):
        return dark
    return None


def expand_products_for_colors(
    products: List[Dict[str, Any]],
    color_choices: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Expand colorable products into one job per frontend color token.

    color_choices example:
      {"tshirt": ["black", "#BE1C24", "rgb(34,120,58)"], "cap": ["#2244AA"]}
    Unknown / unparseable tokens are skipped.
    """
    choices = color_choices or {}
    expanded: List[Dict[str, Any]] = []
    preset_ids = set(COLOR_PRESETS.keys())

    for product in products:
        if not product.get("colorable"):
            expanded.append(dict(product))
            continue

        wanted = choices.get(product["id"])
        if wanted is None:
            wanted = list(product.get("default_colors") or _DEFAULT_SUGGESTED[:2])

        seen_ids = set()
        for token in wanted:
            parsed = parse_color_token(str(token))
            if not parsed:
                continue
            cid = parsed["id"]
            if cid in seen_ids:
                continue
            template = _resolve_job_template(product, parsed)
            if not template:
                continue
            seen_ids.add(cid)

            job = dict(product)
            job["template"] = template
            job["color"] = cid
            job["color_label"] = parsed["label"]
            job["color_hex"] = parsed["hex"]
            # Dark blank already is black — skip tint. Otherwise tint light blank.
            used_dark = (
                parsed.get("prefer_dark")
                and product.get("dark_template")
                and template == product.get("dark_template")
            )
            if parsed.get("tint") and not used_dark:
                job["tint"] = tuple(parsed["tint"])
            else:
                job.pop("tint", None)
            job["variant_id"] = f"{product['id']}-{cid}"
            base_tags = [
                t
                for t in (product.get("tags") or [])
                if str(t).lower() not in preset_ids
            ]
            job["tags"] = base_tags + [parsed["label"]]
            expanded.append(job)

    return expanded
