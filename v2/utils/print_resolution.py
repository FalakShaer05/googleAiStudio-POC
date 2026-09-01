"""Merch print resolution profiles (PPI) for the Print Ready upscale tab."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

Variant = Dict[str, Any]
Profile = Dict[str, Any]


def _variant(
    variant_id: str,
    size_label: str,
    print_w: float,
    print_h: float,
    px_w: int,
    px_h: int,
    ppi: int = 300,
) -> Variant:
    return {
        "id": variant_id,
        "size_label": size_label,
        "print_inches": [print_w, print_h],
        "pixels": [px_w, px_h],
        "ppi": ppi,
    }


MERCH_PROFILES: Dict[str, Profile] = {
    "tshirts": {
        "label": "T-Shirts",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("xs", "XS", 10, 12, 3000, 3600),
            _variant("s", "S", 11, 14, 3300, 4200),
            _variant("m", "M", 12, 16, 3600, 4800),
            _variant("l", "L", 12, 16, 3600, 4800),
            _variant("xl", "XL", 13, 18, 3900, 5400),
            _variant("xxl", "XXL", 14, 18, 4200, 5400),
            _variant("xxxl", "XXXL", 14, 18, 4200, 5400),
        ],
    },
    "hoodies": {
        "label": "Hoodies",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("xs", "XS", 10, 12, 3000, 3600),
            _variant("s", "S", 11, 14, 3300, 4200),
            _variant("m", "M", 12, 16, 3600, 4800),
            _variant("l", "L", 13, 17, 3900, 5100),
            _variant("xl", "XL", 14, 18, 4200, 5400),
            _variant("xxl", "XXL", 14, 18, 4200, 5400),
            _variant("xxxl", "XXXL", 14, 18, 4200, 5400),
        ],
    },
    "tote-bags": {
        "label": "Tote Bags",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("one-size", "One Size", 12, 14, 3600, 4200),
        ],
    },
    "caps": {
        "label": "Caps",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("one-size", "One Size", 4, 2.5, 1200, 750),
        ],
    },
    "aprons": {
        "label": "Aprons",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("one-size", "One Size", 10, 12, 3000, 3600),
        ],
    },
    "stickers": {
        "label": "Stickers",
        "file_formats": ["PNG", "SVG", "PDF"],
        "variants": [
            _variant("2x2", '2 × 2"', 2, 2, 600, 600),
            _variant("2x3", '2 × 3"', 2, 3, 600, 900),
            _variant("3x3", '3 × 3"', 3, 3, 900, 900),
            _variant("3x4", '3 × 4"', 3, 4, 900, 1200),
            _variant("4x4", '4 × 4"', 4, 4, 1200, 1200),
            _variant("5x5", '5 × 5"', 5, 5, 1500, 1500),
        ],
    },
    "magnets": {
        "label": "Magnets",
        "file_formats": ["PNG", "SVG", "PDF"],
        "variants": [
            _variant("2x2", '2 × 2"', 2, 2, 600, 600),
            _variant("2x3", '2 × 3"', 2, 3, 600, 900),
            _variant("3x3", '3 × 3"', 3, 3, 900, 900),
            _variant("3x4", '3 × 4"', 3, 4, 900, 1200),
            _variant("4x4", '4 × 4"', 4, 4, 1200, 1200),
            _variant("5x5", '5 × 5"', 5, 5, 1500, 1500),
        ],
    },
    "mugs": {
        "label": "Mugs",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("11oz", "11 oz", 9, 3.75, 2700, 1125),
        ],
    },
    "tumblers": {
        "label": "Tumblers",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("20oz", "20 oz", 9.3, 8.2, 2790, 2460),
        ],
    },
    "canvas": {
        "label": "Canvas",
        "file_formats": ["TIFF", "PNG", "PSD", "PDF"],
        "variants": [
            _variant("8x10", '8 × 10"', 8, 10, 2400, 3000),
            _variant("11x14", '11 × 14"', 11, 14, 3300, 4200),
            _variant("16x20", '16 × 20"', 16, 20, 4800, 6000),
            _variant("18x24", '18 × 24"', 18, 24, 5400, 7200),
            _variant("24x36", '24 × 36"', 24, 36, 7200, 10800),
            _variant("30x40", '30 × 40"', 30, 40, 9000, 12000),
            _variant("8x8", '8 × 8"', 8, 8, 2400, 2400),
            _variant("12x12", '12 × 12"', 12, 12, 3600, 3600),
            _variant("16x16", '16 × 16"', 16, 16, 4800, 4800),
            _variant("20x20", '20 × 20"', 20, 20, 6000, 6000),
            _variant("24x24", '24 × 24"', 24, 24, 7200, 7200),
            _variant("30x30", '30 × 30"', 30, 30, 9000, 9000),
        ],
    },
    "keychain": {
        "label": "Keychain",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("2x2", '2 × 2" (1.3 × 1.3" print)', 1.3, 1.3, 390, 390),
        ],
    },
}

MERCH_TYPE_IDS = tuple(MERCH_PROFILES.keys())
DEFAULT_MERCH_TYPE = "tshirts"


def normalize_art_type(art_type: str) -> str:
    key = (art_type or DEFAULT_MERCH_TYPE).strip().lower()
    return key if key in MERCH_PROFILES else DEFAULT_MERCH_TYPE


def get_art_type_profile(art_type: str) -> Profile:
    return MERCH_PROFILES[normalize_art_type(art_type)]


def get_variant(art_type: str, variant_id: str) -> Optional[Variant]:
    profile = get_art_type_profile(art_type)
    key = (variant_id or "").strip().lower()
    for variant in profile["variants"]:
        if variant["id"] == key:
            return variant
    return None


def format_inches(w: float, h: float) -> str:
    def _fmt(value: float) -> str:
        return str(int(value)) if value == int(value) else str(value)

    return f'{_fmt(w)} × {_fmt(h)}"'


def orient_pixels(
    pixels: Tuple[int, int],
    src_w: int,
    src_h: int,
) -> Tuple[int, int]:
    target_w, target_h = pixels
    if src_w > src_h and target_w < target_h:
        return target_h, target_w
    if src_h > src_w and target_h < target_w:
        return target_h, target_w
    return target_w, target_h


def list_profiles_for_api() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for merch_id, profile in MERCH_PROFILES.items():
        variants: List[Dict[str, Any]] = []
        for variant in profile["variants"]:
            pw, ph = variant["print_inches"]
            px_w, px_h = variant["pixels"]
            variants.append(
                {
                    "id": variant["id"],
                    "size_label": variant["size_label"],
                    "print_area": format_inches(pw, ph),
                    "print_inches": variant["print_inches"],
                    "pixels": variant["pixels"],
                    "pixels_label": f"{px_w:,} × {px_h:,} px",
                    "ppi": variant["ppi"],
                    "resize_method": "Original preserved (pixel resize)",
                }
            )
        items.append(
            {
                "id": merch_id,
                "label": profile["label"],
                "file_formats": profile["file_formats"],
                "variants": variants,
            }
        )
    return items


def validate_variant(art_type: str, variant_id: str) -> Tuple[bool, str]:
    if not variant_id:
        return False, "Product size (variant_id) is required"
    variant = get_variant(art_type, variant_id)
    if variant is None:
        profile = get_art_type_profile(art_type)
        allowed = [v["id"] for v in profile["variants"]]
        return False, f"Invalid product size. Must be one of: {', '.join(allowed)}"
    return True, ""


def get_target_pixels_for_variant(
    art_type: str,
    variant_id: str,
    src_w: int,
    src_h: int,
) -> Tuple[int, int, int, Variant]:
    variant = get_variant(art_type, variant_id)
    if variant is None:
        raise ValueError(f"Unknown variant {variant_id} for {art_type}")
    px_w, px_h = variant["pixels"]
    target_w, target_h = orient_pixels((px_w, px_h), src_w, src_h)
    return target_w, target_h, variant["ppi"], variant


def resize_preserving_art(
    source: Image.Image,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """
    Scale artwork to fit the print canvas without redrawing or distorting.

    Uses high-quality LANCZOS resampling, centers on a white canvas, and keeps
    alpha when the source has transparency.
    """
    src_w, src_h = source.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    working = source
    if working.mode not in ("RGB", "RGBA"):
        working = working.convert("RGBA" if "A" in working.getbands() else "RGB")

    resized = working.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if working.mode == "RGBA":
        canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
        if resized.mode != "RGBA":
            resized = resized.convert("RGBA")
        canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2), resized)
        return canvas

    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas
