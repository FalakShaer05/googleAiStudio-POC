"""Merch print resolution profiles (PPI) for the Print Ready upscale tab."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

Variant = Dict[str, Any]
Profile = Dict[str, Any]
ArtArea = Tuple[float, float, int, int]


def format_inches(w: float, h: float) -> str:
    def _fmt(value: float) -> str:
        return str(int(value)) if value == int(value) else str(value)

    return f'{_fmt(w)} × {_fmt(h)}"'


def _area(
    print_w: float,
    print_h: float,
    px_w: int,
    px_h: int,
    label: str,
) -> Dict[str, Any]:
    return {
        "label": label,
        "print_inches": [print_w, print_h],
        "pixels": [px_w, px_h],
    }


def _from_tuple(area: ArtArea, label: str) -> Dict[str, Any]:
    print_w, print_h, px_w, px_h = area
    return _area(print_w, print_h, px_w, px_h, label)


def _variant(
    variant_id: str,
    size_label: str,
    max_physical: Tuple[float, float],
    landscape: Optional[ArtArea] = None,
    square: Optional[ArtArea] = None,
    print_only: Optional[ArtArea] = None,
    ppi: int = 300,
) -> Variant:
    aspects: Dict[str, Dict[str, Any]] = {}
    if landscape:
        aspects["landscape"] = _from_tuple(landscape, "Landscape (3:2)")
    if square:
        aspects["square"] = _from_tuple(square, "Square (1:1)")
    if print_only and not aspects:
        print_w, print_h, px_w, px_h = print_only
        if print_w == print_h:
            aspects["square"] = _area(print_w, print_h, px_w, px_h, "Square (1:1)")
        elif print_w > print_h:
            aspects["landscape"] = _area(print_w, print_h, px_w, px_h, "Landscape")
        else:
            aspects["portrait"] = _area(print_w, print_h, px_w, px_h, "Portrait")

    default_key = next(
        (key for key in ("landscape", "square", "portrait") if key in aspects),
        next(iter(aspects)),
    )
    default_area = aspects[default_key]

    return {
        "id": variant_id,
        "size_label": size_label,
        "ppi": ppi,
        "max_physical_inches": [max_physical[0], max_physical[1]],
        "aspects": aspects,
        "default_aspect": default_key,
        "print_inches": default_area["print_inches"],
        "pixels": default_area["pixels"],
    }


MERCH_PROFILES: Dict[str, Profile] = {
    "tshirts": {
        "label": "T-Shirts",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("xs", "XS", (12, 10), landscape=(12, 8, 3600, 2400), square=(10, 10, 3000, 3000)),
            _variant("s", "S", (14, 11), landscape=(14, 9.33, 4200, 2800), square=(11, 11, 3300, 3300)),
            _variant("m", "M", (16, 12), landscape=(16, 10.67, 4800, 3200), square=(12, 12, 3600, 3600)),
            _variant("l", "L", (16, 12), landscape=(16, 10.67, 4800, 3200), square=(12, 12, 3600, 3600)),
            _variant("xl", "XL", (18, 13), landscape=(18, 12, 5400, 3600), square=(13, 13, 3900, 3900)),
            _variant("xxl", "XXL", (18, 14), landscape=(18, 12, 5400, 3600), square=(14, 14, 4200, 4200)),
            _variant("xxxl", "XXXL", (18, 14), landscape=(18, 12, 5400, 3600), square=(14, 14, 4200, 4200)),
        ],
    },
    "hoodies": {
        "label": "Hoodies",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant("xs", "XS", (12, 10), landscape=(12, 8, 3600, 2400), square=(10, 10, 3000, 3000)),
            _variant("s", "S", (14, 11), landscape=(14, 9.33, 4200, 2800), square=(11, 11, 3300, 3300)),
            _variant("m", "M", (16, 12), landscape=(16, 10.67, 4800, 3200), square=(12, 12, 3600, 3600)),
            _variant("l", "L", (16, 12), landscape=(16, 10.67, 4800, 3200), square=(12, 12, 3600, 3600)),
            _variant("xl", "XL", (18, 13), landscape=(18, 12, 5400, 3600), square=(13, 13, 3900, 3900)),
            _variant("xxl", "XXL", (18, 14), landscape=(18, 12, 5400, 3600), square=(14, 14, 4200, 4200)),
            _variant("xxxl", "XXXL", (18, 14), landscape=(18, 12, 5400, 3600), square=(14, 14, 4200, 4200)),
        ],
    },
    "tote-bags": {
        "label": "Tote Bags",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant(
                "one-size",
                "One Size",
                (16, 12),
                landscape=(16, 10.67, 4800, 3200),
                square=(12, 12, 3600, 3600),
            ),
        ],
    },
    "caps": {
        "label": "Caps",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant(
                "one-size",
                "One Size",
                (4, 2.5),
                landscape=(3.75, 2.5, 1125, 750),
                square=(2.5, 2.5, 750, 750),
            ),
        ],
    },
    "aprons": {
        "label": "Aprons",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant(
                "one-size",
                "One Size",
                (10, 12),
                landscape=(10, 6.67, 3000, 2000),
                square=(10, 10, 3000, 3000),
            ),
        ],
    },
    "stickers": {
        "label": "Stickers",
        "file_formats": ["PNG", "SVG", "PDF"],
        "variants": [
            _variant("2x2", '2 × 2"', (2, 2), print_only=(2, 2, 600, 600)),
            _variant("2x3", '2 × 3"', (2, 3), print_only=(2, 3, 600, 900)),
            _variant("3x3", '3 × 3"', (3, 3), print_only=(3, 3, 900, 900)),
            _variant("3x4", '3 × 4"', (3, 4), print_only=(3, 4, 900, 1200)),
            _variant("4x4", '4 × 4"', (4, 4), print_only=(4, 4, 1200, 1200)),
            _variant("5x5", '5 × 5"', (5, 5), print_only=(5, 5, 1500, 1500)),
        ],
    },
    "magnets": {
        "label": "Magnets",
        "file_formats": ["PNG", "SVG", "PDF"],
        "variants": [
            _variant("2x2", '2 × 2"', (2, 2), print_only=(2, 2, 600, 600)),
            _variant("2x3", '2 × 3"', (2, 3), print_only=(2, 3, 600, 900)),
            _variant("3x3", '3 × 3"', (3, 3), print_only=(3, 3, 900, 900)),
            _variant("3x4", '3 × 4"', (3, 4), print_only=(3, 4, 900, 1200)),
            _variant("4x4", '4 × 4"', (4, 4), print_only=(4, 4, 1200, 1200)),
            _variant("5x5", '5 × 5"', (5, 5), print_only=(5, 5, 1500, 1500)),
        ],
    },
    "mugs": {
        "label": "Mugs",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant(
                "11oz",
                "11 oz",
                (9, 3.75),
                landscape=(5.625, 3.75, 1688, 1125),
                square=(3.75, 3.75, 1125, 1125),
            ),
        ],
    },
    "tumblers": {
        "label": "Tumblers",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant(
                "20oz",
                "20 oz",
                (9.3, 8.2),
                landscape=(9.3, 6.2, 2790, 1860),
                square=(8.2, 8.2, 2460, 2460),
            ),
        ],
    },
    "canvas": {
        "label": "Canvas",
        "file_formats": ["TIFF", "PNG", "PSD", "PDF"],
        "variants": [
            _variant("12x8", '12 × 8"', (12, 8), print_only=(12, 8, 3600, 2400)),
            _variant("15x10", '15 × 10"', (15, 10), print_only=(15, 10, 4500, 3000)),
            _variant("18x12", '18 × 12"', (18, 12), print_only=(18, 12, 5400, 3600)),
            _variant("24x16", '24 × 16"', (24, 16), print_only=(24, 16, 7200, 4800)),
            _variant("30x20", '30 × 20"', (30, 20), print_only=(30, 20, 9000, 6000)),
            _variant("36x24", '36 × 24"', (36, 24), print_only=(36, 24, 10800, 7200)),
            _variant("45x30", '45 × 30"', (45, 30), print_only=(45, 30, 13500, 9000)),
            _variant("8x8", '8 × 8"', (8, 8), print_only=(8, 8, 2400, 2400)),
            _variant("12x12", '12 × 12"', (12, 12), print_only=(12, 12, 3600, 3600)),
            _variant("16x16", '16 × 16"', (16, 16), print_only=(16, 16, 4800, 4800)),
            _variant("20x20", '20 × 20"', (20, 20), print_only=(20, 20, 6000, 6000)),
            _variant("24x24", '24 × 24"', (24, 24), print_only=(24, 24, 7200, 7200)),
            _variant("30x30", '30 × 30"', (30, 30), print_only=(30, 30, 9000, 9000)),
        ],
    },
    "keychain": {
        "label": "Keychain",
        "file_formats": ["PNG", "SVG", "PDF", "TIFF"],
        "variants": [
            _variant(
                "3x2",
                '3 × 2"',
                (3, 2),
                landscape=(3, 2, 900, 600),
                square=(2, 2, 600, 600),
            ),
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


def resolve_aspect(variant: Variant, aspect_id: Optional[str] = None) -> Dict[str, Any]:
    aspects: Dict[str, Dict[str, Any]] = variant["aspects"]
    key = (aspect_id or "").strip().lower()
    if key in aspects:
        return {"id": key, **aspects[key]}
    default_key = variant.get("default_aspect") or next(iter(aspects))
    return {"id": default_key, **aspects[default_key]}


def _pixels_label(px_w: int, px_h: int) -> str:
    return f"{px_w:,} × {px_h:,} px"


def _aspect_api(aspect_id: str, area: Dict[str, Any]) -> Dict[str, Any]:
    pw, ph = area["print_inches"]
    px_w, px_h = area["pixels"]
    return {
        "id": aspect_id,
        "label": area["label"],
        "print_area": format_inches(pw, ph),
        "print_inches": area["print_inches"],
        "pixels": area["pixels"],
        "pixels_label": _pixels_label(px_w, px_h),
    }


def list_profiles_for_api() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for merch_id, profile in MERCH_PROFILES.items():
        variants: List[Dict[str, Any]] = []
        for variant in profile["variants"]:
            aspects = [
                _aspect_api(aspect_id, area)
                for aspect_id, area in variant["aspects"].items()
            ]
            default_aspect = resolve_aspect(variant)
            pw, ph = default_aspect["print_inches"]
            px_w, px_h = default_aspect["pixels"]
            mpw, mph = variant["max_physical_inches"]
            landscape = variant["aspects"].get("landscape")
            square = variant["aspects"].get("square")
            portrait = variant["aspects"].get("portrait")
            variants.append(
                {
                    "id": variant["id"],
                    "size_label": variant["size_label"],
                    "print_area": format_inches(pw, ph),
                    "print_inches": default_aspect["print_inches"],
                    "pixels": default_aspect["pixels"],
                    "pixels_label": _pixels_label(px_w, px_h),
                    "ppi": variant["ppi"],
                    "max_physical_print_area": format_inches(mpw, mph),
                    "max_physical_inches": variant["max_physical_inches"],
                    "default_aspect": default_aspect["id"],
                    "aspects": aspects,
                    "landscape_print_area": format_inches(*landscape["print_inches"]) if landscape else None,
                    "landscape_pixels_label": _pixels_label(*landscape["pixels"]) if landscape else None,
                    "square_print_area": format_inches(*square["print_inches"]) if square else None,
                    "square_pixels_label": _pixels_label(*square["pixels"]) if square else None,
                    "portrait_print_area": format_inches(*portrait["print_inches"]) if portrait else None,
                    "portrait_pixels_label": _pixels_label(*portrait["pixels"]) if portrait else None,
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


def validate_aspect(art_type: str, variant_id: str, aspect_id: Optional[str]) -> Tuple[bool, str]:
    variant = get_variant(art_type, variant_id)
    if variant is None:
        return validate_variant(art_type, variant_id)
    if not aspect_id:
        return True, ""
    key = aspect_id.strip().lower()
    if key not in variant["aspects"]:
        allowed = list(variant["aspects"].keys())
        return False, f"Invalid art area. Must be one of: {', '.join(allowed)}"
    return True, ""


def get_target_pixels_for_variant(
    art_type: str,
    variant_id: str,
    src_w: int,
    src_h: int,
    aspect_id: Optional[str] = None,
) -> Tuple[int, int, int, Variant]:
    variant = get_variant(art_type, variant_id)
    if variant is None:
        raise ValueError(f"Unknown variant {variant_id} for {art_type}")
    area = resolve_aspect(variant, aspect_id)
    px_w, px_h = area["pixels"]
    return px_w, px_h, variant["ppi"], variant


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
