"""Generate merch mockups by placing artwork onto blank product templates (fast PIL)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ...shared.gemini import to_rgb
from .catalog import MERCH_PRODUCTS, expand_products_for_colors, product_path
from .compositor import (
    composite_on_template,
    make_canvas_print,
    make_photo_print,
    make_sticker_fan,
    make_sticker_sheet,
    make_sticker_single,
)

_MAX_ART_SIDE = 1400
_TEMPLATE_CACHE: Dict[str, Image.Image] = {}


def _load_artwork(path: str) -> Image.Image:
    img = Image.open(path)
    img = img.convert("RGBA") if img.mode in {"RGBA", "P"} else to_rgb(img)
    if max(img.size) > _MAX_ART_SIDE:
        img = img.copy()
        img.thumbnail((_MAX_ART_SIDE, _MAX_ART_SIDE), Image.Resampling.LANCZOS)
    return img


def _load_template(filename: Optional[str]) -> Optional[Image.Image]:
    path = product_path(filename)
    if not path:
        return None
    cached = _TEMPLATE_CACHE.get(path)
    if cached is not None:
        return cached.copy()
    img = Image.open(path)
    img.load()
    _TEMPLATE_CACHE[path] = img
    return img.copy()


def _save_png(img: Image.Image, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # compress_level=3 is much faster than optimize=True for mockup previews.
    img.save(output_path, format="PNG", compress_level=3)


def _composite_one(
    product: Dict[str, Any],
    artwork: Image.Image,
    output_path: str,
) -> None:
    mode = product["mode"]
    template = _load_template(product.get("template"))

    if mode == "sticker_single":
        out = make_sticker_single(artwork)
    elif mode == "sticker_fan":
        out = make_sticker_fan(artwork, count=int(product.get("sticker_count") or 5))
    elif mode == "sticker_sheet":
        out = make_sticker_sheet(artwork)
    elif mode == "photo_print":
        out = make_photo_print(artwork, template)
    elif mode == "canvas":
        out = make_canvas_print(artwork, template=None)
    elif mode == "mockup":
        if template is None:
            raise FileNotFoundError(f"Missing merch template for {product['id']}")
        out = composite_on_template(
            template,
            artwork,
            print_box=product.get("print_box") or (0.25, 0.25, 0.75, 0.55),
            fit=product.get("fit") or "cover",
            opacity=1.0,
            shade=bool(product.get("shade", False)),
            knockout=bool(product.get("knockout", False)),
            match_art_aspect=bool(product.get("match_art_aspect", True)),
        )
    else:
        raise ValueError(f"Unknown merch mode: {mode}")

    _save_png(out, output_path)


def _generate_one(
    product: Dict[str, Any],
    artwork: Image.Image,
    output_path: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    meta = {
        "id": product["id"],
        "variant_id": product.get("variant_id") or product["id"],
        "label": product["label"],
        "tags": list(product.get("tags") or []),
        "color": product.get("color"),
        "color_label": product.get("color_label"),
        "output_filename": os.path.basename(output_path),
    }
    try:
        _composite_one(product, artwork, output_path)
        return True, "Mockup composed", meta
    except Exception as exc:
        return False, str(exc), meta


def generate_all(
    artwork_path: str,
    output_dir: str,
    filename_prefix: str = "cs_merch",
    max_workers: int = 12,
    product_ids: Optional[List[str]] = None,
    color_choices: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Place one artwork onto every selected product blank (local PIL, no Gemini).

    color_choices selects apparel colors, e.g. {"tshirt": ["black","white"]}.

    Returns (success, message, items).
    """
    os.makedirs(output_dir, exist_ok=True)
    wanted = set(product_ids) if product_ids else None
    products = [
        p for p in MERCH_PRODUCTS
        if wanted is None or p["id"] in wanted
    ]
    if not products:
        return False, "No merch products selected", []

    jobs_spec = expand_products_for_colors(products, color_choices=color_choices)
    if not jobs_spec:
        return False, "No merch variants to generate (check color selection)", []

    artwork = _load_artwork(artwork_path)

    jobs = []
    for product in jobs_spec:
        color = product.get("color")
        suffix = f"{product['id']}_{color}" if color else product["id"]
        out_name = f"{filename_prefix}_{suffix}_{os.urandom(4).hex()}.png"
        out_path = os.path.join(output_dir, out_name)
        jobs.append((product, out_path))

    items: List[Dict[str, Any]] = []
    errors: List[str] = []

    # PIL images are not safely shared across threads — copy per job.
    workers = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_generate_one, product, artwork.copy(), out_path)
            for product, out_path in jobs
        ]
        for future in as_completed(futures):
            ok, message, meta = future.result()
            if ok and os.path.isfile(os.path.join(output_dir, meta["output_filename"])):
                items.append(meta)
            else:
                label = meta["label"]
                if meta.get("color_label"):
                    label = f"{label} ({meta['color_label']})"
                errors.append(f"{label}: {message or 'failed'}")

    order = {p["id"]: i for i, p in enumerate(MERCH_PRODUCTS)}
    color_order = {"black": 0, "white": 1}

    def _sort_key(item: Dict[str, Any]) -> Tuple[int, int]:
        return (
            order.get(item["id"], 999),
            color_order.get(item.get("color") or "", 9),
        )

    items.sort(key=_sort_key)

    if not items:
        return False, "; ".join(errors) or "Merch generation failed", []

    msg = f"Generated {len(items)} merch mockup{'s' if len(items) != 1 else ''}."
    if errors:
        msg += f" {len(errors)} skipped."
    return True, msg, items


def generate(output_path: str, artwork_path: str, **kwargs):
    output_dir = os.path.dirname(output_path) or "."
    prefix = os.path.splitext(os.path.basename(output_path))[0]
    ok, message, items = generate_all(
        artwork_path=artwork_path,
        output_dir=output_dir,
        filename_prefix=prefix,
        product_ids=kwargs.get("product_ids"),
        color_choices=kwargs.get("color_choices"),
    )
    extras = kwargs.get("result_extras")
    if isinstance(extras, dict):
        extras["items"] = items
        extras["merch_count"] = len(items)
    if ok and items:
        first = os.path.join(output_dir, items[0]["output_filename"])
        if os.path.isfile(first) and first != output_path:
            try:
                from shutil import copyfile
                copyfile(first, output_path)
            except OSError:
                pass
    return ok, message
