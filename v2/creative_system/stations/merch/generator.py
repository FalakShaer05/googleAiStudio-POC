"""Generate merch mockups by placing artwork onto blank product templates (fast PIL)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .catalog import MERCH_PRODUCTS, expand_products_for_colors, product_path
from .compositor import (
    composite_on_template,
    make_canvas_print,
    make_photo_print,
    make_sticker_fan,
    make_sticker_sheet,
    make_sticker_single,
    tint_garment_template,
    trim_artwork,
)

_MAX_ART_SIDE = 1400
_TEMPLATE_CACHE: Dict[str, Image.Image] = {}


def _load_artwork(path: str) -> Image.Image:
    img = Image.open(path)
    # Apply EXIF rotation so phone portraits aren't misread as landscape.
    try:
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img) or img
    except Exception:
        pass
    # Always keep alpha — apparel mockups need transparent backdrops intact.
    if img.mode == "P":
        img = img.convert("RGBA")
    elif img.mode != "RGBA":
        img = img.convert("RGBA")
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
    *,
    is_transparent: bool = False,
) -> None:
    mode = product["mode"]
    template = _load_template(product.get("template"))

    if mode == "sticker_single":
        out = make_sticker_single(
            artwork,
            orientation=product.get("orientation"),
        )
    elif mode == "sticker_fan":
        out = make_sticker_fan(
            artwork,
            count=int(product.get("sticker_count") or 5),
            orientation=product.get("orientation"),
        )
    elif mode == "sticker_sheet":
        out = make_sticker_sheet(
            artwork,
            orientation=product.get("orientation"),
        )
    elif mode == "photo_print":
        out = make_photo_print(
            artwork,
            template,
            orientation=product.get("orientation"),
        )
    elif mode == "canvas":
        out = make_canvas_print(
            artwork,
            template=None,
            orientation=product.get("orientation"),
        )
    elif mode == "mockup":
        if template is None:
            raise FileNotFoundError(f"Missing merch template for {product['id']}")
        tint = product.get("tint")
        if tint:
            template = tint_garment_template(template, tuple(tint))
        print_box = product.get("print_box") or (0.25, 0.25, 0.75, 0.55)
        portrait_box = product.get("portrait_print_box")
        if portrait_box and detect_artwork_orientation(artwork) == "portrait":
            print_box = portrait_box
        out = composite_on_template(
            template,
            artwork,
            print_box=print_box,
            fit=product.get("fit") or "cover",
            opacity=1.0,
            shade=bool(product.get("shade", False)),
            knockout=bool(product.get("knockout", False)),
            match_art_aspect=bool(product.get("match_art_aspect", True)),
            orientation=product.get("orientation"),
            is_transparent=is_transparent,
        )
    else:
        raise ValueError(f"Unknown merch mode: {mode}")

    _save_png(out, output_path)


def _generate_one(
    product: Dict[str, Any],
    artwork: Image.Image,
    output_path: str,
    *,
    is_transparent: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    meta = {
        "id": product["id"],
        "variant_id": product.get("variant_id") or product["id"],
        "label": product["label"],
        "tags": list(product.get("tags") or []),
        "color": product.get("color"),
        "color_label": product.get("color_label"),
        "orientation": product.get("orientation"),
        "orientation_label": product.get("orientation_label"),
        "output_filename": os.path.basename(output_path),
    }
    try:
        _composite_one(
            product,
            artwork,
            output_path,
            is_transparent=is_transparent,
        )
        return True, "Mockup composed", meta
    except Exception as exc:
        return False, str(exc), meta


def detect_artwork_orientation(artwork: Image.Image) -> str:
    """
    Detect from the trimmed design, not the full canvas.

    Uploads often sit on a taller/wider studio canvas; stickers and prints
    already trim to content, so orientation must use that same box.

    Square / near-square counts as portrait (sheet brand refs). Only clearly
    wider art (width > height * 1.15) is landscape.
    """
    trimmed = trim_artwork(artwork, pad=2)
    w, h = max(1, trimmed.size[0]), max(1, trimmed.size[1])
    return "portrait" if (w / float(h)) <= 1.15 else "landscape"


def _apply_auto_orientations(
    products: List[Dict[str, Any]],
    artwork: Image.Image,
) -> List[Dict[str, Any]]:
    """Detect artwork orientation and attach it to products that need it."""
    detected = detect_artwork_orientation(artwork)
    label = detected.title()
    out: List[Dict[str, Any]] = []
    for product in products:
        if not product.get("auto_orientation"):
            out.append(product)
            continue
        job = dict(product)
        job["orientation"] = detected
        job["orientation_label"] = label
        job["variant_id"] = f"{product.get('variant_id') or product['id']}-{detected}"
        base_tags = [
            t
            for t in (product.get("tags") or [])
            if str(t).lower() not in {"portrait", "landscape"}
        ]
        job["tags"] = base_tags + [label]
        out.append(job)
    return out


def generate_all(
    artwork_path: str,
    output_dir: str,
    filename_prefix: str = "cs_merch",
    max_workers: int = 12,
    product_ids: Optional[List[str]] = None,
    color_choices: Optional[Dict[str, List[str]]] = None,
    orientation_choices: Optional[Dict[str, str]] = None,
    is_transparent: bool = False,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Place one artwork onto every selected product blank (local PIL, no Gemini).

    color_choices selects apparel colors, e.g. {"tshirt": ["black","white"]}.
    Orientation is auto-detected from artwork size (portrait if taller than wide).
    orientation_choices is ignored (kept for API compatibility).
    is_transparent: when True, apparel/cap/tote keep or key out the art backdrop
    so the garment shows through; when False (default), keep the background plate.

    Returns (success, message, items).
    """
    del orientation_choices  # Auto-detect from artwork; UI no longer collects this.
    os.makedirs(output_dir, exist_ok=True)
    wanted = set(product_ids) if product_ids else None
    products = [
        p for p in MERCH_PRODUCTS
        if (wanted is None and not p.get("hidden")) or (wanted is not None and p["id"] in wanted)
    ]
    if not products:
        return False, "No merch products selected", []

    artwork = _load_artwork(artwork_path)
    jobs_spec = expand_products_for_colors(products, color_choices=color_choices)
    jobs_spec = _apply_auto_orientations(jobs_spec, artwork)
    if not jobs_spec:
        return False, "No merch variants to generate (check color selection)", []

    jobs = []
    for product in jobs_spec:
        color = product.get("color")
        orientation = product.get("orientation")
        parts = [product["id"]]
        if color:
            parts.append(str(color))
        if orientation:
            parts.append(str(orientation))
        suffix = "_".join(parts)
        out_name = f"{filename_prefix}_{suffix}_{os.urandom(4).hex()}.png"
        out_path = os.path.join(output_dir, out_name)
        jobs.append((product, out_path))

    items: List[Dict[str, Any]] = []
    errors: List[str] = []
    use_transparent = bool(is_transparent)

    # PIL images are not safely shared across threads — copy per job.
    workers = max(1, min(max_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _generate_one,
                product,
                artwork.copy(),
                out_path,
                is_transparent=use_transparent,
            )
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
    color_order = {"white": 0, "red": 1, "green": 2, "black": 3, "gray": 4}
    orientation_order = {"portrait": 0, "landscape": 1}

    def _sort_key(item: Dict[str, Any]) -> Tuple[int, int, int]:
        return (
            order.get(item["id"], 999),
            color_order.get(item.get("color") or "", 9),
            orientation_order.get(item.get("orientation") or "", 9),
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
        orientation_choices=kwargs.get("orientation_choices"),
        is_transparent=bool(kwargs.get("is_transparent", False)),
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
