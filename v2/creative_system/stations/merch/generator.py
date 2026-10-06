"""Generate merch mockups from a single source artwork."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ...shared.gemini import generate_composed_image, load_rgb, to_rgb
from .catalog import MERCH_PRODUCTS, product_path
from .compositor import (
    flatten_on_white,
    make_canvas_print,
    make_photo_print,
    make_sticker_fan,
    make_sticker_sheet,
    make_sticker_single,
)
from .prompts import build_prompt

_MAX_ART_SIDE = 1400
_MAX_TEMPLATE_SIDE = 1024
_MAX_STYLE_SIDE = 768
_MAX_GEMINI_WORKERS = 4


def _load_artwork(path: str) -> Image.Image:
    img = Image.open(path)
    img = img.convert("RGBA") if img.mode in {"RGBA", "P"} else to_rgb(img)
    if max(img.size) > _MAX_ART_SIDE:
        img = img.copy()
        img.thumbnail((_MAX_ART_SIDE, _MAX_ART_SIDE), Image.Resampling.LANCZOS)
    return img


def _thumb(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def _thumb_rgb(path: str, max_side: int) -> Image.Image:
    return _thumb(load_rgb(path), max_side)


def _role_images(product: Dict[str, Any], artwork: Image.Image):
    mode = product["mode"]
    art = to_rgb(artwork) if artwork.mode != "RGB" else artwork
    art = _thumb(art, _MAX_ART_SIDE)

    if mode == "mockup":
        template = product_path(product.get("template"))
        if not template:
            raise FileNotFoundError(f"Missing merch template for {product['id']}")
        blank = flatten_on_white(Image.open(template))
        blank = _thumb(blank, _MAX_TEMPLATE_SIDE)
        return [
            (
                "BLANK PRODUCT TEMPLATE. Keep this exact product, color, camera angle, "
                "fabric, and lighting. Only add the source artwork as a print.",
                blank,
            ),
            (
                "SOURCE ARTWORK to print onto the blank product. Use this design exactly "
                "— do not invent a different character, do not stretch or squash it.",
                art,
            ),
        ]

    roles = [
        (
            "SOURCE ARTWORK. This is the only design — keep colors and details exact.",
            art,
        ),
    ]
    style = product_path(product.get("style_ref"))
    if style:
        roles.append(
            (
                "STYLE / LAYOUT REFERENCE only. Match sticker finish and composition. "
                "Replace the reference character with the source artwork.",
                _thumb_rgb(style, _MAX_STYLE_SIDE),
            )
        )
    return roles


def _composite_one(product: Dict[str, Any], artwork: Image.Image, output_path: str) -> None:
    mode = product["mode"]
    template_path = product_path(product.get("template"))
    template = Image.open(template_path) if template_path else None

    if mode == "sticker_single":
        out = make_sticker_single(artwork)
    elif mode == "sticker_fan":
        out = make_sticker_fan(artwork)
    elif mode == "sticker_sheet":
        out = make_sticker_sheet(artwork)
    elif mode == "photo_print":
        out = make_photo_print(artwork, template)
    elif mode == "canvas":
        out = make_canvas_print(artwork, template=None)
    else:
        raise ValueError(f"No composite path for mode={mode}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.save(output_path, format="PNG", optimize=True)


def _generate_one(
    product: Dict[str, Any],
    artwork: Image.Image,
    artwork_path: str,
    output_path: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    meta = {
        "id": product["id"],
        "label": product["label"],
        "tags": list(product.get("tags") or []),
        "output_filename": os.path.basename(output_path),
    }
    try:
        engine = product.get("engine") or "gemini"
        if engine == "composite":
            _composite_one(product, artwork, output_path)
            return True, "Mockup composed", meta

        trailing = None
        if product["id"] in {"tshirt", "hoodie"}:
            trailing = (
                "FINAL CHECK: the print must be LARGE — nearly full width of the front torso "
                "(only small margins before the armholes). A small centered stamp is a failure. "
                "Scale the artwork up to dominate the chest while keeping correct proportions."
            )
        ok, message = generate_composed_image(
            output_path=output_path,
            prompt=build_prompt(product),
            role_images=_role_images(product, artwork),
            style_target=None,
            aspect_ratio=product.get("aspect_ratio") or "1:1",
            temperature=0.2,
            image_size="1K",
            trailing_instruction=trailing,
            operation=f"art_generation:creative:merch:{product['id']}",
        )
        return ok, message, meta
    except Exception as exc:
        return False, str(exc), meta


def generate_all(
    artwork_path: str,
    output_dir: str,
    filename_prefix: str = "cs_merch",
    max_workers: int = 9,
    product_ids: Optional[List[str]] = None,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Stickers/prints compose locally; apparel mockups use Gemini + blank templates.

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

    artwork = _load_artwork(artwork_path)

    jobs = []
    for product in products:
        out_name = f"{filename_prefix}_{product['id']}_{os.urandom(4).hex()}.png"
        out_path = os.path.join(output_dir, out_name)
        jobs.append((product, out_path))

    items: List[Dict[str, Any]] = []
    errors: List[str] = []

    def _collect(ok: bool, message: str, meta: Dict[str, Any]) -> None:
        if ok and os.path.isfile(os.path.join(output_dir, meta["output_filename"])):
            items.append(meta)
        else:
            errors.append(f"{meta['label']}: {message or 'failed'}")

    composite_jobs = [(p, path) for p, path in jobs if p.get("engine") == "composite"]
    gemini_jobs = [(p, path) for p, path in jobs if p.get("engine") != "composite"]

    for product, out_path in composite_jobs:
        _collect(*_generate_one(product, artwork, artwork_path, out_path))

    if gemini_jobs:
        workers = max(1, min(_MAX_GEMINI_WORKERS, max_workers, len(gemini_jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_generate_one, product, artwork, artwork_path, out_path)
                for product, out_path in gemini_jobs
            ]
            for future in as_completed(futures):
                _collect(*future.result())

    order = {p["id"]: i for i, p in enumerate(MERCH_PRODUCTS)}
    items.sort(key=lambda item: order.get(item["id"], 999))

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
