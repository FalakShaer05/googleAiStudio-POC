"""Turn the puzzle source photo into a fillable coloring-book page."""
from __future__ import annotations

import os
from typing import Optional, Tuple

from PIL import Image

from ...shared.gemini import aspect_from_image, generate_composed_image, load_rgb
from .prompts import BACKGROUND_LOCK, build_prompt

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


def _dpi_tuple(path: str) -> Optional[Tuple[float, float]]:
    try:
        with Image.open(path) as img:
            dpi = img.info.get("dpi")
    except Exception:
        return None
    if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
        x, y = float(dpi[0]), float(dpi[1])
        if x > 0 and y > 0:
            return (x, y)
    if isinstance(dpi, (int, float)) and float(dpi) > 0:
        v = float(dpi)
        return (v, v)
    return None


def _whiten_paper(img: Image.Image) -> Image.Image:
    """Force pale / beige paper pixels to pure white; leave ink lines alone."""
    rgb = img.convert("RGB")
    if np is None:
        return rgb

    arr = np.asarray(rgb)
    lum = arr.mean(axis=2)
    # Near-white and warm cream/beige fills become solid white.
    paper = (lum >= 200) | (
        (arr[:, :, 0] >= 210) & (arr[:, :, 1] >= 200) & (arr[:, :, 2] >= 180)
    )
    if not paper.any():
        return rgb
    out = arr.copy()
    out[paper] = (255, 255, 255)
    return Image.fromarray(out, mode="RGB")


def render_coloring_page(image_path: str, output_path: str) -> Tuple[bool, str]:
    """
    Gemini line-art conversion (same pipeline as Selfie Becoming) on a
    solid white field so participants can fill the pieces with color.
    """
    ok, message = generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(),
        role_images=[
            (
                "SOURCE PHOTO. Convert THIS exact scene into a coloring-book "
                "line drawing. Keep the same composition, subjects, and crop.",
                load_rgb(image_path),
            ),
        ],
        aspect_ratio=aspect_from_image(image_path, fallback="3:4"),
        temperature=0.4,
        operation="art_generation:creative:puzzle-collage-lineart",
        isolate_line_art=False,
        trailing_instruction=BACKGROUND_LOCK,
    )
    if not ok:
        return False, message

    with Image.open(image_path) as source:
        target_size = source.size
    dpi = _dpi_tuple(image_path) or (300.0, 300.0)

    with Image.open(output_path) as art:
        page = art.convert("RGB")
    if page.size != target_size:
        page = page.resize(target_size, Image.Resampling.LANCZOS)
    page = _whiten_paper(page)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    page.save(output_path, format="PNG", dpi=dpi, compress_level=0)
    return True, "Coloring-book page ready"
