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

# Print-quality target: long edge at least 4K, tagged 300 DPI.
TARGET_LONG_EDGE = 3840
OUTPUT_DPI = (300.0, 300.0)
# Luminance at/above this is treated as paper (pure white).
PAPER_LUM_THRESHOLD = 210
# Chroma above this (max-min RGB) is forced to ink so tinted dots cannot survive.
CHROMA_INK_THRESHOLD = 18


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


def _target_size(width: int, height: int, long_edge: int = TARGET_LONG_EDGE) -> Tuple[int, int]:
    """Scale so the long edge is at least `long_edge`, preserving aspect ratio."""
    current = max(width, height)
    if current <= 0:
        return width, height
    if current >= long_edge:
        return width, height
    scale = long_edge / float(current)
    return (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )


def _to_pure_line_art(img: Image.Image) -> Image.Image:
    """
    Force pure black ink on pure white paper.

    Any noticeable chroma (blue halftone, yellow flecks) or darkness becomes
    black; everything else becomes white. Keeps a fillable coloring page
    (opaque white), not a transparent cutout.
    """
    rgb = img.convert("RGB")
    if np is None:
        gray = rgb.convert("L")
        # Fallback without numpy: threshold + drop color via L conversion.
        return gray.point(lambda p: 0 if p < PAPER_LUM_THRESHOLD else 255).convert("RGB")

    arr = np.asarray(rgb, dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    ink = (lum < PAPER_LUM_THRESHOLD) | (chroma >= CHROMA_INK_THRESHOLD)
    out = np.full(arr.shape, 255, dtype=np.uint8)
    out[ink] = (0, 0, 0)
    return Image.fromarray(out, mode="RGB")


def render_coloring_page(image_path: str, output_path: str) -> Tuple[bool, str]:
    """
    Gemini line-art conversion on a solid white field so participants can
    fill the pieces with color. Output is pure B&W at >=4K long edge / 300 DPI.
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
        image_size="4K",
    )
    if not ok:
        return False, message

    with Image.open(image_path) as source:
        target_w, target_h = _target_size(*source.size)

    with Image.open(output_path) as art:
        page = art.convert("RGB")
    if page.size != (target_w, target_h):
        page = page.resize((target_w, target_h), Image.Resampling.LANCZOS)
    page = _to_pure_line_art(page)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    page.save(output_path, format="PNG", dpi=OUTPUT_DPI, compress_level=0)
    return True, "Coloring-book page ready"
