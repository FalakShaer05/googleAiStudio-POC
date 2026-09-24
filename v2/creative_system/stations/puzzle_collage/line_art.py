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

# Split work size: fast enough for the API; assemble still upscales to 4K.
SPLIT_LONG_EDGE = 2048
OUTPUT_DPI = (300.0, 300.0)
# Alias for callers that still import TARGET_LONG_EDGE from this module.
TARGET_LONG_EDGE = SPLIT_LONG_EDGE
# Luminance at/above this is treated as paper (pure white).
PAPER_LUM_THRESHOLD = 200
# Any noticeable chroma is treated as a fill → pure white (never keep tint).
CHROMA_FILL_THRESHOLD = 12


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


def _target_size(width: int, height: int, long_edge: int = SPLIT_LONG_EDGE) -> Tuple[int, int]:
    """Downscale so the long edge is at most `long_edge`, preserving aspect ratio."""
    current = max(width, height)
    if current <= 0 or current <= long_edge:
        return width, height
    scale = long_edge / float(current)
    return (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )


def to_pure_line_art(img: Image.Image) -> Image.Image:
    """
    Force pure black ink on pure white paper — zero residual color.

    - Any noticeable chroma (blue halftone, yellow flecks, photo tint) → white
      so fill regions stay empty for coloring.
    - Dark, near-neutral strokes → black ink.
    - Everything else → white.
    """
    rgb = img.convert("RGB")
    if np is None:
        gray = rgb.convert("L")
        return gray.point(lambda p: 0 if p < PAPER_LUM_THRESHOLD else 255).convert("RGB")

    arr = np.array(rgb, dtype=np.float32, copy=True)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    ink = (lum < PAPER_LUM_THRESHOLD) & (chroma < CHROMA_FILL_THRESHOLD)
    out = np.full(arr.shape, 255, dtype=np.uint8)
    out[ink] = 0
    return Image.fromarray(out, mode="RGB")


_to_pure_line_art = to_pure_line_art


def force_bw_rgba(img: Image.Image, preserve_rgb: Optional[Tuple[int, int, int]] = None) -> Image.Image:
    """Apply pure B&W to RGB channels while preserving alpha."""
    rgba = img.convert("RGBA")
    rgb = to_pure_line_art(rgba)
    merged = Image.merge("RGBA", (*rgb.split(), rgba.getchannel("A")))
    if preserve_rgb is None or np is None:
        return merged

    pr, pg, pb = preserve_rgb
    src = np.array(rgba, dtype=np.uint8, copy=True)
    out = np.array(merged, dtype=np.uint8, copy=True)
    keep = (
        (src[:, :, 0] == pr)
        & (src[:, :, 1] == pg)
        & (src[:, :, 2] == pb)
        & (src[:, :, 3] > 0)
    )
    out[keep] = src[keep]
    return Image.fromarray(out, mode="RGBA")


def render_coloring_page(image_path: str, output_path: str) -> Tuple[bool, str]:
    """
    Gemini line-art on white. Uses 2K (much faster than 4K); assemble upscales
    the joined page to 4K @ 300 DPI.
    """
    ok, message = generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(),
        role_images=[
            (
                "SOURCE PHOTO. Convert THIS exact scene into a coloring-book "
                "line drawing. Keep the same composition, subjects, and crop. "
                "Output must be pure black ink on pure white — no color at all.",
                load_rgb(image_path),
            ),
        ],
        aspect_ratio=aspect_from_image(image_path, fallback="3:4"),
        temperature=0.2,
        operation="art_generation:creative:puzzle-collage-lineart",
        isolate_line_art=False,
        trailing_instruction=BACKGROUND_LOCK,
        image_size="2K",
    )
    if not ok:
        return False, message

    with Image.open(output_path) as art:
        page = art.convert("RGB")
    target_w, target_h = _target_size(*page.size, long_edge=SPLIT_LONG_EDGE)
    if page.size != (target_w, target_h):
        page = page.resize((target_w, target_h), Image.Resampling.LANCZOS)
    page = to_pure_line_art(page)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    page.save(output_path, format="PNG", dpi=OUTPUT_DPI, compress_level=1)
    return True, "Coloring-book page ready"
