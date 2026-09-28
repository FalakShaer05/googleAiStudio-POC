"""Palette helpers for origami — extract, classify, soft gamut clamp."""
from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.gemini import to_rgb

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    np = None  # type: ignore
    _HAS_NP = False

RGB = Tuple[int, int, int]


def _is_near_black(rgb: RGB, threshold: int = 32) -> bool:
    return max(rgb) <= threshold


def _is_bg_or_line(rgb: RGB) -> bool:
    """Skip pure black bg, near-white void, and dark crease-line ink."""
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    sat = mx - mn
    lum = (r + g + b) / 3.0
    if mx <= 36:
        return True
    if lum >= 245 and sat <= 18:
        return True
    # Dark dashed crease lines (low sat, dark)
    if lum <= 70 and sat <= 40:
        return True
    return False


def _color_dist(a: RGB, b: RGB) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def extract_palette(image_path: str, max_colors: int = 8) -> List[RGB]:
    """Dominant paper/print colors from the open crease pattern."""
    img = to_rgb(Image.open(image_path))
    sample = img.copy()
    sample.thumbnail((200, 200))
    quantize_kwargs = {"colors": 24}
    try:
        quantize_kwargs["method"] = Image.Quantize.MEDIANCUT
    except AttributeError:
        quantize_kwargs["method"] = getattr(Image, "MEDIANCUT", 0)
    quantized = sample.quantize(**quantize_kwargs).convert("RGB")
    counts: Counter[RGB] = Counter(quantized.getdata())

    picked: List[RGB] = []
    for color, _n in counts.most_common(64):
        if _is_bg_or_line(color):
            continue
        if any(_color_dist(color, p) < 36 for p in picked):
            continue
        picked.append(color)
        if len(picked) >= max_colors:
            break
    return picked


def dominant_paper_color(image_path: str) -> Optional[RGB]:
    """
    If one paper hue covers most of the sheet (solid origami paper), return it.
    Multi-print sheets return None.
    """
    img = to_rgb(Image.open(image_path))
    sample = img.copy()
    sample.thumbnail((160, 160))
    quantize_kwargs = {"colors": 16}
    try:
        quantize_kwargs["method"] = Image.Quantize.MEDIANCUT
    except AttributeError:
        quantize_kwargs["method"] = getattr(Image, "MEDIANCUT", 0)
    quantized = sample.quantize(**quantize_kwargs).convert("RGB")
    counts: Counter[RGB] = Counter(quantized.getdata())

    paper_pixels = [(c, n) for c, n in counts.most_common() if not _is_bg_or_line(c)]
    if not paper_pixels:
        return None
    total = sum(n for _c, n in paper_pixels)
    if total < 20:
        return None

    top, top_n = paper_pixels[0]
    # Merge near-duplicates into the top cluster share
    cluster_n = sum(n for c, n in paper_pixels if _color_dist(c, top) < 55)
    if cluster_n / total >= 0.62:
        return top
    return None


def palette_to_hex(palette: Sequence[RGB]) -> str:
    return ", ".join("#{:02X}{:02X}{:02X}".format(*c) for c in palette)


def build_palette_swatch(palette: Sequence[RGB], width: int = 512, row_h: int = 64) -> Image.Image:
    """Flat color chips — unused in main path; kept for debugging."""
    if not palette:
        return Image.new("RGB", (width, row_h), (200, 200, 200))
    h = row_h * len(palette)
    swatch = Image.new("RGB", (width, h), (0, 0, 0))
    draw = ImageDraw.Draw(swatch)
    for i, color in enumerate(palette):
        draw.rectangle([0, i * row_h, width, (i + 1) * row_h], fill=color)
    return swatch


def _luminance(rgb: RGB) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _shade_of(base: RGB, target_lum: float) -> RGB:
    """Keep hue of base; shift brightness toward target_lum for fold shading."""
    base_lum = max(_luminance(base), 1.0)
    scale = target_lum / base_lum
    # Mild clamp so shadows don't crush to black unless base is dark
    scale = max(0.35, min(1.35, scale))
    return tuple(max(0, min(255, int(round(c * scale)))) for c in base)  # type: ignore[return-value]


def soft_clamp_to_palette(
    image: Image.Image,
    palette: Sequence[RGB],
    threshold: int = 72,
    solid_color: Optional[RGB] = None,
) -> Image.Image:
    """
    Keep in-gamut pixels. Remap only out-of-gamut / invented hues.
    Solid paper mode: force entire subject onto that paper hue family (with shading).
    """
    rgb = to_rgb(image)
    if solid_color is not None:
        palette = [solid_color] + [c for c in palette if _color_dist(c, solid_color) > 30]
        # Build a short shade ladder so folds stay dimensional
        shades = [
            _shade_of(solid_color, _luminance(solid_color) * s)
            for s in (0.45, 0.65, 0.85, 1.0, 1.15)
        ]
        palette = list(dict.fromkeys([*shades, *palette]))

    if not palette:
        return rgb

    if _HAS_NP:
        arr = np.asarray(rgb, dtype=np.int16)
        flat = arr.reshape(-1, 3)
        palette_arr = np.asarray(list(palette), dtype=np.int16)
        dists = np.abs(flat[:, None, :] - palette_arr[None, :, :]).sum(axis=2)
        nearest_idx = dists.argmin(axis=1)
        nearest = palette_arr[nearest_idx]
        min_dist = dists.min(axis=1)

        if solid_color is not None:
            # Always map to paper-family shades; preserve relative brightness
            out_flat = nearest.copy()
            lum = (0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2])
            base = np.asarray(solid_color, dtype=np.float32)
            base_lum = max(float(_luminance(solid_color)), 1.0)
            scale = np.clip(lum / base_lum, 0.35, 1.35)
            shaded = np.clip(base[None, :] * scale[:, None], 0, 255).astype(np.int16)
            # Prefer continuous shading over discrete snap for solid paper
            out_flat = shaded
        else:
            out_flat = flat.copy()
            far = min_dist > threshold
            out_flat[far] = nearest[far]

        return Image.fromarray(out_flat.astype(np.uint8).reshape(arr.shape), mode="RGB")

    pixels = list(rgb.getdata())
    out_px: List[RGB] = []
    for px in pixels:
        if solid_color is not None:
            out_px.append(_shade_of(solid_color, _luminance(px)))
            continue
        nearest = min(palette, key=lambda c: _color_dist(px, c))
        if _color_dist(px, nearest) > threshold:
            out_px.append(nearest)
        else:
            out_px.append(px)  # type: ignore[arg-type]
    out = Image.new("RGB", rgb.size)
    out.putdata(out_px)
    return out


def snap_to_palette(
    image: Image.Image,
    palette: Sequence[RGB],
    black_threshold: int = 36,
) -> Image.Image:
    """Legacy hard snap — prefer soft_clamp_to_palette."""
    if not palette:
        return to_rgb(image)

    rgb = to_rgb(image)
    if _HAS_NP:
        arr = np.asarray(rgb, dtype=np.int16)
        flat = arr.reshape(-1, 3)
        lum = flat.max(axis=1)
        bg = lum <= black_threshold
        palette_arr = np.asarray(palette, dtype=np.int16)
        dists = np.abs(flat[:, None, :] - palette_arr[None, :, :]).sum(axis=2)
        nearest_idx = dists.argmin(axis=1)
        snapped_flat = palette_arr[nearest_idx]
        snapped_flat[bg] = (0, 0, 0)
        out = snapped_flat.astype(np.uint8).reshape(arr.shape)
        return Image.fromarray(out, mode="RGB")

    pixels = list(rgb.getdata())
    out_px: List[RGB] = []
    for px in pixels:
        if _is_near_black(px, black_threshold):
            out_px.append((0, 0, 0))
            continue
        out_px.append(min(palette, key=lambda c: _color_dist(px, c)))
    snapped = Image.new("RGB", rgb.size)
    snapped.putdata(out_px)
    return snapped


def force_black_background(image: Image.Image, threshold: int = 28) -> Image.Image:
    """Crush near-black to pure black so framing stays studio-clean."""
    rgb = to_rgb(image)
    if _HAS_NP:
        arr = np.asarray(rgb, dtype=np.uint8).copy()
        bg = arr.max(axis=2) <= threshold
        arr[bg] = (0, 0, 0)
        return Image.fromarray(arr, mode="RGB")
    pixels = list(rgb.getdata())
    out = [(0, 0, 0) if _is_near_black(px, threshold) else px for px in pixels]
    cleaned = Image.new("RGB", rgb.size)
    cleaned.putdata(out)
    return cleaned
