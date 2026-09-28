"""Strip ownership-map blue/orange leaks from Holding Hands Gemini results."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .layout import CANVAS, CREAM, hand_map_path

RGB = Tuple[int, int, int]

# Dominant fills from holding-hands-hand-map.png
MAP_BLUE: RGB = (51, 107, 234)
MAP_ORANGE: RGB = (244, 135, 7)

_CREAM = np.array(CREAM, dtype=np.float32)
_MAP_BLUE = np.array(MAP_BLUE, dtype=np.float32)
_MAP_ORANGE = np.array(MAP_ORANGE, dtype=np.float32)


def _resize_rgb(img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    return np.array(img.convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32)


def _ownership_masks(map_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    db = np.linalg.norm(map_rgb - _MAP_BLUE, axis=-1)
    do = np.linalg.norm(map_rgb - _MAP_ORANGE, axis=-1)
    return db < 90, do < 90


def _is_background(rgb: np.ndarray) -> np.ndarray:
    lum = rgb.mean(axis=-1)
    sat = rgb.max(axis=-1) - rgb.min(axis=-1)
    near_cream = np.abs(rgb - _CREAM).max(axis=-1) < 28
    near_white = (lum >= 240) & (sat <= 22)
    return near_cream | near_white


def _is_ink(rgb: np.ndarray) -> np.ndarray:
    lum = rgb.mean(axis=-1)
    sat = rgb.max(axis=-1) - rgb.min(axis=-1)
    return (lum < 70) & (sat < 45)


def extract_skin_tone(photo: Image.Image) -> RGB:
    """Median warm skin-like color from a person/hand photo."""
    arr = _resize_rgb(photo, (160, 160))
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = arr.mean(axis=-1)
    sat = arr.max(axis=-1) - arr.min(axis=-1)
    skin = (
        (r > b + 8)
        & (r > g - 12)
        & (g > b - 5)
        & (lum > 45)
        & (lum < 235)
        & (sat > 12)
        & (sat < 160)
        & ~_is_background(arr)
    )
    if skin.sum() < 40:
        # Fall back to non-background midtones.
        skin = (lum > 50) & (lum < 220) & ~_is_background(arr) & (sat > 8)
    if skin.sum() < 20:
        return (210, 170, 140)
    sample = arr[skin]
    med = np.median(sample, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _map_leak_mask(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixels that still look like the ownership-map guide fills."""
    db = np.linalg.norm(rgb - _MAP_BLUE, axis=-1)
    do = np.linalg.norm(rgb - _MAP_ORANGE, axis=-1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = mx - mn
    # Blue / cyan guide: strong blue channel, not ink, not cream.
    blue = (
        ((db < 85) | ((b > r + 28) & (b > g + 12) & (b > 120) & (sat > 40)))
        & ~_is_ink(rgb)
        & ~_is_background(rgb)
    )
    # Pure map orange (very high sat, tiny blue) — not warm brown skin.
    orange = (
        ((do < 85) | ((r > 175) & (g > 60) & (g < r - 25) & (b < 55) & (sat > 140)))
        & ~_is_ink(rgb)
        & ~_is_background(rgb)
    )
    return blue, orange


def _recolor_preserving_luma(rgb: np.ndarray, mask: np.ndarray, target: RGB) -> None:
    """Paint photo skin over guide leaks; keep only relative shading inside the leak."""
    if not mask.any():
        return
    tgt = np.array(target, dtype=np.float32)
    src = rgb[mask]
    local = src.mean(axis=-1, keepdims=True)
    med = float(np.median(local)) or 1.0
    # Guide blue/orange are often brighter than real skin — do not inherit that luma.
    factor = np.clip(local / med, 0.78, 1.18)
    shaded = tgt * factor
    blend = 0.92
    rgb[mask] = np.clip(src * (1.0 - blend) + shaded * blend, 0, 255)


def strip_map_colors(
    result: Image.Image,
    photo_a: Image.Image,
    photo_b: Image.Image,
    hand_map: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Replace leftover ownership-map blue/orange with each person's photo skin tone.

    Safe no-op when the result already uses natural skin (no guide hues).
    """
    map_path = hand_map_path()
    if hand_map is None and not map_path:
        return result

    working = result.convert("RGB").resize(CANVAS, Image.LANCZOS)
    rgb = np.array(working, dtype=np.float32)
    map_rgb = _resize_rgb(hand_map or Image.open(map_path), CANVAS)
    person_a_zone, person_b_zone = _ownership_masks(map_rgb)
    leak_blue, leak_orange = _map_leak_mask(rgb)

    # Prefer fixing inside the matching ownership zone; also catch leaks anywhere.
    fix_a = leak_blue & (person_a_zone | ~person_b_zone)
    fix_b = leak_orange & (person_b_zone | ~person_a_zone)
    # Blue pixels that landed on B's zone (or vice versa) still get corrected.
    fix_a |= leak_blue & person_b_zone
    fix_b |= leak_orange & person_a_zone

    if not fix_a.any() and not fix_b.any():
        return result if result.size == CANVAS else working

    # Grow slightly so soft guide washes at edges are covered.
    if fix_a.any():
        fix_a = ndimage.binary_dilation(fix_a, iterations=2) & ~_is_ink(rgb) & ~_is_background(rgb)
        # Stay inside hands (non-cream).
        fix_a &= person_a_zone | leak_blue
    if fix_b.any():
        fix_b = ndimage.binary_dilation(fix_b, iterations=2) & ~_is_ink(rgb) & ~_is_background(rgb)
        fix_b &= person_b_zone | leak_orange

    skin_a = extract_skin_tone(photo_a)
    skin_b = extract_skin_tone(photo_b)
    _recolor_preserving_luma(rgb, fix_a, skin_a)
    _recolor_preserving_luma(rgb, fix_b, skin_b)

    return Image.fromarray(rgb.round().clip(0, 255).astype(np.uint8), "RGB")
