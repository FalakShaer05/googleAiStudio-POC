"""Thin wrappers around existing Gemini image-generation helpers."""
from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageFilter, ImageOps

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    from scipy import ndimage as _ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    _ndimage = None  # type: ignore
    SCIPY_AVAILABLE = False

from utils.character_utils import (
    _extract_final_image_from_response,
    _generate_content_image,
    generate_seed_from_prompt,
    get_gemini_client,
    get_gemini_image_model,
    normalize_prompt_for_consistency,
    select_gemini_aspect_ratio,
)

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STYLE_TARGETS_DIR = os.path.join(PACKAGE_DIR, "static", "images", "style_targets")

RoleImage = Tuple[str, Image.Image]


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def load_rgb(path: str) -> Image.Image:
    return to_rgb(Image.open(path))


def style_target_path(station_id: str) -> Optional[str]:
    path = os.path.join(STYLE_TARGETS_DIR, f"{station_id}.png")
    return path if os.path.isfile(path) else None


def aspect_from_image(path: Optional[str], fallback: str = "1:1") -> str:
    if not path or not os.path.isfile(path):
        return fallback
    with Image.open(path) as img:
        width, height = img.size
    return select_gemini_aspect_ratio(width, height)


def _mask_is_usable(mask) -> bool:
    frac = float(mask.mean())
    return 0.035 <= frac <= 0.78


def _rembg_hand_mask(rgb: Image.Image):
    """Hand mask via the shared warmed rembg session (no cold model download)."""
    if not NUMPY_AVAILABLE:
        return None
    try:
        from rembg import remove
        from utils.bg_remover import _get_rembg_session, _rembg_max_side, _rembg_model_names
    except Exception:
        return None
    try:
        source = to_rgb(rgb)
        max_side = _rembg_max_side()
        if max(source.size) > max_side:
            source = source.copy()
            source.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        model_name = (_rembg_model_names() or ["u2net"])[0]
        session = _get_rembg_session(model_name)
        cut = remove(source, session=session)
        if not isinstance(cut, Image.Image):
            from io import BytesIO
            cut = Image.open(BytesIO(cut))
        alpha = np.array(cut.convert("RGBA"))[:, :, 3]
        mask = alpha > 40
        if mask.shape[:2] != (rgb.size[1], rgb.size[0]):
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            mask_img = mask_img.resize(rgb.size, Image.Resampling.NEAREST)
            mask = np.array(mask_img) > 127
        return mask if _mask_is_usable(mask) else None
    except Exception as exc:
        print(f"tracing-hand rembg stencil skip: {exc}")
        return None


def _is_sparse_line_drawing(rgb: Image.Image) -> bool:
    """True for canvas / marker outlines: mostly white with thin dark strokes."""
    if not NUMPY_AVAILABLE:
        return False
    arr = np.array(to_rgb(rgb))
    lum = arr.astype(np.int16).mean(axis=2)
    ink = lum <= 150
    frac = float(ink.mean())
    # Thin outline on white — not a filled photo hand.
    return 0.0015 <= frac <= 0.14 and float((lum >= 200).mean()) >= 0.72


def _seal_open_wrist(ink):
    """Close gaps where a canvas outline opens at the image edge (common at wrist)."""
    if not SCIPY_AVAILABLE:
        return ink
    sealed = ink.copy()
    h, w = sealed.shape
    band = max(4, h // 40)
    for edge, slicer in (
        ("bottom", (slice(-band, None), slice(None))),
        ("top", (slice(0, band), slice(None))),
        ("left", (slice(None), slice(0, band))),
        ("right", (slice(None), slice(-band, None))),
    ):
        region = sealed[slicer]
        if not region.any():
            continue
        if edge in {"bottom", "top"}:
            cols = np.where(region.any(axis=0))[0]
            if len(cols) < 2:
                continue
            y = h - 1 if edge == "bottom" else 0
            sealed[y, cols[0] : cols[-1] + 1] = True
            if edge == "bottom":
                sealed[max(0, y - 1), cols[0] : cols[-1] + 1] = True
            else:
                sealed[min(h - 1, y + 1), cols[0] : cols[-1] + 1] = True
        else:
            rows = np.where(region.any(axis=1))[0]
            if len(rows) < 2:
                continue
            x = 0 if edge == "left" else w - 1
            sealed[rows[0] : rows[-1] + 1, x] = True
    return sealed


def _smooth_hand_fill_mask(mask):
    """
    Turn a jagged / holey canvas fill into one solid silhouette.
    Seals pen gaps and vertical seams, keeps the largest blob, grows slightly
    so word packing can reach the outer rough stroke (not stop at every wiggle).
    """
    if not SCIPY_AVAILABLE:
        return mask
    h, w = mask.shape
    k = max(5, min(h, w) // 70)
    if k % 2 == 0:
        k += 1
    closed = _ndimage.binary_closing(mask, structure=np.ones((k, k), dtype=int), iterations=2)
    closed = _ndimage.binary_fill_holes(closed)
    labeled, count = _ndimage.label(closed)
    if count > 1:
        sizes = _ndimage.sum(closed, labeled, index=range(1, count + 1))
        closed = labeled == (int(np.argmax(sizes)) + 1)
    # Grow out to the rough outer edge so fingertips / palm pockets fill.
    grow = max(3, min(h, w) // 90)
    closed = _ndimage.binary_dilation(closed, iterations=grow)
    closed = _ndimage.binary_fill_holes(closed)
    # Soft boundary — avoids razor-cut mid-letter clips from pen wobble.
    sigma = max(1.8, min(h, w) / 160.0)
    soft = _ndimage.gaussian_filter(closed.astype(np.float32), sigma=sigma)
    return soft >= 0.42


def _line_drawing_interior_mask(rgb: Image.Image):
    """Fill the interior of a canvas hand outline (works even if wrist is open)."""
    if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE:
        return None
    arr = np.array(to_rgb(rgb))
    lum = arr.astype(np.int16).mean(axis=2)
    ink = lum <= 150
    if not ink.any():
        return None
    closed = _ndimage.binary_closing(ink, structure=np.ones((5, 5), dtype=int), iterations=5)
    closed = _ndimage.binary_dilation(closed, iterations=3)
    closed = _seal_open_wrist(closed)
    filled = _ndimage.binary_fill_holes(closed)
    # Prefer the filled interior; fall back to edge-flood if holes failed.
    if _mask_is_usable(filled) and float(filled.mean()) >= float(ink.mean()) * 2.5:
        return _smooth_hand_fill_mask(filled)
    paper = lum >= 200
    outer = _flood_from_edges(paper | ~closed)
    interior = ~outer
    if _mask_is_usable(interior):
        return _smooth_hand_fill_mask(interior)
    if _mask_is_usable(filled):
        return _smooth_hand_fill_mask(filled)
    return None


def _ink_or_paper_mask(rgb: Image.Image):
    """Filled interior for a photo hand or a pencil tracing on paper."""
    if not NUMPY_AVAILABLE:
        return None
    if _is_sparse_line_drawing(rgb):
        line_mask = _line_drawing_interior_mask(rgb)
        if line_mask is not None:
            return line_mask
    arr = np.array(rgb)
    lum = arr.astype(np.int16).mean(axis=2)
    sat = np.maximum.reduce([arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]) - np.minimum.reduce(
        [arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]
    )
    ink = (lum <= 142) | ((sat > 38) & (lum < 210))
    if SCIPY_AVAILABLE:
        closed = _ndimage.binary_closing(ink, iterations=2)
        filled = _ndimage.binary_fill_holes(closed)
        if _mask_is_usable(filled):
            return filled
    paper = (lum >= 188) & (sat <= 48)
    outer = _flood_from_edges(paper | (lum >= 222))
    interior = ~outer
    if _mask_is_usable(interior):
        return interior
    return None


def _render_hand_stencil(mask) -> Image.Image:
    """Dark fill used for post-clip only (not shown as a plate to paint)."""
    canvas = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    canvas[mask] = (18, 18, 18)
    if SCIPY_AVAILABLE:
        ring = max(2, min(mask.shape) // 220)
        dilated = _ndimage.binary_dilation(mask, iterations=ring)
        canvas[dilated & ~mask] = (255, 32, 96)
    return Image.fromarray(canvas, "RGB")


def _render_hand_zone_guide(
    photo: Image.Image,
    mask,
    *,
    show_strokes: bool = True,
) -> Image.Image:
    """
    Pale filled silhouette + magenta rim.
    For rough canvas draws, omit jagged stroke overlay so the model packs into
    the solid fill zone (including edge pockets) instead of hugging pen wiggles.
    """
    arr = np.array(to_rgb(photo))
    h, w = mask.shape
    guide = np.full((h, w, 3), 255, dtype=np.uint8)
    guide[mask] = (255, 232, 240)  # pale pink = word zone only
    if SCIPY_AVAILABLE:
        ring = max(3, min(h, w) // 140)
        dilated = _ndimage.binary_dilation(mask, iterations=ring)
        eroded = _ndimage.binary_erosion(mask, iterations=max(1, ring // 2))
        edge = dilated & ~eroded
        guide[edge] = (255, 32, 96)
    if show_strokes:
        lum = arr.astype(np.int16).mean(axis=2)
        strokes = lum <= 150
        guide[strokes] = (25, 25, 25)
    return Image.fromarray(guide, "RGB")


def _overlay_hand_contour(photo: Image.Image, mask) -> Image.Image:
    arr = np.array(to_rgb(photo))
    if SCIPY_AVAILABLE:
        ring = max(3, min(mask.shape) // 180)
        dilated = _ndimage.binary_dilation(mask, iterations=ring)
        eroded = _ndimage.binary_erosion(mask, iterations=max(1, ring // 2))
        edge = dilated & ~eroded
    else:
        edge = mask
    arr[edge] = (255, 32, 96)
    return Image.fromarray(arr, "RGB")


def build_hand_alignment_images(hand_path: str) -> Tuple[list[RoleImage], Optional[Image.Image]]:
    """
    Canvas outline or photo + zone guide so word art follows the upload only.
    Returns (role_images, clip_stencil). Clip stencil is dark; the model sees a
    pale zone guide so it does not paint a solid black hand plate.
    """
    photo = load_rgb(hand_path)
    roles: list[RoleImage] = []
    mask = None
    line_drawing = _is_sparse_line_drawing(photo)
    if NUMPY_AVAILABLE:
        # Canvas sketches: skip rembg (it invents a filled blob).
        if line_drawing:
            mask = _line_drawing_interior_mask(photo)
        else:
            mask = _rembg_hand_mask(photo)
            if mask is not None and SCIPY_AVAILABLE:
                mask = _smooth_hand_fill_mask(mask)
        if mask is None:
            mask = _ink_or_paper_mask(photo)
            if mask is not None and SCIPY_AVAILABLE and not line_drawing:
                mask = _smooth_hand_fill_mask(mask)

    if mask is not None and _mask_is_usable(mask):
        clip_stencil = _render_hand_stencil(mask)
        if line_drawing:
            # Clean filled zone (no jagged stroke overlay) — pack words to the rim.
            zone = _render_hand_zone_guide(photo, mask, show_strokes=False)
            roles.append((
                "USER ROUGH HAND DRAWING. Magenta rim = silhouette to FILL completely. "
                "Pack glossy sticker WORDS densely into every finger tip, gap, and palm "
                "pocket of THIS pose. Scale/rotate whole words to fit — do NOT slice "
                "letters mid-glyph. Keep the same crop/position. No solid black plate. "
                "Do NOT invent a different hand pose.",
                _overlay_hand_contour(photo, mask),
            ))
            roles.append((
                "WORD FILL ZONE (smoothed from the rough drawing). Pale pink = fill this "
                "ENTIRE region with packed sticker words out to the magenta edge — "
                "including rough edge pockets and fingertips. Tiny stars/dots only in "
                "leftover cracks. CRITICAL: words only (no solid plate); outside = transparent.",
                zone,
            ))
        else:
            zone = _render_hand_zone_guide(photo, mask, show_strokes=True)
            roles.append((
                "USER HAND PHOTO with a magenta contour on the real outline. "
                "The word-art hand MUST match this pose exactly (thumb side, finger "
                "count, lengths, gaps, rotation, left vs right). Pack letters densely "
                "INSIDE the magenta outline — complete words, not sliced glyphs. "
                "Keep the same crop/position. No solid black hand plate.",
                _overlay_hand_contour(photo, mask),
            ))
            roles.append((
                "WORD ZONE GUIDE. Pale pink = pack sticker words here out to the edge. "
                "Magenta = rim. Do NOT fill with a solid color plate — letters only.",
                zone,
            ))
        return roles, clip_stencil

    guide = ImageOps.autocontrast(photo.convert("L"))
    guide = guide.point(lambda px: 0 if px < 150 else 255)
    roles.append((
        "USER HAND OUTLINE / PHOTO. Trace THIS exact outline as the word-art silhouette. "
        "Same thumb side, finger lengths, gaps, and rotation. Pack complete words "
        "inside the hand out to the edges. Keep the same crop. No solid black plate. "
        "Do not replace it with a generic open palm.",
        photo,
    ))
    roles.append((
        "HIGH-CONTRAST GUIDE of the same upload. Follow this outline. "
        "Words only inside the hand shape; transparent outside.",
        Image.merge("RGB", (guide, guide, guide)),
    ))
    return roles, None


def clip_image_to_stencil(img: Image.Image, stencil: Image.Image) -> Image.Image:
    """
    Fade pixels outside the stencil. Uses a soft edge so rough canvas outlines
    do not razor-cut mid-letter; only clearly-outside ink is cleared.
    """
    rgba = img.convert("RGBA")
    if not NUMPY_AVAILABLE:
        return rgba
    guide = to_rgb(stencil).resize(rgba.size, Image.Resampling.BILINEAR)
    arr = np.array(rgba)
    dark = np.array(guide).astype(np.float32).mean(axis=2)
    # Soft keep weight: 1 inside hand, 0 far outside (feathered rim).
    if SCIPY_AVAILABLE:
        hard = dark < 150
        # Generous keep so rough-edge packing isn't chopped.
        hard = _ndimage.binary_dilation(hard, iterations=max(4, min(hard.shape) // 100))
        dist_out = _ndimage.distance_transform_edt(~hard)
        feather = max(6.0, min(hard.shape) / 70.0)
        weight = np.ones(hard.shape, dtype=np.float32)
        # Inside stays fully opaque; only fade ink that spilled past the rim.
        weight[~hard] = np.clip(1.0 - (dist_out[~hard] / feather), 0.0, 1.0)
        keep = hard
    else:
        keep = dark < 150
        weight = keep.astype(np.float32)

    rgb = arr[:, :, :3].astype(np.int16)
    lum = rgb.mean(axis=2)
    sat = np.maximum.reduce([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]) - np.minimum.reduce(
        [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
    )
    ink = (arr[:, :, 3] > 16) & (sat > 28) & (lum > 24) & (lum < 250)
    ink_count = int(ink.sum())
    if ink_count < 80:
        ink = (arr[:, :, 3] > 16) & ((sat > 25) | ((lum < 210) & (lum > 18)))
        ink_count = int(ink.sum())
    if ink_count < 80:
        return rgba
    overlap = int((ink & keep).sum()) / ink_count
    if overlap < 0.22:
        return rgba
    alpha = arr[:, :, 3].astype(np.float32) * weight
    arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


DEFAULT_STYLE_INSTRUCTION = (
    "STYLE TARGET (composition and aesthetic reference only). "
    "Match layout, color palette, typography treatment, decorative frame, paper "
    "texture, and overall art style. Do NOT copy any names, dates, locations, "
    "faces, map streets, or other personal details from this reference. "
    "Personalize using the user inputs and uploaded images instead."
)


def _is_page_or_checker_pixel(r: int, g: int, b: int, a: int = 255) -> bool:
    if a < 16:
        return True
    lum = (r + g + b) / 3.0
    sat = max(r, g, b) - min(r, g, b)
    if lum >= 198 and sat <= 85:
        return True
    if r >= 238 and g >= 230 and b >= 210 and sat <= 55:
        return True
    if sat <= 14 and 38 <= lum <= 150:
        return True
    return False


def _flood_from_edges(walkable) -> np.ndarray:
    """4-connected flood fill of True walkable pixels starting from the image border."""
    if SCIPY_AVAILABLE:
        seed = np.zeros_like(walkable)
        seed[0, :] = True
        seed[-1, :] = True
        seed[:, 0] = True
        seed[:, -1] = True
        seed &= walkable
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        return _ndimage.binary_propagation(seed, mask=walkable, structure=structure)

    h, w = walkable.shape
    visited = np.zeros((h, w), dtype=bool)
    queue: deque = deque()

    def try_add(y: int, x: int) -> None:
        if not visited[y, x] and walkable[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    for x in range(w):
        try_add(0, x)
        try_add(h - 1, x)
    for y in range(h):
        try_add(y, 0)
        try_add(y, w - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            try_add(y - 1, x)
        if y + 1 < h:
            try_add(y + 1, x)
        if x > 0:
            try_add(y, x - 1)
        if x + 1 < w:
            try_add(y, x + 1)
    return visited


_WORD_COLOR_PALETTE = np.array(
    [
        (232, 64, 128),
        (255, 140, 40),
        (32, 186, 196),
        (132, 78, 214),
        (76, 186, 72),
        (36, 110, 220),
        (236, 72, 72),
        (240, 186, 36),
    ],
    dtype=np.uint8,
) if NUMPY_AVAILABLE else None


def _drop_large_dark_plate(arr, lum, sat, opaque):
    """Remove a filled hand plate; keep small black doodles and colorful words."""
    colorful = opaque & (sat > 30)
    light_ink = opaque & (lum >= 180)
    if (colorful | light_ink).mean() < 0.01:
        return arr, opaque
    dark = opaque & (lum <= 32) & (sat <= 28)
    h, w = opaque.shape
    min_plate = max(2500, int(w * h * 0.02))
    if SCIPY_AVAILABLE:
        labeled, count = _ndimage.label(dark)
        for index in range(1, count + 1):
            region = labeled == index
            if int(region.sum()) >= min_plate:
                arr[region, 3] = 0
    else:
        arr[dark, 3] = 0
    return arr, arr[:, :, 3] > 0


def _knockout_cream_fill(arr):
    """Clear beige/cream hand FILL so only colored letter ink remains."""
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    a = arr[:, :, 3]
    lum = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    cream = ((lum >= 178) & (sat <= 58)) | (
        (r >= 220) & (g >= 210) & (b >= 190) & (sat <= 48)
    )
    arr[cream & (a > 0), 3] = 0
    return arr


def _drop_uniform_hand_plate(arr):
    """
    If the subject is mostly one flat color (solid embossed hand), clear the
    flat fill and keep only letter-like relief / edges / saturated ink.
    """
    if not SCIPY_AVAILABLE:
        return arr, arr[:, :, 3] > 0
    opaque = arr[:, :, 3] > 0
    if opaque.sum() < 200:
        return arr, opaque
    rgb = arr[:, :, :3].astype(np.float32)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    sat = np.maximum(np.maximum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2]) - np.minimum(
        np.minimum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2]
    )
    med = np.median(rgb[opaque], axis=0)
    dist = np.abs(rgb - med).sum(axis=2)
    near = opaque & (dist <= 60)
    if float(near.sum()) / float(opaque.sum()) < 0.52:
        return arr, opaque  # already a multi-color word cloud

    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)
    local = _ndimage.uniform_filter(lum, size=9)
    relief = np.abs(lum - local)
    letterish = opaque & ((grad >= 3.0) | (relief >= 5.5) | (sat > 48))
    letterish = _ndimage.binary_dilation(letterish, iterations=2)
    clear = near & ~letterish
    kept = opaque & ~clear
    if kept.sum() >= max(120, int(opaque.sum() * 0.04)):
        arr[clear, 3] = 0
    return arr, arr[:, :, 3] > 0


def _keep_bubble_letters_only(arr):
    """
    Words-only cutout for bubble sticker hands: keep saturated letter ink +
    tiny star/dot fillers; drop solid black plates, cream underlays, and
    pink/blue speckled noise backgrounds.
    """
    if not SCIPY_AVAILABLE:
        return arr
    opaque = arr[:, :, 3] > 0
    if opaque.sum() < 80:
        return arr
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    lum = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    # Solid letter bodies + glossy highlights sitting on letters.
    letter_core = opaque & (sat >= 36) & (lum >= 28) & (lum <= 245)
    soft_highlight = opaque & (lum >= 200) & (sat <= 55)
    if SCIPY_AVAILABLE:
        near_core = _ndimage.binary_dilation(letter_core, iterations=2)
        soft_highlight = soft_highlight & near_core

    keep = letter_core | soft_highlight
    if not keep.any():
        return arr

    labeled, count = _ndimage.label(keep)
    sizes = _ndimage.sum(keep, labeled, index=range(1, count + 1)) if count else []
    # Drop isolated speckles smaller than a tiny star; keep letter blobs + dots.
    min_keep = 6
    max_noise = 28
    cleaned = np.zeros_like(keep)
    for index, size in enumerate(sizes, 1):
        size = int(size)
        if size < min_keep:
            continue
        region = labeled == index
        # Tiny saturated dots/stars are OK; tiny low-sat noise is not.
        if size <= max_noise:
            if float(sat[region].mean()) < 40:
                continue
        cleaned[region] = True

    if cleaned.sum() < max(60, int(opaque.sum() * 0.02)):
        return arr

    # Expand so soft letter edges / thin outlines survive, then clear the rest.
    keep = _ndimage.binary_dilation(cleaned, iterations=2)
    arr[~keep, 3] = 0
    return arr


def _knockout_fill_pixels(arr):
    """
    Make paper-like pixels transparent everywhere (not just the page edge).

    Enclosed cream/white inside D, B, O, A, P, R is otherwise kept, then
    recoloring paints it shut. Also opens gaps between words.
    """
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    a = arr[:, :, 3]
    lum = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    paper = ((lum >= 198) & (sat <= 85)) | (
        (r >= 238) & (g >= 230) & (b >= 210) & (sat <= 55)
    )
    arr[paper & (a > 0), 3] = 0
    return arr


def _punch_enclosed_counters(arr):
    """Punch enclosed interiors of D/B/O/A/P/R so letter counters stay hollow."""
    if not SCIPY_AVAILABLE:
        return arr
    opaque = arr[:, :, 3] > 0
    if not opaque.any():
        return arr
    rgb = arr[:, :, :3].astype(np.int16)
    lum = (rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2]) / 3.0
    sat = np.maximum.reduce([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]) - np.minimum.reduce(
        [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
    )
    # Strokes only — dark/cream fills inside letters must not join the blob.
    ink = opaque & ((sat > 28) | ((lum > 45) & (lum < 200)))
    if not ink.any():
        ink = opaque
    labeled, count = _ndimage.label(
        ink,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    )
    for index in range(1, count + 1):
        component = labeled == index
        if int(component.sum()) < 20:
            continue
        filled = _ndimage.binary_fill_holes(component)
        holes = filled & ~component
        if holes.any():
            arr[holes, 3] = 0
    return arr


def _filter_visible_palette(color_palette):
    """Drop black/near-black swatches so lettering never disappears."""
    if color_palette is None:
        return _WORD_COLOR_PALETTE
    visible = []
    for color in color_palette:
        r, g, b = (int(color[0]), int(color[1]), int(color[2]))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        if lum < 72 or max(r, g, b) < 58:
            continue
        visible.append((r, g, b))
    if visible:
        return np.asarray(visible, dtype=np.uint8)
    if _WORD_COLOR_PALETTE is not None:
        return _WORD_COLOR_PALETTE
    return np.asarray([(232, 64, 128), (255, 140, 40), (32, 186, 196)], dtype=np.uint8)


def _recolor_dark_ink_blobs(arr, palette):
    """Force any leftover black/grey words onto the visible user palette."""
    if not SCIPY_AVAILABLE or palette is None or len(palette) == 0:
        return arr
    opaque = arr[:, :, 3] > 0
    if not opaque.any():
        return arr
    rgb = arr[:, :, :3].astype(np.int16)
    lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    sat = np.maximum(np.maximum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2]) - np.minimum(
        np.minimum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2]
    )
    dark = opaque & (lum <= 85) & (sat <= 55)
    if not dark.any():
        return arr
    labeled, count = _ndimage.label(
        dark,
        structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
    )
    n_colors = len(palette)
    usage = [0] * n_colors
    for index in range(1, count + 1):
        region = labeled == index
        if int(region.sum()) < 8:
            continue
        color_idx = min(range(n_colors), key=lambda i: (usage[i], i))
        usage[color_idx] += 1
        base = palette[color_idx]
        arr[region, 0] = base[0]
        arr[region, 1] = base[1]
        arr[region, 2] = base[2]
    return arr


def _colorize_word_blobs(arr, color_palette=None, preserve_shading: bool = False):
    """
    Assign each word-like blob one palette color.

    preserve_shading=True joins letter gaps first so a whole WORD gets one color.
    Colors are balanced across the full user palette (not stuck on 1–2 inks).
    Black/near-black swatches are filtered out so words stay visible.
    """
    if not SCIPY_AVAILABLE:
        return arr
    palette = _filter_visible_palette(color_palette)
    if palette is None or len(palette) == 0:
        return arr
    opaque = arr[:, :, 3] > 0
    if not opaque.any():
        return arr
    # Close gaps between letters so one WORD gets one solid color (not per-letter rainbow).
    # Keep closing mild so neighboring words stay separate and can take different inks.
    if preserve_shading:
        merged = _ndimage.binary_closing(opaque, structure=np.ones((3, 3), dtype=int), iterations=2)
        merged = _ndimage.binary_dilation(merged, iterations=1)
    else:
        merged = opaque
    labeled, count = _ndimage.label(
        merged,
        structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
    )
    if count < 1:
        return arr

    # Collect eligible word blobs (largest first so hero words get distinct inks).
    blobs = []
    for index in range(1, count + 1):
        region = (labeled == index) & opaque
        n = int(region.sum())
        if n < 8:
            continue
        rgb = arr[region, :3].astype(np.float32)
        lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        sat = np.maximum(np.maximum(rgb[:, 0], rgb[:, 1]), rgb[:, 2]) - np.minimum(
            np.minimum(rgb[:, 0], rgb[:, 1]), rgb[:, 2]
        )
        # Keep soft white/cream hand rim — do not force a palette hue onto it.
        if preserve_shading and float(np.median(sat)) < 28 and float(np.median(lum)) > 200:
            continue
        # Skip giant plate blobs so we never paint the whole hand one color.
        if n > max(8000, int(opaque.sum() * 0.28)):
            continue
        cy, cx = _ndimage.center_of_mass(region)
        blobs.append((n, float(cy), float(cx), region))

    if not blobs:
        return _recolor_dark_ink_blobs(arr, palette)

    blobs.sort(key=lambda item: (-item[0], item[1], item[2]))
    n_colors = len(palette)
    usage = [0] * n_colors
    last_idx = -1

    for _, _, _, region in blobs:
        # Prefer the least-used swatch; avoid repeating the previous word's ink.
        ranked = sorted(range(n_colors), key=lambda i: (usage[i], i))
        color_idx = ranked[0]
        if color_idx == last_idx and n_colors > 1:
            color_idx = ranked[1]
        usage[color_idx] += 1
        last_idx = color_idx
        base = palette[color_idx]
        # Flat solid word color from the user palette (one color for the whole word).
        arr[region, 0] = base[0]
        arr[region, 1] = base[1]
        arr[region, 2] = base[2]

    # Second pass: any still-black/grey words get forced onto the palette.
    return _recolor_dark_ink_blobs(arr, palette)


def isolate_word_hand_cutout(
    img: Image.Image,
    crop: bool = True,
    pad: int = 8,
    randomize_colors: bool = True,
    color_palette=None,
    letter_style: str = "marker",
) -> Image.Image:
    """
    Transparent word-art hand: drop page/checkerboard/black and any filled
    silhouette behind the letters. Keep (or assign) per-word colors.

    letter_style:
      - "marker": hollow counters, knockout fills (legacy tracing-hand)
      - "bubble": solid glossy letters; keep fills/highlights; strip black/cream bg
    """
    bubble = str(letter_style or "marker").lower() == "bubble"
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if w < 2 or h < 2:
        return rgba

    if NUMPY_AVAILABLE:
        arr = np.array(rgba)
        r = arr[:, :, 0].astype(np.int16)
        g = arr[:, :, 1].astype(np.int16)
        b = arr[:, :, 2].astype(np.int16)
        a = arr[:, :, 3]
        lum = (r + g + b) / 3.0
        sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        paper = ((lum >= 198) & (sat <= 85)) | (
            (r >= 238) & (g >= 230) & (b >= 210) & (sat <= 55)
        )
        checker = (sat <= 14) & (lum >= 38) & (lum <= 150)
        near_black = (lum <= 28) & (sat <= 45)
        # Speckle / static backdrops (pink-blue noise) are walkable if not letter-like.
        # High local color churn + small structure ≈ noise, not bubble sticker ink.
        if SCIPY_AVAILABLE and bubble:
            local_sat = _ndimage.uniform_filter(sat.astype(np.float32), size=5)
            local_lum = _ndimage.uniform_filter(lum.astype(np.float32), size=5)
            sat_var = _ndimage.uniform_filter((sat.astype(np.float32) - local_sat) ** 2, size=5)
            lum_var = _ndimage.uniform_filter((lum.astype(np.float32) - local_lum) ** 2, size=5)
            speckled = (sat_var > 180) & (lum_var > 120) & (sat < 90)
        else:
            speckled = np.zeros_like(paper, dtype=bool)

        walkable = paper | checker | near_black | speckled | (a < 16)
        outer = _flood_from_edges(walkable)
        if (1.0 - outer.mean()) >= 0.03:
            arr[outer, 3] = 0
            a = arr[:, :, 3]
            r = arr[:, :, 0].astype(np.int16)
            g = arr[:, :, 1].astype(np.int16)
            b = arr[:, :, 2].astype(np.int16)
            lum = (r + g + b) / 3.0
            sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

        opaque = a > 0
        arr, opaque = _drop_large_dark_plate(arr, lum, sat, opaque)
        if bubble:
            # Words-only: strip cream/beige fill + solid plates + leftover noise.
            arr = _knockout_cream_fill(arr)
            arr, opaque = _drop_uniform_hand_plate(arr)
            arr = _keep_bubble_letters_only(arr)
        else:
            arr = _knockout_fill_pixels(arr)
            arr = _punch_enclosed_counters(arr)
        opaque = arr[:, :, 3] > 0
        if opaque.mean() < 0.02:
            return rgba
        if randomize_colors:
            arr = _colorize_word_blobs(
                arr,
                color_palette=color_palette,
                preserve_shading=bubble,
            )
        out = Image.fromarray(arr, "RGBA")
    else:
        pixels = rgba.load()
        walkable = [
            [_is_page_or_checker_pixel(*pixels[x, y]) for x in range(w)]
            for y in range(h)
        ]
        visited = [[False] * w for _ in range(h)]
        queue: deque = deque()
        for x in range(w):
            for y in (0, h - 1):
                if walkable[y][x] and not visited[y][x]:
                    visited[y][x] = True
                    queue.append((x, y))
        for y in range(h):
            for x in (0, w - 1):
                if walkable[y][x] and not visited[y][x]:
                    visited[y][x] = True
                    queue.append((x, y))
        while queue:
            x, y = queue.popleft()
            pixels[x, y] = (0, 0, 0, 0)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h and walkable[ny][nx] and not visited[ny][nx]:
                    visited[ny][nx] = True
                    queue.append((nx, ny))
        out = rgba

    if not crop:
        return out
    bbox = out.getbbox()
    if not bbox:
        return out
    left, top, right, bottom = bbox
    return out.crop((
        max(0, left - pad),
        max(0, top - pad),
        min(w, right + pad),
        min(h, bottom + pad),
    ))


def isolate_paper_background(img: Image.Image, crop: bool = True, pad: int = 8) -> Image.Image:
    """Back-compat alias for tracing-hand cutouts."""
    return isolate_word_hand_cutout(img, crop=crop, pad=pad)


def knockout_cream_card_background(img: Image.Image) -> Image.Image:
    """
    Convert baked card / fake-transparency backgrounds into real PNG alpha.

    Clears edge-connected white, cream, gray, black, checkerboard, and similar
    solid/split backdrops while keeping the heart map, die-cut ridges, green
    underline, and typography.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if w < 2 or h < 2:
        return rgba

    if not NUMPY_AVAILABLE:
        return _knockout_cream_card_background_pil(rgba)

    arr = np.array(rgba)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    a = arr[:, :, 3].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    warm = ((r - b) >= 6) & ((g - b) >= 2)

    cream = (
        ((warm) & (lum >= 175) & (sat <= 75))
        | ((r >= 210) & (g >= 200) & (b >= 185) & (sat <= 60) & warm)
    )
    flat_white = (lum >= 240) & (sat <= 22)
    neutral_light = (sat <= 32) & (lum >= 150)
    neutral_mid = (sat <= 28) & (lum >= 70) & (lum < 150)
    neutral_dark = (sat <= 35) & (lum <= 55)

    checker = _line_art_checker_mask(lum, sat)
    if checker.any() and SCIPY_AVAILABLE:
        checker = _ndimage.binary_dilation(checker, iterations=2)
    elif checker.any():
        grown = checker.copy()
        grown[1:, :] |= checker[:-1, :]
        grown[:-1, :] |= checker[1:, :]
        grown[:, 1:] |= checker[:, :-1]
        grown[:, :-1] |= checker[:, 1:]
        checker = grown
    arr[checker, 3] = 0

    stripe = np.zeros((h, w), dtype=bool)
    edge_w = max(2, w // 40)
    for x0, x1 in ((0, edge_w), (w - edge_w, w)):
        band_lum = lum[:, x0:x1]
        band_sat = sat[:, x0:x1]
        near_bw = (band_sat <= 30) & ((band_lum <= 40) | (band_lum >= 215))
        if float(near_bw.mean()) >= 0.55:
            stripe[:, x0:x1] = near_bw
    arr[stripe, 3] = 0

    chroma = (
        ((g > 200) & (r < 90) & (b < 90))
        | ((r > 200) & (g < 90) & (b > 200))
    )

    # Sample border colors so gray / black / split backdrops key out.
    border_rows = np.concatenate([
        np.zeros(w, dtype=int),
        np.full(w, h - 1, dtype=int),
        np.arange(h),
        np.arange(h),
    ])
    border_cols = np.concatenate([
        np.arange(w),
        np.arange(w),
        np.zeros(h, dtype=int),
        np.full(h, w - 1, dtype=int),
    ])
    step = max(1, len(border_rows) // 400)
    border_rows = border_rows[::step]
    border_cols = border_cols[::step]
    br = r[border_rows, border_cols]
    bg_ = g[border_rows, border_cols]
    bb = b[border_rows, border_cols]
    bl = lum[border_rows, border_cols]
    bs = sat[border_rows, border_cols]
    border_neutral = bs <= 40
    light_share = float(((bl >= 150) & border_neutral).mean()) if len(bl) else 0.0
    dark_share = float(((bl <= 60) & border_neutral).mean()) if len(bl) else 0.0
    mid_share = float(((bl > 60) & (bl < 150) & border_neutral).mean()) if len(bl) else 0.0

    similar_border = np.zeros((h, w), dtype=bool)
    seed_mask = border_neutral & (
        (bl >= 140) | (bl <= 70) | ((bl > 70) & (bl < 140) & (mid_share >= 0.15))
    )
    seed_r, seed_g, seed_b = br[seed_mask], bg_[seed_mask], bb[seed_mask]
    if seed_r.size:
        picks = np.linspace(0, seed_r.size - 1, num=min(12, seed_r.size), dtype=int)
        for i in picks:
            dist = np.abs(r - seed_r[i]) + np.abs(g - seed_g[i]) + np.abs(b - seed_b[i])
            similar_border |= (dist <= 48) & (sat <= 42)

    paper = cream | flat_white | neutral_light | chroma | similar_border
    if mid_share >= 0.12 or light_share >= 0.35:
        paper = paper | neutral_mid
    if dark_share >= 0.12:
        paper = paper | neutral_dark
    paper = paper & (arr[:, :, 3] > 0)

    if SCIPY_AVAILABLE:
        mean = _ndimage.uniform_filter(lum, size=5)
        mean_sq = _ndimage.uniform_filter(lum * lum, size=5)
        local_var = np.clip(mean_sq - mean * mean, 0.0, None)
        dil = _ndimage.maximum_filter(lum, size=3)
        ero = _ndimage.minimum_filter(lum, size=3)
        local_contrast = dil - ero
    else:
        local_var = np.zeros_like(lum)
        local_contrast = np.zeros_like(lum)

    green_accent = (g > r + 18) & (g > b + 18) & (g >= 90) & (sat >= 35)
    map_color = (sat > 30) & ~green_accent
    busy = (local_var > 48) | map_color | green_accent
    text_stroke = (local_contrast > 28) & (sat <= 55)
    yy = np.arange(h)[:, None]
    keep_text = (yy >= int(h * 0.50)) & text_stroke
    if SCIPY_AVAILABLE:
        busy = _ndimage.binary_dilation(busy, iterations=1)
        keep_text = _ndimage.binary_dilation(keep_text, iterations=1)

    map_like = busy | (sat > 22)
    if SCIPY_AVAILABLE:
        map_frac = _ndimage.uniform_filter(map_like.astype(np.float32), size=9)
        internal_ridge = (cream | flat_white | ((sat <= 30) & (lum >= 200))) & (
            (map_frac >= 0.16) & (map_frac <= 0.90)
        )
    else:
        internal_ridge = np.zeros((h, w), dtype=bool)

    walkable = (
        (paper & ~busy & ~internal_ridge & ~keep_text)
        | (arr[:, :, 3] < 16)
        | chroma
        | checker
    )
    outer = _flood_from_edges(walkable)

    if float(outer.mean()) < 0.03:
        walkable = (
            (cream | flat_white | neutral_light | similar_border)
            & ~busy
            & ~internal_ridge
            & ~keep_text
        ) | (arr[:, :, 3] < 16)
        if dark_share >= 0.12:
            walkable = walkable | (neutral_dark & ~busy & ~keep_text)
        if mid_share >= 0.12:
            walkable = walkable | (neutral_mid & ~busy & ~internal_ridge & ~keep_text)
        outer = _flood_from_edges(walkable)

    if float(outer.mean()) >= 0.01:
        arr[outer, 3] = 0
        if SCIPY_AVAILABLE:
            fringe_zone = _ndimage.binary_dilation(outer, iterations=1) & ~outer
            fringe = (
                fringe_zone
                & (sat <= 40)
                & ~busy
                & ~internal_ridge
                & ~keep_text
                & (arr[:, :, 3] > 0)
            )
            arr[fringe, 3] = 0

    soft = (
        (arr[:, :, 3] > 0)
        & (arr[:, :, 3] < 48)
        & (sat <= 40)
        & ~busy
        & ~internal_ridge
        & ~keep_text
    )
    arr[soft, 3] = 0
    return Image.fromarray(arr, "RGBA")


def knockout_sticky_collage_background(img: Image.Image) -> Image.Image:
    """
    True-alpha cutout for sticky-note hearts.

    Keeps yellow paper + red ink (+ warm contact shadows). Clears cream, white,
    black, gray, and speckled fringe so the PNG composites cleanly.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if w < 2 or h < 2:
        return rgba

    if not NUMPY_AVAILABLE:
        return knockout_cream_card_background(rgba)

    arr = np.array(rgba)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    # Post-it yellow (bright body + slightly darker adhesive band).
    yellow = (
        (r >= 155)
        & (g >= 125)
        & (b <= 185)
        & (g > b + 12)
        & (r >= g - 25)
        & (sat >= 22)
        & (lum >= 95)
    )

    # Red marker lettering / tiny hearts on the notes.
    red = (
        (r >= 105)
        & (r > g + 22)
        & (r > b + 22)
        & (g <= 170)
        & (b <= 160)
    )

    keep = yellow | red
    if SCIPY_AVAILABLE:
        yellow_near = _ndimage.binary_dilation(yellow, iterations=3)
        # Soft warm shadows sitting on / under notes (not page noise).
        warm_shadow = (
            yellow_near
            & (lum >= 35)
            & (lum <= 165)
            & (r >= g - 12)
            & (g >= b - 18)
            & (sat >= 8)
            & ~((lum <= 55) & (sat <= 25))  # avoid pure black page
        )
        keep = keep | warm_shadow
        keep = _ndimage.binary_closing(keep, iterations=2)
        keep = _ndimage.binary_dilation(keep, iterations=1)
        keep = _ndimage.binary_erosion(keep, iterations=1)

        labeled, count = _ndimage.label(keep)
        if count:
            sizes = np.bincount(labeled.ravel())
            if sizes.size > 1:
                # Drop dust / fringe islands; keep the heart mass.
                min_size = max(80, int(w * h * 0.00025))
                keep_labels = np.where(sizes >= min_size)[0]
                keep_labels = keep_labels[keep_labels > 0]
                if keep_labels.size:
                    keep = np.isin(labeled, keep_labels)
                # Always retain the largest component even if tiny threshold fails.
                largest = int(np.argmax(sizes[1:]) + 1) if sizes.size > 1 else 0
                if largest:
                    keep = keep | (labeled == largest)

    out = np.zeros_like(arr)
    out[keep, :3] = arr[keep, :3]
    out[keep, 3] = 255

    # Kill near-transparent / pale fringe that survived as "keep".
    if SCIPY_AVAILABLE:
        border = keep & ~_ndimage.binary_erosion(keep, iterations=1)
    else:
        border = keep
    pale_fringe = (
        border
        & (sat <= 35)
        & (lum >= 200)
        & ~yellow
        & ~red
    )
    out[pale_fringe, :] = 0

    # If almost nothing survived, fall back to cream knockout.
    if float((out[:, :, 3] > 0).mean()) < 0.02:
        return knockout_cream_card_background(rgba)

    return Image.fromarray(out, "RGBA")


def knockout_word_ink_background(img: Image.Image) -> Image.Image:
    """
    True-alpha cutout for monochrome red word-heart calligrams.

    Keeps red ink / hatching; clears cream and speckled page noise.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if w < 2 or h < 2:
        return rgba

    if not NUMPY_AVAILABLE:
        return knockout_cream_card_background(rgba)

    arr = np.array(rgba)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    red_ink = (
        (r >= 90)
        & (r > g + 18)
        & (r > b + 18)
        & (g <= 180)
        & (lum <= 210)
    )
    # Darker hatch / contour strokes that are still warm.
    warm_dark = (
        (lum <= 120)
        & (r >= g - 5)
        & (r >= b)
        & (sat >= 12)
        & (r >= 40)
    )
    keep = red_ink | warm_dark

    if SCIPY_AVAILABLE:
        keep = _ndimage.binary_closing(keep, iterations=1)
        labeled, count = _ndimage.label(keep)
        if count:
            sizes = np.bincount(labeled.ravel())
            if sizes.size > 1:
                min_size = max(40, int(w * h * 0.00015))
                keep_labels = np.where(sizes >= min_size)[0]
                keep_labels = keep_labels[keep_labels > 0]
                if keep_labels.size:
                    keep = np.isin(labeled, keep_labels)

    out = np.zeros_like(arr)
    out[keep, :3] = arr[keep, :3]
    out[keep, 3] = 255

    if float((out[:, :, 3] > 0).mean()) < 0.01:
        return knockout_cream_card_background(rgba)
    return Image.fromarray(out, "RGBA")


def _knockout_cream_card_background_pil(rgba: Image.Image) -> Image.Image:
    """PIL fallback when numpy is unavailable."""
    w, h = rgba.size
    pixels = rgba.load()
    walkable = [[False] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            pr, pg, pb, pa = pixels[x, y]
            if pa < 16:
                walkable[y][x] = True
                continue
            lum = (pr + pg + pb) / 3.0
            sat = max(pr, pg, pb) - min(pr, pg, pb)
            warm = (pr - pb) >= 6
            if ((warm and lum >= 175 and sat <= 70)
                    or (lum >= 240 and sat <= 22)
                    or (sat <= 32 and lum >= 70)  # gray / mid card
                    or (sat <= 35 and lum <= 55)  # black backdrop
                    or (pg > 200 and pr < 90 and pb < 90)
                    or (pr > 200 and pg < 90 and pb > 200)):
                walkable[y][x] = True
    visited = [[False] * w for _ in range(h)]
    queue: deque = deque()
    for x in range(w):
        for y in (0, h - 1):
            if walkable[y][x] and not visited[y][x]:
                visited[y][x] = True
                queue.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if walkable[y][x] and not visited[y][x]:
                visited[y][x] = True
                queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        pixels[x, y] = (0, 0, 0, 0)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and walkable[ny][nx] and not visited[ny][nx]:
                visited[ny][nx] = True
                queue.append((nx, ny))
    return rgba


def _is_line_art_paper_pixel(r: int, g: int, b: int, a: int = 255) -> bool:
    """Beige/cream/off-white/white page — not gray or black ink."""
    if a < 16:
        return True
    lum = (r + g + b) / 3.0
    sat = max(r, g, b) - min(r, g, b)
    warm = (r - b >= 8) and (g - b >= 3)
    if warm and lum >= 175 and sat <= 90:
        return True
    if lum >= 242 and sat <= 22:
        return True
    return False


def _line_art_checker_mask(lum, sat):
    """Baked transparency-preview checker (fine B/W dither or gray squares)."""
    h, w = lum.shape
    empty = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return empty

    p00 = lum[:-1, :-1]
    p01 = lum[:-1, 1:]
    p10 = lum[1:, :-1]
    p11 = lum[1:, 1:]
    contrast = 70
    diag_a = (p00 > p01 + contrast) & (p00 > p10 + contrast) & (p11 > p01 + contrast) & (p11 > p10 + contrast)
    diag_b = (p01 > p00 + contrast) & (p10 > p00 + contrast) & (p01 > p11 + contrast) & (p10 > p11 + contrast)
    fine = np.zeros((h, w), dtype=bool)
    pair = diag_a | diag_b
    fine[:-1, :-1] |= pair
    fine[:-1, 1:] |= pair
    fine[1:, :-1] |= pair
    fine[1:, 1:] |= pair

    period = np.zeros((h, w), dtype=bool)
    for step in (1, 2, 4, 8):
        if w <= 2 * step or h <= 2 * step:
            continue
        h_same = np.abs(lum[:, 2 * step:] - lum[:, :-2 * step]) < 32
        h_flip = np.abs(lum[:, step:-step] - lum[:, :-2 * step]) > 70
        v_same = np.abs(lum[2 * step:, :] - lum[:-2 * step, :]) < 32
        v_flip = np.abs(lum[step:-step, :] - lum[:-2 * step, :]) > 70
        hchk = np.zeros((h, w), dtype=bool)
        vchk = np.zeros((h, w), dtype=bool)
        hchk[:, step:-step] = h_same & h_flip
        vchk[step:-step, :] = v_same & v_flip
        period |= hchk & vchk

    detected = (fine | period) & (sat <= 28)
    if float(detected.mean()) < 0.06:
        return empty
    return detected


def isolate_line_art_cutout(img: Image.Image) -> Image.Image:
    """
    Turn a white/beige/checkerboard line drawing into black ink with real alpha.

    Darkness becomes opacity so anti-aliased strokes and the becoming fade
    composite cleanly. Does not crop; the 3:4 canvas stays intact.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    if w < 2 or h < 2:
        return rgba

    if NUMPY_AVAILABLE:
        arr = np.array(rgba)
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)
        a = arr[:, :, 3].astype(np.float32)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        warm = ((r - b) >= 8) & ((g - b) >= 3)
        paper = (
            ((warm) & (lum >= 175) & (sat <= 90))
            | ((lum >= 242) & (sat <= 22))
            | (a < 16)
            | _line_art_checker_mask(lum, sat)
        )
        lum = np.where(paper, 255.0, lum)

        border = np.concatenate([
            lum[0, :], lum[-1, :], lum[:, 0], lum[:, -1],
            lum[min(4, h - 1), :], lum[max(0, h - 5), :],
        ])
        paper_ref = float(np.percentile(border, 92))
        paper_ref = max(paper_ref, 240.0)
        darkness = np.clip(paper_ref - lum, 0.0, 255.0)
        darkness = np.where(darkness < 10.0, 0.0, darkness)
        alpha = np.clip(darkness * (255.0 / max(paper_ref - 18.0, 1.0)), 0.0, 255.0)

        out = np.zeros_like(arr)
        out[:, :, 3] = alpha.astype(np.uint8)
        if (out[:, :, 3] > 16).mean() < 0.005:
            return rgba
        return Image.fromarray(out, "RGBA")

    pixels = rgba.load()
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if _is_line_art_paper_pixel(r, g, b, a):
                pixels[x, y] = (0, 0, 0, 0)
                continue
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            alpha = int(max(0.0, min(255.0, (245.0 - lum) * (255.0 / 227.0))))
            pixels[x, y] = (0, 0, 0, alpha if alpha >= 10 else 0)
    return rgba


def obscure_style_letters(img: Image.Image, radius: int | None = None) -> Image.Image:
    """Keep silhouette and color masses; destroy readable reference vocabulary."""
    rgb = to_rgb(img)
    width, height = rgb.size
    if radius is None:
        radius = max(10, min(width, height) // 50)
    return rgb.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def generate_composed_image(
    output_path: str,
    prompt: str,
    role_images: Optional[Sequence[RoleImage]] = None,
    style_target: Optional[str] = None,
    style_instruction: Optional[str] = None,
    aspect_ratio: str = "1:1",
    temperature: float = 0.8,
    operation: str = "art_generation:creative",
    isolate_subject: bool = False,
    isolate_line_art: bool = False,
    obscure_style_text: bool = False,
    obscure_style_radius: Optional[int] = None,
    trailing_instruction: Optional[str] = None,
    clip_to_stencil: Optional[Image.Image] = None,
    image_size: Optional[str] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    word_color_palette: Optional[Sequence[tuple]] = None,
    letter_style: str = "marker",
) -> Tuple[bool, str]:
    """
    Send prompt + labeled images + optional style target to Gemini and save PNG.
    """
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        normalized = normalize_prompt_for_consistency(prompt)
        contents: list = [normalized]
        next_num = 1

        for label, image in role_images or []:
            contents.extend([
                f"IMAGE {next_num} = {label}",
                image,
            ])
            next_num += 1

        if style_target and os.path.isfile(style_target):
            style_image = Image.open(style_target)
            if isolate_subject:
                style_image = isolate_word_hand_cutout(
                    style_image,
                    randomize_colors=False,
                    letter_style=letter_style,
                )
            if isolate_line_art:
                style_image = isolate_line_art_cutout(style_image)
            if obscure_style_text:
                style_image = obscure_style_letters(style_image, radius=obscure_style_radius)
            else:
                style_image = to_rgb(style_image)
            contents.extend([
                f"IMAGE {next_num} = {style_instruction or DEFAULT_STYLE_INSTRUCTION}",
                style_image,
            ])

        if trailing_instruction:
            contents.append(trailing_instruction)

        client = get_gemini_client()
        # Default: new seed each request so "Generate again" is fresh.
        # Callers may pass seed= for deterministic / locked framing.
        if seed is None:
            seed = (generate_seed_from_prompt(normalized) ^ random.randint(1, 2**31 - 1)) % (2**31)
        else:
            seed = int(seed) % (2**31)
        response = _generate_content_image(
            client=client,
            model=model or get_gemini_image_model(),
            contents=contents,
            seed=seed,
            temperature=temperature,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            operation=operation,
        )
        img = _extract_final_image_from_response(response)
        if img is None:
            return False, "Gemini did not return an image"
        if clip_to_stencil is not None:
            img = clip_image_to_stencil(img, clip_to_stencil)
        if isolate_subject:
            img = isolate_word_hand_cutout(
                img,
                randomize_colors=True,
                color_palette=word_color_palette,
                letter_style=letter_style,
            )
        elif isolate_line_art:
            img = isolate_line_art_cutout(img)
        elif img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img.save(output_path, format="PNG", optimize=True)
        return True, "Artwork generated successfully"
    except Exception as exc:
        print(f"creative generate_composed_image error: {exc}")
        import traceback
        traceback.print_exc()
        return False, str(exc)
