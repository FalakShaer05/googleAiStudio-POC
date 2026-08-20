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
    if not NUMPY_AVAILABLE:
        return None
    try:
        from rembg import remove
    except Exception:
        return None
    try:
        cut = remove(rgb)
        if not isinstance(cut, Image.Image):
            from io import BytesIO
            cut = Image.open(BytesIO(cut))
        alpha = np.array(cut.convert("RGBA"))[:, :, 3]
        mask = alpha > 40
        return mask if _mask_is_usable(mask) else None
    except Exception as exc:
        print(f"tracing-hand rembg stencil skip: {exc}")
        return None


def _ink_or_paper_mask(rgb: Image.Image):
    """Filled interior for a photo hand or a pencil tracing on paper."""
    if not NUMPY_AVAILABLE:
        return None
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
    canvas = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    canvas[mask] = (18, 18, 18)
    if SCIPY_AVAILABLE:
        ring = max(2, min(mask.shape) // 220)
        dilated = _ndimage.binary_dilation(mask, iterations=ring)
        canvas[dilated & ~mask] = (255, 32, 96)
    return Image.fromarray(canvas, "RGB")


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
    """Photo + stencil so word art must follow the uploaded hand only."""
    photo = load_rgb(hand_path)
    roles: list[RoleImage] = []
    mask = None
    if NUMPY_AVAILABLE:
        mask = _rembg_hand_mask(photo)
        if mask is None:
            mask = _ink_or_paper_mask(photo)

    if mask is not None and _mask_is_usable(mask):
        stencil = _render_hand_stencil(mask)
        roles.append((
            "USER HAND PHOTO with a magenta contour on the real outline. "
            "The word-art hand MUST match this pose exactly (thumb side, finger "
            "count, lengths, gaps, rotation, left vs right). Put letters ONLY "
            "inside the magenta outline. Keep the same crop/position as this photo. "
            "Do not invent a generic spread-finger hand.",
            _overlay_hand_contour(photo, mask),
        ))
        roles.append((
            "HAND STENCIL from the same upload. Dark shape = the ONLY region "
            "that may contain words. White = empty/transparent. Fill this "
            "silhouette with lettering. No letters outside the dark shape. "
            "Do not copy any other hand's outline.",
            stencil,
        ))
        return roles, stencil

    guide = ImageOps.autocontrast(photo.convert("L"))
    guide = guide.point(lambda px: 0 if px < 150 else 255)
    roles.append((
        "USER HAND PHOTO. Trace THIS exact outline as the word-art silhouette. "
        "Same thumb side, finger lengths, gaps, and rotation. Letters stay "
        "inside the hand. Keep the same crop as this photo. "
        "Do not replace it with a generic open palm.",
        photo,
    ))
    roles.append((
        "HIGH-CONTRAST GUIDE of the same upload. Follow this outline. "
        "Words only on the hand shape.",
        Image.merge("RGB", (guide, guide, guide)),
    ))
    return roles, None


def clip_image_to_stencil(img: Image.Image, stencil: Image.Image) -> Image.Image:
    """Make pixels outside the dark stencil transparent, if the hand still overlaps."""
    rgba = img.convert("RGBA")
    if not NUMPY_AVAILABLE:
        return rgba
    guide = to_rgb(stencil).resize(rgba.size, Image.Resampling.BILINEAR)
    arr = np.array(rgba)
    keep = np.array(guide).astype(np.int16).mean(axis=2) < 150
    rgb = arr[:, :, :3].astype(np.int16)
    lum = rgb.mean(axis=2)
    sat = np.maximum.reduce([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]) - np.minimum.reduce(
        [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
    )
    ink = (arr[:, :, 3] > 16) & ((sat > 25) | ((lum < 210) & (lum > 18)))
    ink_count = int(ink.sum())
    if ink_count < 80:
        return rgba
    if int((ink & keep).sum()) / ink_count < 0.35:
        return rgba
    arr[~keep, 3] = 0
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


def _colorize_word_blobs(arr):
    """Assign each word-like blob a random palette color."""
    if not SCIPY_AVAILABLE or _WORD_COLOR_PALETTE is None:
        return arr
    opaque = arr[:, :, 3] > 0
    if not opaque.any():
        return arr
    merged = opaque
    labeled, count = _ndimage.label(merged, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))
    if count < 1:
        return arr
    rng = np.random.default_rng()
    palette = _WORD_COLOR_PALETTE
    last_idx = -1
    for index in range(1, count + 1):
        region = (labeled == index) & opaque
        if int(region.sum()) < 8:
            continue
        color_idx = int(rng.integers(0, len(palette)))
        if color_idx == last_idx:
            color_idx = (color_idx + 1) % len(palette)
        last_idx = color_idx
        arr[region, 0] = palette[color_idx][0]
        arr[region, 1] = palette[color_idx][1]
        arr[region, 2] = palette[color_idx][2]
    return arr


def isolate_word_hand_cutout(
    img: Image.Image,
    crop: bool = True,
    pad: int = 8,
    randomize_colors: bool = True,
) -> Image.Image:
    """
    Transparent word-art hand: drop page/checkerboard and any filled silhouette
    behind the letters. Keep (or randomly assign) per-word colors.
    """
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
        walkable = paper | checker | (a < 16)
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
        arr = _knockout_fill_pixels(arr)
        arr = _punch_enclosed_counters(arr)
        opaque = arr[:, :, 3] > 0
        if opaque.mean() < 0.02:
            return rgba
        if randomize_colors:
            arr = _colorize_word_blobs(arr)
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
    obscure_style_text: bool = False,
    obscure_style_radius: Optional[int] = None,
    trailing_instruction: Optional[str] = None,
    clip_to_stencil: Optional[Image.Image] = None,
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
                style_image = isolate_word_hand_cutout(style_image, randomize_colors=False)
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
        # New seed on every request so "Generate again" is a fresh Gemini call.
        seed = (generate_seed_from_prompt(normalized) ^ random.randint(1, 2**31 - 1)) % (2**31)
        response = _generate_content_image(
            client=client,
            model=get_gemini_image_model(),
            contents=contents,
            seed=seed,
            temperature=temperature,
            aspect_ratio=aspect_ratio,
            operation=operation,
        )
        img = _extract_final_image_from_response(response)
        if img is None:
            return False, "Gemini did not return an image"
        if clip_to_stencil is not None:
            img = clip_image_to_stencil(img, clip_to_stencil)
        if isolate_subject:
            img = isolate_word_hand_cutout(img, randomize_colors=True)
        elif img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img.save(output_path, format="PNG", optimize=True)
        return True, "Artwork generated successfully"
    except Exception as exc:
        print(f"creative generate_composed_image error: {exc}")
        import traceback
        traceback.print_exc()
        return False, str(exc)
