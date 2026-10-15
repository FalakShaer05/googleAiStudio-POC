"""Fast PIL mockups: print artwork onto stored product blanks."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

_SHEET_ASSET_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "static",
        "images",
        "merch",
        "sticker_sheet",
    )
)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

Box = Tuple[float, float, float, float]


def _to_rgba(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        return img
    if img.mode == "P":
        return img.convert("RGBA")
    return img.convert("RGBA")


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    return img.convert("RGB")


def flatten_on_white(img: Image.Image) -> Image.Image:
    """RGBA mockups must land on white — convert('RGB') turns holes black."""
    return _to_rgb(_to_rgba(img))


def trim_artwork(img: Image.Image, pad: int = 6) -> Image.Image:
    """Crop away empty / paper-white margins so the design isn't inset."""
    rgba = _to_rgba(img)
    if not NUMPY_AVAILABLE:
        bbox = rgba.getbbox()
        return rgba.crop(bbox) if bbox else rgba
    arr = np.asarray(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    paper = (rgb.min(axis=2) >= 248) & (chroma <= 14)
    content = (alpha > 12) & ~paper
    ys, xs = np.where(content)
    if xs.size == 0:
        bbox = rgba.getbbox()
        return rgba.crop(bbox) if bbox else rgba
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(arr.shape[1], int(xs.max()) + 1 + pad)
    y1 = min(arr.shape[0], int(ys.max()) + 1 + pad)
    return rgba.crop((x0, y0, x1, y1))


def pad_art_margin(
    img: Image.Image,
    frac: float = 0.06,
    *,
    rgba: Tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    """Re-add white breathing room after trim so logos aren't edge-flush."""
    art = _to_rgba(img)
    aw, ah = max(1, art.size[0]), max(1, art.size[1])
    pad = max(10, int(round(min(aw, ah) * max(0.0, frac))))
    canvas = Image.new("RGBA", (aw + pad * 2, ah + pad * 2), rgba)
    canvas.alpha_composite(art, dest=(pad, pad))
    return canvas


def art_for_orientation(
    artwork: Image.Image,
    orientation: Optional[str],
    *,
    margin_frac: float = 0.06,
) -> Image.Image:
    """
    Return the upload as-is.

    White margins are part of the composition (landscape framing, logo inset).
    Never crop them — orientation is taken from the canvas, not a trimmed box.
    """
    del orientation, margin_frac
    return _to_rgba(artwork)


def knockout_paper(img: Image.Image) -> Image.Image:
    """Make studio/paper white transparent so prints sit on the garment."""
    rgba = _to_rgba(img)
    if not NUMPY_AVAILABLE:
        return rgba
    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    mn = rgb.min(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    paper = (mn >= 246) & (chroma <= 14)
    arr[paper, 3] = 0
    fringe = (mn >= 228) & (chroma <= 22) & (arr[:, :, 3] > 0)
    fade = np.clip((246 - mn[fringe]).astype(np.float32) / 18.0, 0, 1)
    arr[fringe, 3] = (arr[fringe, 3].astype(np.float32) * fade).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _has_transparent_backdrop(img: Image.Image, min_frac: float = 0.02) -> bool:
    """True when a meaningful share of pixels are already transparent."""
    rgba = _to_rgba(img)
    if not NUMPY_AVAILABLE:
        alpha = rgba.split()[3]
        # Cheap sample of corners + center strip.
        w, h = rgba.size
        samples = [
            alpha.getpixel((0, 0)),
            alpha.getpixel((w - 1, 0)),
            alpha.getpixel((0, h - 1)),
            alpha.getpixel((w - 1, h - 1)),
        ]
        return sum(1 for a in samples if a < 16) >= 2
    alpha = np.asarray(rgba.split()[3])
    return float((alpha < 128).mean()) >= min_frac


def knockout_corner_backdrop(
    img: Image.Image,
    *,
    soft_fringe: bool = True,
) -> Image.Image:
    """
    Clear solid studio plates (white or black) connected to the image border.

    Flood-fill from the edges so interior black ink (outlines, glasses, cleats)
    stays opaque — only the backdrop plate becomes transparent.

    soft_fringe: fade near-plate edge pixels. Disable for die-cut stickers so
    hair / jersey whites aren't chewed into jagged holes.
    """
    rgba = _to_rgba(img)
    if not NUMPY_AVAILABLE:
        return knockout_paper(rgba)
    try:
        from scipy import ndimage
    except Exception:
        return knockout_paper(rgba)

    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    if h < 4 or w < 4:
        return rgba

    luma = (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ).astype(np.float32)
    chroma = (rgb.max(axis=2) - rgb.min(axis=2)).astype(np.int16)

    # Sample corners to decide plate color (transparent corners already count).
    corners = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
    ]
    corner_luma = []
    for y, x in corners:
        if alpha[y, x] < 16:
            continue
        corner_luma.append(float(luma[y, x]))
    if not corner_luma:
        return rgba  # already transparent at corners

    med = float(np.median(corner_luma))
    if med >= 235:
        plate = (luma >= 232) & (chroma <= 22) & (alpha > 0)
    elif med <= 28:
        # Near-black studio plate (common for caricature / cutout exports).
        plate = (luma <= 32) & (chroma <= 18) & (alpha > 0)
    else:
        # Mixed / photo backdrop — fall back to white paper key only.
        return knockout_paper(rgba)

    if float(plate.mean()) < 0.04:
        return knockout_paper(rgba) if med >= 235 else rgba

    # Seed from any plate pixel on the image border, flood only through plate.
    seed = np.zeros((h, w), dtype=bool)
    seed[0, :] = plate[0, :]
    seed[-1, :] = plate[-1, :]
    seed[:, 0] = plate[:, 0]
    seed[:, -1] = plate[:, -1]
    if not bool(seed.any()):
        return rgba

    structure = np.ones((3, 3), dtype=bool)
    # Restrict flood to plate pixels by masking.
    labeled, _num = ndimage.label(plate, structure=structure)
    border_labels = set(int(v) for v in np.unique(labeled[seed]) if v != 0)
    if not border_labels:
        return rgba
    clear = np.isin(labeled, list(border_labels))
    arr[clear, 3] = 0

    # Soft fringe on remaining near-plate edge pixels (apparel only).
    if soft_fringe:
        if med <= 28:
            fringe = (~clear) & (luma <= 48) & (chroma <= 22) & (arr[:, :, 3] > 0)
            fade = np.clip(luma[fringe] / 48.0, 0, 1)
            arr[fringe, 3] = (arr[fringe, 3].astype(np.float32) * fade).astype(np.uint8)
        else:
            fringe = (~clear) & (luma >= 220) & (chroma <= 24) & (arr[:, :, 3] > 0)
            fade = np.clip((246 - luma[fringe]) / 26.0, 0, 1)
            arr[fringe, 3] = (arr[fringe, 3].astype(np.float32) * fade).astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


def _force_opaque_print(img: Image.Image) -> Image.Image:
    """
    Flatten onto white so cutout / alpha uploads become a solid print plate.

    Setting alpha=255 alone leaves RGB under holes (often black) and can still
    read as a transparent cutout on light garments like the tote.
    """
    return _to_rgb(_to_rgba(img)).convert("RGBA")


def _strip_logo_for_transparent_print(art: Image.Image) -> Image.Image:
    """Remove baked-in MOSIDA marks (same pass stickers use in die-cut mode)."""
    out = remove_mosida_signature(art)
    # After keying, orphan logo islands become separate components — drop those.
    return _drop_floating_logo_islands(out)


def prepare_print_art(
    img: Image.Image,
    knockout: bool = True,
    *,
    is_transparent: bool = False,
) -> Image.Image:
    """
    Prepare art for product placement.

    Opaque prints keep the full upload (including white margins). Knockout /
    transparent modes may key the backdrop, but still avoid cropping the frame
    unless transparency made edge pixels empty.
    """
    art = _to_rgba(img)
    if knockout:
        art = knockout_paper(art)
        # Only crop fully transparent edges after keying — never paper-white trim.
        bbox = art.getbbox()
        return art.crop(bbox) if bbox else art

    # Apparel / cap / tote — only strip backdrop (+ logo) when caller opts in.
    if is_transparent:
        # Key the studio plate FIRST (same strategy as die-cut stickers).
        # Never strip the logo before keying: erasing corner marks creates a
        # few transparent pixels that used to short-circuit keying and leave
        # a white rectangle on the garment.
        if _has_transparent_backdrop(art, min_frac=0.08):
            keyed = art
        else:
            keyed = knockout_corner_backdrop(art, soft_fringe=False)
            if not _has_transparent_backdrop(keyed, min_frac=0.02):
                paper = knockout_paper(art)
                keyed = (
                    paper
                    if _has_transparent_backdrop(paper, min_frac=0.02)
                    else art
                )
        if _has_transparent_backdrop(keyed, min_frac=0.02):
            bbox = keyed.getbbox()
            keyed = keyed.crop(bbox) if bbox else keyed
        # Logo pass after keying — orphans become floating islands we can drop.
        return _strip_logo_for_transparent_print(keyed)

    # Opaque apparel / tote: keep the plate, but never the baked-in MOSIDA mark.
    return strip_mosida_for_opaque_print(art)


def _fit(src: Image.Image, size: Tuple[int, int], mode: str) -> Image.Image:
    tw, th = size
    if tw < 1 or th < 1:
        return src.resize((max(1, tw), max(1, th)), Image.Resampling.LANCZOS)

    src = _to_rgba(src)
    sw, sh = src.size

    if mode == "stretch":
        return src.resize((tw, th), Image.Resampling.LANCZOS)

    if mode == "cover":
        scale = max(tw / sw, th / sh)
        nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
        left = max(0, (nw - tw) // 2)
        top = max(0, (nh - th) // 2)
        return resized.crop((left, top, left + tw, top + th))

    # contain / width: fill the box width, keep the whole design, center it
    scale = tw / sw if mode == "width" else min(tw / sw, th / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    if nh > th:
        scale = th / sh
        nw, nh = max(1, int(round(sw * scale))), th
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2), resized)
    return canvas


def _shade_with_fabric(art: Image.Image, fabric: Image.Image) -> Image.Image:
    """Keep artwork colors, pick up garment wrinkles via luminance."""
    art_rgba = _to_rgba(art)
    art_rgb = _to_rgb(art_rgba)
    fabric_rgb = _to_rgb(fabric).resize(art_rgb.size, Image.Resampling.BILINEAR)
    if not NUMPY_AVAILABLE:
        return art_rgba
    art_arr = np.asarray(art_rgb).astype(np.float32)
    lum = np.asarray(ImageOps.grayscale(fabric_rgb)).astype(np.float32)
    med = float(np.median(lum[lum > 8])) if (lum > 8).any() else 1.0
    shade = np.clip(lum / max(med, 1.0), 0.82, 1.10)
    shade = 0.72 + 0.28 * shade
    out = np.clip(art_arr * shade[:, :, None], 0, 255).astype(np.uint8)
    shaded = Image.fromarray(out, "RGB").convert("RGBA")
    shaded.putalpha(art_rgba.split()[3])
    return shaded


def _studio_backdrop_mask(arr: "np.ndarray") -> "np.ndarray":
    """
    True for studio plate pixels that should not be recolored.

    Handles dark-studio blanks (tee/hoodie/tote) and light-studio blanks (cap)
    by flooding plate-colored pixels connected to the image border.
    """
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(
        np.float32
    )
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    transparent = alpha < 8

    corners = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
    ]
    corner_luma = [float(luma[y, x]) for y, x in corners if alpha[y, x] >= 8]
    if not corner_luma:
        return transparent

    med = float(np.median(corner_luma))
    if med <= 40:
        plate = (luma <= 36) & (chroma <= 22) & (~transparent)
    elif med >= 220:
        # Tight near-white plate so lightly shaded fabric (caps) is preserved.
        plate = (luma >= 248) & (chroma <= 14) & (~transparent)
    else:
        plate = np.zeros((h, w), dtype=bool)

    backdrop = transparent.copy()
    if not plate.any():
        # Fallback: treat near-black OR fully transparent as plate.
        return backdrop | ((luma < 28) & (chroma < 18))

    try:
        from scipy import ndimage
    except Exception:
        return backdrop | plate

    seed = np.zeros((h, w), dtype=bool)
    seed[0, :] = plate[0, :]
    seed[-1, :] = plate[-1, :]
    seed[:, 0] = plate[:, 0]
    seed[:, -1] = plate[:, -1]
    if not seed.any():
        return backdrop | ((luma < 28) & (chroma < 18))

    labeled, _num = ndimage.label(plate, structure=np.ones((3, 3), dtype=bool))
    border_labels = {int(v) for v in np.unique(labeled[seed]) if v != 0}
    if not border_labels:
        return backdrop | ((luma < 28) & (chroma < 18))
    return backdrop | np.isin(labeled, list(border_labels))


def tint_garment_template(
    template: Image.Image,
    rgb: Tuple[int, int, int],
) -> Image.Image:
    """
    Recolor a light product blank to `rgb` while keeping fabric folds.

    Leaves the studio backdrop (dark or light, edge-connected) untouched so
    tee/hoodie/tote/cap blanks all tint cleanly from whatever color the
    frontend sends.
    """
    rgba = _to_rgba(template)
    if not NUMPY_AVAILABLE:
        return rgba
    arr = np.array(rgba)
    rgb_a = arr[:, :, :3].astype(np.float32)
    luma = rgb_a.mean(axis=2)
    backdrop = _studio_backdrop_mask(arr)
    if not (~backdrop).any():
        return rgba

    target = np.array(
        (
            max(0, min(255, int(rgb[0]))),
            max(0, min(255, int(rgb[1]))),
            max(0, min(255, int(rgb[2]))),
        ),
        dtype=np.float32,
    )
    # Pure black collapses fold shading — lift slightly so fabric still reads.
    if float(target.max()) < 1.0:
        target[:] = (18.0, 18.0, 20.0)
    # Normalize against the bright fabric so mid-grays become soft shades of tint.
    fabric_luma = luma[~backdrop]
    ref = float(np.percentile(fabric_luma, 92)) if fabric_luma.size else 255.0
    ref = max(ref, 64.0)
    scale = np.clip(luma / ref, 0.08, 1.15)
    tinted = target[None, None, :] * scale[:, :, None]

    out = arr.copy()
    mask = ~backdrop
    out[mask, 0] = np.clip(tinted[mask, 0], 0, 255).astype(np.uint8)
    out[mask, 1] = np.clip(tinted[mask, 1], 0, 255).astype(np.uint8)
    out[mask, 2] = np.clip(tinted[mask, 2], 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def detect_card_box(img: Image.Image) -> Box:
    """Find the inner face of a white print/canvas card via its drop shadow."""
    if not NUMPY_AVAILABLE:
        return (0.075, 0.28, 0.925, 0.72)
    gray = np.asarray(ImageOps.grayscale(_to_rgb(img)))
    h, w = gray.shape
    dark = gray < 238
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if len(rows) < 8 or len(cols) < 8:
        return (0.075, 0.28, 0.925, 0.72)
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    pad_x = max(2, int((x1 - x0) * 0.006))
    pad_y = max(2, int((y1 - y0) * 0.006))
    return (
        (x0 + pad_x) / w,
        (y0 + pad_y) / h,
        (x1 - pad_x) / w,
        (y1 - pad_y) / h,
    )


def box_for_art_aspect(
    art_size: Tuple[int, int],
    zone: Sequence[float],
) -> Box:
    """
    Largest print rectangle inside `zone` that matches the artwork aspect ratio.
    Keeps proportions — no stretch — and stays inside the front print area.
    """
    aw, ah = max(1, art_size[0]), max(1, art_size[1])
    zx0, zy0, zx1, zy1 = zone
    zw, zh = max(1e-6, zx1 - zx0), max(1e-6, zy1 - zy0)
    art_ar = aw / ah
    zone_ar = zw / zh
    if art_ar >= zone_ar:
        pw, ph = zw, zw / art_ar
    else:
        ph, pw = zh, zh * art_ar
    px0 = zx0 + (zw - pw) / 2
    py0 = zy0 + (zh - ph) / 2
    return (px0, py0, px0 + pw, py0 + ph)


def composite_on_template(
    template: Image.Image,
    artwork: Image.Image,
    print_box: Sequence[float],
    fit: str = "cover",
    opacity: float = 1.0,
    shade: bool = True,
    knockout: bool = True,
    match_art_aspect: bool = False,
    orientation: Optional[str] = None,
    is_transparent: bool = False,
) -> Image.Image:
    """Paste artwork into a normalized print box on the blank product."""
    del orientation  # Frame comes from the upload; do not re-letterbox/crop.
    base = _to_rgba(template)
    art = prepare_print_art(
        artwork,
        knockout=knockout,
        is_transparent=is_transparent,
    )
    tw, th = base.size

    box = print_box
    if match_art_aspect:
        box = box_for_art_aspect(art.size, print_box)

    x0, y0, x1, y1 = box
    px0 = max(0, int(round(x0 * tw)))
    py0 = max(0, int(round(y0 * th)))
    px1 = min(tw, int(round(x1 * tw)))
    py1 = min(th, int(round(y1 * th)))
    box_w, box_h = max(1, px1 - px0), max(1, py1 - py0)

    # Aspect-matched boxes use exact resize (= no distortion).
    place_fit = "stretch" if match_art_aspect else fit
    placed = _fit(art, (box_w, box_h), place_fit)
    fabric = base.crop((px0, py0, px0 + box_w, py0 + box_h))
    if shade:
        placed = _shade_with_fabric(placed, fabric)
    else:
        placed = _to_rgba(placed)

    # Never paint onto transparent studio backdrop — that flattens as a white card.
    # Hard-threshold fabric alpha so soft edges don't bake a white fringe.
    fabric_alpha = fabric.split()[3] if fabric.mode == "RGBA" else None
    if fabric_alpha is not None:
        hard = fabric_alpha.point(lambda p: 255 if p >= 200 else 0)
        r, g, b, a = placed.split()
        a = ImageChops.multiply(a, hard)
        placed = Image.merge("RGBA", (r, g, b, a))

    if opacity < 1:
        r, g, b, a = placed.split()
        a = a.point(lambda p: int(p * opacity))
        placed.putalpha(a)

    out = base.copy()
    out.alpha_composite(placed, dest=(px0, py0))
    return flatten_on_white(out)


def _normalize_orientation(orientation: Optional[str]) -> Optional[str]:
    key = (orientation or "").strip().lower()
    if key in {"portrait", "landscape"}:
        return key
    return None


def fit_art_to_orientation(
    artwork: Image.Image,
    orientation: Optional[str],
    *,
    mode: str = "contain",
    pad_rgba: Tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    """
    Force artwork into a print orientation.
    Portrait → 2:3 (e.g. 4×6). Landscape → 3:2 (e.g. 6×4).
    """
    key = _normalize_orientation(orientation)
    art = _to_rgba(artwork)
    if key is None:
        return art

    aw, ah = art.size
    if aw < 1 or ah < 1:
        return art

    target_ratio = (2.0 / 3.0) if key == "portrait" else (3.0 / 2.0)
    current = aw / float(ah)

    if mode == "cover":
        # Crop to exact orientation (full-bleed posters).
        if current > target_ratio:
            new_w = max(1, int(round(ah * target_ratio)))
            left = max(0, (aw - new_w) // 2)
            return art.crop((left, 0, left + new_w, ah))
        new_h = max(1, int(round(aw / target_ratio)))
        top = max(0, (ah - new_h) // 2)
        return art.crop((0, top, aw, top + new_h))

    # contain — letterbox so the whole design stays visible.
    if abs(current - target_ratio) < 0.01:
        return art
    if current > target_ratio:
        new_w = aw
        new_h = max(1, int(round(aw / target_ratio)))
    else:
        new_h = ah
        new_w = max(1, int(round(ah * target_ratio)))
    canvas = Image.new("RGBA", (new_w, new_h), pad_rgba)
    canvas.alpha_composite(art, dest=((new_w - aw) // 2, (new_h - ah) // 2))
    return canvas


def make_photo_print(
    artwork: Image.Image,
    template: Image.Image | None = None,
    orientation: Optional[str] = None,
    *,
    is_transparent: bool = False,
) -> Image.Image:
    """
    Photo print: artwork on a white mat, flat on a light grey studio backdrop.
    No drop shadow or dark outline.
    Always strips baked-in MOSIDA marks (holes filled white).
    """
    del template  # Build a matted photo print; ignore blank template.
    del is_transparent  # Photo print is always an opaque plate; still strip logos.
    art = strip_mosida_for_opaque_print(_to_rgba(artwork))
    # Keep upload margins — only letterbox if forcing the opposite orientation.
    art = fit_art_to_orientation(art, orientation, mode="contain")
    max_side = 1100
    if max(art.size) > max_side:
        art = art.copy()
        art.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    aw, ah = art.size
    # Thick white mat — matches store "PHOTO PRINT" look.
    mat = max(90, min(aw, ah) // 5)
    card_w = aw + mat * 2
    card_h = ah + mat * 2
    card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
    card.alpha_composite(art, dest=(mat, mat))

    studio_pad = max(40, min(card_w, card_h) // 16)
    out_w = card_w + studio_pad * 2
    out_h = card_h + studio_pad * 2
    backdrop = Image.new("RGBA", (out_w, out_h), (242, 242, 244, 255))
    backdrop.alpha_composite(card, dest=(studio_pad, studio_pad))
    return flatten_on_white(backdrop)


def make_canvas_print(
    artwork: Image.Image,
    template: Image.Image | None = None,
    orientation: Optional[str] = None,
    *,
    is_transparent: bool = False,
) -> Image.Image:
    """
    Poster: large full-bleed artwork on a light grey studio backdrop.
    No drop shadow, no black border, no white mat.
    Always strips baked-in MOSIDA marks (holes filled white).
    """
    del template
    del is_transparent  # Poster is always an opaque plate; still strip logos.
    art = strip_mosida_for_opaque_print(_to_rgba(artwork))
    art = fit_art_to_orientation(
        art,
        orientation,
        mode="contain",
        pad_rgba=(242, 242, 244, 255),
    )
    max_side = 1600
    if max(art.size) > max_side:
        art = art.copy()
        art.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    card_w, card_h = art.size
    # Tight margin so the poster fills most of the mockup.
    frame_pad = max(28, min(card_w, card_h) // 22)
    out_w = card_w + frame_pad * 2
    out_h = card_h + frame_pad * 2
    backdrop = Image.new("RGBA", (out_w, out_h), (242, 242, 244, 255))
    backdrop.alpha_composite(art, dest=(frame_pad, frame_pad))
    return flatten_on_white(backdrop)


def _white_stroke(art: Image.Image, px: int) -> Image.Image:
    """
    Smooth die-cut white border around the artwork silhouette.

    Uses a distance-transform (or Gaussian expand) at 2× resolution so the
    contour follows curves cleanly instead of MaxFilter stair-steps.
    """
    rgba = _to_rgba(art)
    scale = 2
    hi_w = max(1, rgba.size[0] * scale)
    hi_h = max(1, rgba.size[1] * scale)
    hi = rgba.resize((hi_w, hi_h), Image.Resampling.LANCZOS)
    stroke_hi = max(2, int(round(px * scale)))
    pad = stroke_hi + 8
    canvas = Image.new("RGBA", (hi_w + pad * 2, hi_h + pad * 2), (0, 0, 0, 0))
    canvas.paste(hi, (pad, pad), hi)
    alpha = np.asarray(canvas.split()[3], dtype=np.float32) if NUMPY_AVAILABLE else None

    outline_img: Image.Image
    if NUMPY_AVAILABLE and alpha is not None:
        solid = alpha >= 128.0
        try:
            from scipy.ndimage import distance_transform_edt, gaussian_filter

            # Distance outside the subject → expand by stroke_hi pixels.
            dist_out = distance_transform_edt(~solid)
            expanded = solid | (dist_out <= float(stroke_hi))
            # Soft AA rim so downscale stays smooth.
            soft = gaussian_filter(expanded.astype(np.float32), sigma=1.1)
            outline_a = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
        except Exception:
            # Gaussian expand fallback (still much smoother than MaxFilter).
            hard = Image.fromarray((solid.astype(np.uint8) * 255), mode="L")
            blur_r = max(1.0, stroke_hi * 0.62)
            expanded = hard.filter(ImageFilter.GaussianBlur(radius=blur_r))
            outline_a = np.asarray(
                expanded.point(lambda p: 255 if p > 36 else 0),
                dtype=np.uint8,
            )
            outline_a = np.asarray(
                Image.fromarray(outline_a, mode="L").filter(
                    ImageFilter.GaussianBlur(radius=1.0)
                ),
                dtype=np.uint8,
            )
        outline_img = Image.fromarray(outline_a, mode="L")
    else:
        hard = canvas.split()[3].point(lambda p: 255 if p > 128 else 0)
        blur_r = max(1.0, stroke_hi * 0.62)
        expanded = hard.filter(ImageFilter.GaussianBlur(radius=blur_r))
        outline_img = expanded.point(lambda p: 255 if p > 36 else 0).filter(
            ImageFilter.GaussianBlur(radius=1.0)
        )

    sticker_hi = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sticker_hi.paste(Image.new("RGBA", canvas.size, (255, 255, 255, 255)), mask=outline_img)
    sticker_hi.alpha_composite(canvas)

    out_w = max(1, int(round(sticker_hi.size[0] / scale)))
    out_h = max(1, int(round(sticker_hi.size[1] / scale)))
    return sticker_hi.resize((out_w, out_h), Image.Resampling.LANCZOS)


def _sticker_shadow(sticker: Image.Image, blur: int = 10, offset: Tuple[int, int] = (4, 6)) -> Image.Image:
    """Soft drop shadow so the white die-cut border reads on a white background."""
    alpha = sticker.split()[3]
    shadow = Image.new("RGBA", sticker.size, (0, 0, 0, 0))
    shadow.paste(Image.new("RGBA", sticker.size, (0, 0, 0, 70)), mask=alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    ox, oy = offset
    pad = max(ox, oy) + blur * 2
    out = Image.new("RGBA", (sticker.size[0] + pad * 2, sticker.size[1] + pad * 2), (0, 0, 0, 0))
    out.alpha_composite(shadow, dest=(pad + ox, pad + oy))
    out.alpha_composite(sticker, dest=(pad, pad))
    return out


_MOSIDA_LOGO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "constants")
)
# Edge placements only — never search the image center (that punches holes
# through faces / jerseys when contrast matching misfires).
_MOSIDA_LOGO_POSITIONS = (
    "top_right",
    "top_left",
    "top_center",
    "bottom_center",
    "bottom_right",
    "bottom_left",
)


def _trim_logo_rgba(logo: Image.Image) -> Image.Image:
    rgba = _to_rgba(logo)
    bbox = rgba.split()[3].getbbox()
    return rgba.crop(bbox) if bbox else rgba


def _mosida_logo_geometry(
    img_w: int,
    img_h: int,
    sig_aspect: float,
    position: str,
    *,
    scale: float = 1.0,
    nudge_x: int = 0,
    nudge_y: int = 0,
) -> Tuple[int, int, int, int]:
    """Match character_utils signature placement (20% box + padding), optional scale."""
    sig_width = int(img_w * 0.20 * scale)
    sig_height = int(img_h * 0.20 * scale)
    target_aspect = sig_width / sig_height if sig_height else sig_aspect
    if sig_aspect > target_aspect:
        new_w = max(1, sig_width)
        new_h = max(1, int(sig_width / sig_aspect))
    else:
        new_h = max(1, sig_height)
        new_w = max(1, int(sig_height * sig_aspect))

    # Keep in lockstep with character_utils._signature_overlay_geometry.
    pad_x = max(12, int(img_w * 0.025))
    pad_y = max(12, int(img_h * 0.025))
    bottom_inset = max(2, int(img_h * 0.015))
    vertical, horizontal = position.split("_", 1)

    if horizontal == "left":
        x = pad_x
    elif horizontal == "center":
        x = (img_w - new_w) // 2
    else:
        x = img_w - new_w - pad_x

    if vertical == "top":
        y = pad_y
    elif vertical == "center":
        y = (img_h - new_h) // 2
    else:
        y = img_h - new_h - bottom_inset

    x = max(0, min(img_w - new_w, x + nudge_x))
    y = max(0, min(img_h - new_h, y + nudge_y))
    return x, y, new_w, new_h


def _load_mosida_logo_templates() -> List[Image.Image]:
    templates: List[Image.Image] = []
    for name in ("cartoon.png", "others.png"):
        path = os.path.join(_MOSIDA_LOGO_DIR, name)
        if not os.path.isfile(path):
            continue
        try:
            img = Image.open(path)
            img.load()
            templates.append(_trim_logo_rgba(img))
        except Exception:
            continue
    return templates


def _score_logo_region(
    arr: "np.ndarray",
    logo: Image.Image,
    x: int,
    y: int,
    lw: int,
    lh: int,
    *,
    allow_plate_fallback: bool = False,
) -> Tuple[float, Optional["np.ndarray"], Tuple[int, int, int, int]]:
    """
    Score one logo placement. Returns (score, erase_mask_or_None, box).

    RGB template match is preferred. Dark-ink-on-light-plate fallback is only
    used for corner placements (never top_center) so hair highlights survive.
    """
    h, w = arr.shape[:2]
    empty = (0.0, None, (x, y, lw, lh))
    if lw < 8 or lh < 8 or x < 0 or y < 0 or x + lw > w or y + lh > h:
        return empty
    logo_r = logo.resize((lw, lh), Image.Resampling.LANCZOS)
    logo_arr = np.asarray(logo_r)
    logo_a = logo_arr[:, :, 3].astype(np.float32) / 255.0
    ink = logo_a >= 0.40
    ink_n = int(ink.sum())
    if ink_n < 40:
        return empty

    region = arr[y : y + lh, x : x + lw]
    # Skip placements that land mostly on already-transparent pixels.
    if float((region[:, :, 3] > 16).mean()) < 0.15:
        return empty

    art_rgb = region[:, :, :3].astype(np.float32)
    logo_rgb = logo_arr[:, :, :3].astype(np.float32)
    solid = region[:, :, 3] > 16
    plate = solid & (~ink)
    if int(plate.sum()) < 80 or int((solid & ink).sum()) < 40:
        return empty

    def _luma(rgb: "np.ndarray") -> "np.ndarray":
        return (
            0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        )

    logo_ink_lum = float(np.median(_luma(logo_rgb[ink])))
    art_ink_lum = float(np.median(_luma(art_rgb[solid & ink])))
    plate_med = float(np.median(_luma(art_rgb[plate])))
    plate_std = float(np.std(_luma(art_rgb[plate])))
    contrast = abs(art_ink_lum - plate_med)

    # White logo on white plate (or dark logo on dark plate) is not a mark.
    if contrast < 40.0:
        return empty
    if logo_ink_lum >= 170.0 and art_ink_lum < 150.0:
        return empty  # light template but art ink is dark
    if logo_ink_lum <= 110.0 and art_ink_lum > 170.0:
        return empty  # dark template but art ink is light
    if logo_ink_lum >= 170.0 and plate_med >= 180.0:
        return empty  # white logo invisible on light studio plate
    if logo_ink_lum <= 110.0 and plate_med <= 80.0:
        return empty  # dark logo invisible on dark plate

    dist = np.sqrt(((art_rgb - logo_rgb) ** 2).sum(axis=2))
    matched = ink & solid & (dist <= 65.0)
    match_ratio = float(matched.sum()) / float(ink_n)
    if match_ratio >= 0.50:
        return (match_ratio + contrast / 255.0 + 1.0, matched, (x, y, lw, lh))

    if not allow_plate_fallback:
        return empty

    # Dark brush logo on a near-white studio plate (common merch upload).
    if logo_ink_lum > 140.0:
        return empty
    ink_lum = _luma(art_rgb[solid & ink])
    dark_frac = float((ink_lum < (plate_med - 50.0)).sum()) / float(
        max(1, ink_lum.size)
    )
    if not (
        plate_med >= 210.0
        and plate_std <= 22.0
        and art_ink_lum <= 130.0
        and (plate_med - art_ink_lum) >= 70.0
        and dark_frac >= 0.45
    ):
        return empty
    lum = _luma(art_rgb)
    erase = ink & solid & (lum < (plate_med - 50.0))
    if not bool(erase.any()):
        return empty
    # Plate fallback scores below RGB matches so a real template win wins.
    return (0.35 + dark_frac * 0.5 + contrast / 255.0, erase, (x, y, lw, lh))


def _apply_logo_erase(
    arr: "np.ndarray",
    erase: "np.ndarray",
    box: Tuple[int, int, int, int],
) -> None:
    x, y, lw, lh = box
    region = arr[y : y + lh, x : x + lw]
    try:
        from scipy import ndimage

        erase = ndimage.binary_dilation(erase, iterations=1)
    except Exception:
        pass
    region[erase, 3] = 0
    arr[y : y + lh, x : x + lw] = region


def _drop_floating_logo_islands(img: Image.Image) -> Image.Image:
    """
    Drop logo leftovers and speckles (the stray white dots beside stickers).

    Keeps the largest component and nearby sizable props (ball/grass/hand).
    Always removes tiny disconnected blobs — even inside the hero pad box.
    """
    rgba = _to_rgba(img)
    if not NUMPY_AVAILABLE:
        return rgba
    try:
        from scipy import ndimage
    except Exception:
        return rgba

    arr = np.array(rgba)
    solid = arr[:, :, 3] >= 128
    if not bool(solid.any()):
        return rgba

    labeled, num = ndimage.label(solid)
    if num <= 1:
        return rgba

    sizes = ndimage.sum(solid, labeled, index=list(range(1, num + 1)))
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    largest_i = int(np.argmax(sizes_arr)) + 1
    largest_n = float(sizes_arr[largest_i - 1])
    keep = {largest_i}
    # Specks / registration dots — drop anything this small anywhere.
    speck_max = max(80.0, largest_n * 0.008)

    ys, xs = np.where(labeled == largest_i)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(8, int((x1 - x0) * 0.18))
    pad_y = max(8, int((y1 - y0) * 0.12))
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(arr.shape[1] - 1, x1 + pad_x), min(arr.shape[0] - 1, y1 + pad_y)

    for i, sz in enumerate(sizes_arr, start=1):
        if i == largest_i:
            continue
        area = float(sz)
        if area <= speck_max:
            continue  # stray white dots / dust
        cys, cxs = np.where(labeled == i)
        cx = float(cxs.mean())
        cy = float(cys.mean())
        inside = x0 <= cx <= x1 and y0 <= cy <= y1
        if inside or area >= largest_n * 0.35:
            keep.add(i)

    mask = np.isin(labeled, list(keep))
    dropped = num - len(keep)
    if dropped <= 0:
        return rgba
    arr[~mask, 3] = 0
    print(f"🧹 Dropped {dropped} floating logo/speck component(s)")
    return Image.fromarray(arr, "RGBA")


def remove_mosida_signature(img: Image.Image) -> Image.Image:
    """Erase baked-in MOSIDA MIAMI marks via multi-scale template / plate match."""
    rgba = _to_rgba(img).copy()
    templates = _load_mosida_logo_templates()
    if not templates or not NUMPY_AVAILABLE:
        return rgba

    arr = np.array(rgba)
    h, w = arr.shape[:2]
    # Prefer the real overlay corners; skip top_center (hits hair/faces).
    positions = (
        "top_right",
        "top_left",
        "bottom_center",
        "bottom_right",
        "bottom_left",
    )
    scales = (0.85, 1.0, 1.15, 1.30)
    nudges = (
        (0, 0),
        (int(w * 0.01), 0),
        (-int(w * 0.01), 0),
        (0, int(h * 0.01)),
        (0, -int(h * 0.01)),
    )

    # Prefer the usual overlay corner slightly when scores are close.
    position_bonus = {
        "top_right": 0.08,
        "bottom_center": 0.05,
        "top_left": 0.02,
        "bottom_right": 0.01,
        "bottom_left": 0.01,
    }
    candidates: List[Tuple[float, "np.ndarray", Tuple[int, int, int, int]]] = []
    for logo in templates:
        aspect = logo.size[0] / max(1, logo.size[1])
        for position in positions:
            allow_plate = position in {"top_right", "top_left", "bottom_center"}
            for scale in scales:
                for nx, ny in nudges:
                    x, y, lw, lh = _mosida_logo_geometry(
                        w,
                        h,
                        aspect,
                        position,
                        scale=scale,
                        nudge_x=nx,
                        nudge_y=ny,
                    )
                    score, erase, box = _score_logo_region(
                        arr,
                        logo,
                        x,
                        y,
                        lw,
                        lh,
                        allow_plate_fallback=allow_plate,
                    )
                    if erase is not None and score > 0:
                        candidates.append(
                            (
                                score + position_bonus.get(position, 0.0),
                                erase.copy(),
                                box,
                            )
                        )

    if not candidates:
        return rgba

    # Keep only the best hit — applying every scale/nudge was chewing hair.
    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, best_erase, best_box = candidates[0]
    _apply_logo_erase(arr, best_erase, best_box)
    print(
        f"🧹 Removed MOSIDA signature (merch print) "
        f"score={best_score:.2f} box={best_box}"
    )
    return Image.fromarray(arr, "RGBA")


def strip_mosida_for_opaque_print(img: Image.Image) -> Image.Image:
    """Remove baked-in MOSIDA marks and flatten holes onto a white plate."""
    return _force_opaque_print(remove_mosida_signature(img))


def build_diecut_sticker(
    artwork: Image.Image,
    stroke_frac: float = 0.045,
    *,
    strip_logo: bool = False,
) -> Image.Image:
    """Knock out studio plate and add a thick smooth white die-cut rim."""
    art = _to_rgba(artwork)
    if strip_logo:
        art = remove_mosida_signature(art)

    # Work at a healthy size for smooth contours, then stroke.
    if max(art.size) > 900:
        art = art.copy()
        art.thumbnail((900, 900), Image.Resampling.LANCZOS)

    # If art is already a cutout, keep its alpha — don't re-key and chew edges.
    if _has_transparent_backdrop(art, min_frac=0.08):
        keyed = art
    else:
        # No soft fringe — fringe fade was carving jagged holes in hair/jersey.
        keyed = knockout_corner_backdrop(art, soft_fringe=False)
        if not _has_transparent_backdrop(keyed, min_frac=0.02):
            paper = knockout_paper(art)
            keyed = paper if _has_transparent_backdrop(paper, min_frac=0.02) else art
    art = keyed

    if strip_logo:
        # Logos that survive keying become separate islands — drop those only.
        art = remove_mosida_signature(art)
        art = _drop_floating_logo_islands(art)

    bbox = art.getbbox()
    if bbox:
        # Small pad so soft fringe / hair isn't cropped flush before stroke.
        pad = 4
        x0 = max(0, bbox[0] - pad)
        y0 = max(0, bbox[1] - pad)
        x1 = min(art.size[0], bbox[2] + pad)
        y1 = min(art.size[1], bbox[3] + pad)
        art = art.crop((x0, y0, x1, y1))

    stroke = max(14, min(32, int(round(min(art.size) * max(stroke_frac, 0.055)))))
    sticker = _white_stroke(art, stroke)
    # Stroke turns leftover dust into visible white dots — scrub again.
    return _drop_floating_logo_islands(sticker)


def _studio_pad_sticker(sticker: Image.Image, pad_frac: float = 0.10) -> Image.Image:
    """Place a die-cut sticker on a white studio card with soft margin."""
    pad = max(48, int(min(sticker.size) * pad_frac))
    canvas = Image.new(
        "RGBA",
        (sticker.size[0] + pad * 2, sticker.size[1] + pad * 2),
        (255, 255, 255, 255),
    )
    canvas.paste(sticker, (pad, pad), sticker)
    return flatten_on_white(canvas)


def _rect_sticker_card(
    artwork: Image.Image,
    max_side: int = 900,
) -> Image.Image:
    """Full rectangular artwork with a thin dark edge — upload margins kept."""
    # Flatten to an opaque card; strip baked-in MOSIDA; keep upload margins.
    art = strip_mosida_for_opaque_print(_to_rgba(artwork))
    if max(art.size) > max_side:
        art = art.copy()
        art.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    # Thin dark border — helps separate identical stickers in pack mockups.
    border = max(2, min(4, min(art.size) // 140))
    draw = ImageDraw.Draw(art)
    draw.rectangle(
        (0, 0, art.size[0] - 1, art.size[1] - 1),
        outline=(35, 35, 35, 235),
        width=border,
    )
    return art


def _sticker_art_for_orientation(
    artwork: Image.Image,
    orientation: Optional[str],
) -> Image.Image:
    """Keep the upload frame intact (no white-space crop)."""
    return art_for_orientation(artwork, orientation)


def make_sticker_single(
    artwork: Image.Image,
    diecut: Optional[Image.Image] = None,
    orientation: Optional[str] = None,
    *,
    is_transparent: bool = False,
) -> Image.Image:
    """
    Single sticker product.

    Default: full rectangular artwork with a thin dark edge + drop shadow
    (baked-in MOSIDA stripped). Transparent mode: die-cut white contour PNG.
    """
    art = _sticker_art_for_orientation(artwork, orientation)
    if is_transparent:
        sticker = diecut if diecut is not None else build_diecut_sticker(
            art,
            stroke_frac=0.055,
            strip_logo=True,
        )
        sticker = _sticker_shadow(sticker, blur=14, offset=(6, 9))
        pad = max(40, int(min(sticker.size) * 0.08))
        canvas = Image.new(
            "RGBA",
            (sticker.size[0] + pad * 2, sticker.size[1] + pad * 2),
            (0, 0, 0, 0),
        )
        canvas.alpha_composite(sticker, dest=(pad, pad))
        return canvas

    del diecut  # Rectangular listing keeps the full scene (logo stripped).
    sticker = _rect_sticker_card(art, max_side=900)
    sticker = _sticker_shadow(sticker, blur=14, offset=(6, 9))
    return _studio_pad_sticker(sticker)


def _free_transform_sticker(
    sticker: Image.Image,
    max_w: int,
    max_h: int,
    *,
    cell: bool = False,
) -> Image.Image:
    """
    Uniform free-transform: scale a die-cut to fit max_w×max_h (contain).

    When cell=True, center the result on a transparent max_w×max_h tile so
    every sticker in a grid occupies the same aligned bounding box.
    """
    tile = _to_rgba(sticker)
    tw, th = max(1, tile.size[0]), max(1, tile.size[1])
    mw, mh = max(1, max_w), max(1, max_h)
    scale = min(mw / float(tw), mh / float(th))
    nw = max(1, int(round(tw * scale)))
    nh = max(1, int(round(th * scale)))
    resized = tile.resize((nw, nh), Image.Resampling.LANCZOS)
    if not cell:
        return resized
    canvas = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
    canvas.alpha_composite(resized, dest=((mw - nw) // 2, (mh - nh) // 2))
    return canvas


def make_sticker_fan(
    artwork: Image.Image,
    count: int = 5,
    diecut: Optional[Image.Image] = None,
    orientation: Optional[str] = None,
    *,
    is_transparent: bool = False,
) -> Image.Image:
    """
    Pack mockup of identical stickers.

    Default: rectangular cards stacked diagonally (store listing).
    Transparent mode: die-cut white contour, no MOSIDA logo, uniform
    free-transform — neat evenly-spaced row on a transparent canvas.
    """
    art = _sticker_art_for_orientation(artwork, orientation)
    if is_transparent:
        card = diecut if diecut is not None else build_diecut_sticker(
            art,
            stroke_frac=0.055,
            strip_logo=True,
        )
        # One free-transform size for every copy — perfect alignment.
        target_h = 480
        tile = _free_transform_sticker(card, int(target_h * 1.35), target_h)
        # Soft contact shadow (keeps rim readable on light / dark UIs).
        tile = _sticker_shadow(tile, blur=12, offset=(5, 8))
        sw, sh = tile.size
        gap = max(28, int(sw * 0.10))
        margin = max(48, int(min(sw, sh) * 0.12))
        w = count * sw + max(0, count - 1) * gap + margin * 2
        h = sh + margin * 2
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for i in range(count):
            x = margin + i * (sw + gap)
            y = margin
            canvas.alpha_composite(tile, dest=(x, y))
        return canvas

    del diecut
    card = _rect_sticker_card(art, max_side=520)
    card = _sticker_shadow(card, blur=10, offset=(4, 6))
    sw, sh = card.size
    # Diagonal stack offset (matches the fanned pack reference).
    step_x = max(28, int(sw * 0.11))
    step_y = max(24, int(sh * 0.11))
    margin = 56
    w = sw + step_x * (count - 1) + margin * 2
    h = sh + step_y * (count - 1) + margin * 2
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    # Draw back to front so the last sticker sits on top.
    for i in range(count):
        x = margin + i * step_x
        y = margin + i * step_y
        canvas.alpha_composite(card, dest=(x, y))
    return flatten_on_white(canvas)


def _load_sheet_asset(name: str) -> Image.Image:
    path = os.path.join(_SHEET_ASSET_DIR, name)
    img = Image.open(path)
    img.load()
    return _to_rgba(img)


def _sheet_art_tile(artwork: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    """
    Opaque rectangular sticker tile.

    Cover-fit fills the cell (keeps aspect, never stretches). Does not trim the
    upload first — composition margins stay on single-sticker products.
    """
    art = _to_rgb(_to_rgba(artwork)).convert("RGBA")
    fitted = _fit(art, (max(1, cell_w), max(1, cell_h)), "cover")
    tile = Image.new("RGBA", (cell_w, cell_h), (255, 255, 255, 255))
    tile.paste(fitted, (0, 0), fitted)
    border = max(2, min(3, min(cell_w, cell_h) // 120))
    ImageDraw.Draw(tile).rectangle(
        (0, 0, cell_w - 1, cell_h - 1),
        outline=(30, 30, 30, 220),
        width=border,
    )
    return tile


def _grid_cell_size(
    area_w: int,
    area_h: int,
    cols: int,
    rows: int,
    gap: int,
    art_aspect: float,
) -> Tuple[int, int]:
    """Largest cell that tiles into the area while keeping artwork aspect."""
    max_w = max(1, (area_w - gap * max(0, cols - 1)) // max(1, cols))
    max_h = max(1, (area_h - gap * max(0, rows - 1)) // max(1, rows))
    aspect = max(0.05, art_aspect)
    if max_w / max_h > aspect:
        cell_h = max_h
        cell_w = max(1, int(round(cell_h * aspect)))
    else:
        cell_w = max_w
        cell_h = max(1, int(round(cell_w / aspect)))
    return cell_w, cell_h


def _paste_art_grid(
    sheet: Image.Image,
    tile: Image.Image,
    origin: Tuple[int, int],
    area: Tuple[int, int],
    cols: int,
    rows: int,
    gap: int,
    *,
    valign: str = "center",
) -> None:
    """Place a cols×rows grid of identical tiles inside the area."""
    x0, y0 = origin
    area_w, area_h = area
    tw, th = tile.size
    grid_w = cols * tw + gap * max(0, cols - 1)
    grid_h = rows * th + gap * max(0, rows - 1)
    x = x0 + max(0, (area_w - grid_w) // 2)
    if valign == "top":
        y = y0
    else:
        y = y0 + max(0, (area_h - grid_h) // 2)
    for row in range(rows):
        for col in range(cols):
            sheet.alpha_composite(
                tile,
                dest=(x + col * (tw + gap), y + row * (th + gap)),
            )


def _knockout_side_backdrop(img: Image.Image) -> Image.Image:
    """
    Clear gray/black card backdrops around die-cut side stickers.

    Floods from the image edges through low-saturation dark/gray noise and
    stops at white die-cut borders and saturated artwork colors. A second
    grow pass clears speckled gray islands left behind by noisy cards.
    """
    rgba = _to_rgba(img)
    if not NUMPY_AVAILABLE:
        return rgba

    arr = np.asarray(rgba).copy()
    h, w = arr.shape[:2]
    if h < 4 or w < 4:
        return rgba

    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    lum = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)

    # Gray/black card noise (speckled) — never the bright white die-cut rim.
    is_backdrop = (alpha < 40) | ((sat <= 40) & (lum < 235))

    clear = np.zeros((h, w), dtype=bool)
    stack: List[Tuple[int, int]] = []

    def _push(y: int, x: int) -> None:
        if 0 <= y < h and 0 <= x < w and not clear[y, x] and is_backdrop[y, x]:
            clear[y, x] = True
            stack.append((y, x))

    for x in range(w):
        _push(0, x)
        _push(h - 1, x)
    for y in range(h):
        _push(y, 0)
        _push(y, w - 1)

    while stack:
        y, x = stack.pop()
        _push(y - 1, x)
        _push(y + 1, x)
        _push(y, x - 1)
        _push(y, x + 1)
        # 8-connected helps bridge speckled noise gaps.
        _push(y - 1, x - 1)
        _push(y - 1, x + 1)
        _push(y + 1, x - 1)
        _push(y + 1, x + 1)

    # Grow into remaining backdrop-colored pixels that touch cleared ones.
    for _ in range(8):
        if not clear.any():
            break
        grow = np.zeros_like(clear)
        # Neighbor of cleared via shifts
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            shifted = np.zeros_like(clear)
            y0, y1 = max(0, dy), h + min(0, dy)
            x0, x1 = max(0, dx), w + min(0, dx)
            sy0, sy1 = max(0, -dy), h - max(0, dy)
            sx0, sx1 = max(0, -dx), w - max(0, dx)
            shifted[y0:y1, x0:x1] = clear[sy0:sy1, sx0:sx1]
            grow |= shifted & is_backdrop & ~clear
        if not grow.any():
            break
        clear |= grow

    if clear.any():
        arr[clear, 3] = 0
        arr[clear, 0:3] = 0

    out = Image.fromarray(arr, "RGBA")
    bbox = out.getbbox()
    return out.crop(bbox) if bbox else out


def _fit_side_sticker(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    sticker = _knockout_side_backdrop(_to_rgba(img))
    bbox = sticker.getbbox()
    if bbox:
        sticker = sticker.crop(bbox)
    sticker.thumbnail((max(1, max_w), max(1, max_h)), Image.Resampling.LANCZOS)
    return sticker


def _paste_stack(
    sheet: Image.Image,
    items: List[Image.Image],
    box: Tuple[int, int, int, int],
    *,
    vertical: bool = True,
    gap: int = 10,
    align: str = "center",
) -> None:
    """Place a row or column of stickers inside box (center or start-aligned)."""
    x0, y0, x1, y1 = box
    area_w, area_h = max(1, x1 - x0), max(1, y1 - y0)
    if vertical:
        total = sum(im.size[1] for im in items) + gap * max(0, len(items) - 1)
        if align == "start":
            y = y0
        else:
            y = y0 + max(0, (area_h - total) // 2)
        for im in items:
            x = x0 + max(0, (area_w - im.size[0]) // 2)
            sheet.alpha_composite(im, dest=(x, y))
            y += im.size[1] + gap
    else:
        total = sum(im.size[0] for im in items) + gap * max(0, len(items) - 1)
        if align == "start":
            x = x0
        else:
            x = x0 + max(0, (area_w - total) // 2)
        for im in items:
            y = y0 + max(0, (area_h - im.size[1]) // 2)
            sheet.alpha_composite(im, dest=(x, y))
            x += im.size[0] + gap


def _paste_side_column(
    sheet: Image.Image,
    box: Tuple[int, int, int, int],
    *,
    mosida_frac: float = 0.52,
    align: str = "start",
) -> None:
    """
    Right-column die-cuts: vertical MOSIDA, red WYNWOOD, green MIAMI, CCTV.

    MOSIDA is the hero brand mark (larger). Other stickers keep natural
    proportions and are not stretched to fill the column.
    """
    x0, y0, x1, y1 = box
    col_w, col_h = max(1, x1 - x0), max(1, y1 - y0)
    gap = max(12, int(col_h * 0.018))
    mosida = _knockout_side_backdrop(
        _load_sheet_asset("side_mosida.png").rotate(-90, expand=True)
    )
    red = _load_sheet_asset("side_wynwood_red.png")
    miami = _load_sheet_asset("side_wynwood_miami.png")
    cctv = _load_sheet_asset("side_cctv.png")

    mosida_h = max(120, int(col_h * mosida_frac))
    mosida_w = max(40, int(col_w * 0.98))
    rest = max(100, col_h - mosida_h - gap * 3)
    small_h = max(44, min(int(col_h * 0.14), rest // 3))
    sticker_w = max(40, int(col_w * 0.90))

    items = [
        _fit_side_sticker(mosida, mosida_w, mosida_h),
        _fit_side_sticker(red, sticker_w, small_h),
        _fit_side_sticker(miami, sticker_w, small_h),
        _fit_side_sticker(cctv, sticker_w, int(small_h * 1.15)),
    ]
    _paste_stack(sheet, items, box, vertical=True, gap=gap, align=align)


def _paste_portrait_top_brand(
    sheet: Image.Image,
    box: Tuple[int, int, int, int],
) -> None:
    """Portrait sheet top row: horizontal MOSIDA, red WYNWOOD, circular badge."""
    x0, y0, x1, y1 = box
    row_w, row_h = max(1, x1 - x0), max(1, y1 - y0)
    gap = max(12, row_w // 40)
    mosida = _fit_side_sticker(_load_sheet_asset("side_mosida.png"), int(row_w * 0.42), row_h)
    red = _fit_side_sticker(_load_sheet_asset("side_wynwood_red.png"), int(row_w * 0.30), row_h)
    badge = _fit_side_sticker(_load_sheet_asset("top_badge.png"), int(row_w * 0.18), row_h)
    _paste_stack(sheet, [mosida, red, badge], box, vertical=False, gap=gap)


def _sheet_canvas() -> Tuple[Image.Image, int, int]:
    sheet_w = 1100
    sheet_h = int(round(sheet_w * 1024 / 791))
    bg = _load_sheet_asset("sheet_bg.jpg").resize(
        (sheet_w, sheet_h),
        Image.Resampling.LANCZOS,
    )
    return bg.convert("RGBA"), sheet_w, sheet_h


def _cell_for_aspect(
    max_w: int,
    max_h: int,
    art_aspect: float,
) -> Tuple[int, int]:
    """Largest cell inside max_w×max_h that keeps art_aspect (w/h)."""
    aspect = max(0.05, art_aspect)
    if max_w / float(max(1, max_h)) > aspect:
        cell_h = max(1, max_h)
        cell_w = max(1, int(round(cell_h * aspect)))
    else:
        cell_w = max(1, max_w)
        cell_h = max(1, int(round(cell_w / aspect)))
    return cell_w, cell_h


def _make_sticker_sheet_landscape(art: Image.Image) -> Image.Image:
    """
    Landscape-art sheet matching the store reference:
    2×3 top (beside logos) + larger full-width 3×2 bottom.

    Both grids keep the artwork aspect (cover-fit, never stretched).
    Bottom cells are sized to fill remaining page height.
    """
    sheet, sheet_w, sheet_h = _sheet_canvas()
    aw, ah = max(1, art.size[0]), max(1, art.size[1])
    art_aspect = max(0.5, aw / float(ah))

    pad_x = int(sheet_w * 0.028)
    pad_y = int(sheet_h * 0.020)
    pad_bot = int(sheet_h * 0.022)
    gap_x = max(10, int(sheet_w * 0.011))
    gap_y = max(8, int(sheet_h * 0.008))
    section_gap = max(10, int(sheet_h * 0.012))
    side_gap = max(8, int(sheet_w * 0.010))
    side_w = int(sheet_w * 0.20)

    main_w = max(1, sheet_w - pad_x * 2 - side_gap - side_w)
    side_left = pad_x + main_w + side_gap
    full_w = max(1, sheet_w - pad_x * 2)
    usable_h = sheet_h - pad_y - pad_bot

    # Bottom 3×2 FIRST — full width, as large as aspect allows (no tiny leftovers).
    bot_cw = max(1, (full_w - gap_x * 2) // 3)
    bot_ch = max(1, int(round(bot_cw / art_aspect)))
    bot_grid_w = 3 * bot_cw + gap_x * 2
    bot_grid_h = 2 * bot_ch + gap_y

    # Top 2×3 gets whatever height remains above the bottom band — fills main_w.
    top_budget = max(1, usable_h - bot_grid_h - section_gap)
    top_cw, top_ch = _cell_for_aspect(
        max(1, (main_w - gap_x) // 2),
        max(1, (top_budget - gap_y * 2) // 3),
        art_aspect,
    )
    top_grid_h = 3 * top_ch + gap_y * 2

    large = _sheet_art_tile(art, top_cw, top_ch)
    small = _sheet_art_tile(art, bot_cw, bot_ch)

    y_top = pad_y
    y_bot = y_top + top_grid_h + section_gap
    _paste_art_grid_xy(sheet, large, pad_x, y_top, 2, 3, gap_x, gap_y)
    bot_x = pad_x + max(0, (full_w - bot_grid_w) // 2)
    _paste_art_grid_xy(sheet, small, bot_x, y_bot, 3, 2, gap_x, gap_y)
    # Logos sit beside the top block only — bottom band is full-width.
    _paste_side_column(
        sheet,
        (side_left, pad_y, sheet_w - pad_x, y_bot - max(4, section_gap // 2)),
        mosida_frac=0.55,
        align="start",
    )
    print(
        f"🧾 landscape cells top={top_cw}x{top_ch} bot={bot_cw}x{bot_ch} "
        f"aspect={art_aspect:.3f} bot_h={bot_grid_h} top_h={top_grid_h}"
    )
    return flatten_on_white(sheet)


def _paste_art_grid_xy(
    sheet: Image.Image,
    tile: Image.Image,
    x: int,
    y: int,
    cols: int,
    rows: int,
    gap_x: int,
    gap_y: int,
) -> None:
    """Place a cols×rows grid at an exact top-left origin with separate gutters."""
    tw, th = tile.size
    for row in range(rows):
        for col in range(cols):
            sheet.alpha_composite(
                tile,
                dest=(x + col * (tw + gap_x), y + row * (th + gap_y)),
            )


def _make_sticker_sheet_portrait(art: Image.Image) -> Image.Image:
    """
    Portrait-art sheet (reference): top brand row, 2×2 main art,
    right decorative column, full-width 3×1 bottom art row.
    """
    sheet, sheet_w, sheet_h = _sheet_canvas()
    aw, ah = max(1, art.size[0]), max(1, art.size[1])
    # Prefer portrait/square cells even if content is near-square.
    art_aspect = min(aw / float(ah), 0.92)

    top_brand = (0.03, 0.02, 0.97, 0.13)
    main = (0.04, 0.15, 0.76, 0.72)
    side = (0.78, 0.15, 0.96, 0.72)
    bottom = (0.04, 0.74, 0.96, 0.97)
    gap = max(10, int(min(sheet_w, sheet_h) * 0.014))

    def _box(norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = norm
        return (
            int(x0 * sheet_w),
            int(y0 * sheet_h),
            int(x1 * sheet_w),
            int(y1 * sheet_h),
        )

    _paste_portrait_top_brand(sheet, _box(top_brand))

    mx0, my0, mx1, my1 = _box(main)
    bx0, by0, bx1, by1 = _box(bottom)
    main_w, main_h = max(1, mx1 - mx0), max(1, my1 - my0)
    bot_w, bot_h = max(1, bx1 - bx0), max(1, by1 - by0)

    top_cw, top_ch = _grid_cell_size(main_w, main_h, 2, 2, gap, art_aspect)
    bot_cw, bot_ch = _grid_cell_size(bot_w, bot_h, 3, 1, gap, art_aspect)
    # Bottom row should read smaller than the 2×2 block.
    if bot_cw > int(top_cw * 0.92):
        scale = (top_cw * 0.92) / float(bot_cw)
        bot_cw = max(1, int(bot_cw * scale))
        bot_ch = max(1, int(bot_ch * scale))

    large = _sheet_art_tile(art, top_cw, top_ch)
    small = _sheet_art_tile(art, bot_cw, bot_ch)
    _paste_art_grid(sheet, large, (mx0, my0), (main_w, main_h), 2, 2, gap)
    _paste_art_grid(sheet, small, (bx0, by0), (bot_w, bot_h), 3, 1, gap)
    _paste_side_column(sheet, _box(side), mosida_frac=0.40, align="center")
    return flatten_on_white(sheet)


def _sheet_art_is_portrait(art: Image.Image) -> bool:
    """Portrait sheet when art is taller, square, or only mildly wide."""
    aw, ah = max(1, art.size[0]), max(1, art.size[1])
    return (aw / float(ah)) <= 1.15


def _fit_diecut_tile(sticker: Image.Image, max_h: int) -> Image.Image:
    """Scale a die-cut sticker to a target height (keep aspect)."""
    return _free_transform_sticker(sticker, int(max_h * 2.5), max_h)


def _diecut_cell_tile(
    sticker: Image.Image,
    cell_w: int,
    cell_h: int,
) -> Image.Image:
    """
    Identical free-transformed die-cut centered in a fixed cell_w×cell_h box.

    Inner inset keeps white rims from kissing neighbors while every grid
    slot stays the same size for perfect column/row alignment.
    """
    cw, ch = max(1, cell_w), max(1, cell_h)
    inset = max(4, int(min(cw, ch) * 0.04))
    fitted = _free_transform_sticker(
        sticker,
        max(1, cw - inset * 2),
        max(1, ch - inset * 2),
        cell=False,
    )
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    fw, fh = fitted.size
    canvas.alpha_composite(fitted, dest=((cw - fw) // 2, (ch - fh) // 2))
    return canvas


def _make_sticker_sheet_diecut_landscape(diecut: Image.Image) -> Image.Image:
    """
    Transparent-mode landscape sheet matching the store reference:
    paint-splatter page, 2×3 die-cuts beside brand logos, full-width 3×2
    bottom band — every hero sticker free-transformed to one uniform size
    per band so the grid reads perfectly aligned.
    """
    sheet, sheet_w, sheet_h = _sheet_canvas()
    aw, ah = max(1, diecut.size[0]), max(1, diecut.size[1])
    art_aspect = max(0.5, aw / float(ah))

    pad_x = int(sheet_w * 0.028)
    pad_y = int(sheet_h * 0.020)
    pad_bot = int(sheet_h * 0.022)
    gap_x = max(12, int(sheet_w * 0.014))
    gap_y = max(10, int(sheet_h * 0.010))
    section_gap = max(14, int(sheet_h * 0.014))
    side_gap = max(10, int(sheet_w * 0.012))
    side_w = int(sheet_w * 0.20)

    main_w = max(1, sheet_w - pad_x * 2 - side_gap - side_w)
    side_left = pad_x + main_w + side_gap
    full_w = max(1, sheet_w - pad_x * 2)
    usable_h = sheet_h - pad_y - pad_bot

    # Bottom 3×2 first — largest uniform cells that keep aspect.
    bot_cw = max(1, (full_w - gap_x * 2) // 3)
    bot_ch = max(1, int(round(bot_cw / art_aspect)))
    bot_grid_w = 3 * bot_cw + gap_x * 2
    bot_grid_h = 2 * bot_ch + gap_y

    top_budget = max(1, usable_h - bot_grid_h - section_gap)
    top_cw, top_ch = _cell_for_aspect(
        max(1, (main_w - gap_x) // 2),
        max(1, (top_budget - gap_y * 2) // 3),
        art_aspect,
    )
    top_grid_h = 3 * top_ch + gap_y * 2

    large = _diecut_cell_tile(diecut, top_cw, top_ch)
    small = _diecut_cell_tile(diecut, bot_cw, bot_ch)

    y_top = pad_y
    y_bot = y_top + top_grid_h + section_gap
    _paste_art_grid_xy(sheet, large, pad_x, y_top, 2, 3, gap_x, gap_y)
    bot_x = pad_x + max(0, (full_w - bot_grid_w) // 2)
    _paste_art_grid_xy(sheet, small, bot_x, y_bot, 3, 2, gap_x, gap_y)
    _paste_side_column(
        sheet,
        (side_left, pad_y, sheet_w - pad_x, y_bot - max(4, section_gap // 2)),
        mosida_frac=0.55,
        align="start",
    )
    print(
        f"🧾 diecut landscape top={top_cw}x{top_ch} bot={bot_cw}x{bot_ch} "
        f"aspect={art_aspect:.3f}"
    )
    return flatten_on_white(sheet)


def _make_sticker_sheet_diecut_portrait(diecut: Image.Image) -> Image.Image:
    """
    Transparent-mode portrait sheet: brand row, 2×2 die-cuts, side logos,
    full-width 3×1 bottom — uniform free-transform per band.
    """
    sheet, sheet_w, sheet_h = _sheet_canvas()
    aw, ah = max(1, diecut.size[0]), max(1, diecut.size[1])
    art_aspect = min(aw / float(ah), 0.92)

    top_brand = (0.03, 0.02, 0.97, 0.13)
    main = (0.04, 0.15, 0.76, 0.72)
    side = (0.78, 0.15, 0.96, 0.72)
    bottom = (0.04, 0.74, 0.96, 0.97)
    gap = max(12, int(min(sheet_w, sheet_h) * 0.016))

    def _box(norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = norm
        return (
            int(x0 * sheet_w),
            int(y0 * sheet_h),
            int(x1 * sheet_w),
            int(y1 * sheet_h),
        )

    _paste_portrait_top_brand(sheet, _box(top_brand))

    mx0, my0, mx1, my1 = _box(main)
    bx0, by0, bx1, by1 = _box(bottom)
    main_w, main_h = max(1, mx1 - mx0), max(1, my1 - my0)
    bot_w, bot_h = max(1, bx1 - bx0), max(1, by1 - by0)

    top_cw, top_ch = _grid_cell_size(main_w, main_h, 2, 2, gap, art_aspect)
    bot_cw, bot_ch = _grid_cell_size(bot_w, bot_h, 3, 1, gap, art_aspect)
    if bot_cw > int(top_cw * 0.92):
        scale = (top_cw * 0.92) / float(bot_cw)
        bot_cw = max(1, int(bot_cw * scale))
        bot_ch = max(1, int(bot_ch * scale))

    large = _diecut_cell_tile(diecut, top_cw, top_ch)
    small = _diecut_cell_tile(diecut, bot_cw, bot_ch)
    _paste_art_grid(sheet, large, (mx0, my0), (main_w, main_h), 2, 2, gap)
    _paste_art_grid(sheet, small, (bx0, by0), (bot_w, bot_h), 3, 1, gap)
    _paste_side_column(sheet, _box(side), mosida_frac=0.40, align="center")
    print(f"🧾 diecut portrait top={top_cw}x{top_ch} bot={bot_cw}x{bot_ch}")
    return flatten_on_white(sheet)


def _make_sticker_sheet_diecut_pack(diecut: Image.Image) -> Image.Image:
    """
    Fallback pack sheet: transparent canvas, 4 + 5 + 8 uniform rows.
    Each row free-transforms once so every sticker in that row matches.
    """
    sheet_w = 1400
    side_pad = int(sheet_w * 0.028)
    usable_w = sheet_w - side_pad * 2
    gap_frac = 0.045

    def _row(count: int, gap_f: float) -> Tuple[Image.Image, int]:
        count = max(1, count)
        denom = count + gap_f * max(0, count - 1)
        tw = max(1.0, usable_w / denom)
        tile = _free_transform_sticker(diecut, int(tw), int(tw * 2.2))
        # Re-fit width if height-bound shrunk it.
        if tile.size[0] != int(round(tw)):
            scale = tw / float(max(1, tile.size[0]))
            tile = tile.resize(
                (
                    max(1, int(round(tile.size[0] * scale))),
                    max(1, int(round(tile.size[1] * scale))),
                ),
                Image.Resampling.LANCZOS,
            )
        gap = max(4, int(round(tile.size[0] * gap_f)))
        return tile, gap

    large, gap_l = _row(4, gap_frac)
    medium, gap_m = _row(5, gap_frac)
    small, gap_s = _row(8, 0.040)

    row_gap = max(16, int(large.size[1] * 0.06))
    top_pad = max(28, int(large.size[1] * 0.08))
    content_h = large.size[1] + medium.size[1] + small.size[1] + row_gap * 2
    sheet_h = content_h + top_pad * 2

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))

    def _paste_row(tile: Image.Image, count: int, y: int, gap: int) -> None:
        tw = tile.size[0]
        total_w = count * tw + max(0, count - 1) * gap
        x = max(side_pad, (sheet_w - total_w) // 2)
        for i in range(count):
            sheet.alpha_composite(tile, dest=(x + i * (tw + gap), y))

    y = top_pad
    _paste_row(large, 4, y, gap_l)
    y += large.size[1] + row_gap
    _paste_row(medium, 5, y, gap_m)
    y += medium.size[1] + row_gap
    _paste_row(small, 8, y, gap_s)
    print(f"🧾 diecut pack 4+5+8 transparent {sheet_w}x{sheet_h}")
    return sheet


def _make_sticker_sheet_diecut(
    artwork: Image.Image,
    orientation: Optional[str] = None,
    *,
    pack_layout: bool = False,
) -> Image.Image:
    """
    Transparent-mode sticker sheet.

    Default: paint-splatter page with brand column + uniform die-cut grids
    (matches the Wynwood / MOSIDA sheet reference). Set pack_layout=True for
    a plain transparent 4+5+8 size pack with no brand marks.
    """
    diecut = build_diecut_sticker(
        artwork,
        stroke_frac=0.055,
        strip_logo=True,
    )
    if pack_layout:
        return _make_sticker_sheet_diecut_pack(diecut)

    orient = _normalize_orientation(orientation)
    use_portrait = (
        orient == "portrait"
        if orient is not None
        else _sheet_art_is_portrait(artwork)
    )
    if use_portrait:
        return _make_sticker_sheet_diecut_portrait(diecut)
    return _make_sticker_sheet_diecut_landscape(diecut)


def make_sticker_sheet(
    artwork: Image.Image,
    diecut: Optional[Image.Image] = None,
    orientation: Optional[str] = None,
    *,
    is_transparent: bool = False,
) -> Image.Image:
    """
    Sticker sheet layouts:
    - transparent → die-cut grids on paint page (2×3+3×2 or portrait) + logos
    - landscape art → rectangular 2×3 + full-width 3×2 bottom + right logos
    - portrait / square art → top brand row + 2×2 + 3×1 bottom + right column
    """
    del diecut
    art = _to_rgba(artwork)
    if is_transparent:
        print("🧾 Sticker sheet layout=transparent die-cut (aligned free-transform)")
        return _make_sticker_sheet_diecut(art, orientation=orientation)

    # Opaque sheet tiles keep the studio plate but never the baked-in mark.
    art = strip_mosida_for_opaque_print(art)
    orient = _normalize_orientation(orientation)
    use_portrait = (
        orient == "portrait"
        if orient is not None
        else _sheet_art_is_portrait(art)
    )
    if use_portrait:
        print("🧾 Sticker sheet layout=portrait (2×2 + 3×1)")
        return _make_sticker_sheet_portrait(art)
    print("🧾 Sticker sheet layout=landscape (2×3 + large full-width 3×2)")
    return _make_sticker_sheet_landscape(art)
