"""Fast PIL mockups: print artwork onto stored product blanks."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

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


def prepare_print_art(img: Image.Image, knockout: bool = True) -> Image.Image:
    art = trim_artwork(img)
    if knockout:
        art = knockout_paper(art)
        art = trim_artwork(art, pad=2)
        return art
    # Apparel / prints: keep a solid opaque rectangle so stretch/cover
    # fills every edge of the print box (no floating transparent card).
    return _to_rgb(art).convert("RGBA")


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
) -> Image.Image:
    """Paste trimmed artwork into a normalized print box on the blank product."""
    base = _to_rgba(template)
    art = prepare_print_art(artwork, knockout=knockout)
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


def make_photo_print(artwork: Image.Image, template: Image.Image | None = None) -> Image.Image:
    """
    Photo print: artwork on a white mat, flat on a light grey studio backdrop.
    No drop shadow or dark outline.
    """
    del template  # Build a matted photo print; ignore blank template.
    art = trim_artwork(artwork, pad=2)
    art = _to_rgb(art).convert("RGBA")
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


def make_canvas_print(artwork: Image.Image, template: Image.Image | None = None) -> Image.Image:
    """
    Poster: large full-bleed artwork on a light grey studio backdrop.
    No drop shadow, no black border, no white mat.
    """
    del template
    art = trim_artwork(artwork, pad=2)
    art = _to_rgb(art).convert("RGBA")
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
    """Die-cut white border around the artwork silhouette."""
    rgba = _to_rgba(art)
    # Grow canvas so the border isn't clipped at the edges.
    pad = px + 4
    canvas = Image.new("RGBA", (rgba.size[0] + pad * 2, rgba.size[1] + pad * 2), (0, 0, 0, 0))
    canvas.paste(rgba, (pad, pad), rgba)
    alpha = canvas.split()[3]
    # MaxFilter kernel must be odd.
    kernel = max(3, px * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    outline = alpha.filter(ImageFilter.MaxFilter(kernel))
    # Soften the rim slightly for a cleaner die-cut edge.
    outline = outline.filter(ImageFilter.GaussianBlur(0.6))
    sticker = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    sticker.paste(Image.new("RGBA", canvas.size, (255, 255, 255, 255)), mask=outline)
    sticker.alpha_composite(canvas)
    return sticker


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


def build_diecut_sticker(artwork: Image.Image, stroke_frac: float = 0.045) -> Image.Image:
    """Knock out paper and add a thick white die-cut rim (no shadow yet)."""
    # Keep rectangular prints as solid cards with a white rim (don't punch
    # holes through light areas of the art — that breaks Wynwood-style scenes).
    art = trim_artwork(artwork, pad=4)
    # Cap size before MaxFilter — large kernels on big images are very slow.
    if max(art.size) > 720:
        art = art.copy()
        art.thumbnail((720, 720), Image.Resampling.LANCZOS)
    rgba = _to_rgba(art)
    alpha = rgba.split()[3]
    opaque_frac = 0.0
    if NUMPY_AVAILABLE:
        opaque_frac = float((np.asarray(alpha) > 200).mean())
    if opaque_frac < 0.85:
        # Cutout / character art — knock out studio white, then rim silhouette.
        art = prepare_print_art(art, knockout=True)
        if max(art.size) > 720:
            art = art.copy()
            art.thumbnail((720, 720), Image.Resampling.LANCZOS)
    else:
        art = _to_rgb(art).convert("RGBA")

    # Keep stroke modest so MaxFilter stays fast.
    stroke = max(10, min(22, int(round(min(art.size) * stroke_frac))))
    return _white_stroke(art, stroke)


def _rect_sticker_card(
    artwork: Image.Image,
    max_side: int = 900,
) -> Image.Image:
    """Full rectangular artwork with a thin dark edge so stacked stickers read clearly."""
    art = trim_artwork(artwork, pad=2)
    # Flatten to an opaque card so splash backgrounds / white highlights stay.
    art = _to_rgb(art).convert("RGBA")
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


def make_sticker_single(
    artwork: Image.Image,
    diecut: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Single sticker product: full rectangular artwork with a thin dark edge,
    soft drop shadow — matches store listing style.
    """
    del diecut  # Don't use silhouette die-cut; keep the full rectangular scene.
    sticker = _rect_sticker_card(artwork, max_side=900)
    sticker = _sticker_shadow(sticker, blur=14, offset=(6, 9))

    # Studio-style white margin around the product.
    pad = max(48, min(sticker.size) // 10)
    canvas = Image.new(
        "RGBA",
        (sticker.size[0] + pad * 2, sticker.size[1] + pad * 2),
        (255, 255, 255, 255),
    )
    canvas.paste(sticker, (pad, pad), sticker)
    return flatten_on_white(canvas)


def make_sticker_fan(
    artwork: Image.Image,
    count: int = 5,
    diecut: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Pack mockup: identical rectangular stickers stacked diagonally
    (back-left → front-right) with drop shadows — store listing style.
    """
    del diecut
    card = _rect_sticker_card(artwork, max_side=520)
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


def _rounded_rect_mask(size: Tuple[int, int], radius: int) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    r = max(0, min(radius, w // 2, h // 2))
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
    return mask


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int, int],
    width: int = 1,
    dash: int = 7,
    gap: int = 5,
) -> None:
    """Kiss-cut / die guide line around a sticker tile."""
    x0, y0, x1, y1 = box

    def _segments(a: int, b: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        pos = a
        while pos < b:
            end = min(pos + dash, b)
            out.append((pos, end))
            pos = end + gap
        return out

    for xa, xb in _segments(x0, x1):
        draw.line([(xa, y0), (xb, y0)], fill=color, width=width)
        draw.line([(xa, y1), (xb, y1)], fill=color, width=width)
    for ya, yb in _segments(y0, y1):
        draw.line([(x0, ya), (x0, yb)], fill=color, width=width)
        draw.line([(x1, ya), (x1, yb)], fill=color, width=width)


def _kiss_cut_rect_sticker(artwork: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """
    Rectangular sticker tile: opaque art, thin solid edge, dashed cut line.
    Matches store / Canva sticker-sheet layout.
    """
    art = trim_artwork(artwork, pad=2)
    art = _to_rgb(art).convert("RGBA")
    # Fit whole art inside the cell (no crop).
    fitted = _fit(art, (max_w, max_h), "contain")
    bbox = fitted.getbbox() or (0, 0, fitted.size[0], fitted.size[1])
    tile = fitted.crop(bbox)

    # Solid border on the art edge.
    border = max(2, min(4, min(tile.size) // 90))
    cut_gap = max(6, min(10, min(tile.size) // 40))
    dash_pad = cut_gap + 2
    canvas = Image.new(
        "RGBA",
        (tile.size[0] + dash_pad * 2, tile.size[1] + dash_pad * 2),
        (0, 0, 0, 0),
    )
    ox, oy = dash_pad, dash_pad
    canvas.paste(tile, (ox, oy), tile)

    draw = ImageDraw.Draw(canvas)
    # Thin dark outline on the sticker face.
    draw.rectangle(
        (ox, oy, ox + tile.size[0] - 1, oy + tile.size[1] - 1),
        outline=(40, 40, 40, 220),
        width=border,
    )
    # Dashed kiss-cut guide just outside the sticker.
    _draw_dashed_rect(
        draw,
        (
            ox - cut_gap,
            oy - cut_gap,
            ox + tile.size[0] - 1 + cut_gap,
            oy + tile.size[1] - 1 + cut_gap,
        ),
        color=(150, 150, 150, 200),
        width=1,
        dash=8,
        gap=5,
    )
    return canvas


def make_sticker_sheet(
    artwork: Image.Image,
    diecut: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Sticker sheet like the store layout: 3 large + 4 medium + 4 medium
    rectangular kiss-cut stickers on a white backing card.
    """
    del diecut  # Sheet uses rectangular kiss-cut tiles, not silhouette die-cuts.
    art = trim_artwork(artwork, pad=2)

    sheet_w = 1100
    pad_x, pad_y = 56, 56
    gap_x, gap_y = 28, 36
    usable_w = sheet_w - pad_x * 2

    # Top: 3 large; middle+bottom: 4 medium each (same size).
    large_w = (usable_w - gap_x * 2) // 3
    med_w = (usable_w - gap_x * 3) // 4
    aw, ah = max(1, art.size[0]), max(1, art.size[1])
    large_h = int(round(large_w * ah / aw))
    med_h = int(round(med_w * ah / aw))

    # Build one tile per size so sheet height fits cut-line padding exactly.
    large_tile = _kiss_cut_rect_sticker(art, large_w, large_h)
    med_tile = _kiss_cut_rect_sticker(art, med_w, med_h)
    rows = [
        (3, large_tile),
        (4, med_tile),
        (4, med_tile),
    ]
    content_h = sum(tile.size[1] for _, tile in rows) + gap_y * (len(rows) - 1)
    sheet_h = content_h + pad_y * 2

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (255, 255, 255, 255))
    y = pad_y
    for count, proto in rows:
        tw, th = proto.size
        row_w = count * tw + (count - 1) * gap_x
        x = max(0, (sheet_w - row_w) // 2)
        for _ in range(count):
            sheet.alpha_composite(proto, dest=(x, y))
            x += tw + gap_x
        y += th + gap_y

    # Soft rounded corners + faint edge so the sheet reads as a cut card.
    radius = 14
    mask = _rounded_rect_mask((sheet_w, sheet_h), radius)
    sheet.putalpha(mask)
    edge = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    ImageDraw.Draw(edge).rounded_rectangle(
        (0, 0, sheet_w - 1, sheet_h - 1),
        radius=radius,
        outline=(210, 210, 210, 255),
        width=2,
    )
    sheet.alpha_composite(edge)

    # Studio backdrop + card shadow — distinct from page white.
    frame_pad = 64
    out_w = sheet_w + frame_pad * 2
    out_h = sheet_h + frame_pad * 2
    backdrop = Image.new("RGBA", (out_w, out_h), (236, 236, 238, 255))

    shadow = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sx0 = frame_pad + 10
    sy0 = frame_pad + 14
    sdraw.rounded_rectangle(
        (sx0, sy0, sx0 + sheet_w, sy0 + sheet_h),
        radius=radius + 2,
        fill=(0, 0, 0, 55),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    backdrop.alpha_composite(shadow)
    backdrop.alpha_composite(sheet, dest=(frame_pad, frame_pad))
    return flatten_on_white(backdrop)
