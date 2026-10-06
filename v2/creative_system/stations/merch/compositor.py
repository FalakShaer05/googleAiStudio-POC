"""Fast PIL mockups: print artwork onto stored product blanks."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

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
    """Photo print — art keeps aspect ratio, fills the card as much as possible."""
    art = trim_artwork(artwork)
    if template is not None:
        return composite_on_template(
            template,
            art,
            print_box=detect_card_box(template),
            fit="cover",
            opacity=1.0,
            shade=False,
            knockout=False,
            match_art_aspect=True,
        )
    w, h = 1600, 1067
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    card = (90, 210, w - 90, h - 210)
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rectangle((card[0] + 10, card[1] + 12, card[2] + 10, card[3] + 12), fill=(0, 0, 0, 50))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    canvas.paste(shadow.convert("RGB"), mask=shadow.split()[3])
    filled = _fit(art, (card[2] - card[0], card[3] - card[1]), "contain")
    canvas.paste(flatten_on_white(filled), (card[0], card[1]))
    return canvas


def make_canvas_print(artwork: Image.Image, template: Image.Image | None = None) -> Image.Image:
    """Square canvas — art keeps aspect ratio, centered, no hands or props."""
    art = trim_artwork(artwork)
    size = 1400
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    margin = 70
    card = (margin, margin, size - margin, size - margin)
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rectangle((card[0] + 14, card[1] + 18, card[2] + 14, card[3] + 18), fill=(0, 0, 0, 55))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    canvas.paste(shadow.convert("RGB"), mask=shadow.split()[3])
    # Cover the canvas face without distorting proportions (may crop edges).
    filled = _fit(art, (card[2] - card[0], card[3] - card[1]), "cover").convert("RGB")
    canvas.paste(filled, (card[0], card[1]))
    return canvas


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


def _build_diecut_sticker(artwork: Image.Image, stroke_frac: float = 0.045) -> Image.Image:
    """Knock out paper, add a thick white die-cut rim, then a soft shadow."""
    # Keep rectangular prints as solid cards with a white rim (don't punch
    # holes through light areas of the art — that breaks Wynwood-style scenes).
    art = trim_artwork(artwork, pad=4)
    # If the art is mostly opaque rectangle, keep it opaque and rim the outer edge.
    rgba = _to_rgba(art)
    alpha = rgba.split()[3]
    opaque_frac = 0.0
    if NUMPY_AVAILABLE:
        opaque_frac = float((np.asarray(alpha) > 200).mean())
    if opaque_frac < 0.85:
        # Cutout / character art — knock out studio white, then rim silhouette.
        art = prepare_print_art(artwork, knockout=True)
    else:
        art = _to_rgb(art).convert("RGBA")

    stroke = max(14, int(round(min(art.size) * stroke_frac)))
    return _white_stroke(art, stroke)


def make_sticker_single(artwork: Image.Image) -> Image.Image:
    sticker = _build_diecut_sticker(artwork, stroke_frac=0.05)
    sticker = _sticker_shadow(sticker, blur=12, offset=(5, 8))
    # Extra white margin around the product shot.
    pad = max(36, min(sticker.size) // 12)
    canvas = Image.new("RGBA", (sticker.size[0] + pad * 2, sticker.size[1] + pad * 2), (255, 255, 255, 255))
    canvas.paste(sticker, (pad, pad), sticker)
    return flatten_on_white(canvas)


def make_sticker_fan(artwork: Image.Image, count: int = 5) -> Image.Image:
    sticker = _build_diecut_sticker(artwork, stroke_frac=0.048)
    sticker.thumbnail((440, 560), Image.Resampling.LANCZOS)
    # Rebuild a lighter shadow after resize so the rim stays crisp.
    sticker = _sticker_shadow(sticker, blur=8, offset=(3, 5))
    sw, sh = sticker.size
    step_x, step_y = int(sw * 0.16), int(sh * 0.035)
    margin = 48
    w = sw + step_x * (count - 1) + margin * 2
    h = sh + abs(step_y) * (count - 1) + margin * 2
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    for i in range(count):
        x = margin + i * step_x
        y = margin + (count - 1 - i) * step_y
        canvas.alpha_composite(sticker, dest=(x, y))
    return flatten_on_white(canvas)


def make_sticker_sheet(artwork: Image.Image) -> Image.Image:
    sticker = _build_diecut_sticker(artwork, stroke_frac=0.05)
    canvas = Image.new("RGBA", (1400, 1000), (255, 255, 255, 255))
    rows = [(4, 340), (5, 230), (8, 140)]
    y = 36
    for count, height in rows:
        item = sticker.copy()
        item.thumbnail((int(height * 0.9), height), Image.Resampling.LANCZOS)
        item = _sticker_shadow(item, blur=6, offset=(2, 3))
        gap = 28
        row_w = count * item.size[0] + (count - 1) * gap
        x = max(16, (canvas.size[0] - row_w) // 2)
        for _ in range(count):
            canvas.alpha_composite(item, dest=(x, y))
            x += item.size[0] + gap
        y += item.size[1] + 22
    return flatten_on_white(canvas)
