"""Typography layout for the Holding Hands print, matched to the reference artwork.

All coordinates are in plate space (928 x 1152, 4:5). The Gemini hands image is resized
to that canvas, cut out of its cream background, and the lettering is drawn on top.
The result is an RGBA PNG with a fully transparent background.
"""
from __future__ import annotations

import math
import os
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from ...shared.fonts import FONTS_DIR, font_path

CANVAS = (928, 1152)
CREAM = (253, 251, 238)
INK = (17, 17, 17)
SS = 2

DATE_FONT = os.path.join(FONTS_DIR, "Poppins-Regular.ttf")

# Where the arm outlines begin on the plate. The name strokes flow into these points.
LEFT_ARM_TOP = (373, 322)
RIGHT_ARM_TOP = (579, 286)
LEFT_ARM_DIR = (0.7, 0.7)
RIGHT_ARM_DIR = (-0.7, 0.7)

LEFT_NAME = {"start": (150, 115), "angle": 52.0, "max_len": 255}
RIGHT_NAME = {"start": (628, 262), "angle": -32.0, "max_len": 250}
LEFT_EDGE_ENTRY = (72, -6)
RIGHT_EDGE_EXIT = (866, -6)

NAME_SIZE = 104
NAME_MIN_SIZE = 54
PHRASE_SIZE = 150
PHRASE_MIN_SIZE = 64
PHRASE_MAX_WIDTH = 600
PHRASE_CENTER_X = 464
PHRASE_GAP = 14
DATE_GAP = 30
DATE_BOTTOM_MARGIN = 70
HANDS_BOTTOM = 842
FADE_CUT_LEFT = 322
FADE_CUT_RIGHT = 286
FADE_SPLIT = (420, 520)
FADE_LENGTH = 80
FADE_DEPTH = 150
DATE_CAP_SIZE = 34
DATE_TRACKING_EM = 0.3


Point = Tuple[float, float]


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def pretty_date(date_text: str) -> str:
    text = (date_text or "").strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            return f"{parsed.strftime('%B').upper()} {parsed.day}, {parsed.year}"
        except ValueError:
            continue
    return text.upper()


def _fit_font(path: str, text: str, size: int, min_size: int, max_width: float) -> ImageFont.FreeTypeFont:
    font = _font(path, size)
    while size > min_size and font.getlength(text) > max_width:
        size -= 2
        font = _font(path, size)
    return font


def _rotate_point(p: Point, center: Point, new_center: Point, angle_deg: float) -> Point:
    theta = math.radians(angle_deg)
    dx, dy = p[0] - center[0], p[1] - center[1]
    return (
        new_center[0] + dx * math.cos(theta) + dy * math.sin(theta),
        new_center[1] - dx * math.sin(theta) + dy * math.cos(theta),
    )


def _text_layer(text: str, font: ImageFont.FreeTypeFont):
    """Render text on a transparent layer and return (layer, start, end) at mid-x-height."""
    left, top, right, bottom = font.getbbox(text)
    pad = int(font.size * 0.4)
    layer = Image.new("L", (right - left + pad * 2, bottom - top + pad * 2), 0)
    ImageDraw.Draw(layer).text((pad - left, pad - top), text, font=font, fill=255)
    ascent, _ = font.getmetrics()
    baseline = pad - top + ascent
    mid = baseline - font.size * 0.14
    return layer, (pad, mid), (pad + right - left, mid)


def _paste_rotated_name(
    overlay: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    start: Point,
    angle_deg: float,
) -> Tuple[Point, Point]:
    layer, s_pt, e_pt = _text_layer(text, font)
    center = (layer.width / 2, layer.height / 2)
    rotated = layer.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)
    new_center = (rotated.width / 2, rotated.height / 2)
    s_rot = _rotate_point(s_pt, center, new_center, -angle_deg)
    e_rot = _rotate_point(e_pt, center, new_center, -angle_deg)
    offset = (round(start[0] - s_rot[0]), round(start[1] - s_rot[1]))
    overlay.paste(INK + (255,), offset, rotated)
    return (
        (offset[0] + s_rot[0], offset[1] + s_rot[1]),
        (offset[0] + e_rot[0], offset[1] + e_rot[1]),
    )


def _bezier(p0: Point, p1: Point, p2: Point, p3: Point, steps: int):
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        yield (
            u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1],
            t,
        )


def _stroke(
    draw: ImageDraw.ImageDraw,
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    width: float,
    taper_start: bool = False,
    taper_end: bool = False,
) -> None:
    length = math.dist(p0, p3) + math.dist(p1, p2)
    steps = max(40, int(length / 2))
    for x, y, t in _bezier(p0, p1, p2, p3, steps):
        w = width
        if taper_start:
            w *= min(1.0, 0.25 + t * 4)
        if taper_end:
            w *= min(1.0, 0.25 + (1 - t) * 4)
        r = max(w / 2, 0.6 * SS)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=INK + (255,))


def _scale(p: Point) -> Point:
    return (p[0] * SS, p[1] * SS)


def _toward(p: Point, direction: Point, dist: float) -> Point:
    return (p[0] + direction[0] * dist, p[1] + direction[1] * dist)


def _unit(angle_deg: float) -> Point:
    theta = math.radians(angle_deg)
    return (math.cos(theta), math.sin(theta))


def _find_arm_top(
    img: np.ndarray,
    default: Point,
    box: Tuple[int, int, int, int],
    inner: str,
) -> Point:
    """Top of the arm's inner outline (the edge facing the center), where the name stroke joins."""
    x0, y0, x1, y1 = box
    region = img[y0:y1, x0:x1, :3].mean(axis=2) < 110
    ys, xs = np.where(region)
    if len(ys) < 8:
        return default
    near_top = ys <= ys.min() + 25
    idx = np.argmax(xs[near_top]) if inner == "right" else np.argmin(xs[near_top])
    return (x0 + float(xs[near_top][idx]), y0 + float(ys[near_top][idx]))


def cut_out_hands(img: Image.Image) -> Tuple[Image.Image, int]:
    """Remove the cream background so only the hands remain. Returns (RGBA image, hands bottom y)."""
    rgb = np.array(img.convert("RGB").resize(CANVAS, Image.LANCZOS)).astype(np.float32)
    cream = np.array(CREAM, dtype=np.float32)
    lum = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    bg_like = (lum >= 232) & (sat <= 28)

    # Background is the cream reachable from the canvas edge. Eroding first keeps the fill
    # from leaking through small gaps in the outlines into highlights and white nails.
    sealed = ndimage.binary_erosion(bg_like, iterations=2, border_value=1)
    labels, _ = ndimage.label(sealed)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, edge_labels[edge_labels > 0])
    outside = ndimage.binary_dilation(outside, iterations=3) & bg_like
    hands = ~outside

    ys, xs = np.where(hands[:, 120:-120])
    hands_bottom = HANDS_BOTTOM
    if len(ys) > 500:
        top, bottom = ys.min() - 12, ys.max() + 18
        left, right = xs.min() + 120 - 18, xs.max() + 120 + 18
        box = np.zeros_like(hands)
        box[max(top, 0):bottom, max(left, 0):right] = True
        hands &= box
        hands_bottom = int(ys.max())

    distance = np.abs(rgb - cream).max(axis=2)
    alpha = hands.astype(np.float32)

    # Anti-aliased outline edges are black blended with cream: make them black with partial opacity.
    rim = hands & ndimage.binary_dilation(outside, iterations=3)
    ink_edge = rim & (sat < 18)
    alpha[ink_edge] = np.clip(1 - lum[ink_edge] / cream.mean(), 0, 1)
    skin_edge = rim & ~ink_edge
    alpha[skin_edge] = np.clip(distance[skin_edge] / 45.0, 0, 1)

    safe = np.maximum(alpha, 0.05)[..., None]
    unmixed = np.clip(cream + (rgb - cream) / safe, 0, 255)
    rgb = np.where((skin_edge & (alpha > 0) & (alpha < 1))[..., None], unmixed, rgb)
    rgb[ink_edge] = INK

    _fade_arm_tops(alpha, hands, lum < 110)

    rgba = np.dstack([rgb, alpha * 255]).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA"), hands_bottom


def _fade_arm_tops(alpha: np.ndarray, hands: np.ndarray, ink: np.ndarray) -> None:
    """Clear above the arm-top line, then fade each arm in from transparent.

    Ink outlines inside the fade stay fully opaque so they meet the name strokes."""
    h, w = alpha.shape
    xs = np.arange(w)
    t = np.clip((xs - FADE_SPLIT[0]) / (FADE_SPLIT[1] - FADE_SPLIT[0]), 0, 1)
    cut = (FADE_CUT_LEFT + (FADE_CUT_RIGHT - FADE_CUT_LEFT) * t).astype(int)
    ys = np.arange(h)[:, None]
    below_cut = ys >= cut[None, :]
    alpha[~below_cut] = 0

    skin = hands & ~ink & below_cut
    has_skin = skin.any(axis=0)
    if not has_skin.any():
        return
    raw_top = np.where(has_skin, np.argmax(skin, axis=0), h).astype(np.float32)
    top = raw_top.copy()
    top[has_skin] = ndimage.median_filter(raw_top[has_skin], size=31, mode="nearest")
    top[has_skin] = ndimage.gaussian_filter1d(top[has_skin], sigma=6, mode="nearest")

    # Opacity ramps up over FADE_LENGTH rows below each arm's top, then hands back to the
    # original opacity before FADE_DEPTH.
    rel = ys - top[None, :]
    ramp = np.clip(rel / FADE_LENGTH, 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)
    handoff = np.clip((rel - (FADE_DEPTH - 40)) / 40.0, 0, 1)
    in_fade = below_cut & (rel < FADE_DEPTH)

    # The faint fade sits right on the background threshold, so its silhouette is ragged:
    # close gaps and blur it into a soft edge.
    closed = ndimage.binary_closing(hands, structure=np.ones((3, 3)), iterations=4)
    closed = ndimage.binary_opening(closed, structure=np.ones((3, 3)), iterations=2)
    silhouette = np.clip(ndimage.gaussian_filter(closed.astype(np.float32), sigma=2.5) * 1.15, 0, 1)
    zone = (silhouette > 0.01) & ~ink & in_fade
    alpha[zone] = ((ramp * silhouette) * (1 - handoff) + alpha * handoff)[zone]

    # Long outlines pass through the fade and meet the name strokes; ragged fragments fade out.
    labels, count = ndimage.label(ink & below_cut, structure=np.ones((3, 3)))
    if count:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
        small = np.isin(labels, np.where(sizes < 150)[0] + 1)
        fragments = small & in_fade
        alpha[fragments] *= ramp[fragments]


def compose_artwork(
    hands: Image.Image,
    name_a: str,
    name_b: str,
    caption: str,
    date_text: str,
    font_id: str,
) -> Image.Image:
    base, hands_bottom = cut_out_hands(hands)
    arr = np.array(base).astype(np.int16)
    arr[arr[..., 3] < 128, :3] = 255
    left_top = _find_arm_top(arr, LEFT_ARM_TOP, (300, FADE_CUT_LEFT, 480, 440), inner="right")
    right_top = _find_arm_top(arr, RIGHT_ARM_TOP, (480, FADE_CUT_RIGHT, 660, 390), inner="left")

    script = font_path(font_id)
    overlay = Image.new("RGBA", (CANVAS[0] * SS, CANVAS[1] * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_w = 4.2 * SS

    for name, spec, side in ((name_a, LEFT_NAME, "left"), (name_b, RIGHT_NAME, "right")):
        font = _fit_font(script, name, NAME_SIZE * SS, NAME_MIN_SIZE * SS, spec["max_len"] * SS)
        start, end = _paste_rotated_name(overlay, name, font, _scale(spec["start"]), spec["angle"])
        along = _unit(spec["angle"])
        if side == "left":
            entry = _scale(LEFT_EDGE_ENTRY)
            _stroke(draw, entry, (entry[0] + 4 * SS, entry[1] + 70 * SS),
                    _toward(start, along, -60 * SS), start, line_w, taper_start=True)
            arm = _scale(left_top)
            _stroke(draw, end, _toward(end, along, 45 * SS),
                    _toward(arm, LEFT_ARM_DIR, -55 * SS), arm, line_w, taper_end=True)
        else:
            arm = _scale(right_top)
            _stroke(draw, arm, _toward(arm, RIGHT_ARM_DIR, -45 * SS),
                    _toward(start, along, -40 * SS), start, line_w, taper_start=True)
            exit_pt = _scale(RIGHT_EDGE_EXIT)
            _stroke(draw, end, _toward(end, along, 55 * SS),
                    (exit_pt[0] - 4 * SS, exit_pt[1] + 70 * SS), exit_pt, line_w, taper_end=True)

    date_text = (date_text or "").strip()
    date_font = _font(DATE_FONT, DATE_CAP_SIZE * SS)
    date_cap = date_font.getbbox("S")
    date_h = (date_cap[3] - date_cap[1]) if date_text else 0
    text_bottom_limit = (CANVAS[1] - DATE_BOTTOM_MARGIN) * SS - (date_h + DATE_GAP * SS if date_text else 0)
    date_top = (hands_bottom + PHRASE_GAP + 40) * SS

    caption = (caption or "").strip()
    if caption:
        font = _fit_font(script, caption, PHRASE_SIZE * SS, PHRASE_MIN_SIZE * SS, PHRASE_MAX_WIDTH * SS)
        left, top, right, bottom = font.getbbox(caption)
        size = font.size
        while size > PHRASE_MIN_SIZE * SS and (hands_bottom + PHRASE_GAP) * SS + (bottom - top) > text_bottom_limit:
            size -= 2 * SS
            font = _font(script, size)
            left, top, right, bottom = font.getbbox(caption)
        x = PHRASE_CENTER_X * SS - (right - left) / 2 - left
        y = (hands_bottom + PHRASE_GAP) * SS - top
        date_top = y + bottom + DATE_GAP * SS
        draw.text((x, y), caption, font=font, fill=INK + (255,))
        ascent, _ = font.getmetrics()
        swash_y = y + ascent - font.size * 0.2
        ink_left, ink_right = x + left, x + right
        _stroke(draw, (-10 * SS, swash_y + 16 * SS), (ink_left * 0.35, swash_y + 18 * SS),
                (ink_left * 0.75, swash_y + 2 * SS), (ink_left + 6 * SS, swash_y), line_w * 0.9,
                taper_start=True)
        far = CANVAS[0] * SS * 0.95
        _stroke(draw, (ink_right - 6 * SS, swash_y), (ink_right + (far - ink_right) * 0.3, swash_y + 12 * SS),
                (ink_right + (far - ink_right) * 0.7, swash_y + 10 * SS), (far, swash_y - 2 * SS),
                line_w * 0.9, taper_end=True)

    if date_text:
        label = pretty_date(date_text)
        tracking = date_font.size * DATE_TRACKING_EM
        widths = [date_font.getlength(ch) for ch in label]
        total = sum(widths) + tracking * (len(label) - 1)
        x = CANVAS[0] * SS / 2 - total / 2
        top = min(date_top, (CANVAS[1] - DATE_BOTTOM_MARGIN) * SS - date_h)
        y = top - date_cap[1]
        for ch, w in zip(label, widths):
            draw.text((x, y), ch, font=date_font, fill=INK + (255,))
            x += w + tracking

    overlay = overlay.resize(CANVAS, Image.LANCZOS)
    base.alpha_composite(overlay)
    return base


def _asset(name: str) -> Optional[str]:
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "static", "images", "style_targets", name
    ))
    return path if os.path.isfile(path) else None


def plate_path() -> Optional[str]:
    return _asset("holding-hands-plate.png")


def hand_map_path() -> Optional[str]:
    return _asset("holding-hands-hand-map.png")
