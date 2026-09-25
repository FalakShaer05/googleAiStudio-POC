"""Selfie Becoming / Me Remix — half color photo | half fine pencil line art."""
from __future__ import annotations

import os
import tempfile
from typing import Tuple

from PIL import Image, ImageOps

from utils.bg_remover import _decontaminate_cutout_edges, _get_rembg_session, _has_real_transparency, _rembg_model_names
from utils.character_utils import get_gemini_fallback_image_model

from ...shared.gemini import generate_composed_image, load_rgb, to_rgb
from .prompts import BACKGROUND_LOCK, STYLE_INSTRUCTION, build_split_prompt

# Final portrait canvas (3:4).
TARGET_W, TARGET_H = 768, 1024


def _fit_portrait(img: Image.Image, size: Tuple[int, int] = (TARGET_W, TARGET_H)) -> Image.Image:
    """Cover-fit any upload (front / three-quarter / profile) onto the 3:4 canvas."""
    return ImageOps.fit(to_rgb(img), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.42))


def _cutout_on_white(img: Image.Image) -> Image.Image:
    """rembg cutout on pure white; falls back to original if rembg fails."""
    rgb = to_rgb(img)
    try:
        from rembg import remove
    except Exception as exc:
        print(f"selfie cutout: rembg unavailable ({exc}) — keeping original background")
        return rgb

    model_name = (_rembg_model_names() or ["u2net"])[0]
    try:
        session = _get_rembg_session(model_name)
        cut = remove(rgb, session=session)
        if not isinstance(cut, Image.Image):
            from io import BytesIO

            cut = Image.open(BytesIO(cut))
        cut = cut.convert("RGBA")
        if _has_real_transparency(cut):
            cut = _decontaminate_cutout_edges(cut)
            white = Image.new("RGB", cut.size, (255, 255, 255))
            white.paste(cut, mask=cut.split()[3])
            return white
        print(f"selfie cutout: rembg model={model_name} returned opaque alpha — keeping original")
    except Exception as exc:
        print(f"selfie cutout: rembg failed ({exc}) — keeping original background")
    return rgb


def _fast_image_model() -> str | None:
    return get_gemini_fallback_image_model() or None


def _whiten_paper(img: Image.Image) -> Image.Image:
    """
    Grayscale graphite with the paper tone pushed to pure white, so Gemini's
    gray "sketchbook paper" never shows up as a box on the right half.
    """
    import numpy as np

    arr = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
    paper = max(180.0, float(np.percentile(arr, 85)))
    out = np.clip(arr * (255.0 / paper), 0, 255)
    out[out >= 242] = 255
    return Image.fromarray(out.astype(np.uint8), mode="L").convert("RGB")


def _find_seam_x(edited: Image.Image) -> int:
    """
    Column where Gemini actually switched from color photo to pencil. It rarely
    cuts at exactly 50%, and cutting at our own midline leaves a gray strip of photo.
    """
    import numpy as np

    arr = np.asarray(edited, dtype=np.int16)
    width = arr.shape[1]
    mid = width // 2
    sat = (arr.max(axis=2) - arr.min(axis=2)).astype(np.float32)
    rows = sat.mean(axis=1) > 2.0
    if rows.sum() < 10:
        return mid
    col_sat = sat[rows].mean(axis=0)

    band = 6
    lo, hi = int(width * 0.40), int(width * 0.62)
    best_x, best_drop = mid, 0.0
    for x in range(max(band, lo), min(width - band, hi)):
        drop = float(col_sat[x - band:x].mean() - col_sat[x:x + band].mean())
        if drop > best_drop:
            best_drop, best_x = drop, x
    return best_x if best_drop >= 6.0 else mid


def _photo_contours(photo: Image.Image) -> "np.ndarray":
    """
    Light-gray line art taken from the photo itself, so every stroke sits on the
    real glasses, hair, nose, lips and beard. True where a contour should be drawn.
    """
    import numpy as np
    from PIL import ImageFilter

    gray = ImageOps.grayscale(photo)
    fine = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=0.6)), dtype=np.float32)
    soft = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32)
    # Difference-of-Gaussians: zero-crossings are the photo's own edges.
    dog = fine - soft
    mag = np.abs(dog)
    subject = fine < 250
    if int(subject.sum()) < 50:
        return np.zeros(fine.shape, dtype=bool)
    thresh = float(np.percentile(mag[subject], 86))
    lines = (mag >= max(thresh, 8.0)) & subject
    # Drop speckle: keep a pixel only if it has a few line neighbors.
    pad = np.pad(lines.astype(np.uint8), 1, mode="constant")
    neighbors = (
        pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
        + pad[1:-1, 0:-2] + pad[1:-1, 2:]
        + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
    )
    return lines & (neighbors >= 2)


def _lock_pencil_to_photo(photo: Image.Image, pencil: Image.Image) -> Image.Image:
    """
    Keep Gemini's pencil texture only where it already lies on the photo contour.
    Everywhere else, draw the photo contour so the face cannot drift.
    """
    import numpy as np

    contours = _photo_contours(photo)
    pen = np.asarray(ImageOps.grayscale(pencil), dtype=np.float32)
    height, width = contours.shape
    if pen.shape != contours.shape:
        return pencil

    # A few pixels of slack so a stroke that is almost on the edge still counts.
    pad = np.pad(contours.astype(np.uint8), 3, mode="constant")
    near = np.zeros_like(contours, dtype=bool)
    for dy in range(7):
        for dx in range(7):
            near |= pad[dy:dy + height, dx:dx + width].astype(bool)

    ink = pen < 235
    kept = ink & near
    out = np.full((height, width), 255, dtype=np.uint8)
    # Photo contour is the alignment lock (light graphite).
    out[contours] = 176
    # Gemini stroke wins when it sits on that contour, so the line stays a sketch.
    tone = np.clip(pen, 150, 210).astype(np.uint8)
    out[kept] = tone[kept]
    # Hair and beard: keep Gemini's hatching, but only inside the dark photo mass
    # and only if it is close to a real edge so it cannot redraw the silhouette.
    photo_g = np.asarray(ImageOps.grayscale(photo), dtype=np.float32)
    dark = photo_g < 110
    hatch = ink & dark & near
    out[hatch] = tone[hatch]
    return Image.fromarray(out, mode="L").convert("RGB")


def composite_photo_line_split(photo: Image.Image, edited: Image.Image) -> Image.Image:
    """Hard cut: left = original photo, right = pencil locked to the photo contours."""
    photo_rgb = to_rgb(photo)
    edited_rgb = to_rgb(edited)
    if edited_rgb.size != photo_rgb.size:
        edited_rgb = edited_rgb.resize(photo_rgb.size, Image.Resampling.LANCZOS)
    width, height = photo_rgb.size
    mid = _find_seam_x(edited_rgb)
    locked = _lock_pencil_to_photo(photo_rgb, edited_rgb)
    pencil_half = _whiten_paper(locked.crop((mid, 0, width, height)))
    out = Image.new("RGB", (width, height), (255, 255, 255))
    out.paste(photo_rgb.crop((0, 0, mid, height)), (0, 0))
    out.paste(pencil_half, (mid, 0))
    return out


def generate_half_photo_half_line_art(
    output_path: str,
    selfie_path: str,
    *,
    style_target_id: str = "selfie-becoming",
    operation: str = "art_generation:creative:selfie-becoming",
) -> Tuple[bool, str]:
    """
    rembg photo on white → Gemini edits ONLY the right half into pencil, in place,
    using the untouched left half as its alignment anchor → re-stamp the real photo
    on the left. One image edit instead of two separate drawings keeps the face aligned
    for any pose.
    """
    _ = style_target_id
    prepared = _cutout_on_white(_fit_portrait(load_rgb(selfie_path)))

    with tempfile.TemporaryDirectory(prefix="cs_me_remix_") as tmp:
        edited_path = os.path.join(tmp, "edited.png")

        ok, message = generate_composed_image(
            output_path=edited_path,
            prompt=build_split_prompt(),
            role_images=[
                (
                    "PHOTO TO EDIT IN PLACE. Keep the left half exactly as-is. "
                    "Convert only the right half into pencil line art traced directly over "
                    "the photo, so the face continues seamlessly across the center line. "
                    f"{STYLE_INSTRUCTION}",
                    prepared,
                ),
            ],
            style_target=None,
            aspect_ratio="3:4",
            temperature=0.1,
            operation=operation,
            isolate_line_art=False,
            trailing_instruction=BACKGROUND_LOCK,
            image_size="1K",
            model=_fast_image_model(),
        )
        if not ok:
            return False, message

        result = composite_photo_line_split(prepared, Image.open(edited_path))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.save(output_path, format="PNG", optimize=True)
    return True, "Artwork generated successfully"


def generate(output_path: str, selfie_path: str, **_kwargs):
    return generate_half_photo_half_line_art(
        output_path=output_path,
        selfie_path=selfie_path,
        style_target_id="selfie-becoming",
        operation="art_generation:creative:selfie-becoming",
    )
