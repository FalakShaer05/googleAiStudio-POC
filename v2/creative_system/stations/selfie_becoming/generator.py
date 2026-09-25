"""Selfie Becoming / Me Remix — half color photo | half fine pencil line art."""
from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

from PIL import Image, ImageFilter, ImageOps

from utils.bg_remover import _decontaminate_cutout_edges, _get_rembg_session, _has_real_transparency, _rembg_model_names
from utils.character_utils import get_gemini_fallback_image_model

from ...shared.gemini import generate_composed_image, load_rgb, style_target_path, to_rgb
from .prompts import BACKGROUND_LOCK, STYLE_INSTRUCTION, build_line_art_prompt

# Final portrait canvas (3:4).
TARGET_W, TARGET_H = 768, 1024

# Style-target graphite: soft light strokes on pure white paper.
_GRAPHITE_DARK = 165.0
_GRAPHITE_LIGHT = 228.0
_PAPER_WHITE = 255

_STYLE_REF_INSTRUCTION = (
    "STYLE TARGET — match ONLY the right-half pencil technique: fine continuous "
    "light-gray graphite line art on pure white, delicate hair strands, clean "
    "feature contours, minimal shading. Trace the uploaded selfie in place; "
    "do NOT copy the reference person's face, hair, jewelry, or pose."
)


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


def _fast_image_model() -> Optional[str]:
    return get_gemini_fallback_image_model() or None


def _xdog_response(gray: Image.Image, sigma: float, k: float, tau: float) -> "np.ndarray":
    """Extended Difference-of-Gaussians response."""
    import numpy as np

    fine = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(radius=max(sigma, 0.3))),
        dtype=np.float32,
    )
    soft = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(radius=max(sigma * k, 0.6))),
        dtype=np.float32,
    )
    return fine - tau * soft


def _render_pencil_from_photo(photo: Image.Image) -> Image.Image:
    """
    Fallback light graphite line art traced from the photo pixels.
    Used when Gemini is unavailable — alignment is perfect by construction.
    """
    import numpy as np

    gray = ImageOps.grayscale(to_rgb(photo)).filter(ImageFilter.MedianFilter(size=3))
    g = np.asarray(gray, dtype=np.float32)
    subject = g < 250.0
    if int(subject.sum()) < 50:
        return Image.new("RGB", photo.size, (_PAPER_WHITE, _PAPER_WHITE, _PAPER_WHITE))

    energy = np.zeros_like(g)
    for sigma, k, tau, weight in (
        (0.40, 1.6, 0.98, 1.20),
        (0.70, 1.6, 0.97, 1.00),
        (1.20, 1.6, 0.97, 0.80),
        (2.00, 1.6, 0.96, 0.55),
    ):
        dog = _xdog_response(gray, sigma=sigma, k=k, tau=tau)
        energy = np.maximum(energy, np.clip(-dog, 0.0, None) * weight)

    energy[~subject] = 0.0
    vals = energy[subject]
    lo = float(np.percentile(vals, 62))
    hi = float(np.percentile(vals, 99))
    if hi <= lo + 1e-3:
        hi = lo + 1.0
    strength = np.clip((energy - lo) / (hi - lo), 0.0, 1.0)

    alive = strength > 0.12
    pad = np.pad(alive.astype(np.uint8), 1, mode="constant")
    neighbors = (
        pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
        + pad[1:-1, 0:-2] + pad[1:-1, 2:]
        + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
    )
    strength[(strength < 0.40) & (neighbors < 2)] = 0.0

    core = (strength > 0.35).astype(np.uint8) * 255
    widened = np.asarray(
        Image.fromarray(core, mode="L")
        .filter(ImageFilter.MaxFilter(size=3))
        .filter(ImageFilter.GaussianBlur(radius=0.55)),
        dtype=np.float32,
    ) / 255.0
    soft = np.clip(widened * 0.65 + strength * 0.70, 0.0, 1.0)
    soft = np.asarray(
        Image.fromarray((soft * 255.0).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=0.35)
        ),
        dtype=np.float32,
    ) / 255.0
    soft[~subject] = 0.0

    ink = _PAPER_WHITE - soft * (_PAPER_WHITE - _GRAPHITE_DARK)
    drawn = soft > 0.04
    ink = np.where(drawn, ink, float(_PAPER_WHITE))
    ink[drawn] = np.clip(
        ink[drawn] * 0.25 + 218.0 * 0.75 + (ink[drawn] - 218.0) * 0.20,
        170.0,
        255.0,
    )
    ink[ink >= 244] = _PAPER_WHITE
    ink[~subject] = _PAPER_WHITE
    return Image.fromarray(ink.astype(np.uint8), mode="L").convert("RGB")


def _whiten_paper(img: Image.Image) -> Image.Image:
    """Force near-white paper to pure white; keep graphite strokes."""
    import numpy as np

    arr = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
    paper = max(200.0, float(np.percentile(arr, 88)))
    out = np.clip(arr * (255.0 / paper), 0, 255)
    out[out >= 242] = 255
    return Image.fromarray(out.astype(np.uint8), mode="L").convert("RGB")


def _pencil_looks_usable(pencil: Image.Image, photo: Image.Image) -> bool:
    """
    Reject Gemini redraws that drifted: need enough ink, and the right-half
    silhouette should still overlap the photo silhouette.
    """
    import numpy as np

    if pencil.size != photo.size:
        return False
    width, height = photo.size
    mid = width // 2
    photo_r = np.asarray(ImageOps.grayscale(photo.crop((mid, 0, width, height))), dtype=np.float32)
    pen_r = np.asarray(ImageOps.grayscale(pencil.crop((mid, 0, width, height))), dtype=np.float32)
    photo_subj = photo_r < 248
    pen_ink = pen_r < 235
    if float(pen_ink.mean()) < 0.02:
        return False
    if float(pen_ink.mean()) > 0.55:
        return False  # charcoal slab / gray fill — not line art
    overlap = float((pen_ink & photo_subj).sum()) / max(float(pen_ink.sum()), 1.0)
    return overlap >= 0.55


def _align_pencil_to_photo(photo: Image.Image, pencil: Image.Image, limit: int = 8) -> Image.Image:
    """
    Tiny global translation of the pencil so contours meet the photo at the seam.
    Capped so a bad estimate cannot invent a new face.
    """
    import numpy as np

    if pencil.size != photo.size or limit <= 0:
        return pencil

    width, height = photo.size
    mid = width // 2
    # Match on a band around the seam + central face where glasses/nose live.
    x0 = max(0, mid - width // 5)
    x1 = min(width, mid + width // 5)
    y0 = int(height * 0.18)
    y1 = int(height * 0.72)

    def _edges(img: Image.Image) -> "np.ndarray":
        g = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1, :] = g[2:, :] - g[:-2, :]
        return np.hypot(gx, gy)

    a = _edges(photo)[y0:y1, x0:x1]
    b = _edges(pencil)[y0:y1, x0:x1]
    if a.size < 100 or b.size < 100:
        return pencil

    a = a - a.mean()
    b = b - b.mean()
    corr = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    peak_i = int(np.argmax(corr))
    peak_v = float(corr.flat[peak_i])
    if peak_v < float(corr.mean()) + 3.5 * float(corr.std()):
        return pencil
    peak = np.unravel_index(peak_i, corr.shape)
    dy = int(peak[0])
    dx = int(peak[1])
    h, w = corr.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    # corr peak at (dy,dx) means b shifted by that lands on a → move pencil by -dy,-dx
    dy, dx = -dy, -dx
    if abs(dy) > limit or abs(dx) > limit or (dy == 0 and dx == 0):
        return pencil

    canvas = Image.new("RGB", pencil.size, (255, 255, 255))
    canvas.paste(pencil, (dx, dy))
    return canvas


def composite_photo_line_split(photo: Image.Image, edited: Image.Image | None = None) -> Image.Image:
    """Hard mid cut: left = photo, right = whitened pencil (Gemini or photo-traced)."""
    photo_rgb = to_rgb(photo)
    width, height = photo_rgb.size
    mid = width // 2

    pencil: Optional[Image.Image] = None
    if edited is not None:
        edited_rgb = to_rgb(edited)
        if edited_rgb.size != photo_rgb.size:
            edited_rgb = edited_rgb.resize(photo_rgb.size, Image.Resampling.LANCZOS)
        edited_rgb = _align_pencil_to_photo(photo_rgb, edited_rgb)
        if _pencil_looks_usable(edited_rgb, photo_rgb):
            pencil = edited_rgb

    if pencil is None:
        pencil = _render_pencil_from_photo(photo_rgb)

    right = _whiten_paper(pencil.crop((mid, 0, width, height)))
    out = Image.new("RGB", (width, height), (255, 255, 255))
    out.paste(photo_rgb.crop((0, 0, mid, height)), (0, 0))
    out.paste(right, (mid, 0))
    return out


def generate_half_photo_half_line_art(
    output_path: str,
    selfie_path: str,
    *,
    style_target_id: str = "selfie-becoming",
    operation: str = "art_generation:creative:selfie-becoming",
) -> Tuple[bool, str]:
    """
    rembg photo on white → Gemini traces the FULL face into pencil line art in place
    (style-target technique) → hard split keeps the real photo on the left.

    Tracing the whole portrait (instead of only the right half) keeps features
    locked to the photo. If Gemini drifts or fails, fall back to photo-traced lines.
    """
    prepared = _cutout_on_white(_fit_portrait(load_rgb(selfie_path)))
    style_path = style_target_path(style_target_id) or style_target_path("me-remix")

    edited: Optional[Image.Image] = None
    with tempfile.TemporaryDirectory(prefix="cs_me_remix_") as tmp:
        edited_path = os.path.join(tmp, "line_art.png")
        ok, message = generate_composed_image(
            output_path=edited_path,
            prompt=build_line_art_prompt(),
            role_images=[
                (
                    "SOURCE SELFIE on white. Trace THIS exact person into fine light-gray "
                    "pencil line art on pure white. Keep every feature in the SAME place, "
                    "size, and pose — do not redraw a new face. "
                    f"{STYLE_INSTRUCTION}",
                    prepared,
                ),
            ],
            style_target=style_path,
            style_instruction=_STYLE_REF_INSTRUCTION,
            aspect_ratio="3:4",
            temperature=0.15,
            operation=operation,
            isolate_line_art=False,
            trailing_instruction=BACKGROUND_LOCK,
            image_size="1K",
            model=_fast_image_model(),
        )
        if ok:
            edited = Image.open(edited_path)
        else:
            print(f"me-remix: Gemini line art failed ({message}) — using photo-traced fallback")

    result = composite_photo_line_split(prepared, edited)
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
