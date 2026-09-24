"""Selfie Becoming — half color photo (no bg) | half pencil sketch."""
from __future__ import annotations

import os
import tempfile
from typing import Tuple

from PIL import Image, ImageOps

from utils.bg_remover import _decontaminate_cutout_edges, _get_rembg_session, _has_real_transparency, _rembg_model_names
from utils.character_utils import get_gemini_fallback_image_model

from ...shared.gemini import generate_composed_image, load_rgb, style_target_path, to_rgb
from .prompts import BACKGROUND_LOCK, STYLE_INSTRUCTION, build_split_prompt

# Final portrait canvas (3:4). Gemini sees a smaller copy for speed.
TARGET_W, TARGET_H = 768, 1024
GEMINI_W, GEMINI_H = 512, 683


def _fit_portrait(img: Image.Image, size: Tuple[int, int] = (TARGET_W, TARGET_H)) -> Image.Image:
    """Center-crop / cover-fit to a fixed 3:4 RGB canvas."""
    return ImageOps.fit(to_rgb(img), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.4))


def _cutout_on_white(img: Image.Image) -> Image.Image:
    """
    Remove background locally (rembg) and paste the subject on pure white.
    Uses only the first configured rembg model for speed.
    Falls back to the original image if rembg is unavailable.
    """
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


def composite_photo_line_split(photo: Image.Image, line_art: Image.Image) -> Image.Image:
    """Hard vertical split helper (same-geometry sources only)."""
    photo_rgb = to_rgb(photo)
    line_rgb = to_rgb(line_art)
    if line_rgb.size != photo_rgb.size:
        line_rgb = ImageOps.fit(
            line_rgb,
            photo_rgb.size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.4),
        )
    width, height = photo_rgb.size
    mid = width // 2
    out = Image.new("RGB", (width, height), (255, 255, 255))
    out.paste(photo_rgb.crop((0, 0, mid, height)), (0, 0))
    out.paste(line_rgb.crop((mid, 0, width, height)), (mid, 0))
    return out


def _fast_image_model() -> str | None:
    """Prefer the flash image model when configured."""
    return get_gemini_fallback_image_model() or None


def generate_half_photo_half_line_art(
    output_path: str,
    selfie_path: str,
    *,
    style_target_id: str = "selfie-becoming",
    operation: str = "art_generation:creative:selfie-becoming",
) -> Tuple[bool, str]:
    """
    rembg cutout on white, then ONE Gemini pass for the full half-photo | half-pencil
    image on a single canvas so features stay aligned at the midline.
    """
    prepared = _cutout_on_white(_fit_portrait(load_rgb(selfie_path)))
    gemini_input = prepared.resize((GEMINI_W, GEMINI_H), Image.Resampling.LANCZOS)

    with tempfile.TemporaryDirectory(prefix="cs_selfie_split_") as tmp:
        split_path = os.path.join(tmp, "split.png")

        ok, message = generate_composed_image(
            output_path=split_path,
            prompt=build_split_prompt(),
            role_images=[
                (
                    "SELFIE / BASE IMAGE. Edit THIS exact portrait in place. "
                    "Keep the same person, crop, scale, and head position. "
                    "Left half = color photo of THIS person; "
                    "right half = light-gray pencil sketch of THIS same face, aligned. "
                    "Do not copy the face from the style target.",
                    gemini_input,
                ),
            ],
            style_target=style_target_path(style_target_id),
            style_instruction=STYLE_INSTRUCTION,
            aspect_ratio="3:4",
            temperature=0.2,
            operation=operation,
            isolate_line_art=False,
            trailing_instruction=BACKGROUND_LOCK,
            image_size="1K",
            model=_fast_image_model(),
        )
        if not ok:
            return False, message

        # Use Gemini's single canvas as-is — do NOT paste a separate photo half
        # (that was causing the midline break when framing differed).
        result = _fit_portrait(Image.open(split_path), prepared.size)

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
