"""Origami station — wrap open-sheet print onto locked fold; transparent PNG."""
from __future__ import annotations

import hashlib
from typing import Optional

from PIL import Image

from ...shared.gemini import aspect_from_image, generate_composed_image, load_rgb
from .finalize import finalize_origami
from .palette import dominant_paper_color, extract_palette
from .plate import build_color_plate, describe_layout_hint
from .prompts import build_prompt, build_trailing_instruction

CREAM = (245, 240, 230)


def _content_seed(*paths: Optional[str]) -> int:
    h = hashlib.sha256()
    for path in paths:
        if not path:
            continue
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            continue
    return int(h.hexdigest()[:8], 16) % (2**31)


def _template_on_cream(template_path: str) -> Image.Image:
    """Present the pose lock on cream so Gemini matches product framing, not black void."""
    rgb = load_rgb(template_path)
    w, h = rgb.size
    side = max(max(w, h), 768)
    canvas = Image.new("RGB", (side, side), CREAM)
    scale = min((side * 0.88) / max(w, 1), (side * 0.88) / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((side - nw) // 2, (side - nh) // 2))
    return canvas


def generate(
    output_path: str,
    artwork_path: str,
    template_path: Optional[str] = None,
    user_prompt: str = "",
    **_kwargs,
):
    palette = extract_palette(artwork_path)
    solid = dominant_paper_color(artwork_path)
    layout_hint = describe_layout_hint(artwork_path)

    role_images = []
    canvas_size = None

    if template_path:
        pose = _template_on_cream(template_path)
        canvas_size = pose.size
        role_images.append(
            (
                "TARGET FOLDED MODEL — LOCKED camera, pose, subject, scale, centering. "
                "Edit this image in place. Never switch to top-down. Ignore its paper color.",
                pose,
            )
        )

    # Keep print reference smaller than the pose lock so orientation doesn't drift
    color_plate = build_color_plate(artwork_path, max_side=512)
    role_images.append(
        (
            "COLOR PLATE — print colors/regions to WRAP onto the locked model faces. "
            "Do not change camera to match this flat sheet.",
            color_plate,
        )
    )

    crease = load_rgb(artwork_path)
    crease = crease.copy()
    crease.thumbnail((360, 360))
    role_images.append(
        (
            "CREASE PATTERN — topology hint only. Do NOT use its top-down framing.",
            crease,
        )
    )

    ok, message = generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(user_prompt, layout_hint=layout_hint),
        role_images=role_images,
        style_target=None,
        aspect_ratio=aspect_from_image(template_path or artwork_path, fallback="1:1"),
        temperature=0.0,
        seed=_content_seed(template_path, artwork_path),
        trailing_instruction=build_trailing_instruction(layout_hint=layout_hint),
        operation="art_generation:creative:origami",
        image_size="2K",
    )
    if not ok:
        return ok, message

    try:
        finalize_origami(
            output_path,
            palette,
            canvas_size=canvas_size,
            solid_color=solid,
            clamp_colors=bool(solid),
        )
    except Exception as exc:
        print(f"origami finalize error: {exc}")
        import traceback
        traceback.print_exc()
        return True, message

    return True, "Artwork generated successfully (transparent PNG)"
