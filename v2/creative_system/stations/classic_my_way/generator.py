from typing import Optional

from ...shared.gemini import aspect_from_image, generate_composed_image, load_rgb
from .prompts import build_prompt


def generate(
    output_path: str,
    artwork_path: str,
    user_prompt: str,
    template_path: Optional[str] = None,
    **_kwargs,
):
    role_images = []
    if template_path:
        role_images.append(
            (
                "ORIGINAL ART / TEMPLATE",
                load_rgb(template_path),
            )
        )
    role_images.append(
        (
            "USER ART",
            load_rgb(artwork_path),
        )
    )
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(user_prompt),
        role_images=role_images,
        style_target=None,
        aspect_ratio=aspect_from_image(artwork_path, fallback="3:4"),
        temperature=0.35,
        operation="art_generation:creative:classic-my-way",
    )
