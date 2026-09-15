from ...shared.gemini import aspect_from_image, generate_composed_image, load_rgb
from .prompts import build_prompt


def generate(output_path: str, template_path: str, artwork_path: str, user_prompt: str, **_kwargs):
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(user_prompt),
        role_images=[
            (
                "ORIGINAL ART / TEMPLATE",
                load_rgb(template_path),
            ),
            (
                "USER ART",
                load_rgb(artwork_path),
            ),
        ],
        style_target=None,
        aspect_ratio=aspect_from_image(artwork_path, fallback="3:4"),
        temperature=0.35,
        operation="art_generation:creative:classic-my-way",
    )
