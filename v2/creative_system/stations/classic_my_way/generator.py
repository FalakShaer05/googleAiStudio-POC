from ...shared.gemini import aspect_from_image, generate_composed_image, load_rgb
from .prompts import build_prompt


def generate(output_path: str, artwork_path: str, **_kwargs):
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(),
        role_images=[
            (
                "SOURCE ARTWORK. Keep this exact creative idea and layout. "
                "Aggressively finish rough edits with richer color, deeper shadows, "
                "and clearer lighting so the enhancement is obvious at a glance.",
                load_rgb(artwork_path),
            ),
        ],
        style_target=None,
        aspect_ratio=aspect_from_image(artwork_path, fallback="3:4"),
        temperature=0.5,
        operation="art_generation:creative:classic-my-way",
    )
