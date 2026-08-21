from ...shared.gemini import aspect_from_image, generate_composed_image, load_rgb
from .prompts import build_prompt


def generate(output_path: str, artwork_path: str, user_prompt: str, **_kwargs):
    prompt = build_prompt(user_prompt)
    return generate_composed_image(
        output_path=output_path,
        prompt=prompt,
        role_images=[
            (
                "SOURCE ARTWORK to edit. This is the base image. Transform it using the user prompt.",
                load_rgb(artwork_path),
            ),
        ],
        style_target=None,
        aspect_ratio=aspect_from_image(artwork_path, fallback="3:4"),
        temperature=0.85,
        operation="art_generation:creative:make-art-yours",
    )
