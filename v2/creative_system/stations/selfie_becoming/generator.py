from ...shared.gemini import generate_composed_image, load_rgb, style_target_path
from .prompts import build_prompt


def generate(output_path: str, selfie_path: str, **_kwargs):
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(),
        role_images=[
            (
                "SELFIE / IDENTITY PHOTO. Draw THIS person as line art. "
                "Do not copy the face from the style target.",
                load_rgb(selfie_path),
            ),
        ],
        style_target=style_target_path("selfie-becoming"),
        aspect_ratio="3:4",
        temperature=0.55,
        operation="art_generation:creative:selfie-becoming",
    )
