from ...shared.gemini import generate_composed_image, load_rgb, style_target_path
from .prompts import build_prompt


def generate(
    output_path: str,
    map_image_path: str,
    message: str,
    location_label: str = "",
    latitude=None,
    longitude=None,
    **_kwargs,
):
    prompt = build_prompt(message, location_label, latitude, longitude)
    return generate_composed_image(
        output_path=output_path,
        prompt=prompt,
        role_images=[
            (
                "MAP SNAPSHOT of the chosen coordinates. Use this street grid inside the heart. "
                "Restyle it rust-red / cream. Keep the pin location consistent with the marker.",
                load_rgb(map_image_path),
            ),
        ],
        style_target=style_target_path("graphic-heart"),
        aspect_ratio="1:1",
        temperature=0.7,
        operation="art_generation:creative:graphic-heart",
    )
