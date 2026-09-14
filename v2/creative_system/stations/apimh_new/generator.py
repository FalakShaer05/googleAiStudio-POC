from ...shared.gemini import generate_composed_image, load_rgb, style_target_path
from .prompts import (
    build_prompt,
    layout_lock,
    map_lock,
    normalize_type,
    style_instruction,
)


def generate(
    output_path: str,
    map_image_path: str,
    message: str,
    location_label: str = "",
    latitude=None,
    longitude=None,
    map_type: str = "explorer",
    **_kwargs,
):
    chosen = normalize_type(map_type)
    user_map = load_rgb(map_image_path)
    map_role = (
        "USER MAP SCREENSHOT — LITERAL SOURCE. "
        "Crop/mask THIS exact image into the FULL heart silhouette, including top vessel lobes. "
        "No blank cream/gray cropped pieces. Keep the same roads, parks, labels, and icons. "
        "Do not redraw a different city. Do not use the sample/layout map geography."
    )
    # Heavy style blur so sample streets cannot override the uploaded map.
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(message, location_label, latitude, longitude, chosen),
        role_images=[
            (map_role, user_map),
            (map_role + " Repeat: the heart fill must be THIS screenshot, not a new map.", user_map),
            (map_role + " Final check: same streets as this image inside the heart.", user_map),
        ],
        style_target=style_target_path(f"apimh-new-{chosen}"),
        style_instruction=style_instruction(chosen),
        aspect_ratio="1:1",
        temperature=0.2,
        operation=f"art_generation:creative:apimh-new:{chosen}",
        obscure_style_text=True,
        obscure_style_radius=64,
        trailing_instruction=map_lock(chosen) + " " + layout_lock(chosen),
    )
