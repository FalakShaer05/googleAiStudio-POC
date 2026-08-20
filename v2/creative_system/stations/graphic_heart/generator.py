from ...shared.gemini import generate_composed_image, load_rgb, style_target_path
from .prompts import STYLE_INSTRUCTION, build_prompt, map_lock


def generate(
    output_path: str,
    map_image_path: str,
    message: str,
    location_label: str = "",
    latitude=None,
    longitude=None,
    **_kwargs,
):
    user_map = load_rgb(map_image_path)
    map_role = (
        "USER MAP SNAPSHOT — this is the ONLY street grid allowed inside the heart. "
        "Trace these roads, blocks, interchanges, water, and parks. Recolor to rust-red / cream. "
        "Keep any existing pin in the same relative spot. Do not replace this geography with the sample print."
    )
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(message, location_label, latitude, longitude),
        role_images=[
            (map_role, user_map),
            (map_role + " Repeat: fill the heart from THIS image.", user_map),
        ],
        style_target=style_target_path("graphic-heart"),
        style_instruction=STYLE_INSTRUCTION,
        aspect_ratio="1:1",
        temperature=0.35,
        operation="art_generation:creative:graphic-heart",
        obscure_style_text=True,
        obscure_style_radius=56,
        trailing_instruction=map_lock(),
    )
