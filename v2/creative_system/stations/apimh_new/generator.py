from PIL import Image

from ...shared.gemini import (
    generate_composed_image,
    knockout_cream_card_background,
    load_rgb,
    style_target_path,
)
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
    message: str = "",
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
    ok, message_out = generate_composed_image(
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
        # Keep ridge structure visible in the explorer layout sample.
        obscure_style_radius=20 if chosen == "explorer" else 64,
        trailing_instruction=map_lock(chosen) + " " + layout_lock(chosen),
    )
    if not ok:
        return ok, message_out

    try:
        with Image.open(output_path) as raw:
            transparent = knockout_cream_card_background(raw)
            # Force true PNG alpha (no accidental RGB flatten).
            if transparent.mode != "RGBA":
                transparent = transparent.convert("RGBA")
            transparent.save(output_path, format="PNG", optimize=True)
    except Exception as exc:
        print(f"apimh-new transparency knockout failed: {exc}")
        return True, message_out

    return True, "Artwork generated successfully (transparent PNG)"
