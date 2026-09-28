from PIL import Image

from ...shared.fonts import DEFAULT_FONT_ID, get_font
from ...shared.gemini import generate_composed_image, load_rgb
from .colorfix import strip_map_colors
from .layout import compose_artwork, hand_map_path, plate_path
from .prompts import (
    COLOR_LOCK,
    HAND_MAP_LABEL,
    PERSON_A_LABEL,
    PERSON_B_LABEL,
    PLATE_LABEL,
    build_prompt,
)


def generate(
    output_path: str,
    photo_a_path: str,
    photo_b_path: str,
    name_a: str,
    name_b: str,
    date_text: str = "",
    caption: str = "",
    font_id: str = DEFAULT_FONT_ID,
    **_kwargs,
):
    font = get_font(font_id)
    plate = plate_path()
    hand_map = hand_map_path()
    if not plate or not hand_map:
        return False, "Holding Hands template is missing"

    photo_a = load_rgb(photo_a_path)
    photo_b = load_rgb(photo_b_path)

    success, message = generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(),
        role_images=[
            (PLATE_LABEL, load_rgb(plate)),
            (HAND_MAP_LABEL, load_rgb(hand_map)),
            (PERSON_A_LABEL, photo_a),
            (PERSON_B_LABEL, photo_b),
        ],
        aspect_ratio="4:5",
        temperature=0.3,
        operation="art_generation:creative:holding-hands",
        trailing_instruction=COLOR_LOCK,
    )
    if not success:
        return success, message

    with Image.open(output_path) as hands:
        cleaned = strip_map_colors(hands, photo_a, photo_b)
        art = compose_artwork(cleaned, name_a, name_b, caption, date_text, font["id"])
    art.save(output_path, format="PNG", optimize=True)
    return True, message
