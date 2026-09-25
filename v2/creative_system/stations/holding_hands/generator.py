from PIL import Image

from ...shared.fonts import DEFAULT_FONT_ID, get_font
from ...shared.gemini import generate_composed_image, load_rgb
from .layout import compose_artwork, hand_map_path, plate_path
from .prompts import HAND_MAP_LABEL, PERSON_A_LABEL, PERSON_B_LABEL, PLATE_LABEL, build_prompt


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

    success, message = generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(),
        role_images=[
            (PLATE_LABEL, load_rgb(plate)),
            (HAND_MAP_LABEL, load_rgb(hand_map)),
            (PERSON_A_LABEL, load_rgb(photo_a_path)),
            (PERSON_B_LABEL, load_rgb(photo_b_path)),
        ],
        aspect_ratio="4:5",
        temperature=0.3,
        operation="art_generation:creative:holding-hands",
    )
    if not success:
        return success, message

    with Image.open(output_path) as hands:
        art = compose_artwork(hands, name_a, name_b, caption, date_text, font["id"])
    art.save(output_path, format="PNG", optimize=True)
    return True, message
