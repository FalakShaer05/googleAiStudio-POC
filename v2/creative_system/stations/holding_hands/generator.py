from ...shared.gemini import generate_composed_image, load_rgb, style_target_path
from .prompts import STYLE_INSTRUCTION, build_prompt


def generate(
    output_path: str,
    photo_a_path: str,
    photo_b_path: str,
    name_a: str,
    name_b: str,
    date_text: str,
    caption: str = "",
    **_kwargs,
):
    prompt = build_prompt(name_a, name_b, date_text, caption)
    return generate_composed_image(
        output_path=output_path,
        prompt=prompt,
        role_images=[
            (
                "PERSON A PHOTO. Use this person ONLY for the LEFT hand/arm skin tone, "
                f"age impression, and identity cues belonging to the name {name_a}. "
                "Do not paste the original photo. Draw an illustrated hand instead.",
                load_rgb(photo_a_path),
            ),
            (
                "PERSON B PHOTO. Use this person ONLY for the RIGHT hand/arm skin tone, "
                f"age impression, and identity cues belonging to the name {name_b}. "
                "Do not paste the original photo. Draw an illustrated hand instead.",
                load_rgb(photo_b_path),
            ),
        ],
        style_target=style_target_path("holding-hands"),
        style_instruction=STYLE_INSTRUCTION,
        aspect_ratio="1:1",
        temperature=0.35,
        operation="art_generation:creative:holding-hands",
    )
