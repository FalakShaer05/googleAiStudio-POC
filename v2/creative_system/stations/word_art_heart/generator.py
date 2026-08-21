from ...shared.gemini import aspect_from_image, generate_composed_image, style_target_path
from .prompts import STYLE_INSTRUCTION, build_prompt, vocabulary_lock


def generate(output_path: str, words: list, **_kwargs):
    selected = list(words)
    style = style_target_path("word-art-heart")
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(selected),
        role_images=[],
        style_target=style,
        style_instruction=STYLE_INSTRUCTION,
        aspect_ratio=aspect_from_image(style, fallback="4:5"),
        temperature=0.5,
        operation="art_generation:creative:word-art-heart",
        trailing_instruction=vocabulary_lock(selected),
    )
