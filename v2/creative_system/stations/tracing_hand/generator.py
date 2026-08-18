from ...shared.gemini import (
    aspect_from_image,
    build_hand_alignment_images,
    generate_composed_image,
    style_target_path,
)
from .prompts import STYLE_INSTRUCTION, build_prompt, vocabulary_lock


def generate(output_path: str, hand_path: str, words: list, **_kwargs):
    selected = list(words)
    role_images, stencil = build_hand_alignment_images(hand_path)
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(selected),
        role_images=role_images,
        style_target=style_target_path("tracing-hand"),
        style_instruction=STYLE_INSTRUCTION,
        aspect_ratio=aspect_from_image(hand_path, fallback="1:1"),
        temperature=0.35,
        operation="art_generation:creative:tracing-hand",
        isolate_subject=True,
        obscure_style_text=True,
        trailing_instruction=vocabulary_lock(selected),
        clip_to_stencil=stencil,
    )
