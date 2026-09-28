from ...shared.gemini import (
    aspect_from_image,
    build_hand_alignment_images,
    generate_composed_image,
    style_target_path,
)
from .prompts import STYLE_INSTRUCTION, build_prompt, hex_to_rgb, lettering_colors, vocabulary_lock


def generate(
    output_path: str,
    hand_path: str,
    words: list,
    colors: list | None = None,
    **_kwargs,
):
    selected = list(words)
    # Drop black/near-black — those words vanish on cutouts.
    palette = lettering_colors(colors)
    role_images, stencil = build_hand_alignment_images(hand_path)
    return generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(selected, colors=palette),
        role_images=role_images,
        style_target=style_target_path("tracing-hand"),
        style_instruction=STYLE_INSTRUCTION,
        aspect_ratio=aspect_from_image(hand_path, fallback="1:1"),
        temperature=0.4,
        operation="art_generation:creative:tracing-hand",
        isolate_subject=True,
        # Light blur: keep multi-color density masses, hide readable style vocabulary.
        obscure_style_text=True,
        obscure_style_radius=6,
        trailing_instruction=vocabulary_lock(selected, colors=palette),
        clip_to_stencil=stencil,
        word_color_palette=[hex_to_rgb(c) for c in palette],
        letter_style="bubble",
    )
