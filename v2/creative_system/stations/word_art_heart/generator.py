from PIL import Image

from ...shared.gemini import (
    aspect_from_image,
    generate_composed_image,
    knockout_sticky_collage_background,
    knockout_word_ink_background,
    style_target_path,
)
from .prompts import (
    ART_STYLE_META,
    build_prompt,
    normalize_art_style,
    style_instruction,
    vocabulary_lock,
)


def generate(output_path: str, words: list, art_style: str = "word-heart", **_kwargs):
    selected = list(words)
    chosen = normalize_art_style(art_style)
    meta = ART_STYLE_META[chosen]
    style = style_target_path(meta["style_target_id"])
    # Sticky heart needs a readable ♥ silhouette; light blur only so lettering is unreadable.
    # Word heart is denser type — stronger blur to stop sample vocabulary leaking.
    obscure_radius = 12 if chosen == "sticky-heart" else 28
    ok, message_out = generate_composed_image(
        output_path=output_path,
        prompt=build_prompt(selected, chosen),
        role_images=[],
        style_target=style,
        style_instruction=style_instruction(chosen),
        aspect_ratio=aspect_from_image(style, fallback="1:1"),
        temperature=0.3 if chosen == "sticky-heart" else 0.35,
        operation=f"art_generation:creative:word-art-heart:{chosen}",
        obscure_style_text=True,
        obscure_style_radius=obscure_radius,
        trailing_instruction=vocabulary_lock(selected, chosen),
    )
    if not ok:
        return ok, message_out

    try:
        with Image.open(output_path) as raw:
            if chosen == "sticky-heart":
                transparent = knockout_sticky_collage_background(raw)
            else:
                transparent = knockout_word_ink_background(raw)
            if transparent.mode != "RGBA":
                transparent = transparent.convert("RGBA")
            transparent.save(output_path, format="PNG", optimize=True)
    except Exception as exc:
        print(f"word-art-heart transparency knockout failed: {exc}")
        return True, message_out

    return True, "Artwork generated successfully (transparent PNG)"
