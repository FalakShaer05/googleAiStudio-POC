WORD_CHIPS = [
    "kindness",
    "strength",
    "hope",
    "dream",
    "love",
    "joy",
    "kind",
    "honest",
    "brave",
    "confident",
    "ambitious",
    "caring",
    "patient",
    "friendly",
    "humble",
    "optimistic",
    "creative",
    "positive",
    "determined",
    "hardworking",
    "passionate",
    "respectful",
    "intelligent",
    "compassionate",
    "grateful",
    "loyal",
    "responsible",
    "supportive",
    "understanding",
    "happy",
    "generous",
    "thoughtful",
    "inspiring",
    "wise",
]


# Words printed on the tracing-hand style-target image. Copied unless forbidden.
STYLE_TARGET_WORDS = [
    "caring",
    "patient",
    "friendly",
    "humble",
    "optimistic",
    "kind",
    "honest",
    "brave",
    "confident",
    "ambitious",
    "creative",
    "positive",
    "determined",
    "hardworking",
    "passionate",
    "loyal",
    "responsible",
    "grateful",
    "supportive",
    "understanding",
    "happy",
    "generous",
    "thoughtful",
    "inspiring",
    "wise",
    "respectful",
    "intelligent",
    "compassionate",
    "amazing",
    "sincere",
]

# Common invented / garbled labels to block even if they are not on the reference.
INVENTED_WORDS = [
    "wonderful",
    "conderful",
    "inspiriful",
]


STYLE_INSTRUCTION = (
    "LETTERING AND COLOR TARGET only. Copy marker-letter weight, hollow counters, "
    "mixed print/script, per-word bright colors, and tiny heart/star fillers. "
    "IGNORE every word printed on this reference. "
    "Do NOT copy this reference's hand pose, finger lengths, thumb side, or silhouette. "
    "The uploaded hand photo and stencil are the only outline to fill. "
    "Do NOT copy the cream page. Each word its own bright color."
)

POSE_LOCK = (
    "POSE LOCK (highest priority with vocabulary).\n"
    "The uploaded hand is the ONLY silhouette. Match its thumb side, finger count, "
    "finger lengths, gaps, rotation, and left vs right. "
    "Put every letter INSIDE that outline. White/background stays empty. "
    "Do not draw a generic open palm and do not copy the style-target hand shape."
)


def _normalized_words(words: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for word in words:
        cleaned = " ".join(str(word).split()).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def numbered_word_list(words: list[str]) -> str:
    return "\n".join(
        f"{index}. {word.upper()}"
        for index, word in enumerate(_normalized_words(words), 1)
    )


def forbidden_style_words(words: list[str]) -> list[str]:
    allowed = {word.lower() for word in _normalized_words(words)}
    blocked: list[str] = []
    seen: set[str] = set()
    for word in (*STYLE_TARGET_WORDS, *INVENTED_WORDS):
        key = word.lower()
        if key not in allowed and key not in seen:
            seen.add(key)
            blocked.append(word.upper())
    return blocked


def vocabulary_lock(words: list[str]) -> str:
    listed = numbered_word_list(words)
    forbidden = forbidden_style_words(words)
    forbid_line = (
        "Never write these (style-reference or invented): " + ", ".join(forbidden) + "."
        if forbidden
        else "Do not invent extra adjectives."
    )
    return (
        "VOCABULARY LOCK (final instruction, highest priority).\n"
        "The layout/style image may contain other words — ignore them completely.\n"
        "Every readable word in the output MUST be one of these, spelled exactly:\n"
        f"{listed}\n"
        f"{forbid_line}\n"
        "Fill leftover space by REPEATING words from this list. "
        "Do not add synonyms, related traits, or new spellings.\n\n"
        f"{POSE_LOCK}"
    )


def build_prompt(words: list[str]) -> str:
    cleaned = _normalized_words(words)
    hero = cleaned[0].upper() if cleaned else "KINDNESS"
    listed = numbered_word_list(cleaned)
    forbidden = forbidden_style_words(cleaned)
    forbid_line = (
        "Never write: " + ", ".join(forbidden) + "."
        if forbidden
        else "Do not invent extra adjectives."
    )
    return f"""Fill the uploaded hand's exact outline with colorful word art. The hand is made ONLY of lettering.

{POSE_LOCK}

VOCABULARY (closed list):
Every readable word MUST be copied character-for-character from this numbered list.
Use these words only. Repeat from this list to fill leftover space inside the outline. Never add a new word.
{listed}

{forbid_line}
Do not blend or misspell words (no INSPIRIFUL, CONDERFUL, WONDERFUL unless listed).

LETTERFORMS (text instructions, not the style image's vocabulary):
- Medium-weight hand-drawn marker letters. Mix simple print caps with a little brush script for 1-2 palm hero words.
- HOLLOW COUNTERS ARE MANDATORY. The enclosed spaces in A, D, B, P, R, O, Q, E, g must stay EMPTY — you can see through them. Never fill those holes with the letter color.
- Think outlined strokes / drawn letters, not solid blobs. If a D looks like a filled D-pad, it is wrong.
- Letters do not melt together. Small gaps between letters and between words.
- Comfortable tracking. Open bowls and loops.

ALIGNMENT (follow the uploaded hand, not a stock pose):
- Words live only inside the uploaded outline. Nothing outside it.
- Thumb: words curve/tilt along THAT thumb.
- Fingers: stacks follow each real finger's length and angle. Short fingers get fewer/smaller words.
- Palm: larger horizontal words in THAT palm. Hero word "{hero}" in the palm, slightly arched if needed.
- Tiny line-art hearts, stars, smileys, and dots fill leftover edge gaps inside the outline only.

FORBIDDEN:
- A different hand than the upload (generic spread palm, style-target silhouette)
- Letters crossing outside the uploaded outline
- Filled-in D, B, O, A, P, R (counters closed)
- Extra-bold / ultra-heavy rounded display sans
- Tight tracking that turns letters into blocks
- A solid hand plate behind the words
- Any word that is not on the numbered list above

COLOR:
- Each WORD is one solid color from: hot pink, coral, orange, gold, lime, teal, sky blue, royal blue, violet.
- Neighboring words different colors. Icons use the same palette. No black or cream ink.

BACKGROUND:
- Letters and doodles only. Fully transparent outside them (pure white only if transparency is impossible).
- No cream paper, pale fill, checkerboard, watercolor, frame, or title.

OUTPUT: isolated word-art of the uploaded hand only, hollow letter counters, transparent outside the outline. Keep the same crop as the photo.
Before finishing, check every word against the numbered list. Remove anything that is not on it."""
