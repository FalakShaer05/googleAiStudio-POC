WORD_CHIPS = [
    "family",
    "love",
    "happiness",
    "joy",
    "smile",
    "laughter",
    "cheerful",
    "delight",
    "bliss",
    "peace",
    "harmony",
    "hope",
    "faith",
    "gratitude",
    "blessing",
    "kindness",
    "sunshine",
    "rainbow",
    "adventure",
    "celebration",
    "success",
    "art",
    "artist",
    "painting",
    "sketch",
    "drawing",
    "palette",
    "brush",
    "creativity",
    "imagination",
    "fantasy",
    "friends",
    "togetherness",
    "dream",
    "story",
    "character",
    "portrait",
]


# Readable labels on the dense word-cloud style target. Forbidden unless the user picked them.
STYLE_TARGET_WORDS = [
    "joy",
    "happiness",
    "smile",
    "laughter",
    "cheerful",
    "delight",
    "bliss",
    "love",
    "family",
    "friends",
    "togetherness",
    "peace",
    "harmony",
    "hope",
    "faith",
    "gratitude",
    "blessing",
    "kindness",
    "sunshine",
    "rainbow",
    "adventure",
    "celebration",
    "success",
    "art",
    "artist",
    "painting",
    "sketch",
    "drawing",
    "color",
    "palette",
    "canvas",
    "brush",
    "watercolor",
    "acrylic",
    "sculpture",
    "design",
    "illustration",
    "creativity",
    "imagination",
    "fantasy",
    "dreamscape",
    "story",
    "character",
    "portrait",
    "dream",
]


STYLE_INSTRUCTION = (
    "LOCKED STYLE TARGET — clone THIS look, not a medical sketch. "
    "The heart is a DENSE handwritten word cloud: every chamber and the top vessels "
    "are packed with mixed print/cursive words plus tiny matching doodles. "
    "Sketchy multi-color colored-pencil outlines (red, coral, green, blue overlapping). "
    "About 8–12 irregular color-zoned compartments. Soft saturated reds, oranges, "
    "golds, greens, teals, blues, and purples. Pure WHITE background. "
    "Do NOT copy cream paper, watercolor corner blobs, black anatomical hatching, "
    "or large empty chambers. "
    "IGNORE every printed word on this reference — use only the user's word list."
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
    for word in STYLE_TARGET_WORDS:
        key = word.lower()
        if key not in allowed and key not in seen:
            seen.add(key)
            blocked.append(word.upper())
    return blocked


def vocabulary_lock(words: list[str]) -> str:
    listed = numbered_word_list(words)
    forbidden = forbidden_style_words(words)
    forbid_line = (
        "Never write these (style-reference leftovers): " + ", ".join(forbidden) + "."
        if forbidden
        else "Do not invent extra words."
    )
    return (
        "VOCABULARY LOCK (final instruction, highest priority).\n"
        "The style image may show other words — ignore them completely.\n"
        "Every readable word MUST be one of these, spelled exactly:\n"
        f"{listed}\n"
        f"{forbid_line}\n"
        "Fill leftover interior by REPEATING words from this list at different sizes. "
        "Do not add synonyms unless they are on the list.\n"
        "Keep the dense word-cloud layout of the style target on a pure white page."
    )


def build_prompt(words: list[str]) -> str:
    cleaned = _normalized_words(words)
    listed = numbered_word_list(cleaned)
    forbidden = forbidden_style_words(cleaned)
    forbid_line = (
        "Never write: " + ", ".join(forbidden) + "."
        if forbidden
        else "Do not invent extra words."
    )
    return f"""Draw an ANATOMICAL human heart that IS a dense handwritten word cloud.

Clone the attached style target: packed lettering + tiny doodles forming the heart.
This is NOT a medical illustration with a few words dropped into empty chambers.

WHAT TO DRAW:
- Recognizable anatomical heart (atria, ventricles, aorta and pulmonary vessels on top). Slight tilt. NOT a valentine ♥.
- The entire silhouette is filled with handwritten words. Vessels at the top are also packed with words.
- 8–12 irregular compartments divided by thin sketchy colored-pencil strokes (several overlapping colors, never one solid black line).
- Almost no empty interior. If you can see a large patch of paper inside the heart, keep adding words and doodles.
- Group related user-words into color-coded clusters (joy/smile together, art words together, family/love together, etc.).
- Informal mixed lettering: print caps mixed with brushy cursive. Sizes vary. Words follow the curves of the walls and vessels.
- Each word is one color. Neighbors use different colors from: coral red, pink, orange, gold, olive, teal, sky blue, royal blue, violet, dusty purple.
- Sprinkle many tiny line-art doodles next to matching words: mini hearts, leaves, sun, bird, rainbow, stars, mountains, paint palette, brush, book, lightbulb, cloud.

VOCABULARY (closed list):
Use these words only. Repeat from this list at several sizes to pack the heart densely. Never add a new word.
{listed}

{forbid_line}

FORBIDDEN (this is the wrong look):
- Cream / off-white paper texture
- Peach or watercolor splashes in the corners
- Black fine-liner medical outlines with pink/blue muscle hatching
- Only 4 large chambers with sparse cursive and lots of blank space
- Visible red/blue vein diagrams as the main texture
- UI, title, buttons, caption, or frame

BACKGROUND:
- Pure white. No paper grain. No watercolor. Heart centered with a modest white margin.

OUTPUT: finished illustration matching the dense colorful word-cloud heart, not a sketchbook anatomy study.
Before finishing, check every readable word against the numbered list and pack any empty chamber."""
