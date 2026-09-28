from __future__ import annotations

import json

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

# Palette from the "Choose your colors" station UI (4 rows × 5).
COLOR_OPTIONS = [
    {"hex": "#C8D42E", "label": "Lime"},
    {"hex": "#2A9B96", "label": "Teal"},
    {"hex": "#E88B70", "label": "Coral"},
    {"hex": "#D63A8C", "label": "Magenta"},
    {"hex": "#1C1C1C", "label": "Black"},
    {"hex": "#C45A28", "label": "Burnt Orange"},
    {"hex": "#D62828", "label": "Red"},
    {"hex": "#F0A878", "label": "Peach"},
    {"hex": "#F08C20", "label": "Orange"},
    {"hex": "#8B7355", "label": "Taupe"},
    {"hex": "#1A3A6E", "label": "Navy"},
    {"hex": "#2E5FCF", "label": "Royal Blue"},
    {"hex": "#2D5A4A", "label": "Pine"},
    {"hex": "#7EC8A3", "label": "Mint"},
    {"hex": "#C8CDD2", "label": "Silver"},
    {"hex": "#111111", "label": "Ink Black"},
    {"hex": "#4A4A4A", "label": "Charcoal"},
    {"hex": "#7A7A7A", "label": "Slate"},
    {"hex": "#E8E4DC", "label": "Cream"},
    {"hex": "#9A9A9A", "label": "Grey"},
]

ALLOWED_COLOR_HEX = {c["hex"].upper() for c in COLOR_OPTIONS}
MAX_COLORS = 5
DEFAULT_COLORS = ["#D63A8C", "#E88B70", "#F08C20", "#C8D42E", "#2E5FCF"]


# Words printed on the tracing-hand style-target image. Copied unless forbidden.
STYLE_TARGET_WORDS = [
    "dream",
    "explore",
    "inspire",
    "joy",
    "love",
    "believe",
    "share",
    "create",
    "hope",
    "rage",
    "grow",
    "change",
    "play",
    "engage",
    "future",
    "peace",
    "unity",
    "imagine",
    "conagine",
    "copp",
    "be you",
    "take steps",
]

# Common invented / garbled labels to block even if they are not on the reference.
INVENTED_WORDS = [
    "wonderful",
    "conderful",
    "inspiriful",
    "conagine",
    "copp",
]


STYLE_INSTRUCTION = (
    "LETTER STYLE TARGET only. "
    "Copy glossy bubble-sticker caps: bold rounded 3D letters, "
    "ONE solid ink color per whole word, tiny star/dot fillers, soft light outer rim. "
    "Spread ALL user-selected inks across neighboring words. "
    "WRONG: rainbow letters inside one word, solid black/colored hand plate, "
    "speckle/noise backgrounds, digits/numbers, or writing color codes as text. "
    "IGNORE every word printed on this reference and IGNORE its black/speckled backdrop. "
    "Do NOT copy this reference's hand pose or its repeated words. "
    "User canvas outline / photo = the ONLY silhouette to fill with words."
)

POSE_LOCK = (
    "POSE LOCK (highest priority with vocabulary).\n"
    "The upload is a ROUGH HAND DRAWING or photo — that silhouette is the ONLY shape. "
    "Match its thumb side, finger count, finger lengths, gaps, rotation, and left vs right. "
    "PACK complete sticker words densely into the WHOLE silhouette — fingertips, "
    "finger gaps, palm pockets, and rough edge regions. Scale and rotate whole words "
    "to fit; do NOT slice or crop letters mid-glyph. Tiny stars/dots fill leftover cracks. "
    "Outside the silhouette stays fully transparent. "
    "Do NOT paint a solid black (or any solid) hand plate behind the words. "
    "Do NOT add speckles, noise, checkerboard, or a filled backdrop. "
    "Do not draw a generic open palm and do not copy the style-target hand shape."
)


def _normalized_words(words: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for word in words:
        cleaned = " ".join(str(word).split()).strip()
        # Drop pure numbers / digit strings — never paint numerals into the hand.
        if cleaned.isdigit() or (cleaned and all(ch.isdigit() or ch in ".,-+#" for ch in cleaned)):
            continue
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def numbered_word_list(words: list[str]) -> str:
    # Bullet list — never "1. 2. 3." prefixes (models copy those as literal digits).
    return "\n".join(
        f"- {word.upper()}"
        for word in _normalized_words(words)
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


def _normalize_hex(value: str) -> str:
    cleaned = str(value or "").strip().upper().replace("0X", "")
    if cleaned.startswith("#"):
        body = cleaned[1:]
    else:
        body = cleaned
    body = "".join(ch for ch in body if ch in "0123456789ABCDEF")
    if len(body) == 3:
        body = "".join(ch * 2 for ch in body)
    if len(body) != 6:
        return ""
    return f"#{body}"


def _is_valid_hex(hex_color: str) -> bool:
    return bool(hex_color) and len(hex_color) == 7 and hex_color.startswith("#")


def normalize_colors(colors: list[str] | None, *, fallback: bool = True) -> list[str]:
    """Accept any valid #RRGGBB values (hub swatches or custom API arrays), max 5."""
    picked: list[str] = []
    seen: set[str] = set()
    for raw in colors or []:
        hex_color = _normalize_hex(raw)
        if not _is_valid_hex(hex_color) or hex_color in seen:
            continue
        seen.add(hex_color)
        picked.append(hex_color)
        if len(picked) >= MAX_COLORS:
            break
    if picked:
        return picked
    return list(DEFAULT_COLORS) if fallback else []


def parse_color_list(*sources: str | list | None) -> list[str]:
    """
    Merge color inputs from JSON arrays, repeated form fields, or comma lists.
    Examples: '["#FF0000","#00FF00"]', ['#FF0000', '#00FF00'], '#FF0000,#00FF00'
    """
    collected: list[str] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, (list, tuple)):
            for item in source:
                collected.extend(parse_color_list(item))
            continue
        text = str(source).strip()
        if not text:
            continue
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                collected.extend(str(item) for item in parsed)
            continue
        if "," in text:
            collected.extend(part.strip() for part in text.split(","))
            continue
        collected.append(text)
    return collected


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = _normalize_hex(hex_color).lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def is_too_dark_for_lettering(hex_color: str) -> bool:
    """Black/near-black inks vanish on dark/transparent cutouts — never use for words."""
    r, g, b = hex_to_rgb(hex_color)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return lum < 72 or max(r, g, b) < 58


def lettering_colors(colors: list[str] | None, *, fallback: bool = True) -> list[str]:
    """User palette with black/near-black removed so every word stays visible."""
    picked = [c for c in normalize_colors(colors, fallback=False) if not is_too_dark_for_lettering(c)]
    if picked:
        return picked
    defaults = [c for c in DEFAULT_COLORS if not is_too_dark_for_lettering(c)]
    return defaults if fallback else []


def _annotate_color_option(option: dict) -> dict:
    annotated = dict(option)
    annotated["too_dark"] = is_too_dark_for_lettering(option["hex"])
    return annotated


COLOR_ROWS = [
    [_annotate_color_option(c) for c in COLOR_OPTIONS[i : i + 5]]
    for i in range(0, len(COLOR_OPTIONS), 5)
]


def color_label(hex_color: str) -> str:
    key = _normalize_hex(hex_color)
    for option in COLOR_OPTIONS:
        if option["hex"].upper() == key:
            return option["label"]
    return "Custom"


def palette_swatch_lines(colors: list[str]) -> str:
    """Describe ink swatches without #hex (models copy hex as literal text)."""
    # Letter tags instead of "Ink 1/2/3" — digit labels leak into the artwork.
    tags = "ABCDEFGH"
    lines: list[str] = []
    for index, hex_color in enumerate(lettering_colors(colors)):
        r, g, b = hex_to_rgb(hex_color)
        tag = tags[index] if index < len(tags) else chr(ord("A") + index)
        lines.append(
            f"  Swatch {tag} — {color_label(hex_color)} "
            f"(red {r}, green {g}, blue {b})"
        )
    return "\n".join(lines)


def vocabulary_lock(words: list[str], colors: list[str] | None = None) -> str:
    vocab = _normalized_words(words)
    listed = numbered_word_list(vocab)
    forbidden = forbidden_style_words(vocab)
    forbid_line = (
        "Never write these (style-reference or invented): " + ", ".join(forbidden) + "."
        if forbidden
        else "Do not invent extra adjectives."
    )
    swatches = palette_swatch_lines(colors)
    count = len(vocab)
    return (
        "VOCABULARY LOCK (final instruction, highest priority).\n"
        "The layout/style image may contain other words — ignore them completely.\n"
        f"Use EACH of these {count} words EXACTLY ONCE — no duplicates, no repeats:\n"
        f"{listed}\n"
        f"{forbid_line}\n"
        "Do NOT repeat any word. Do NOT invent fillers. "
        "Leftover space = tiny stars/dots only (no extra words).\n"
        "NEVER write digits, numerals, numbers (0-9), years, or strings like 69. "
        "NEVER write hex codes, hash marks, RGB values, or ink/swatch names as text.\n\n"
        "STRUCTURE LOCK — WORDS ONLY:\n"
        "Do NOT fill a solid black/colored hand plate. Do NOT emboss into a monochrome hand.\n"
        "Do NOT add noise/speckle backgrounds. Outside the letters = transparent.\n"
        "Pack COMPLETE words to fill the silhouette including rough edges and fingertips — "
        "do not slice letters mid-glyph; leftover cracks get stars/dots only.\n"
        "ONLY letter glyphs + tiny star/dot fillers are opaque.\n\n"
        "COLOR LOCK:\n"
        f"Paint lettering using ONLY these user-selected ink swatches:\n{swatches}\n"
        "Use EVERY swatch above across the composition — do not stick to only one or two inks.\n"
        "Every word MUST be one of these visible colors — never black, charcoal, or near-black ink.\n"
        "Each WORD = exactly ONE ink for EVERY letter in that word (solid, no rainbow). "
        "Neighboring words MUST use different inks; rotate through ALL swatches evenly. "
        "Stars/dots also cycle through every swatch. No other hues allowed.\n\n"
        f"{POSE_LOCK}"
    )


def build_prompt(
    words: list[str],
    colors: list[str] | None = None,
) -> str:
    cleaned = _normalized_words(words)
    hero = cleaned[0].upper() if cleaned else "KINDNESS"
    listed = numbered_word_list(cleaned)
    forbidden = forbidden_style_words(cleaned)
    forbid_line = (
        "Never write: " + ", ".join(forbidden) + "."
        if forbidden
        else "Do not invent extra adjectives."
    )
    swatches = palette_swatch_lines(colors)
    word_count = len(cleaned)
    return f"""Fill the user's ROUGH HAND DRAWING with glossy bubble sticker words — letters only, no solid hand fill.

The upload is a rough canvas outline. Treat the filled silhouette as a packing region:
pack COMPLETE words densely so fingertips, edge pockets, and the palm all read as filled.
Nestle / scale / rotate whole words to the shape — never razor-cut letters mid-stroke.

MATCH THE STYLE TARGET LOOK (letters only, not its black background):
- Thick puffy 3D bubble / plastic sticker capitals with soft top highlight and depth.
- Soft light cream rim around the outer hand edge only (optional).
- Tiny colored stars and dots only in leftover edge cracks (not words).
- One solid ink color per whole word.
- Dense packed composition like the style target — the hand shape is made OF words.

VOCABULARY — NO DUPLICATES:
- Write each of these {word_count} words EXACTLY ONCE. Never repeat a word.
{listed}
{forbid_line}
- Do not invent new words. Do not duplicate. Leftover cracks get dots/stars only.
- LETTERS ONLY — never write digits, numerals, or numbers (no 0-9, no "69", no years).

CRITICAL — NEVER DO THIS:
- Solid black (or any solid) hand silhouette / plate behind the words.
- Speckle, noise, pink/blue static, checkerboard, or filled page backgrounds.
- Sliced / cropped letters at the silhouette edge (fit whole words instead).
- Empty fingertips or empty edge pockets — fill them with a word or stars.
- Any digits / numerals / numbers anywhere in the artwork.
- Duplicate any word from the list.
- Write color codes, hash tags, RGB values, or ink names as readable text.
- Rainbow / multi-color letters inside a single word.
- A different hand shape than the uploaded drawing.

{POSE_LOCK}

LETTERFORMS:
- Bold rounded uppercase sticker letters, glossy 3D, solid filled shapes.
- Soft light outer rim hugging the hand silhouette (not a filled interior plate).

ALIGNMENT:
- Follow the rough drawing's pose/crop exactly.
- Thumb / fingers: vertical or angled words that fit EACH digit fully.
- Palm: larger words; hero "{hero}" largest in the center.
- Pack out to the magenta/pale-pink fill zone edge.

COLOR (user-selected inks only — paint with them, never write them as text):
{swatches}
- Use ALL of these inks across the hand — spread them evenly; do not favor only 1–2 colors.
- EVERY word must use one of these inks — no colorless, black, grey, or near-black words.
- Each WORD uses exactly ONE of these inks for the entire word (every letter same color).
- Neighboring words MUST use different inks; rotate through the full swatch list.
- Icons/dots/stars also cycle through every swatch — no other colors.

BACKGROUND:
- Fully TRANSPARENT outside letters/rim and between words.
- No black page, no cream underlay plate, no checkerboard, no noise.

OUTPUT: densely packed sticker-word hand on transparency matching the rough drawing — each listed word once, each word one solid user-selected color."""
