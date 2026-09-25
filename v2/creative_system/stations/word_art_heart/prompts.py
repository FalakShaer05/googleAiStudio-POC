WORD_CHIPS = [
    "love",
    "family",
    "passion",
    "kindness",
    "compassion",
    "courage",
    "strength",
    "hope",
    "dreams",
    "community",
    "peace",
    "home",
    "art",
    "creativity",
    "joy",
    "happiness",
    "smile",
    "friends",
    "gratitude",
    "faith",
    "harmony",
    "adventure",
    "celebration",
    "success",
    "imagination",
    "dream",
    "togetherness",
    "blessing",
    "sunshine",
    "you",
    "trust",
    "care",
    "create",
    "light",
    "hugs",
    "memory",
    "unity",
]

ART_STYLES = ("word-heart", "sticky-heart")

ART_STYLE_META = {
    "word-heart": {
        "label": "Word Heart",
        "tagline": "A heart made of your words.",
        "style_target_id": "word-art-heart",
        "sample": "word-art-heart.png",
    },
    "sticky-heart": {
        "label": "Sticky Heart",
        "tagline": "A heart filled with your words.",
        "style_target_id": "word-art-sticky-heart",
        "sample": "word-art-sticky-heart.png",
    },
}

# Readable labels on each style target. Forbidden unless the user picked them.
STYLE_TARGET_WORDS = {
    "word-heart": [
        "love",
        "family",
        "passion",
        "kindness",
        "compassion",
        "courage",
        "strength",
        "hope",
        "dreams",
        "community",
        "peace",
        "home",
        "art",
        "creativity",
        "joy",
    ],
    "sticky-heart": [
        "love",
        "joy",
        "together forever",
        "dream",
        "kindness",
        "you",
        "hope",
        "create",
        "trust",
        "care",
        "smile",
        "peace",
        "light",
        "support each other",
        "hugs",
        "memory",
        "unity",
        "friends",
    ],
}


def normalize_art_style(art_style: str | None) -> str:
    value = (art_style or "word-heart").strip().lower().replace("_", "-")
    aliases = {
        "word": "word-heart",
        "words": "word-heart",
        "anatomical": "word-heart",
        "calligram": "word-heart",
        "sticky": "sticky-heart",
        "stickies": "sticky-heart",
        "sticky-notes": "sticky-heart",
        "notes": "sticky-heart",
    }
    value = aliases.get(value, value)
    return value if value in ART_STYLES else "word-heart"


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


def coverage_checklist(words: list[str]) -> str:
    cleaned = _normalized_words(words)
    lines = [
        f"[ ] {index}. {word.upper()}"
        for index, word in enumerate(cleaned, 1)
    ]
    return "\n".join(lines)


def comma_word_list(words: list[str]) -> str:
    return ", ".join(word.upper() for word in _normalized_words(words))


def forbidden_style_words(words: list[str], art_style: str | None = None) -> list[str]:
    chosen = normalize_art_style(art_style)
    allowed = {word.lower() for word in _normalized_words(words)}
    # Common style-target / invented leaks beyond the active sample.
    extras = (
        "love",
        "family",
        "passion",
        "kindness",
        "compassion",
        "courage",
        "strength",
        "hope",
        "joy",
        "create",
        "you",
        "unity",
        "together forever",
        "support each other",
    )
    blocked: list[str] = []
    seen: set[str] = set()
    for word in (*STYLE_TARGET_WORDS.get(chosen, ()), *extras):
        key = word.lower()
        if key not in allowed and key not in seen:
            seen.add(key)
            blocked.append(word.upper())
    return blocked


def _hero_word(words: list[str]) -> str:
    cleaned = _normalized_words(words)
    if not cleaned:
        return "LOVE"
    preferred = ("love", "family", "home", "hope", "peace", "joy", "dreams", "happiness")
    lower_map = {w.lower(): w for w in cleaned}
    for key in preferred:
        if key in lower_map:
            return lower_map[key].upper()
    singles = [w for w in cleaned if " " not in w]
    pick = min(singles or cleaned, key=len)
    return pick.upper()


def style_instruction(art_style: str | None = None) -> str:
    chosen = normalize_art_style(art_style)
    if chosen == "sticky-heart":
        return (
            "LAYOUT / COLOR TARGET only (text is intentionally blurred). "
            "Clone THIS compact sticky-note ♥: overlapping yellow notes form the silhouette themselves "
            "(not a giant solid yellow heart with tiny notes stuck inside). "
            "Heart is medium-sized with generous page margin on all sides — about half the frame height. "
            "Deep top cleft, two rounded lobes, sharp pointed tip at the bottom. "
            "Bright yellow notes + red handwritten lettering. Cream/off-white page. "
            "Do NOT copy any words from this image — they are unreadable on purpose. "
            "Print ONLY words from the user's numbered vocabulary list."
        )
    return (
        "LAYOUT / COLOR TARGET only (text is intentionally blurred). "
        "Clone the anatomical heart silhouette, monochrome vibrant red ink, warped typography density, "
        "thin red vessel hatching, and cream page. Do NOT copy any words from this image — "
        "they are unreadable on purpose. Print ONLY words from the user's numbered vocabulary list."
    )


def shape_lock(art_style: str | None = None) -> str:
    chosen = normalize_art_style(art_style)
    if chosen != "sticky-heart":
        return ""
    return (
        "SHAPE + SIZE LOCK (highest priority with vocabulary).\n"
        "SIZE: Keep the whole heart COMPACT — roughly 45–55% of the canvas height, centered, "
        "with generous empty cream margin on all four sides. "
        "Do NOT let the heart fill most of the frame. Do NOT make a giant yellow plate.\n"
        "STRUCTURE: The sticky notes ARE the heart. Overlapping yellow notes create the ♥ outline. "
        "Optional: a VERY thin yellow under-edge may peek in tiny gaps / at the tip only — "
        "never a large solid yellow heart showing around the notes.\n"
        "Silhouette:\n"
        "- Deep V cleft at TOP CENTER between two rounded lobes\n"
        "- Sides curve in to a SHARP acute POINT at the bottom (heart tail)\n"
        "- Tip from a downward-pointing note corner (and tiny yellow peek if needed) — "
        "not a blunt square bottom, not a wide rounded blob\n"
        "Notes should look relatively LARGE within the heart (like the style target), "
        "not tiny stickers floating on a huge yellow field."
    )


def vocabulary_lock(words: list[str], art_style: str | None = None) -> str:
    chosen = normalize_art_style(art_style)
    cleaned = _normalized_words(words)
    count = len(cleaned)
    listed = numbered_word_list(cleaned)
    checklist = coverage_checklist(cleaned)
    forbidden = forbidden_style_words(cleaned, chosen)
    hero = _hero_word(cleaned)
    forbid_line = (
        "BAN these words if they are not on the list: " + ", ".join(forbidden) + "."
        if forbidden
        else "Do not invent extra words."
    )
    coverage = (
        f"COMPLETE COVERAGE (non-negotiable): all {count} list entries MUST appear "
        "at least once as readable text. Missing even one word = failed output.\n"
        "Mentally tick every box before finishing:\n"
        f"{checklist}\n"
        "Only AFTER every box is covered may you repeat words to fill leftover space."
    )
    shape = shape_lock(chosen)
    if chosen == "sticky-heart":
        # Prefer one note per word; a few extras only if needed to complete the ♥ outline.
        note_count = count if count >= 18 else max(count, 18)
        return (
            "VOCABULARY LOCK (final instruction, highest priority).\n"
            "The style image is blurred — ignore any guessed reference words.\n"
            f"Closed list of {count} words (spell exactly):\n"
            f"{listed}\n"
            f"{forbid_line}\n"
            f"{coverage}\n"
            f"{shape}\n"
            f"Use about {note_count} sticky notes: one note per list entry first, "
            "then only add repeat notes if the ♥ outline still has holes.\n"
            "Keep the heart compact (~half the frame) with cream margin around it — "
            "notes form the silhouette, not a giant yellow plate.\n"
            "Plain cream or white page behind the heart (real transparency is applied after). "
            "Never use a black page."
        )
    return (
        "VOCABULARY LOCK (final instruction, highest priority).\n"
        "The style image is blurred — ignore any guessed reference words.\n"
        f"Closed list of {count} words (spell exactly):\n"
        f"{listed}\n"
        f"{forbid_line}\n"
        f"{coverage}\n"
        f"Make \"{hero}\" the single largest central word inside the heart body.\n"
        "Do not add synonyms unless they are on the list.\n"
        "Keep the monochrome red anatomical calligram on a plain cream page "
        "(real transparency is applied after generation)."
    )


def build_prompt(words: list[str], art_style: str | None = None) -> str:
    chosen = normalize_art_style(art_style)
    cleaned = _normalized_words(words)
    count = len(cleaned)
    listed = numbered_word_list(cleaned)
    checklist = coverage_checklist(cleaned)
    csv_words = comma_word_list(cleaned)
    forbidden = forbidden_style_words(cleaned, chosen)
    hero = _hero_word(cleaned)
    forbid_line = (
        "Never write: " + ", ".join(forbidden) + "."
        if forbidden
        else "Do not invent extra words."
    )
    if chosen == "sticky-heart":
        note_count = count if count >= 18 else max(count, 18)
        return f"""Draw a COMPACT sticky-note valentine ♥ (notes form the shape).

Clone the attached style target for LAYOUT/COLOR only (its text is blurred — do not invent words from it).
Match THAT scale and look: medium heart with margin, notes overlapping into a ♥ — NOT a giant yellow heart plate.

{shape_lock(chosen)}

COMPLETE COVERAGE FIRST:
You have exactly {count} required words. Every one MUST appear as readable text on its own sticky note before any repeats:
{listed}

Quick scan: {csv_words}

Tick before finishing:
{checklist}

WHAT TO DRAW:
- ~{note_count} bright yellow sticky notes, slightly rotated, overlapping with soft drop shadows.
- Arrange notes into a classic ♥: deep top cleft, two lobes, sharp pointed tip at bottom.
- Heart height about 45–55% of the canvas; keep clear cream margin on every side.
- Notes should be large enough to read easily and should define the outline themselves.
- Optional thin yellow under-edge only in tiny gaps / tip — never a big solid yellow heart showing around the collage.
- Each note shows ONE list entry in casual red handwritten capital letters.
- Most notes include a tiny hand-drawn red heart under the text (optional dashes/underlines like the sample).
- No anatomical vessels, chambers, aorta, or medical sketch lines.

{forbid_line}

FORBIDDEN:
- Oversized heart filling most of the frame
- Giant solid yellow ♥ plate with tiny notes stuck on top
- Blunt / flat / square bottom tip
- Circle / oval / blob arrangements that do not read as ♥
- Skipping any numbered word
- Copying style-sample vocabulary (LOVE, JOY, YOU, CREATE, UNITY, etc. unless listed above)
- Anatomical heart / dense red calligram / UI / captions / buttons
- Black background

BACKGROUND:
- Plain soft cream / off-white field only. No paper grain, no UI, no frame, no black page.
- Do NOT fake checkerboard transparency — leave a solid cream page; real PNG alpha is added afterward.

OUTPUT: compact sticky-note ♥ with sharp tip, generous margin, all {count} words visible. Huge yellow plate = wrong."""

    return f"""Draw an ANATOMICAL human heart that IS a monochrome red typography calligram.

Clone the attached style target for LAYOUT/COLOR only (its text is blurred — do not invent words from it).
Vibrant RED warped lettering forming a realistic anatomical heart on soft cream paper.

COMPLETE COVERAGE FIRST (highest priority after style):
You have exactly {count} required words. Place EVERY numbered word at least once as readable text BEFORE repeating any word to fill gaps:
{listed}

Quick scan of required vocabulary: {csv_words}

Coverage checklist — every box must be satisfied:
{checklist}

WHAT TO DRAW:
- Anatomical heart (atria, ventricles, aorta/pulmonary vessels on top). Slight tilt. NOT a valentine ♥.
- Silhouette built from words; vessels also packed with curved/vertical words.
- "{hero}" is the HERO word: largest, boldest, centered in the main heart body.
- Distribute the other {count - 1} words across chambers and vessels so none are omitted.
- Warp/rotate lettering to follow walls. Thin red contour/hatching between clusters.
- ALL lettering and linework ONE vibrant red. Dense packing after full coverage.
- Clean bold mixed typography — readable, not messy scribbles.

{forbid_line}

FORBIDDEN:
- Skipping any numbered word
- Copying style-sample leftovers (FAMILY, COURAGE, STRENGTH, HOPE, JOY, PASSION, KINDNESS, COMPASSION, LOVE unless listed above)
- Multi-color doodles, sticky-note collage, UI, captions, buttons

BACKGROUND:
- Plain soft cream / off-white field only. No paper grain, no watercolor corners, no UI, no frame.
- Do NOT fake checkerboard transparency — leave a solid cream page; real PNG alpha is added afterward.

OUTPUT: monochrome red anatomical word-heart where all {count} user words appear at least once.
Missing even one list word = failed output. Hero "{hero}" must stay visually dominant."""
