STYLE_IDS = ("rings", "heart", "bars")

STYLE_LABELS = {
    "rings": "Circular rings",
    "heart": "Wave heart",
    "bars": "Type waveform",
}

TYPE_FROM_AUDIO_PROMPT = (
    "Transcribe the attached audio faithfully. Return only the spoken words "
    "as a single line, with normal spaces and no line breaks. If the language "
    "is not English, transcribe in the original language. Do not add a title, "
    "summary, translation, or commentary."
)

TYPE_TEXT_MAX_CHARS = 400


def normalize_style(style: str) -> str:
    key = (style or "rings").strip().lower()
    return key if key in STYLE_IDS else "rings"


def type_from_transcript(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if len(cleaned) <= TYPE_TEXT_MAX_CHARS:
        return cleaned
    trimmed = cleaned[:TYPE_TEXT_MAX_CHARS].rsplit(" ", 1)[0].strip()
    return trimmed or cleaned[:TYPE_TEXT_MAX_CHARS]
