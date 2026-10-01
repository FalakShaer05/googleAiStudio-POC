import re

STYLE_IDS = ("rings", "heart", "bars")

STYLE_LABELS = {
    "rings": "Circular rings",
    "heart": "Wave heart",
    "bars": "Type waveform",
}

TYPE_FROM_AUDIO_PROMPT = (
    "You are a precise speech-to-text engine.\n"
    "Transcribe the attached audio from the first spoken sound to the last.\n"
    "Return EVERY spoken word in order as one single line of plain text, "
    "with normal spaces between words and no line breaks.\n"
    "Keep fillers that were clearly spoken (um, uh, yeah) only if they are "
    "audible words; otherwise skip pure noise.\n"
    "If the language is not English, transcribe in the original language.\n"
    "Never invent or infer words that were not clearly spoken.\n"
    "If a short segment is not clear, use [inaudible] instead of guessing.\n"
    "Do NOT summarize, title, translate, shorten, or pick a keyword.\n"
    "Do NOT return only the first word or a theme word.\n"
    "Do NOT add quotes, labels, bullets, timestamps, or commentary.\n"
    "Output must be the full spoken line only."
)

TYPE_TEXT_MAX_CHARS = 2000

_PREFIX_RE = re.compile(
    r"^(?:transcript|transcription|spoken(?:\s+line)?|text|output)\s*[:\-–]\s*",
    re.IGNORECASE,
)


def normalize_style(style: str) -> str:
    key = (style or "rings").strip().lower()
    return key if key in STYLE_IDS else "rings"


def type_from_transcript(text: str) -> str:
    cleaned = str(text or "").replace("\r", "\n").strip()
    if not cleaned:
        return ""

    # Prefer the longest plain line if the model wrapped the answer.
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    if len(lines) > 1:
        cleaned = max(lines, key=len)

    cleaned = _PREFIX_RE.sub("", cleaned).strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()

    cleaned = " ".join(cleaned.split()).strip()
    if len(cleaned) <= TYPE_TEXT_MAX_CHARS:
        return cleaned
    trimmed = cleaned[:TYPE_TEXT_MAX_CHARS].rsplit(" ", 1)[0].strip()
    return trimmed or cleaned[:TYPE_TEXT_MAX_CHARS]
