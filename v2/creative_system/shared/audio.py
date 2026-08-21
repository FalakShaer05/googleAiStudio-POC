"""Audio upload helpers and Gemini transcription."""
from __future__ import annotations

import os
from typing import Optional, Tuple

from utils.character_utils import get_gemini_client, get_gemini_text_model, log_gemini_token_usage

try:
    from google.genai import types
except Exception:  # pragma: no cover
    types = None  # type: ignore

from .io import ALLOWED_AUDIO_EXTENSIONS, file_extension

AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "wav": "audio/wav",
    "wave": "audio/wav",
    "webm": "audio/webm",
    "opus": "audio/opus",
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "aiff": "audio/aiff",
    "aif": "audio/aiff",
    "amr": "audio/amr",
    "3gp": "audio/3gpp",
}

INLINE_AUDIO_MAX_BYTES = int(os.getenv("GEMINI_INLINE_AUDIO_MAX_BYTES", str(18 * 1024 * 1024)))


def mime_for_audio(path: str, content_type: Optional[str] = None) -> str:
    ext = file_extension(path)
    if ext in AUDIO_MIME_TYPES:
        return AUDIO_MIME_TYPES[ext]
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime.startswith("audio/") or mime.startswith("video/"):
            return mime
    return "application/octet-stream"


def transcribe_audio(audio_path: str, prompt: str, content_type: Optional[str] = None) -> Tuple[bool, str]:
    if not audio_path or not os.path.isfile(audio_path):
        return False, "Audio file was not found"
    ext = file_extension(audio_path)
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return False, f"Unsupported audio format .{ext}" if ext else "Unsupported audio format"

    mime = mime_for_audio(audio_path, content_type)
    client = get_gemini_client()
    model = get_gemini_text_model()

    try:
        response = _generate_transcript(client, model, audio_path, mime, prompt)
    except Exception as exc:
        print(f"creative transcribe_audio error: {exc}")
        import traceback
        traceback.print_exc()
        return False, str(exc)

    log_gemini_token_usage(response, operation="transcription:creative:audio-to-text", model=model)
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        return False, "Gemini did not return a transcript"
    return True, text


def _generate_transcript(client, model: str, audio_path: str, mime: str, prompt: str):
    size = os.path.getsize(audio_path)
    uploaded = None
    try:
        if size > INLINE_AUDIO_MAX_BYTES and hasattr(client, "files"):
            uploaded = _upload_audio(client, audio_path, mime)
            return client.models.generate_content(model=model, contents=[prompt, uploaded])

        with open(audio_path, "rb") as handle:
            data = handle.read()
        if types is not None and hasattr(types, "Part"):
            try:
                audio_part = types.Part.from_bytes(data=data, mime_type=mime)
                return client.models.generate_content(model=model, contents=[prompt, audio_part])
            except Exception as exc:
                print(f"inline audio send failed ({mime}), trying Files API: {exc}")

        if hasattr(client, "files"):
            uploaded = _upload_audio(client, audio_path, mime)
            return client.models.generate_content(model=model, contents=[prompt, uploaded])
        raise RuntimeError("This Gemini SDK cannot send audio parts")
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=getattr(uploaded, "name", None) or uploaded)
            except Exception:
                pass


def _upload_audio(client, audio_path: str, mime: str):
    try:
        return client.files.upload(file=audio_path, config={"mime_type": mime})
    except TypeError:
        return client.files.upload(file=audio_path)
