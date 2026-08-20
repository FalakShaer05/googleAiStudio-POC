"""Upload, word parsing, and JSON response helpers for the creative system."""
from __future__ import annotations

import json
import os
from typing import Optional

from flask import current_app, jsonify, request
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from utils.character_utils import cleanup_file, generate_unique_filename
from utils.s3_utils import upload_image_to_s3

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}
ALLOWED_AUDIO_EXTENSIONS = {
    "mp3",
    "mpeg",
    "mpga",
    "wav",
    "wave",
    "webm",
    "opus",
    "ogg",
    "oga",
    "m4a",
    "mp4",
    "aac",
    "flac",
    "aiff",
    "aif",
    "amr",
    "3gp",
}

AUDIO_MIME_TO_EXT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mpeg3": "mp3",
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/opus": "opus",
    "audio/ogg": "ogg",
    "application/ogg": "ogg",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/vnd.wave": "wav",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/aac": "aac",
    "audio/x-aac": "aac",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/aiff": "aiff",
    "audio/x-aiff": "aiff",
    "audio/amr": "amr",
    "audio/3gpp": "3gp",
    "video/mp4": "mp4",
    "video/3gpp": "3gp",
}


def file_extension(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def allowed_file(filename: str, allowed: Optional[set] = None) -> bool:
    ext = file_extension(filename)
    return bool(ext) and ext in (allowed or ALLOWED_EXTENSIONS)


def _filename_from_audio_mime(file_storage: FileStorage, original: str) -> Optional[str]:
    mime = (file_storage.mimetype or "").split(";", 1)[0].strip().lower()
    ext = AUDIO_MIME_TO_EXT.get(mime)
    if not ext:
        return None
    base = original.rsplit(".", 1)[0] if original and "." in original else (original or "audio")
    return f"{base}.{ext}"


def normalize_upload_filename(file_storage: FileStorage, allowed: Optional[set]) -> str:
    original = file_storage.filename or ""
    if allowed_file(original, allowed=allowed):
        return original
    if allowed == ALLOWED_AUDIO_EXTENSIONS:
        inferred = _filename_from_audio_mime(file_storage, original)
        if inferred:
            return inferred
    return original


def upload_folder() -> str:
    return current_app.config["UPLOAD_FOLDER"]


def output_folder() -> str:
    return current_app.config["OUTPUT_FOLDER"]


def save_upload(
    file_storage: Optional[FileStorage],
    prefix: str,
    allowed: Optional[set] = None,
) -> Optional[str]:
    if not file_storage:
        return None
    source_name = normalize_upload_filename(file_storage, allowed)
    if not source_name:
        return None
    if not allowed_file(source_name, allowed=allowed):
        raise ValueError(
            f"Invalid file type for {prefix}. Use a supported format such as mp3, webm, opus, wav, or m4a."
            if allowed == ALLOWED_AUDIO_EXTENSIONS
            else f"Invalid file type for {prefix}"
        )
    safe_name = secure_filename(source_name) or source_name
    filename = generate_unique_filename(safe_name, prefix)
    path = os.path.join(upload_folder(), filename)
    file_storage.save(path)
    return path


def save_named_upload(
    field: str,
    prefix: str,
    required: bool = False,
    allowed: Optional[set] = None,
    kind: str = "image",
) -> Optional[str]:
    path = save_upload(request.files.get(field), prefix, allowed=allowed)
    if required and not path:
        raise ValueError(f"{field} {kind} is required")
    return path


def parse_word_list() -> list[str]:
    """Merge JSON/chip words with a free-text custom_words field."""
    collected: list[str] = []
    raw = (request.form.get("words") or "").strip()
    custom = (request.form.get("custom_words") or "").strip()

    if raw:
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    collected.extend(str(item) for item in parsed)
                else:
                    raise ValueError("words must be a JSON array")
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid words JSON") from exc
        else:
            collected.extend(raw.replace("\n", ",").split(","))

    if custom:
        collected.extend(custom.replace("\n", ",").split(","))

    seen = set()
    unique: list[str] = []
    for word in collected:
        cleaned = " ".join(str(word).split()).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def cleanup_paths(*paths: Optional[str]) -> None:
    for path in paths:
        cleanup_file(path)


def success_payload(output_filename: str, message: str) -> dict:
    data = {
        "success": True,
        "message": message,
        "output_filename": output_filename,
        "local_path": f"/outputs/{output_filename}",
    }
    output_path = os.path.join(output_folder(), output_filename)
    if output_filename.lower().endswith(".txt") and os.path.isfile(output_path):
        data["result_type"] = "text"
        with open(output_path, encoding="utf-8") as handle:
            data["transcript"] = handle.read()
        return data
    data["result_type"] = "image"
    cloudfront_url = upload_image_to_s3(output_path)
    if cloudfront_url:
        data["image_url"] = cloudfront_url
    return data


def json_error(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status
