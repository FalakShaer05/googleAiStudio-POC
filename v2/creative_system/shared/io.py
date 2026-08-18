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


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_folder() -> str:
    return current_app.config["UPLOAD_FOLDER"]


def output_folder() -> str:
    return current_app.config["OUTPUT_FOLDER"]


def save_upload(file_storage: Optional[FileStorage], prefix: str) -> Optional[str]:
    if not file_storage or not file_storage.filename:
        return None
    if not allowed_file(file_storage.filename):
        raise ValueError(f"Invalid file type for {prefix}")
    filename = generate_unique_filename(secure_filename(file_storage.filename), prefix)
    path = os.path.join(upload_folder(), filename)
    file_storage.save(path)
    return path


def save_named_upload(field: str, prefix: str, required: bool = False) -> Optional[str]:
    path = save_upload(request.files.get(field), prefix)
    if required and not path:
        raise ValueError(f"{field} image is required")
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
    cloudfront_url = upload_image_to_s3(output_path)
    if cloudfront_url:
        data["image_url"] = cloudfront_url
    return data


def json_error(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status
