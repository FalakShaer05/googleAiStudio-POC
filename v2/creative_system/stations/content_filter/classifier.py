"""Gemini-backed classifier for Wish and Wisdom submissions."""
from __future__ import annotations

import json
from typing import Any

from google.genai import types

from utils.character_utils import get_gemini_client, get_gemini_text_model

from .prompts import build_classification_prompt

ENTRY_TYPES = {"wish", "wisdom"}
MAX_ENTRY_LENGTH = 5000

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "blocked": {"type": "BOOLEAN"},
        "reason": {
            "type": "STRING",
            "nullable": True,
        },
    },
    "required": ["blocked", "reason"],
}


def validate_input(entry_type: str, text: str) -> tuple[str, str]:
    normalized_type = (entry_type or "").strip().lower()
    normalized_text = (text or "").strip()
    if normalized_type not in ENTRY_TYPES:
        raise ValueError("Entry type must be either 'wish' or 'wisdom'.")
    if not normalized_text:
        raise ValueError("Enter a sentence or paragraph to check.")
    if len(normalized_text) > MAX_ENTRY_LENGTH:
        raise ValueError(f"Entry must be {MAX_ENTRY_LENGTH} characters or fewer.")
    return normalized_type, normalized_text


def normalize_result(payload: Any) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("blocked"), bool):
        raise ValueError("The content filter returned an invalid response.")

    blocked = payload["blocked"]
    reason = payload.get("reason")
    if blocked:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A blocked decision must include a reason.")
        return {"blocked": True, "reason": reason.strip()}
    return {"blocked": False, "reason": None}


def classify_content(entry_type: str, text: str) -> dict:
    normalized_type, normalized_text = validate_input(entry_type, text)
    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_gemini_text_model(),
        contents=build_classification_prompt(normalized_type, normalized_text),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        ),
    )

    response_text = getattr(response, "text", None)
    if not response_text:
        raise RuntimeError(
            "The entry could not be checked because the AI service returned no decision. "
            "Please revise it and try again."
        )

    try:
        payload = json.loads(response_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("The content filter returned an unreadable decision.") from exc
    try:
        return normalize_result(payload)
    except ValueError as exc:
        raise RuntimeError("The content filter returned an invalid decision.") from exc
