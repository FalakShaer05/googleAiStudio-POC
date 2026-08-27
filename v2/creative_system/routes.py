"""Creative System HTTP routes."""
from __future__ import annotations

import os
import traceback

from flask import jsonify, render_template, request

from utils.auth import require_api_key
from utils.character_utils import generate_unique_filename

from .blueprint import api_bp, bp
from .shared.io import (
    ALLOWED_AUDIO_EXTENSIONS,
    cleanup_paths,
    json_error,
    output_folder,
    parse_word_list,
    save_named_upload,
    success_payload,
    upload_folder,
)
from .shared.maps import fetch_static_map
from .shared.registry import STATION_IDS, STATIONS, get_generator


def _page_context():
    return {
        "stations": STATIONS,
        "word_chips": {
            "tracing-hand": _station_chips("tracing_hand"),
            "word-art-heart": _station_chips("word_art_heart"),
        },
    }


def _station_chips(module: str) -> list[str]:
    prompts = __import__(
        f"creative_system.stations.{module}.prompts",
        fromlist=["WORD_CHIPS"],
    )
    return list(getattr(prompts, "WORD_CHIPS", []))


@bp.route("/")
def hub():
    return render_template("hub.html", **_page_context())


def _generate_impl():
    temp_paths = []
    try:
        station_id = (request.form.get("station") or "").strip().lower()
        if station_id not in STATION_IDS:
            return json_error("Invalid station. Choose a valid creative station.")

        kwargs = {"station_id": station_id}

        if station_id == "holding-hands":
            kwargs["photo_a_path"] = save_named_upload("photo_a", "cs_holding_a", required=True)
            kwargs["photo_b_path"] = save_named_upload("photo_b", "cs_holding_b", required=True)
            temp_paths.extend([kwargs["photo_a_path"], kwargs["photo_b_path"]])
            kwargs["name_a"] = (request.form.get("name_a") or "").strip()
            kwargs["name_b"] = (request.form.get("name_b") or "").strip()
            kwargs["date_text"] = (request.form.get("date") or "").strip()
            kwargs["caption"] = (request.form.get("caption") or "").strip()
            if not kwargs["name_a"] or not kwargs["name_b"] or not kwargs["date_text"]:
                return json_error("Both names and a date are required")

        elif station_id == "make-art-yours":
            kwargs["artwork_path"] = save_named_upload("artwork", "cs_artwork", required=True)
            temp_paths.append(kwargs["artwork_path"])
            kwargs["user_prompt"] = (request.form.get("prompt") or "").strip()
            if not kwargs["user_prompt"]:
                return json_error("A prompt is required")

        elif station_id == "selfie-becoming":
            kwargs["selfie_path"] = save_named_upload("selfie", "cs_selfie", required=True)
            temp_paths.append(kwargs["selfie_path"])

        elif station_id == "tracing-hand":
            kwargs["hand_path"] = save_named_upload("hand", "cs_hand", required=True)
            temp_paths.append(kwargs["hand_path"])
            kwargs["words"] = parse_word_list()
            if len(kwargs["words"]) < 3:
                return json_error("Pick or enter at least 3 words")

        elif station_id == "word-art-heart":
            kwargs["words"] = parse_word_list()
            if len(kwargs["words"]) < 3:
                return json_error("Pick or enter at least 3 words")

        elif station_id == "graphic-heart":
            kwargs["message"] = (request.form.get("message") or "").strip()
            kwargs["location_label"] = (request.form.get("location_label") or "").strip()
            lat_raw = (request.form.get("latitude") or "").strip()
            lng_raw = (request.form.get("longitude") or "").strip()
            if not kwargs["message"]:
                return json_error("A message is required")
            try:
                kwargs["latitude"] = float(lat_raw) if lat_raw else None
                kwargs["longitude"] = float(lng_raw) if lng_raw else None
            except ValueError:
                return json_error("Latitude and longitude must be numbers")

            map_path = save_named_upload("map_image", "cs_map", required=False)
            if map_path:
                temp_paths.append(map_path)
            elif kwargs["latitude"] is not None and kwargs["longitude"] is not None:
                static_name = generate_unique_filename("static_map.png", "cs_static_map")
                static_path = os.path.join(upload_folder(), static_name)
                try:
                    fetched = fetch_static_map(kwargs["latitude"], kwargs["longitude"], static_path)
                except Exception as map_exc:
                    return json_error(
                        f"Could not fetch a map snapshot ({map_exc}). Upload a map screenshot instead."
                    )
                if not fetched:
                    return json_error(
                        "Could not fetch a map image. Upload a map screenshot or set GOOGLE_MAPS_API_KEY."
                    )
                map_path = fetched
                temp_paths.append(map_path)
            else:
                return json_error("Pick a map location or upload a map screenshot with latitude and longitude")
            kwargs["map_image_path"] = map_path

        elif station_id == "audio-to-text":
            kwargs["audio_path"] = save_named_upload(
                "audio",
                "cs_audio",
                required=True,
                allowed=ALLOWED_AUDIO_EXTENSIONS,
                kind="audio",
            )
            temp_paths.append(kwargs["audio_path"])

        elif station_id == "audio-type":
            kwargs["audio_path"] = save_named_upload(
                "audio",
                "cs_audio_type",
                required=True,
                allowed=ALLOWED_AUDIO_EXTENSIONS,
                kind="audio",
            )
            temp_paths.append(kwargs["audio_path"])
            kwargs["style"] = (request.form.get("style") or "rings").strip().lower()
            if kwargs["style"] not in {"rings", "heart", "bars"}:
                return json_error("Choose a visualization style: rings, heart, or bars")

        if station_id == "audio-to-text":
            out_filename = generate_unique_filename("creative.txt", f"output_{station_id.replace('-', '_')}")
        else:
            out_filename = generate_unique_filename("creative.png", f"output_{station_id.replace('-', '_')}")
        out_path = os.path.join(output_folder(), out_filename)
        kwargs["output_path"] = out_path

        success, message = get_generator(station_id)(**kwargs)
        if not success:
            return json_error(message or "Generation failed", 500)
        return jsonify(success_payload(out_filename, message))
    except ValueError as exc:
        return json_error(str(exc))
    except Exception as exc:
        print("Error in creative-system generate:", exc)
        print(traceback.format_exc())
        return json_error(str(exc), 500)
    finally:
        cleanup_paths(*temp_paths)


@bp.route("/generate", methods=["POST"])
def generate():
    return _generate_impl()


@api_bp.route("/generate-creative", methods=["POST"])
@require_api_key
def api_generate():
    """
    Generate a Creative System artwork.
    ---
    tags:
      - Creative System
    consumes:
      - multipart/form-data
    parameters:
      - in: header
        name: X-API-Key
        type: string
      - in: formData
        name: station
        type: string
        required: true
        description: holding-hands, make-art-yours, selfie-becoming, tracing-hand, word-art-heart, graphic-heart, audio-to-text, audio-type
    responses:
      200:
        description: Artwork generated
      400:
        description: Invalid input
      401:
        description: Missing API key
      500:
        description: Generation failed
    """
    return _generate_impl()
