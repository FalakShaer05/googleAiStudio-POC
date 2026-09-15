"""Creative System HTTP routes."""
from __future__ import annotations

import json
import os
import traceback

from flask import jsonify, render_template, request

from utils.auth import require_api_key
from utils.character_utils import generate_unique_filename
from utils.s3_utils import upload_image_to_s3

from .blueprint import api_bp, bp
from .shared.io import (
    ALLOWED_AUDIO_EXTENSIONS,
    cleanup_paths,
    json_error,
    output_folder,
    parse_word_list,
    save_named_upload,
    save_upload,
    success_payload,
    upload_folder,
)
from .shared.maps import fetch_static_map
from .shared.registry import STATION_IDS, STATIONS, get_generator
from .stations.content_filter import classify_content
from .stations.puzzle_collage import assemble_puzzle, split_puzzle


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
        if station_id == "content-filter":
            return json_error("Use the Wish & Wisdom content filter endpoint for this station.")
        if station_id == "puzzle-collage":
            return json_error(
                "Use the Puzzle Collage split and assemble endpoints for this station."
            )

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

        elif station_id == "classic-my-way":
            kwargs["template_path"] = save_named_upload("template", "cs_classic_template", required=True)
            kwargs["artwork_path"] = save_named_upload("artwork", "cs_classic_my_way", required=True)
            temp_paths.extend([kwargs["template_path"], kwargs["artwork_path"]])
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

        elif station_id in {"graphic-heart", "apimh-new"}:
            kwargs["message"] = (request.form.get("message") or "").strip()
            kwargs["location_label"] = (request.form.get("location_label") or "").strip()
            lat_raw = (request.form.get("latitude") or "").strip()
            lng_raw = (request.form.get("longitude") or "").strip()
            if station_id != "apimh-new" and not kwargs["message"]:
                return json_error("A message is required")
            try:
                kwargs["latitude"] = float(lat_raw) if lat_raw else None
                kwargs["longitude"] = float(lng_raw) if lng_raw else None
            except ValueError:
                return json_error("Latitude and longitude must be numbers")

            marker_color = "0x111111"
            if station_id == "apimh-new":
                map_type = (request.form.get("map_type") or request.form.get("type") or "explorer").strip().lower()
                if map_type not in {"explorer", "dreamer"}:
                    return json_error("Choose a type: explorer or dreamer")
                kwargs["map_type"] = map_type
                if map_type == "dreamer":
                    marker_color = "0xE85A2D"

            map_path = save_named_upload("map_image", "cs_map", required=False)
            if map_path:
                temp_paths.append(map_path)
            elif kwargs["latitude"] is not None and kwargs["longitude"] is not None:
                static_name = generate_unique_filename("static_map.png", "cs_static_map")
                static_path = os.path.join(upload_folder(), static_name)
                try:
                    fetched = fetch_static_map(
                        kwargs["latitude"],
                        kwargs["longitude"],
                        static_path,
                        marker_color=marker_color,
                    )
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


def _filter_content_impl():
    try:
        data = request.get_json(silent=True) if request.is_json else request.form
        if data is None or not hasattr(data, "get"):
            raise ValueError("Request body must contain entry_type and text.")
        entry_type = str(data.get("entry_type") or "")
        text = str(data.get("text") or "")
        result = classify_content(entry_type, text)
        return jsonify(
            {
                "success": True,
                "entry_type": entry_type.strip().lower(),
                **result,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        print("Error in Wish & Wisdom content filter:", exc)
        print(traceback.format_exc())
        return jsonify(
            {
                "success": False,
                "error": (
                    "The entry could not be checked right now. "
                    "Please try again before submitting it."
                ),
            }
        ), 502


@bp.route("/filter-content", methods=["POST"])
def filter_content():
    return _filter_content_impl()


@api_bp.route("/filter-wish-wisdom", methods=["POST"])
@require_api_key
def api_filter_content():
    """
    Check whether a Wish or Wisdom entry is eligible for public display.
    ---
    tags:
      - Creative System
    consumes:
      - application/json
      - application/x-www-form-urlencoded
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required: [entry_type, text]
          properties:
            entry_type:
              type: string
              enum: [wish, wisdom]
            text:
              type: string
    responses:
      200:
        description: Classification with blocked boolean and reason
      400:
        description: Invalid input
      401:
        description: Missing or invalid API key
      502:
        description: AI classification unavailable
    """
    return _filter_content_impl()


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
        description: holding-hands, make-art-yours, classic-my-way, selfie-becoming, tracing-hand, word-art-heart, graphic-heart, apimh-new, audio-to-text, audio-type
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


def _parse_participants() -> int:
    raw = (request.form.get("participants") or request.form.get("participant_count") or "").strip()
    if not raw:
        raise ValueError("participants is required (number of people)")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("participants must be an integer") from exc
    if value < 1:
        raise ValueError("participants must be at least 1")
    if value > 40:
        raise ValueError("participants cannot exceed 40 for this POC")
    return value


def _puzzle_split_impl():
    temp_paths = []
    try:
        image_path = save_named_upload("image", "cs_puzzle_src", required=True)
        temp_paths.append(image_path)
        participants = _parse_participants()
        seed_raw = (request.form.get("seed") or "").strip()
        seed = int(seed_raw) if seed_raw else None

        result = split_puzzle(
            image_path=image_path,
            participants=participants,
            output_dir=output_folder(),
            seed=seed,
        )

        pieces_payload = []
        for piece in result["pieces"]:
            item = {
                "piece_id": piece["piece_id"],
                "person": piece["person"],
                "person_tag": piece["person_tag"],
                "row": piece["row"],
                "col": piece["col"],
                "index": piece["index"],
                "bbox": piece["bbox"],
                "output_filename": piece["output_filename"],
                "local_path": piece["local_path"],
            }
            cloudfront_url = upload_image_to_s3(piece["abs_path"])
            if cloudfront_url:
                item["image_url"] = cloudfront_url
            pieces_payload.append(item)

        return jsonify(
            {
                "success": True,
                "message": (
                    f"Split into {result['total_pieces']} pieces for "
                    f"{participants} participant(s)."
                ),
                "participants": participants,
                "rows": result["rows"],
                "cols": result["cols"],
                "total_pieces": result["total_pieces"],
                "pieces_per_person": result["pieces_per_person"],
                "pieces": pieces_payload,
                "layout": result["layout"],
                "layout_filename": result["layout_filename"],
                "layout_local_path": result["layout_local_path"],
            }
        )
    except ValueError as exc:
        return json_error(str(exc))
    except Exception as exc:
        print("Error in puzzle-collage split:", exc)
        print(traceback.format_exc())
        return json_error(str(exc), 500)
    finally:
        cleanup_paths(*temp_paths)


def _collect_piece_uploads(temp_paths: list) -> tuple[list[str], list[str]]:
    """Collect piece image uploads and optional parallel piece_ids."""
    piece_paths: list[str] = []
    piece_ids: list[str] = []

    # Multi-file field "pieces"
    for storage in request.files.getlist("pieces"):
        path = save_upload(storage, "cs_puzzle_decorated")
        if path:
            temp_paths.append(path)
            piece_paths.append(path)

    # Indexed fields piece_0, piece_1, ...
    if not piece_paths:
        index = 0
        while True:
            path = save_named_upload(f"piece_{index}", "cs_puzzle_decorated", required=False)
            if not path:
                break
            temp_paths.append(path)
            piece_paths.append(path)
            index += 1

    # Single "piece" fallback
    if not piece_paths:
        path = save_named_upload("piece", "cs_puzzle_decorated", required=False)
        if path:
            temp_paths.append(path)
            piece_paths.append(path)

    raw_ids = (request.form.get("piece_ids") or "").strip()
    if raw_ids:
        if raw_ids.startswith("["):
            parsed = json.loads(raw_ids)
            if not isinstance(parsed, list):
                raise ValueError("piece_ids must be a JSON array of strings")
            piece_ids = [str(item).strip() for item in parsed]
        else:
            piece_ids = [part.strip() for part in raw_ids.split(",") if part.strip()]

    # Optional parallel form fields piece_id_0 ...
    if not piece_ids:
        index = 0
        collected = []
        while True:
            value = (request.form.get(f"piece_id_{index}") or "").strip()
            if not value and index >= len(piece_paths):
                break
            collected.append(value)
            index += 1
            if index > len(piece_paths) + 5:
                break
        if any(collected):
            piece_ids = collected[: len(piece_paths)]

    return piece_paths, piece_ids


def _puzzle_assemble_impl():
    temp_paths = []
    try:
        original_path = save_named_upload("original", "cs_puzzle_original", required=False)
        if original_path:
            temp_paths.append(original_path)
        # Also accept "image" as the original.
        if not original_path:
            original_path = save_named_upload("image", "cs_puzzle_original", required=False)
            if original_path:
                temp_paths.append(original_path)

        piece_paths, piece_ids = _collect_piece_uploads(temp_paths)
        if not piece_paths:
            return json_error("Upload at least one decorated puzzle piece (pieces)")

        layout = None
        layout_path = None
        layout_raw = (request.form.get("layout") or "").strip()
        layout_filename = (request.form.get("layout_filename") or "").strip()

        if layout_raw:
            try:
                layout = json.loads(layout_raw)
            except json.JSONDecodeError as exc:
                raise ValueError("layout must be valid JSON") from exc
        elif layout_filename:
            safe_name = os.path.basename(layout_filename)
            candidate = os.path.join(output_folder(), safe_name)
            if not os.path.isfile(candidate):
                return json_error(f"layout_filename not found: {safe_name}")
            layout_path = candidate
        else:
            uploaded_layout = save_named_upload(
                "layout_file",
                "cs_puzzle_layout_in",
                required=False,
                allowed={"json"},
                kind="layout",
            )
            if uploaded_layout:
                temp_paths.append(uploaded_layout)
                layout_path = uploaded_layout

        # layout is optional now — assembler can read placement from piece PNG metadata
        out_filename = generate_unique_filename("creative.png", "output_puzzle_collage")
        out_path = os.path.join(output_folder(), out_filename)
        success, message = assemble_puzzle(
            piece_paths=piece_paths,
            output_path=out_path,
            layout=layout,
            layout_path=layout_path,
            original_path=original_path,
            piece_ids=piece_ids or None,
        )
        if not success:
            return json_error(message or "Assembly failed", 500)
        return jsonify(success_payload(out_filename, message))
    except ValueError as exc:
        return json_error(str(exc))
    except Exception as exc:
        print("Error in puzzle-collage assemble:", exc)
        print(traceback.format_exc())
        return json_error(str(exc), 500)
    finally:
        cleanup_paths(*temp_paths)


@bp.route("/puzzle-collage/split", methods=["POST"])
def puzzle_split():
    return _puzzle_split_impl()


@bp.route("/puzzle-collage/assemble", methods=["POST"])
def puzzle_assemble():
    return _puzzle_assemble_impl()


@api_bp.route("/puzzle-collage-split", methods=["POST"])
@require_api_key
def api_puzzle_split():
    """
    Split an image into jigsaw pieces assigned to participants.
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
        name: image
        type: file
        required: true
      - in: formData
        name: participants
        type: integer
        required: true
        description: Number of people (each gets 3-6 pieces)
    responses:
      200:
        description: Pieces tagged by person plus layout JSON for assemble
      400:
        description: Invalid input
      401:
        description: Missing or invalid API key
      500:
        description: Split failed
    """
    return _puzzle_split_impl()


@api_bp.route("/puzzle-collage-assemble", methods=["POST"])
@require_api_key
def api_puzzle_assemble():
    """
    Join decorated puzzle pieces back into one image.
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
        name: original
        type: file
        required: false
        description: Optional original image for canvas sizing
      - in: formData
        name: pieces
        type: file
        required: true
        description: One or more decorated piece images (multi-file). Split PNGs embed placement metadata.
      - in: formData
        name: layout_filename
        type: string
        required: false
        description: Filename returned by split (e.g. cs_puzzle_layout_….json). Preferred over full layout JSON.
      - in: formData
        name: layout
        type: string
        required: false
        description: Optional full layout JSON from split
      - in: formData
        name: layout_file
        type: file
        required: false
      - in: formData
        name: piece_ids
        type: string
        required: false
        description: Optional comma-separated piece ids matching pieces order (r0_c0,r0_c1). Only needed if files were renamed.
    responses:
      200:
        description: Assembled image
      400:
        description: Invalid input
      401:
        description: Missing or invalid API key
      500:
        description: Assembly failed
    """
    return _puzzle_assemble_impl()
