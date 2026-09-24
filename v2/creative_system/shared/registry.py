"""Station metadata and generator dispatch."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

GenerateFn = Callable[..., Tuple[bool, str]]

STATIONS = [
    {
        "id": "holding-hands",
        "label": "Holding Hands",
        "tagline": "Two photos, two names, one keepsake.",
        "module": "holding_hands",
    },
    {
        "id": "make-art-yours",
        "label": "Make Art Yours",
        "tagline": "Upload a masterpiece. Prompt the rest.",
        "module": "make_art_yours",
    },
    {
        "id": "classic-my-way",
        "label": "Classic My Way",
        "tagline": "Template + user art + prompt. Enhance the additions.",
        "module": "classic_my_way",
    },
    {
        "id": "selfie-becoming",
        "label": "Selfie Becoming",
        "tagline": "Half photo, half light line art — a portrait becoming a sketch.",
        "module": "selfie_becoming",
    },
    {
        "id": "me-remix",
        "label": "Me Remix",
        "tagline": "Your selfie remixed: color photo meets light line art.",
        "module": "me_remix",
    },
    {
        "id": "tracing-hand",
        "label": "Tracing Hand",
        "tagline": "Your hand, filled with words that matter.",
        "module": "tracing_hand",
    },
    {
        "id": "word-art-heart",
        "label": "Word Art — Heart",
        "tagline": "A colorful word-cloud heart made of your words.",
        "module": "word_art_heart",
    },
    {
        "id": "graphic-heart",
        "label": "Graphic Heart",
        "tagline": "A map of the place it all began.",
        "module": "graphic_heart",
    },
    {
        "id": "apimh-new",
        "label": "APIMH-New",
        "tagline": "Explorer or Dreamer heart map.",
        "module": "apimh_new",
    },
    {
        "id": "audio-to-text",
        "label": "Audio to Text",
        "tagline": "Upload a recording. Get a transcript.",
        "module": "audio_to_text",
    },
    {
        "id": "audio-type",
        "label": "Audio Type",
        "tagline": "A recording mapped into neon waveform art.",
        "module": "audio_type",
    },
    {
        "id": "content-filter",
        "label": "Wish & Wisdom Filter",
        "tagline": "Check an entry before it is displayed.",
        "module": "content_filter",
        "kind": "filter",
    },
    {
        "id": "puzzle-collage",
        "label": "Puzzle Collage",
        "tagline": "Turn a photo into fillable line-art pieces, then reassemble the art.",
        "module": "puzzle_collage",
        "kind": "puzzle",
    },
]

STATION_IDS = {item["id"] for item in STATIONS}

_GENERATOR_CACHE: Dict[str, GenerateFn] = {}


def get_station(station_id: str) -> dict:
    for item in STATIONS:
        if item["id"] == station_id:
            return item
    raise KeyError(station_id)


def get_generator(station_id: str) -> GenerateFn:
    if station_id in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[station_id]

    station = get_station(station_id)
    module_name = station["module"]
    generator = __import__(
        f"creative_system.stations.{module_name}.generator",
        fromlist=["generate"],
    )
    _GENERATOR_CACHE[station_id] = generator.generate
    return generator.generate
