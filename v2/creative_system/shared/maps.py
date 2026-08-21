"""Google Static Maps + reverse geocode helpers."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import requests

STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def maps_api_key() -> str:
    return (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip()


def fetch_static_map(
    latitude: float,
    longitude: float,
    dest_path: str,
    zoom: int = 16,
    size: str = "640x640",
) -> Optional[str]:
    key = maps_api_key()
    if not key:
        return None
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom,
        "size": size,
        "scale": 2,
        "maptype": "roadmap",
        "markers": f"color:0x111111|{latitude},{longitude}",
        "key": key,
    }
    response = requests.get(STATIC_MAP_URL, params=params, timeout=20)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type or not response.content:
        raise RuntimeError("Static Maps request failed")
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "wb") as handle:
        handle.write(response.content)
    return dest_path


def reverse_geocode_label(latitude: float, longitude: float) -> str:
    key = maps_api_key()
    if not key:
        return ""
    response = requests.get(
        GEOCODE_URL,
        params={"latlng": f"{latitude},{longitude}", "key": key},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    if not results:
        return ""

    components = results[0].get("address_components") or []

    def _component(types: Tuple[str, ...]) -> str:
        for item in components:
            if any(t in item.get("types", []) for t in types):
                return item.get("short_name") or item.get("long_name") or ""
        return ""

    neighborhood = _component(("neighborhood", "sublocality", "sublocality_level_1"))
    city = _component(("locality", "postal_town", "administrative_area_level_2"))
    region = _component(("administrative_area_level_1",))
    place = neighborhood or city
    if place and region:
        return f"{place}, {region}".upper()
    if place:
        return place.upper()
    formatted = results[0].get("formatted_address") or ""
    return formatted.upper()
