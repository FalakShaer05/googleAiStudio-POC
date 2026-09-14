"""Reassemble decorated jigsaw pieces into a full image."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

_PIECE_ID_RE = re.compile(r"r(\d+)_c(\d+)", re.IGNORECASE)


def _load_layout(layout: Optional[Dict[str, Any]] = None, layout_path: Optional[str] = None) -> Dict[str, Any]:
    if layout and isinstance(layout, dict):
        return layout
    if layout_path:
        with open(layout_path, encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError("layout JSON or layout file is required to assemble the puzzle")


def _piece_lookup(layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in layout.get("pieces") or []:
        piece_id = str(item.get("piece_id") or "").strip()
        if piece_id:
            lookup[piece_id] = item
            lookup[piece_id.lower()] = item
    return lookup


def _infer_piece_id(filename: str, piece_ids: Optional[List[str]], index: int) -> Optional[str]:
    if piece_ids and index < len(piece_ids) and piece_ids[index]:
        return str(piece_ids[index]).strip()
    name = os.path.basename(filename or "")
    match = _PIECE_ID_RE.search(name)
    if match:
        return f"r{match.group(1)}_c{match.group(2)}"
    return None


def assemble_puzzle(
    piece_paths: List[str],
    output_path: str,
    layout: Optional[Dict[str, Any]] = None,
    layout_path: Optional[str] = None,
    original_path: Optional[str] = None,
    piece_ids: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Paste decorated puzzle pieces onto a canvas using split layout metadata.

    `original_path` (optional) locks canvas size to the source image.
    `piece_ids` (optional) maps each uploaded piece file to a layout piece_id
    in the same order as `piece_paths`.
    """
    meta = _load_layout(layout=layout, layout_path=layout_path)
    lookup = _piece_lookup(meta)
    if not lookup:
        return False, "Layout has no piece metadata"

    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if original_path and os.path.isfile(original_path):
        with Image.open(original_path) as original:
            ow, oh = original.size
        if width <= 0 or height <= 0:
            width, height = ow, oh
    if width <= 0 or height <= 0:
        return False, "Layout is missing canvas size"

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    if not piece_paths:
        return False, "At least one puzzle piece image is required"

    placed = 0
    missing: List[str] = []
    for index, path in enumerate(piece_paths):
        if not path or not os.path.isfile(path):
            missing.append(f"missing file at index {index}")
            continue
        piece_id = _infer_piece_id(path, piece_ids, index)
        if not piece_id:
            missing.append(f"could not determine piece_id for {os.path.basename(path)}")
            continue
        info = lookup.get(piece_id) or lookup.get(piece_id.lower())
        if not info:
            missing.append(f"unknown piece_id '{piece_id}'")
            continue
        bbox = info.get("bbox") or {}
        left = int(bbox.get("left", 0))
        top = int(bbox.get("top", 0))
        right = int(bbox.get("right", left))
        bottom = int(bbox.get("bottom", top))
        expected_w = max(1, right - left)
        expected_h = max(1, bottom - top)

        piece = Image.open(path).convert("RGBA")
        if piece.size != (expected_w, expected_h):
            piece = piece.resize((expected_w, expected_h), Image.Resampling.LANCZOS)

        # BBox may extend past the canvas (transparent padding around tabs).
        paste_x, paste_y = left, top
        src_l = src_t = 0
        src_r, src_b = piece.size
        if paste_x < 0:
            src_l = -paste_x
            paste_x = 0
        if paste_y < 0:
            src_t = -paste_y
            paste_y = 0
        if paste_x + (src_r - src_l) > width:
            src_r = src_l + max(0, width - paste_x)
        if paste_y + (src_b - src_t) > height:
            src_b = src_t + max(0, height - paste_y)
        if src_r <= src_l or src_b <= src_t:
            missing.append(f"piece '{piece_id}' is outside the canvas")
            continue
        clipped = piece.crop((src_l, src_t, src_r, src_b))
        canvas.alpha_composite(clipped, dest=(paste_x, paste_y))
        placed += 1

    if placed == 0:
        detail = "; ".join(missing[:5]) if missing else "no pieces placed"
        return False, f"Could not assemble puzzle: {detail}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.convert("RGB").save(output_path, "PNG")
    note = f"Assembled {placed} puzzle piece(s) into one image."
    if missing:
        note += f" Skipped {len(missing)}: " + "; ".join(missing[:3])
    return True, note
