"""Reassemble decorated jigsaw pieces into a full image."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

META_KEY = "puzzle_collage"
OUTPUT_DPI = 300
# Minimal PNG compression keeps files closer to source size (lossless).
PNG_COMPRESS_LEVEL = 0
# Must match splitter.BORDER_* so assemble can undo the decorative stroke.
BORDER_WIDTH_PX = 3
BORDER_COLOR = (0x1B, 0x1B, 0x1B)
_PIECE_ID_RE = re.compile(r"r(\d+)_c(\d+)", re.IGNORECASE)


def _dpi_tuple(value: Any, fallback: int = OUTPUT_DPI) -> Tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            x, y = float(value[0]), float(value[1])
            if x > 0 and y > 0:
                return (x, y)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (int, float)) and float(value) > 0:
        v = float(value)
        return (v, v)
    return (float(fallback), float(fallback))


def _dpi_from_image(path: Optional[str], fallback: int = OUTPUT_DPI) -> Tuple[float, float]:
    """Prefer the source image DPI when present; otherwise use OUTPUT_DPI."""
    if path and os.path.isfile(path):
        try:
            with Image.open(path) as img:
                return _dpi_tuple(img.info.get("dpi"), fallback=fallback)
        except Exception:
            pass
    return (float(fallback), float(fallback))


def _load_layout(layout: Optional[Dict[str, Any]] = None, layout_path: Optional[str] = None) -> Dict[str, Any]:
    if layout and isinstance(layout, dict):
        return layout
    if layout_path:
        with open(layout_path, encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError("layout JSON or layout file is required to assemble the puzzle")


def _read_piece_meta(path: str) -> Optional[Dict[str, Any]]:
    try:
        with Image.open(path) as img:
            raw = (img.info or {}).get(META_KEY)
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _layout_from_pieces(
    piece_paths: List[str],
    piece_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Rebuild a minimal layout from PNG metadata embedded at split time."""
    pieces: List[Dict[str, Any]] = []
    width = height = 0
    rows = cols = 0

    for index, path in enumerate(piece_paths):
        meta = _read_piece_meta(path)
        if not meta:
            continue
        piece_id = str(meta.get("piece_id") or "").strip()
        if piece_ids and index < len(piece_ids) and piece_ids[index]:
            piece_id = str(piece_ids[index]).strip()
        if not piece_id:
            continue
        bbox = meta.get("bbox") or {}
        pieces.append(
            {
                "piece_id": piece_id,
                "person": meta.get("person"),
                "row": meta.get("row"),
                "col": meta.get("col"),
                "index": meta.get("index", index),
                "bbox": bbox,
            }
        )
        width = int(meta.get("width") or width or 0)
        height = int(meta.get("height") or height or 0)
        rows = int(meta.get("rows") or rows or 0)
        cols = int(meta.get("cols") or cols or 0)

    if not pieces or width <= 0 or height <= 0:
        return None
    return {
        "version": 2,
        "rows": rows,
        "cols": cols,
        "width": width,
        "height": height,
        "pieces": pieces,
    }


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
    meta = _read_piece_meta(filename)
    if meta and meta.get("piece_id"):
        return str(meta["piece_id"]).strip()
    name = os.path.basename(filename or "")
    match = _PIECE_ID_RE.search(name)
    if match:
        return f"r{match.group(1)}_c{match.group(2)}"
    return None


def _strip_split_border(piece: Image.Image, width_px: int = BORDER_WIDTH_PX) -> Image.Image:
    """
    Drop the decorative outward stroke by eroding the silhouette.

    Split paints the rim outside the exclusive photo pixels, so shrinking
    alpha by the border width restores the unique tile that tessellates
    back into the original image.
    """
    piece = piece.convert("RGBA")
    width_px = max(0, int(width_px))
    if width_px <= 0:
        return piece
    alpha = piece.getchannel("A")
    eroded = alpha
    for _ in range(width_px):
        eroded = eroded.filter(ImageFilter.MinFilter(3))
    piece.putalpha(eroded)

    # MaxFilter/MinFilter are not perfect inverses at corners — any leftover
    # stroke is exactly BORDER_COLOR and would otherwise show as a dark seam.
    pixels = piece.load()
    width, height = piece.size
    tr, tg, tb = BORDER_COLOR
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a < 8:
                continue
            if r == tr and g == tg and b == tb:
                pixels[x, y] = (0, 0, 0, 0)
    return piece


def _strip_dark_seam_outline(piece: Image.Image) -> Image.Image:
    """
    Remove dark fringe pixels along the transparent boundary.

    Older splits baked a black outline into each piece; those show up as
    jigsaw 'lines' after assemble. Also clears near-black edge anti-alias.
    """
    piece = piece.convert("RGBA")
    pixels = piece.load()
    width, height = piece.size
    clear: List[Tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a < 8:
                continue
            # Only consider dark ink-like pixels.
            if (r + g + b) > 110 or max(r, g, b) > 55:
                continue
            near_empty = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        near_empty = True
                        break
                    if pixels[nx, ny][3] < 40:
                        near_empty = True
                        break
                if near_empty:
                    break
            if near_empty:
                clear.append((x, y))

    for x, y in clear:
        pixels[x, y] = (0, 0, 0, 0)
    return piece


def _expand_piece_for_seams(piece: Image.Image, radius: int = 2) -> Image.Image:
    """
    Grow the opaque silhouette slightly so neighboring pieces overlap and
    hide 1px raster gaps along shared jigsaw edges.
    """
    piece = piece.convert("RGBA")
    if radius < 1:
        return piece

    alpha = piece.getchannel("A")
    # Odd kernel size required by MaxFilter.
    kernel = radius * 2 + 1
    grown = alpha.filter(ImageFilter.MaxFilter(kernel))

    # Spread color into the newly covered fringe from nearby opaque pixels.
    rgb = piece.convert("RGB")
    # Masked composite: start transparent, paste original, then fill fringe
    # by copying RGB where old alpha was solid and using grown alpha.
    out = Image.new("RGBA", piece.size, (0, 0, 0, 0))
    out.paste(piece, (0, 0), mask=alpha)

    # For fringe pixels (grown but not original), sample nearest opaque via
    # a simple box blur of RGB weighted by alpha, then restore hard alpha.
    weighted = Image.merge(
        "RGBA",
        (
            rgb.getchannel("R"),
            rgb.getchannel("G"),
            rgb.getchannel("B"),
            alpha,
        ),
    )
    blurred = weighted.filter(ImageFilter.BoxBlur(radius))
    blurred_px = blurred.load()
    out_px = out.load()
    grown_px = grown.load()
    alpha_px = alpha.load()
    width, height = piece.size
    for y in range(height):
        for x in range(width):
            if grown_px[x, y] < 128:
                continue
            if alpha_px[x, y] >= 128:
                continue
            br, bg, bb, ba = blurred_px[x, y]
            if ba < 1:
                continue
            out_px[x, y] = (br, bg, bb, 255)
    return out


def assemble_puzzle(
    piece_paths: List[str],
    output_path: str,
    layout: Optional[Dict[str, Any]] = None,
    layout_path: Optional[str] = None,
    original_path: Optional[str] = None,
    piece_ids: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Paste decorated puzzle pieces onto a canvas.

    Decorative split-time borders are stripped so exclusive pieces tessellate
    back into the original photo (plus any art drawn on the pieces).

    Placement comes from (first match):
      1. `layout` / `layout_path`
      2. PNG metadata embedded in the piece files at split time
    """
    if layout is None and not layout_path:
        layout = _layout_from_pieces(piece_paths, piece_ids=piece_ids)
        if layout is None:
            return (
                False,
                "No layout found. Pass layout_filename from split, or upload the "
                "original split PNGs (they contain placement metadata).",
            )

    meta = _load_layout(layout=layout, layout_path=layout_path)
    lookup = _piece_lookup(meta)
    if not lookup:
        return False, "Layout has no piece metadata"

    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    # DPI preference: original upload → layout metadata → first piece → 300.
    if original_path and os.path.isfile(original_path):
        out_dpi = _dpi_from_image(original_path, fallback=OUTPUT_DPI)
    elif meta.get("dpi") is not None:
        out_dpi = _dpi_tuple(meta.get("dpi"), fallback=OUTPUT_DPI)
    elif piece_paths:
        out_dpi = _dpi_from_image(piece_paths[0], fallback=OUTPUT_DPI)
    else:
        out_dpi = (float(OUTPUT_DPI), float(OUTPUT_DPI))

    # Prefer the original photo as underlay so any tiny gaps stay invisible.
    if original_path and os.path.isfile(original_path):
        base = Image.open(original_path).convert("RGBA")
        if width <= 0 or height <= 0:
            width, height = base.size
        elif base.size != (width, height):
            base = base.resize((width, height), Image.Resampling.LANCZOS)
        canvas = base
    else:
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
            file_meta = _read_piece_meta(path)
            if file_meta and file_meta.get("bbox"):
                info = file_meta
            else:
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
        border_px = int(meta.get("border_px") or info.get("border_px") or BORDER_WIDTH_PX)
        piece = _strip_split_border(piece, width_px=border_px)
        if piece.size != (expected_w, expected_h):
            # Prefer upscaling decorated art with high-quality filter; never
            # downscale below the layout slot when the upload is larger — crop
            # via paste bounds instead only when sizes already match layout.
            piece = piece.resize((expected_w, expected_h), Image.Resampling.LANCZOS)

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
    # Lossless PNG at print DPI; minimal compression preserves file size / fidelity.
    canvas.convert("RGB").save(
        output_path,
        "PNG",
        compress_level=PNG_COMPRESS_LEVEL,
        dpi=out_dpi,
    )
    dpi_label = int(round(out_dpi[0]))
    note = (
        f"Assembled {placed} puzzle piece(s) into one image "
        f"({width}x{height}px @ {dpi_label} DPI)."
    )
    if missing:
        note += f" Skipped {len(missing)}: " + "; ".join(missing[:3])
    return True, note
