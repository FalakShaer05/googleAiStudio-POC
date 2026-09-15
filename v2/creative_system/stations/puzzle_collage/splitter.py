"""Split an image into classic interlocking jigsaw pieces."""
from __future__ import annotations

import json
import math
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo

META_KEY = "puzzle_collage"


MIN_PIECES_PER_PERSON = 3
MAX_PIECES_PER_PERSON = 6
# Knob size relative to the shorter cell side — classic die-cut proportion.
TAB_SIZE_RATIO = 0.26
# How circular the knob head is (1.0 = pure circle). Mild undercut via center offset.
TAB_HEAD_RADIUS_RATIO = 0.50  # of tab_size
TAB_CENTER_OUT_RATIO = 0.40  # of tab_size; must be < head radius for neck undercut
TAB_SHOULDER_RATIO = 0.055  # along-edge lead-in before the neck (smooth fillet)
TAB_ARC_SAMPLES = 28
# Solid outline around the entire piece silhouette (inward stroke).
BORDER_WIDTH_PX = 3
BORDER_COLOR = (0x1B, 0x1B, 0x1B)  # #1b1b1b
# Padding so the crop does not clip anti-aliased edge pixels.
PIECE_RENDER_PAD = 4


Point = Tuple[float, float]


def _unique_filename(original_name: str, prefix: str) -> str:
    _, ext = os.path.splitext(original_name)
    if not ext:
        ext = ".png"
    return f"{prefix}_{uuid.uuid4().hex}{ext}"


def _cell_aspect_score(rows: int, cols: int, img_w: int, img_h: int) -> float:
    """Lower is better — 0 means perfectly square cells."""
    if rows < 1 or cols < 1:
        return float("inf")
    cell_w = img_w / cols
    cell_h = img_h / rows
    cell_aspect = cell_w / max(cell_h, 1e-6)
    score = abs(math.log(max(cell_aspect, 1e-6)))
    # Ribbon / strip grids (1×N or N×1) always look like long pieces — hard reject.
    if min(rows, cols) == 1:
        score += 12.0
    # Mildly prefer chunkier grids over long 2×N banners when scores are close.
    elif min(rows, cols) == 2 and max(rows, cols) >= 5:
        score += 0.35
    return score


def _choose_grid(total: int, img_w: int, img_h: int) -> Tuple[int, int]:
    """
    Pick rows×cols = total so individual cells are as square as possible
    for this image aspect ratio (avoids ribbon / strip pieces).
    """
    if total <= 1:
        return 1, 1

    best = (1, total)
    best_score = float("inf")
    for cols in range(1, total + 1):
        if total % cols:
            continue
        rows = total // cols
        score = _cell_aspect_score(rows, cols, img_w, img_h)
        if score < best_score:
            best_score = score
            best = (rows, cols)
    return best


def _counts_in_range(counts: List[int]) -> bool:
    return all(MIN_PIECES_PER_PERSON <= c <= MAX_PIECES_PER_PERSON for c in counts)


def _distribute_counts(participants: int, total: int, rng: random.Random) -> List[int]:
    """
    Split `total` into `participants` buckets, each in [MIN, MAX].

    Starts even, then randomly walks leftovers so assignments stay varied.
    """
    lo, hi = MIN_PIECES_PER_PERSON, MAX_PIECES_PER_PERSON
    if total < lo * participants or total > hi * participants:
        raise ValueError(f"cannot distribute {total} pieces across {participants} people")

    base = total // participants
    counts = [base] * participants
    for i in range(total - base * participants):
        counts[i] += 1

    # Fisher-Yates shuffle keeps sum fixed while randomizing who gets extras.
    rng.shuffle(counts)

    # If any bucket drifted outside [lo, hi] (shouldn't with valid total), repair.
    while True:
        low_i = next((i for i, c in enumerate(counts) if c < lo), None)
        high_i = next((i for i, c in enumerate(counts) if c > hi), None)
        if low_i is None and high_i is None:
            break
        if low_i is not None and high_i is not None:
            counts[low_i] += 1
            counts[high_i] -= 1
            continue
        # Extremely defensive — borrow from a mid bucket.
        if low_i is not None:
            donor = max(range(participants), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[low_i] += 1
        else:
            recv = min(range(participants), key=lambda i: counts[i])
            counts[high_i] -= 1
            counts[recv] += 1

    assert sum(counts) == total
    assert _counts_in_range(counts)
    return counts


def _plan_assignment(
    participants: int,
    rng: random.Random,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, List[int]]:
    """
    Pick a piece total (and per-person counts in [3, 6]) so the grid cells
    are as square as possible — avoids long strip / ribbon pieces.
    """
    if participants < 1:
        raise ValueError("participants must be at least 1")

    lo = MIN_PIECES_PER_PERSON * participants
    hi = MAX_PIECES_PER_PERSON * participants

    best: Optional[Tuple[float, int, int, int]] = None  # score, total, rows, cols
    for total in range(lo, hi + 1):
        rows, cols = _choose_grid(total, img_w, img_h)
        score = _cell_aspect_score(rows, cols, img_w, img_h)
        # Tiny jitter so ties don't always pick the smallest total.
        score += rng.random() * 0.02
        if best is None or score < best[0]:
            best = (score, total, rows, cols)

    assert best is not None
    _, total, rows, cols = best
    counts = _distribute_counts(participants, total, rng)

    assert _counts_in_range(counts)
    assert rows * cols == sum(counts) == total

    assignment: List[int] = []
    for person_idx, count in enumerate(counts, start=1):
        assignment.extend([person_idx] * count)
    rng.shuffle(assignment)
    return rows, cols, assignment


def _tab_sign(rng: random.Random) -> int:
    return 1 if rng.random() < 0.5 else -1


def _build_edge_tabs(rows: int, cols: int, rng: random.Random) -> Dict[str, int]:
    """
    Shared edge orientation:
      h:{r}:{c} — edge between (r,c) and (r+1,c); +1 = tab into the lower cell
      v:{r}:{c} — edge between (r,c) and (r,c+1); +1 = tab into the right cell
    """
    tabs: Dict[str, int] = {}
    for r in range(rows - 1):
        for c in range(cols):
            tabs[f"h:{r}:{c}"] = _tab_sign(rng)
    for r in range(rows):
        for c in range(cols - 1):
            tabs[f"v:{r}:{c}"] = _tab_sign(rng)
    return tabs


def _edge_point(
    ax: float,
    ay: float,
    ux: float,
    uy: float,
    nx: float,
    ny: float,
    along: float,
    out: float,
) -> Point:
    return (ax + ux * along + nx * out, ay + uy * along + ny * out)


def _jigsaw_tab_points(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    direction: int,
    tab_size: float,
) -> List[Point]:
    """
    Classic die-cut interlocking knob along edge A→B.

    Shape matches traditional jigsaw cuts: a near-circular bulb with a mild
    undercut neck and soft shoulders into the flat edge (refs: round-tab style).

    `direction` is +1 / -1 along the left-hand normal of A→B (0 = flat).
    """
    if direction == 0:
        return [(bx, by)]

    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy * direction, ux * direction  # unit normal * sign

    r = tab_size * TAB_HEAD_RADIUS_RATIO
    cy = tab_size * TAB_CENTER_OUT_RATIO
    if cy >= r:
        cy = r * 0.85

    mid = length * 0.5
    neck_half = math.sqrt(max(r * r - cy * cy, 0.0))
    shoulder = min(length * TAB_SHOULDER_RATIO, max(neck_half * 0.45, 1.0))
    # Angle from the outward (+out) axis where the circle meets the edge line.
    cos_neck = max(-1.0, min(1.0, -cy / r))
    theta_neck = math.acos(cos_neck)
    # Start the arc slightly past the geometric edge hit so the neck undercuts
    # like a real die-cut tab, then fillet back to the flat edge.
    theta_arc = min(theta_neck * 1.08, math.pi * 0.92)

    points: List[Point] = []

    def add(along: float, out: float) -> None:
        points.append(_edge_point(ax, ay, ux, uy, nx, ny, along, out))

    # Left shoulder: flat edge → undercut neck entry (smoothstep).
    left_flat = mid - neck_half - shoulder
    left_neck_along = mid + r * math.sin(-theta_arc)
    left_neck_out = cy + r * math.cos(-theta_arc)
    for i in range(6):
        u = i / 6.0
        s = u * u * (3.0 - 2.0 * u)
        add(
            left_flat + (left_neck_along - left_flat) * s,
            left_neck_out * s,
        )

    # Circular bulb (classic round tab / blank).
    for i in range(1, TAB_ARC_SAMPLES):
        theta = -theta_arc + (2.0 * theta_arc) * (i / TAB_ARC_SAMPLES)
        add(mid + r * math.sin(theta), cy + r * math.cos(theta))

    # Right shoulder: undercut neck exit → flat edge.
    right_neck_along = mid + r * math.sin(theta_arc)
    right_neck_out = cy + r * math.cos(theta_arc)
    right_flat = mid + neck_half + shoulder
    for i in range(1, 7):
        u = i / 6.0
        s = u * u * (3.0 - 2.0 * u)
        add(
            right_neck_along + (right_flat - right_neck_along) * s,
            right_neck_out * (1.0 - s),
        )

    points.append((bx, by))
    return points


def _piece_polygon(
    row: int,
    col: int,
    cell_w: float,
    cell_h: float,
    tab_size: float,
    tabs: Dict[str, int],
    rows: int,
    cols: int,
) -> List[Point]:
    """Clockwise outline with interlocking tabs/blanks on shared edges."""
    x0 = col * cell_w
    y0 = row * cell_h
    x1 = x0 + cell_w
    y1 = y0 + cell_h

    # For each edge, direction is relative to walking clockwise around the piece.
    # Top L→R: left-normal points up (−Y). Shared h:{row-1}:{col}: +1 = tab into lower
    #   (= into this piece from above) → blank on our top → we need tab into −normal? 
    #   Walking L→R, left normal is UP. Tab protruding UP from this piece = +normal.
    #   Shared +1 means tab into THIS cell from the upper piece's bottom = blank on our top
    #   = protrusion into us from above = our top edge curves inward = −normal for us.
    if row == 0:
        top_dir = 0
    else:
        shared = tabs[f"h:{row - 1}:{col}"]
        top_dir = -shared  # +shared → blank on our top

    # Right T→B: left-normal points right (+X). Shared v:{row}:{col}: +1 = tab into right cell
    #   = tab out of this piece on the right = +normal.
    if col == cols - 1:
        right_dir = 0
    else:
        right_dir = tabs[f"v:{row}:{col}"]

    # Bottom R→L: left-normal points down (+Y). Shared h:{row}:{col}: +1 = tab into lower cell
    #   = tab out of this piece downward = +normal while walking R→L.
    if row == rows - 1:
        bottom_dir = 0
    else:
        bottom_dir = tabs[f"h:{row}:{col}"]

    # Left B→T: left-normal points left (−X). Shared v:{row}:{col-1}: +1 = tab into this cell
    #   = blank on our left = −normal (protrusion would be leftward / + our left-normal).
    if col == 0:
        left_dir = 0
    else:
        left_dir = -tabs[f"v:{row}:{col - 1}"]

    points: List[Point] = [(x0, y0)]
    points.extend(_jigsaw_tab_points(x0, y0, x1, y0, top_dir, tab_size)[:-1])
    points.append((x1, y0))
    points.extend(_jigsaw_tab_points(x1, y0, x1, y1, right_dir, tab_size)[:-1])
    points.append((x1, y1))
    points.extend(_jigsaw_tab_points(x1, y1, x0, y1, bottom_dir, tab_size)[:-1])
    points.append((x0, y1))
    points.extend(_jigsaw_tab_points(x0, y1, x0, y0, left_dir, tab_size)[:-1])
    points.append((x0, y0))
    return points


def _bbox_from_points(points: List[Point], pad: int = 2) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        int(math.floor(min(xs))) - pad,
        int(math.floor(min(ys))) - pad,
        int(math.ceil(max(xs))) + pad,
        int(math.ceil(max(ys))) + pad,
    )


def _apply_piece_border(piece: Image.Image, mask: Image.Image, width_px: int) -> None:
    """
    Paint a solid inward border along the entire piece silhouette (in place).

    The stroke sits on the piece content around the whole outline (tabs, blanks,
    and outer edges) — not only as an outward cut-line ring.

    Caller must leave empty margin around `mask` so MinFilter can erode every
    side (pieces that touch the source image edge otherwise skip that side).
    """
    width_px = max(0, int(round(width_px)))
    if width_px <= 0:
        return

    eroded = mask
    for _ in range(width_px):
        eroded = eroded.filter(ImageFilter.MinFilter(3))
    ring = ImageChops.subtract(mask, eroded)
    if ring.getbbox() is None:
        return

    r, g, b = BORDER_COLOR
    # Alpha-composite so border ink reliably replaces art on every rim pixel.
    ink = Image.new("RGBA", piece.size, (r, g, b, 0))
    ink.putalpha(ring)
    piece.alpha_composite(ink)


def _render_piece(
    source: Image.Image,
    polygon: List[Point],
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Mask the piece and return an RGBA crop with a full-perimeter border."""
    width, height = source.size
    left, top, right, bottom = bbox

    # Pad the working buffer so edge-of-image pieces (row/col 0 or last) still
    # get a full inward border — MinFilter cannot erode a silhouette that sits
    # flush against the bitmap boundary.
    work_pad = BORDER_WIDTH_PX + 2
    work_w = width + 2 * work_pad
    work_h = height + 2 * work_pad

    mask = Image.new("L", (work_w, work_h), 0)
    draw = ImageDraw.Draw(mask)
    int_poly = [
        (int(round(px)) + work_pad, int(round(py)) + work_pad) for px, py in polygon
    ]
    draw.polygon(int_poly, fill=255)

    piece_full = Image.new("RGBA", (work_w, work_h), (0, 0, 0, 0))
    piece_full.paste(source, (work_pad, work_pad))
    # Clear outside the silhouette.
    clear = Image.new("RGBA", (work_w, work_h), (0, 0, 0, 0))
    piece_full = Image.composite(piece_full, clear, mask)
    _apply_piece_border(piece_full, mask, BORDER_WIDTH_PX)

    # Map the requested source-space bbox into the padded working image.
    work_l = left + work_pad
    work_t = top + work_pad
    work_r = right + work_pad
    work_b = bottom + work_pad

    clamp_l = max(0, work_l)
    clamp_t = max(0, work_t)
    clamp_r = min(work_w, work_r)
    clamp_b = min(work_h, work_b)

    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    canvas = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
    src_crop = piece_full.crop((clamp_l, clamp_t, clamp_r, clamp_b))
    canvas.paste(src_crop, (clamp_l - work_l, clamp_t - work_t), src_crop)
    return canvas


def split_puzzle(
    image_path: str,
    participants: int,
    output_dir: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Cut `image_path` into interlocking jigsaw pieces and assign 3–6 per participant.
    """
    rng = random.Random(seed)
    with Image.open(image_path) as opened:
        source_dpi = opened.info.get("dpi")
        source = opened.convert("RGBA")
    width, height = source.size
    rows, cols, assignment = _plan_assignment(participants, rng, width, height)

    save_dpi: Optional[Tuple[float, float]] = None
    if isinstance(source_dpi, (tuple, list)) and len(source_dpi) >= 2:
        x, y = float(source_dpi[0]), float(source_dpi[1])
        if x > 0 and y > 0:
            save_dpi = (x, y)
    elif isinstance(source_dpi, (int, float)) and float(source_dpi) > 0:
        v = float(source_dpi)
        save_dpi = (v, v)
    if save_dpi is None:
        save_dpi = (300.0, 300.0)

    cell_w = width / cols
    cell_h = height / rows
    tab_size = min(cell_w, cell_h) * TAB_SIZE_RATIO
    tabs = _build_edge_tabs(rows, cols, rng)

    pieces_meta: List[Dict[str, Any]] = []
    os.makedirs(output_dir, exist_ok=True)

    for index, person in enumerate(assignment):
        row, col = divmod(index, cols)
        polygon = _piece_polygon(row, col, cell_w, cell_h, tab_size, tabs, rows, cols)
        # Pad so anti-aliased edges are not clipped by the crop.
        bbox = _bbox_from_points(polygon, pad=PIECE_RENDER_PAD)
        piece_crop = _render_piece(source, polygon, bbox)

        piece_id = f"r{row}_c{col}"
        left, top, right, bottom = bbox
        piece_meta = {
            "piece_id": piece_id,
            "person": person,
            "row": row,
            "col": col,
            "index": index,
            "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
            "width": width,
            "height": height,
            "rows": rows,
            "cols": cols,
        }
        pnginfo = PngInfo()
        pnginfo.add_text(META_KEY, json.dumps(piece_meta))

        filename = f"cs_puzzle_piece_p{person}_{piece_id}_{uuid.uuid4().hex}.png"
        out_path = os.path.join(output_dir, filename)
        piece_crop.save(out_path, "PNG", pnginfo=pnginfo, dpi=save_dpi, compress_level=0)

        pieces_meta.append(
            {
                "piece_id": piece_id,
                "person": person,
                "person_tag": f"person {person}",
                "row": row,
                "col": col,
                "index": index,
                "bbox": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
                "output_filename": filename,
                "local_path": f"/outputs/{filename}",
                "abs_path": out_path,
            }
        )

    layout = {
        "version": 2,
        "rows": rows,
        "cols": cols,
        "width": width,
        "height": height,
        "dpi": [save_dpi[0], save_dpi[1]],
        "cell_w": cell_w,
        "cell_h": cell_h,
        "tab_size": tab_size,
        "tabs": tabs,
        "participants": participants,
        "pieces": [
            {
                "piece_id": p["piece_id"],
                "person": p["person"],
                "row": p["row"],
                "col": p["col"],
                "index": p["index"],
                "bbox": p["bbox"],
            }
            for p in pieces_meta
        ],
    }

    layout_filename = _unique_filename("puzzle_layout.json", "cs_puzzle_layout")
    layout_path = os.path.join(output_dir, layout_filename)
    with open(layout_path, "w", encoding="utf-8") as handle:
        json.dump(layout, handle, indent=2)

    by_person: Dict[int, List[str]] = {}
    for p in pieces_meta:
        by_person.setdefault(p["person"], []).append(p["piece_id"])

    return {
        "participants": participants,
        "rows": rows,
        "cols": cols,
        "total_pieces": len(pieces_meta),
        "pieces_per_person": {str(k): len(v) for k, v in sorted(by_person.items())},
        "pieces": pieces_meta,
        "layout": layout,
        "layout_filename": layout_filename,
        "layout_local_path": f"/outputs/{layout_filename}",
        "layout_abs_path": layout_path,
    }
