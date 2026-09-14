"""Split an image into classic interlocking jigsaw pieces."""
from __future__ import annotations

import json
import math
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo

META_KEY = "puzzle_collage"


MIN_PIECES_PER_PERSON = 3
MAX_PIECES_PER_PERSON = 6
# Knob size relative to the shorter cell side — large enough to read as a real tab.
TAB_SIZE_RATIO = 0.28


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
    cell_aspect = (img_w / cols) / max(img_h / rows, 1e-6)
    score = abs(math.log(max(cell_aspect, 1e-6)))
    # Strongly prefer at least 2×2 when we have enough pieces (avoids ribbon strips).
    if min(rows, cols) == 1 and rows * cols > 6:
        score += 2.5
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


def _plan_assignment(
    participants: int,
    rng: random.Random,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, List[int]]:
    """
    Each participant gets a strict random count in [3, 6] — not forced to 6.

    Counts are kept as rolled. We only nudge within [3, 6] when the total would
    otherwise form a 1×N ribbon that cannot interlock cleanly.
    """
    if participants < 1:
        raise ValueError("participants must be at least 1")

    counts = [rng.randint(MIN_PIECES_PER_PERSON, MAX_PIECES_PER_PERSON) for _ in range(participants)]
    rows, cols = _choose_grid(sum(counts), img_w, img_h)

    # Fix ribbon layouts only (e.g. prime totals like 7 → 1×7), still inside [3, 6].
    if min(rows, cols) == 1 and sum(counts) > 4:
        best = (counts, rows, cols, _cell_aspect_score(rows, cols, img_w, img_h))
        for _ in range(60):
            trial = [rng.randint(MIN_PIECES_PER_PERSON, MAX_PIECES_PER_PERSON) for _ in range(participants)]
            t_rows, t_cols = _choose_grid(sum(trial), img_w, img_h)
            if min(t_rows, t_cols) == 1:
                continue
            score = _cell_aspect_score(t_rows, t_cols, img_w, img_h)
            if score < best[3]:
                best = (trial, t_rows, t_cols, score)
                if score < 0.4:
                    break
        counts, rows, cols, _ = best

    assert _counts_in_range(counts)
    assert rows * cols == sum(counts)

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


def _jigsaw_tab_points(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    direction: int,
    tab_size: float,
) -> List[Point]:
    """
    Classic interlocking knob along edge A→B.

    `direction` is +1 / -1 along the left-hand normal of A→B (0 = flat).
    The path undercuts slightly so the neck locks like a real jigsaw tab.
    """
    if direction == 0:
        return [(bx, by)]

    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy * direction, ux * direction  # unit normal * sign

    # (along_edge fraction, out_normal fraction of tab_size)
    # t goes slightly backward around the neck to form the classic bulb.
    profile = [
        (0.28, 0.00),
        (0.34, 0.05),
        (0.37, 0.20),
        (0.34, 0.36),
        (0.30, 0.48),
        (0.34, 0.60),
        (0.42, 0.70),
        (0.50, 0.74),
        (0.58, 0.70),
        (0.66, 0.60),
        (0.70, 0.48),
        (0.66, 0.36),
        (0.63, 0.20),
        (0.66, 0.05),
        (0.72, 0.00),
    ]

    points: List[Point] = []
    for t, n in profile:
        points.append(
            (
                ax + ux * length * t + nx * tab_size * n,
                ay + uy * length * t + ny * tab_size * n,
            )
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


def _render_piece(
    source: Image.Image,
    polygon: List[Point],
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Mask the piece and return a transparent RGBA crop (no outline stroke)."""
    width, height = source.size
    left, top, right, bottom = bbox
    clamp_l = max(0, left)
    clamp_t = max(0, top)
    clamp_r = min(width, right)
    clamp_b = min(height, bottom)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    int_poly = [(int(round(px)), int(round(py))) for px, py in polygon]
    draw.polygon(int_poly, fill=255)
    # Slight dilate so neighboring pieces share a 1–2px overlap (seamless join).
    mask = mask.filter(ImageFilter.MaxFilter(3))

    piece_full = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    piece_full.paste(source, (0, 0), mask=mask)

    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    canvas = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))
    src_crop = piece_full.crop((clamp_l, clamp_t, clamp_r, clamp_b))
    canvas.paste(src_crop, (clamp_l - left, clamp_t - top), src_crop)
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
    source = Image.open(image_path).convert("RGBA")
    width, height = source.size
    rows, cols, assignment = _plan_assignment(participants, rng, width, height)

    cell_w = width / cols
    cell_h = height / rows
    tab_size = min(cell_w, cell_h) * TAB_SIZE_RATIO
    tabs = _build_edge_tabs(rows, cols, rng)

    pieces_meta: List[Dict[str, Any]] = []
    os.makedirs(output_dir, exist_ok=True)

    for index, person in enumerate(assignment):
        row, col = divmod(index, cols)
        polygon = _piece_polygon(row, col, cell_w, cell_h, tab_size, tabs, rows, cols)
        bbox = _bbox_from_points(polygon, pad=3)
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
        piece_crop.save(out_path, "PNG", pnginfo=pnginfo)

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
