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
# Thin die-cut stroke (kept light so bevel/swell stay visible).
BORDER_WIDTH_PX = 1.6
BORDER_COLOR = (0x2A, 0x2A, 0x2A)  # #2a2a2a
# Cardboard swell / elevation (light from top-left).
SWELL_STEPS = 10  # erosion cascade depth → soft dome height map
SWELL_CENTER_LIFT = 0.14  # brighten toward piece center
SWELL_EDGE_SETTLE = 0.22  # darken near the cut rim
ELEVATION_RIM_PX = 6
ELEVATION_HIGHLIGHT = 0.55  # white overlay on lit rim
ELEVATION_SHADE = 0.62  # black overlay on shaded rim
ELEVATION_OFFSET = 3  # px shift used to find lit vs shaded edges
SPECULAR_STRENGTH = 0.18  # soft sheen across the swollen face
DROP_SHADOW_OFFSET = (3, 5)
DROP_SHADOW_BLUR = 4.0
DROP_SHADOW_ALPHA = 140
# Extra bbox pad: border + drop shadow offset/blur.
PIECE_RENDER_PAD = 14


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


def _shift_gray(img: Image.Image, dx: int, dy: int, fill: int = 0) -> Image.Image:
    """Translate an L image; vacated area filled with `fill`."""
    out = Image.new("L", img.size, fill)
    out.paste(img, (dx, dy))
    return out


def _height_map(mask: Image.Image, steps: int) -> Image.Image:
    """
    Soft distance-from-edge map (0 at rim → 255 toward interior).

    Built by successive MinFilter erosions so the face reads as a gentle dome
    — the classic cardboard “swell” on die-cut puzzle tiles.
    """
    steps = max(1, steps)
    acc = Image.new("L", mask.size, 0)
    eroded = mask
    # Weight later (more interior) layers slightly higher for a rounder crown.
    for i in range(steps):
        eroded = eroded.filter(ImageFilter.MinFilter(3))
        weight = int(round(255.0 * ((i + 1) / steps)))
        layer = eroded.point(lambda p, w=weight: w if p > 0 else 0)
        acc = ImageChops.lighter(acc, layer)
    return ImageChops.multiply(acc, mask).filter(ImageFilter.GaussianBlur(1.6))


def _composite_overlay(
    piece: Image.Image,
    alpha: Image.Image,
    color: Tuple[int, int, int],
) -> None:
    """Alpha-composite a solid color overlay (alpha already premultiplied 0–255)."""
    piece_alpha = piece.getchannel("A")
    overlay_a = ImageChops.multiply(alpha, piece_alpha)
    layer = Image.new("RGBA", piece.size, (*color, 0))
    layer.putalpha(overlay_a)
    piece.paste(Image.alpha_composite(piece, layer))


def _apply_elevation(piece: Image.Image, mask: Image.Image) -> None:
    """
    In-place cardboard swell so the piece reads as a raised jigsaw tile.

    Combines:
      • pillow dome (center lifts, rim settles)
      • directional rim bevel (NW highlight / SE shade)
      • soft specular sheen across the swollen face
    """
    height = _height_map(mask, SWELL_STEPS)

    # --- Pillow / swell across the face ---
    # Brighten where height is high; darken a soft band near the rim.
    center = height.point(
        lambda p: min(255, int(p * SWELL_CENTER_LIFT))
    )
    rim_settle = ImageChops.invert(height)
    rim_settle = ImageChops.multiply(rim_settle, mask).point(
        lambda p: min(255, int(p * SWELL_EDGE_SETTLE))
    )
    _composite_overlay(piece, center, (255, 255, 255))
    _composite_overlay(piece, rim_settle, (0, 0, 0))

    # --- Directional bevel on the cut rim ---
    eroded = mask
    for _ in range(max(1, ELEVATION_RIM_PX)):
        eroded = eroded.filter(ImageFilter.MinFilter(3))
    rim = ImageChops.subtract(mask, eroded)

    off = max(1, ELEVATION_OFFSET)
    # Opaque here but empty up-left → facing the light.
    highlight = ImageChops.multiply(
        ImageChops.subtract(mask, _shift_gray(mask, off, off)),
        rim,
    )
    # Opaque here but empty down-right → facing away from the light.
    shade = ImageChops.multiply(
        ImageChops.subtract(mask, _shift_gray(mask, -off, -off)),
        rim,
    )
    highlight = highlight.filter(ImageFilter.GaussianBlur(1.2))
    shade = shade.filter(ImageFilter.GaussianBlur(1.4))
    highlight = ImageChops.multiply(highlight, mask)
    shade = ImageChops.multiply(shade, mask)

    _composite_overlay(
        piece,
        highlight.point(lambda p: min(255, int(p * ELEVATION_HIGHLIGHT))),
        (255, 255, 255),
    )
    _composite_overlay(
        piece,
        shade.point(lambda p: min(255, int(p * ELEVATION_SHADE))),
        (0, 0, 0),
    )

    # --- Soft specular sheen (printed cardboard catch-light) ---
    if SPECULAR_STRENGTH > 0:
        # Bias the height map toward the lit side with a shifted copy.
        lit = ImageChops.subtract(height, _shift_gray(height, off, off))
        lit = lit.filter(ImageFilter.GaussianBlur(2.5))
        lit = ImageChops.multiply(lit, mask)
        _composite_overlay(
            piece,
            lit.point(lambda p: min(255, int(p * SPECULAR_STRENGTH))),
            (255, 255, 255),
        )


def _apply_drop_shadow(canvas: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Soft drop shadow under the silhouette (elevated-tile cue on plain backgrounds).
    Returns a new image with shadow behind the piece.
    """
    ox, oy = DROP_SHADOW_OFFSET
    shadow = mask.filter(ImageFilter.GaussianBlur(DROP_SHADOW_BLUR))
    shadow = _shift_gray(shadow, ox, oy)
    shadow = shadow.point(lambda p: min(255, int(p * (DROP_SHADOW_ALPHA / 255.0))))
    # Don't draw shadow where the piece body already covers it.
    shadow = ImageChops.subtract(shadow, mask)

    out = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shade.putalpha(shadow)
    out = Image.alpha_composite(out, shade)
    out = Image.alpha_composite(out, canvas)
    return out


def _apply_cut_border(piece: Image.Image, mask: Image.Image, width_px: float) -> None:
    """
    Paint a ~width_px border along the piece silhouette (in place).

    Floor(width) is a solid outward ring; the fractional part is a soft-alpha
    outer ring. Kept thin so the swell bevel stays readable.
    """
    if width_px <= 0:
        return

    edge = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(3)))
    full_px = max(1, int(math.floor(width_px)))
    expanded = mask.filter(ImageFilter.MaxFilter(full_px * 2 + 1))
    solid = ImageChops.lighter(edge, ImageChops.subtract(expanded, mask))

    r, g, b = BORDER_COLOR
    ink = Image.new("RGBA", piece.size, (r, g, b, 255))
    piece.paste(ink, (0, 0), mask=solid)

    frac = width_px - math.floor(width_px)
    if frac > 0.01:
        outer = mask.filter(ImageFilter.MaxFilter((full_px + 1) * 2 + 1))
        soft_ring = ImageChops.subtract(outer, expanded)
        soft = Image.new("RGBA", piece.size, (r, g, b, int(round(255 * frac))))
        piece.paste(soft, (0, 0), mask=soft_ring)


def _render_piece(
    source: Image.Image,
    polygon: List[Point],
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Mask the piece and return a raised RGBA crop (swell + border + soft shadow)."""
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

    piece_full = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    piece_full.paste(source, (0, 0), mask=mask)
    # Border first, then elevation so NW/SE light catches the cut edge itself.
    _apply_cut_border(piece_full, mask, BORDER_WIDTH_PX)
    _apply_elevation(piece_full, mask)
    piece_full = _apply_drop_shadow(piece_full, mask)

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
        # Pad for border ring + drop shadow blur/offset.
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
