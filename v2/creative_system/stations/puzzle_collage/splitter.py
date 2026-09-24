"""Split an image into unique interlocking jigsaw pieces that reconstruct it."""
from __future__ import annotations

import json
import math
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo

from .line_art import SPLIT_LONG_EDGE, to_pure_line_art

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

META_KEY = "puzzle_collage"

# Station rules: 1–5 people; each gets 3–6 unique pieces covering the full image.
MAX_PARTICIPANTS = 5
MIN_PIECES_PER_PERSON = 3
MAX_PIECES_PER_PERSON = 6
# Split at moderate res for API speed; assemble upscales to 4K.
TARGET_LONG_EDGE = SPLIT_LONG_EDGE
OUTPUT_DPI = (300.0, 300.0)
PNG_COMPRESS_LEVEL = 1

# Knob size relative to the shorter cell side — classic die-cut proportion.
TAB_SIZE_RATIO = 0.26
# How circular the knob head is (1.0 = pure circle). Mild undercut via center offset.
TAB_HEAD_RADIUS_RATIO = 0.50  # of tab_size
TAB_CENTER_OUT_RATIO = 0.40  # of tab_size; must be < head radius for neck undercut
TAB_SHOULDER_RATIO = 0.055  # along-edge lead-in before the neck (smooth fillet)
TAB_ARC_SAMPLES = 28
# Solid outline around the entire piece silhouette (outward stroke so
# exclusive photo pixels stay intact and can reconstruct the original).
BORDER_WIDTH_PX = 3
BORDER_COLOR = (0x1B, 0x1B, 0x1B)  # #1b1b1b
# Padding so the crop holds the outward border plus anti-aliased edge pixels.
PIECE_RENDER_PAD = BORDER_WIDTH_PX + 3


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


def _partition_bounds(length: int, parts: int) -> List[int]:
    """
    Split [0, length] into `parts` adjoining integer spans.

    Cells are as equal as possible, with no gap and no overlap, so the union
    is the original interval and each pixel belongs to exactly one cell.
    """
    if parts < 1:
        raise ValueError("parts must be at least 1")
    if length < 1:
        raise ValueError("length must be at least 1")
    bounds = [0]
    base, rem = divmod(length, parts)
    for i in range(parts):
        bounds.append(bounds[-1] + base + (1 if i < rem else 0))
    bounds[-1] = length
    return bounds


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
    Cover the full image with a near-square grid of unique pieces.

    - 1 participant: whole image split into 4–6 pieces (2D grid; no 1×N strips).
    - More participants: more total pieces (up to 6 each) so tiles get smaller.
    - Participants capped at MAX_PARTICIPANTS; equal counts when possible.
    """
    if participants < 1:
        raise ValueError("participants must be at least 1")
    if participants > MAX_PARTICIPANTS:
        raise ValueError(
            f"participants cannot exceed {MAX_PARTICIPANTS} for Puzzle Collage"
        )

    lo = MIN_PIECES_PER_PERSON * participants
    hi = MAX_PIECES_PER_PERSON * participants
    # One person: never use a 3-piece strip — need a real 2D cover of the page.
    if participants == 1:
        lo = max(lo, 4)

    def _candidate(total: int) -> Optional[Tuple[float, int, int, int]]:
        rows, cols = _choose_grid(total, img_w, img_h)
        if rows * cols != total:
            return None
        # Full-image cover must be a real grid, not a ribbon of giant pieces.
        if min(rows, cols) < 2:
            return None
        score = _cell_aspect_score(rows, cols, img_w, img_h)
        score += rng.random() * 0.02
        # Prefer more pieces (smaller tiles), especially as participant count rises.
        score += (hi - total) * 0.08
        return (score, total, rows, cols)

    best: Optional[Tuple[float, int, int, int]] = None
    # Equal counts first; scan high→low so we lock onto smaller tiles when tied.
    for total in range(hi, lo - 1, -1):
        if total % participants != 0:
            continue
        cand = _candidate(total)
        if cand is None:
            continue
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        for total in range(hi, lo - 1, -1):
            cand = _candidate(total)
            if cand is None:
                continue
            if best is None or cand[0] < best[0]:
                best = cand

    if best is None:
        raise ValueError(
            f"cannot build a unique {lo}–{hi} piece grid for {participants} participant(s)"
        )

    _, total, rows, cols = best
    if total < lo or rows * cols != total:
        raise RuntimeError("planned grid does not cover the image with unique pieces")
    if min(rows, cols) < 2:
        raise RuntimeError("grid must be at least 2×2 so the whole image is covered")
    counts = _distribute_counts(participants, total, rng)

    assert _counts_in_range(counts)
    assert rows * cols == sum(counts) == total
    if total % participants == 0 and len(set(counts)) != 1:
        raise RuntimeError("equal grid must give every participant the same piece count")

    # One owner per unique grid cell — never reuse a piece across people.
    assignment: List[int] = []
    for person_idx, count in enumerate(counts, start=1):
        assignment.extend([person_idx] * count)
    if len(assignment) != rows * cols:
        raise RuntimeError("assignment length must equal unique grid cells")
    if len(set(assignment)) != participants:
        raise RuntimeError("every participant must receive at least one unique piece")
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


def _build_shared_edges(
    rows: int,
    cols: int,
    xs: List[int],
    ys: List[int],
    tabs: Dict[str, int],
    tab_size: float,
) -> Tuple[Dict[Tuple[int, int], List[Point]], Dict[Tuple[int, int], List[Point]]]:
    """
    One polyline per internal edge, reused by both adjacent pieces.

    h_edges[(r, c)] walks L→R between (r,c) and (r+1,c).
      tabs +1 (into lower) → the curve bulges down.
    v_edges[(r, c)] walks T→B between (r,c) and (r,c+1).
      tabs +1 (into right) → the curve bulges right.
    """
    h_edges: Dict[Tuple[int, int], List[Point]] = {}
    v_edges: Dict[Tuple[int, int], List[Point]] = {}
    for r in range(rows - 1):
        for c in range(cols):
            direction = tabs[f"h:{r}:{c}"]
            # Walking L→R, left-normal is UP; bulge down when tab goes into lower.
            h_edges[(r, c)] = _jigsaw_tab_points(
                float(xs[c]),
                float(ys[r + 1]),
                float(xs[c + 1]),
                float(ys[r + 1]),
                -direction,
                tab_size,
            )
    for r in range(rows):
        for c in range(cols - 1):
            direction = tabs[f"v:{r}:{c}"]
            # Walking T→B, left-normal is RIGHT; +direction bulges into the right cell.
            v_edges[(r, c)] = _jigsaw_tab_points(
                float(xs[c + 1]),
                float(ys[r]),
                float(xs[c + 1]),
                float(ys[r + 1]),
                direction,
                tab_size,
            )
    return h_edges, v_edges


def _walk_reverse(edge: List[Point], end: Point) -> List[Point]:
    """Walk a shared A→B edge backwards, ending at A."""
    return list(reversed(edge[:-1])) + [end]


def _piece_polygon(
    row: int,
    col: int,
    xs: List[int],
    ys: List[int],
    h_edges: Dict[Tuple[int, int], List[Point]],
    v_edges: Dict[Tuple[int, int], List[Point]],
    rows: int,
    cols: int,
) -> List[Point]:
    """Clockwise outline; neighbors share the exact same edge polyline."""
    x0, y0 = float(xs[col]), float(ys[row])
    x1, y1 = float(xs[col + 1]), float(ys[row + 1])

    if row == 0:
        top: List[Point] = [(x1, y0)]
    else:
        top = h_edges[(row - 1, col)]

    if col == cols - 1:
        right: List[Point] = [(x1, y1)]
    else:
        right = v_edges[(row, col)]

    if row == rows - 1:
        bottom: List[Point] = [(x0, y1)]
    else:
        bottom = _walk_reverse(h_edges[(row, col)], (x0, y1))

    if col == 0:
        left: List[Point] = [(x0, y0)]
    else:
        left = _walk_reverse(v_edges[(row, col - 1)], (x0, y0))

    points: List[Point] = [(x0, y0)]
    points.extend(top[:-1])
    points.append((x1, y0))
    points.extend(right[:-1])
    points.append((x1, y1))
    points.extend(bottom[:-1])
    points.append((x0, y1))
    points.extend(left[:-1])
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
    Paint a solid outward border around the piece silhouette (in place).

    The stroke sits in the transparent margin so exclusive photo pixels are
    not overwritten — assembling after stripping the rim reconstructs the
    original image.

    Caller must leave empty margin around `mask` so MaxFilter can grow every
    side (pieces that sit flush against the crop otherwise skip that side).
    """
    width_px = max(0, int(round(width_px)))
    if width_px <= 0:
        return

    dilated = mask
    for _ in range(width_px):
        dilated = dilated.filter(ImageFilter.MaxFilter(3))
    ring = ImageChops.subtract(dilated, mask)
    if ring.getbbox() is None:
        return

    r, g, b = BORDER_COLOR
    ink = Image.new("RGBA", piece.size, (r, g, b, 0))
    ink.putalpha(ring)
    piece.alpha_composite(ink)


def _init_cell_labels(
    width: int,
    height: int,
    rows: int,
    cols: int,
    xs: List[int],
    ys: List[int],
) -> bytearray:
    """Every source pixel starts owned by exactly one grid cell (1-based index)."""
    labels = bytearray(width * height)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c + 1
            x0, x1 = xs[c], xs[c + 1]
            y0, y1 = ys[r], ys[r + 1]
            if x1 <= x0 or y1 <= y0:
                continue
            fill = bytes([idx]) * (x1 - x0)
            for y in range(y0, y1):
                start = y * width + x0
                labels[start : start + (x1 - x0)] = fill
    return labels


def _claim_tabs(
    labels: bytearray,
    width: int,
    height: int,
    piece_idx: int,
    polygon: List[Point],
    cell: Tuple[int, int, int, int],
) -> None:
    """
    Transfer tab pixels to this piece.

    Pixels inside the polygon but outside this cell are this piece's tab —
    they move from the neighbor's cell owner to `piece_idx`. Cell-body pixels
    stay with the cell owner until a neighbor claims them as a tab.
    """
    left, top, right, bottom = _bbox_from_points(polygon, pad=2)
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right <= left or bottom <= top:
        return

    # Local mask only (full-image masks at 2K+ are far too slow per piece).
    mw, mh = right - left, bottom - top
    mask = Image.new("L", (mw, mh), 0)
    ImageDraw.Draw(mask).polygon(
        [(int(round(px)) - left, int(round(py)) - top) for px, py in polygon],
        fill=255,
    )
    cx0, cy0, cx1, cy1 = cell
    pixels = mask.load()
    for y in range(top, bottom):
        row = y * width
        ly = y - top
        for x in range(left, right):
            if pixels[x - left, ly] < 128:
                continue
            if cx0 <= x < cx1 and cy0 <= y < cy1:
                continue
            labels[row + x] = piece_idx


def _labels_bbox(
    labels: bytearray,
    width: int,
    height: int,
    piece_idx: int,
    hint: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = hint
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    minx, miny, maxx, maxy = right, bottom, left, top
    found = False
    for y in range(top, bottom):
        row = y * width
        for x in range(left, right):
            if labels[row + x] != piece_idx:
                continue
            found = True
            if x < minx:
                minx = x
            if x + 1 > maxx:
                maxx = x + 1
            if y < miny:
                miny = y
            if y + 1 > maxy:
                maxy = y + 1
    if not found:
        return hint
    return (minx, miny, maxx, maxy)


def _render_exclusive_piece(
    source: Image.Image,
    labels: bytearray,
    piece_idx: int,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    """Crop one exclusive piece and paint the decorative outward border."""
    width, height = source.size
    left, top, right, bottom = bbox
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)

    # Extra margin so MaxFilter can grow the outline on every side.
    work_pad = BORDER_WIDTH_PX + 2
    work_w = crop_w + 2 * work_pad
    work_h = crop_h + 2 * work_pad
    mask = Image.new("L", (work_w, work_h), 0)
    piece = Image.new("RGBA", (work_w, work_h), (0, 0, 0, 0))

    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(width, right)
    y1 = min(height, bottom)

    if np is not None and x1 > x0 and y1 > y0:
        label_arr = np.frombuffer(labels, dtype=np.uint8).reshape(height, width)
        region = label_arr[y0:y1, x0:x1]
        hit = region == piece_idx
        if hit.any():
            src_arr = np.array(source, dtype=np.uint8, copy=False)
            src_region = src_arr[y0:y1, x0:x1]
            piece_arr = np.array(piece, dtype=np.uint8, copy=True)
            mask_arr = np.array(mask, dtype=np.uint8, copy=True)
            dy = y0 - top + work_pad
            dx = x0 - left + work_pad
            dest = piece_arr[dy : dy + (y1 - y0), dx : dx + (x1 - x0)]
            dest_m = mask_arr[dy : dy + (y1 - y0), dx : dx + (x1 - x0)]
            dest[hit] = src_region[hit]
            dest_m[hit] = 255
            piece = Image.fromarray(piece_arr, mode="RGBA")
            mask = Image.fromarray(mask_arr, mode="L")
    else:
        src_px = source.load()
        mp = mask.load()
        pp = piece.load()
        for y in range(y0, y1):
            row = y * width
            cy = y - top + work_pad
            for x in range(x0, x1):
                if labels[row + x] != piece_idx:
                    continue
                cx = x - left + work_pad
                mp[cx, cy] = 255
                pp[cx, cy] = src_px[x, y]

    _apply_piece_border(piece, mask, BORDER_WIDTH_PX)
    return piece.crop((work_pad, work_pad, work_pad + crop_w, work_pad + crop_h))


def _verify_exclusive_labels(
    labels: bytearray,
    width: int,
    height: int,
    piece_count: int,
) -> None:
    """Every pixel must belong to exactly one piece in 1..piece_count."""
    expected = set(range(1, piece_count + 1))
    if np is not None:
        arr = np.frombuffer(labels, dtype=np.uint8)
        if arr.min() < 1 or arr.max() > piece_count:
            raise RuntimeError(
                f"split label out of range ({int(arr.min())}–{int(arr.max())}); "
                "pieces would be incomplete"
            )
        seen = set(int(v) for v in np.unique(arr))
    else:
        seen = set()
        for value in labels:
            if value < 1 or value > piece_count:
                raise RuntimeError(
                    f"split label out of range ({value}); pieces would be incomplete"
                )
            seen.add(value)
    if seen != expected:
        missing = sorted(expected - seen)
        raise RuntimeError(
            f"split missed piece index(es) {missing[:8]}; pieces would be incomplete"
        )


def _ensure_print_resolution(source: Image.Image) -> Image.Image:
    """Downscale only when larger than TARGET_LONG_EDGE (keeps split fast)."""
    width, height = source.size
    long_edge = max(width, height)
    if long_edge <= 0 or long_edge <= TARGET_LONG_EDGE:
        return source
    scale = TARGET_LONG_EDGE / float(long_edge)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return source.resize(new_size, Image.Resampling.LANCZOS)


def piece_silhouette_mask(
    layout: Dict[str, Any],
    piece_id: str,
    size: Tuple[int, int],
) -> Optional[Image.Image]:
    """
    Rebuild the jigsaw silhouette for `piece_id` as an L mask matching `size`.

    Used at assemble time to punch out photo backgrounds that appear inside
    blank (innie) holes when decorated pieces are re-photographed.
    """
    rows = int(layout.get("rows") or 0)
    cols = int(layout.get("cols") or 0)
    xs = layout.get("xs")
    ys = layout.get("ys")
    tabs = layout.get("tabs")
    tab_size = float(layout.get("tab_size") or 0)
    pieces = layout.get("pieces") or []
    if rows < 1 or cols < 1 or not xs or not ys or not tabs or tab_size <= 0:
        return None

    info = None
    for item in pieces:
        if str(item.get("piece_id") or "").strip().lower() == piece_id.lower():
            info = item
            break
    if not info:
        return None

    row = int(info.get("row", -1))
    col = int(info.get("col", -1))
    bbox = info.get("bbox") or {}
    if row < 0 or col < 0:
        return None

    try:
        xs_i = [int(v) for v in xs]
        ys_i = [int(v) for v in ys]
        tabs_i = {str(k): int(v) for k, v in tabs.items()}
    except (TypeError, ValueError):
        return None

    h_edges, v_edges = _build_shared_edges(rows, cols, xs_i, ys_i, tabs_i, tab_size)
    polygon = _piece_polygon(row, col, xs_i, ys_i, h_edges, v_edges, rows, cols)
    left = int(bbox.get("left", 0))
    top = int(bbox.get("top", 0))
    local = [(px - left, py - top) for px, py in polygon]
    mask = Image.new("L", size, 0)
    if len(local) >= 3:
        ImageDraw.Draw(mask).polygon(
            [(int(round(px)), int(round(py))) for px, py in local],
            fill=255,
        )
    return mask


def split_puzzle(
    image_path: str,
    participants: int,
    output_dir: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Cut `image_path` into a uniform grid of unique interlocking pieces.

    Each source pixel belongs to exactly one piece. Pieces are then assigned
    to participants (3–6 each) with no reuse, so the set reconstructs the
    original image when assembled.
    """
    rng = random.Random(seed)
    split_id = uuid.uuid4().hex
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    # Safety net: force pure B&W even if line-art step left tints.
    source = to_pure_line_art(source)
    source = _ensure_print_resolution(source).convert("RGBA")
    width, height = source.size
    rows, cols, assignment = _plan_assignment(participants, rng, width, height)
    if len(assignment) != rows * cols:
        raise RuntimeError("planned assignment does not cover the grid")

    # Always tag print DPI; pixel size is already >=4K long edge above.
    save_dpi = OUTPUT_DPI

    xs = _partition_bounds(width, cols)
    ys = _partition_bounds(height, rows)
    cell_widths = [xs[i + 1] - xs[i] for i in range(cols)]
    cell_heights = [ys[i + 1] - ys[i] for i in range(rows)]
    min_cell_w = min(cell_widths)
    min_cell_h = min(cell_heights)
    max_cell_w = max(cell_widths)
    max_cell_h = max(cell_heights)
    if min_cell_w < 8 or min_cell_h < 8:
        raise ValueError(
            f"Image is too small to cut into a {rows}×{cols} grid of unique pieces"
        )

    cell_w = width / cols
    cell_h = height / rows
    tab_size = min(min_cell_w, min_cell_h) * TAB_SIZE_RATIO
    # Same export canvas for every piece so tabs/blanks don't change file size.
    tab_pad = int(math.ceil(tab_size)) + PIECE_RENDER_PAD + BORDER_WIDTH_PX
    export_w = max_cell_w + 2 * tab_pad
    export_h = max_cell_h + 2 * tab_pad
    tabs = _build_edge_tabs(rows, cols, rng)
    h_edges, v_edges = _build_shared_edges(rows, cols, xs, ys, tabs, tab_size)

    labels = _init_cell_labels(width, height, rows, cols, xs, ys)
    cells: List[Tuple[int, int, int, int]] = []
    for index in range(rows * cols):
        row, col = divmod(index, cols)
        polygon = _piece_polygon(row, col, xs, ys, h_edges, v_edges, rows, cols)
        cell = (xs[col], ys[row], xs[col + 1], ys[row + 1])
        cells.append(cell)
        _claim_tabs(labels, width, height, index + 1, polygon, cell)

    _verify_exclusive_labels(labels, width, height, rows * cols)

    pieces_meta: List[Dict[str, Any]] = []
    os.makedirs(output_dir, exist_ok=True)
    seen_ids: set[str] = set()
    seen_cells: set[Tuple[int, int]] = set()
    seen_persons: Dict[int, List[str]] = {}

    for index, person in enumerate(assignment):
        row, col = divmod(index, cols)
        piece_id = f"r{row}_c{col}"
        piece_number = index + 1  # stable 1..N in row-major grid order
        if piece_id in seen_ids or (row, col) in seen_cells:
            raise RuntimeError(f"duplicate piece generated for {piece_id}")
        seen_ids.add(piece_id)
        seen_cells.add((row, col))

        x0, y0, _, _ = cells[index]
        left = x0 - tab_pad
        top = y0 - tab_pad
        right = left + export_w
        bottom = top + export_h
        piece_crop = _render_exclusive_piece(
            source, labels, index + 1, (left, top, right, bottom)
        )
        if piece_crop.size != (export_w, export_h):
            raise RuntimeError(
                f"piece {piece_id} is {piece_crop.size}, expected {export_w}x{export_h}"
            )

        piece_meta = {
            "piece_id": piece_id,
            "piece_number": piece_number,
            "person": person,
            "row": row,
            "col": col,
            "index": index,
            "split_id": split_id,
            "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
            "width": width,
            "height": height,
            "rows": rows,
            "cols": cols,
        }
        pnginfo = PngInfo()
        pnginfo.add_text(META_KEY, json.dumps(piece_meta))

        filename = (
            f"cs_puzzle_piece_p{person}_n{piece_number:02d}_{piece_id}_"
            f"{uuid.uuid4().hex}.png"
        )
        out_path = os.path.join(output_dir, filename)
        piece_crop.save(
            out_path,
            "PNG",
            pnginfo=pnginfo,
            dpi=save_dpi,
            compress_level=PNG_COMPRESS_LEVEL,
        )

        seen_persons.setdefault(person, []).append(piece_id)
        pieces_meta.append(
            {
                "piece_id": piece_id,
                "piece_number": piece_number,
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

    if len(seen_ids) != rows * cols:
        raise RuntimeError("split did not produce one unique piece per grid cell")
    if len(pieces_meta) != rows * cols:
        raise RuntimeError("piece list does not match grid size")
    if len(seen_persons) != participants:
        raise RuntimeError("not every participant received unique pieces")
    for person, ids in seen_persons.items():
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"person {person} received duplicate piece ids")
    if len(pieces_meta) < MIN_PIECES_PER_PERSON * participants:
        raise RuntimeError(
            f"split produced {len(pieces_meta)} pieces for {participants} "
            f"participant(s); need at least {MIN_PIECES_PER_PERSON * participants}"
        )

    layout = {
        "version": 4,
        "split_id": split_id,
        "rows": rows,
        "cols": cols,
        "width": width,
        "height": height,
        "dpi": [save_dpi[0], save_dpi[1]],
        "cell_w": cell_w,
        "cell_h": cell_h,
        "xs": xs,
        "ys": ys,
        "piece_width": export_w,
        "piece_height": export_h,
        "tab_size": tab_size,
        "tabs": tabs,
        "partition": "exclusive",
        "border_px": BORDER_WIDTH_PX,
        "participants": participants,
        "total_pieces": len(pieces_meta),
        "pieces": [
            {
                "piece_id": p["piece_id"],
                "piece_number": p["piece_number"],
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
        "split_id": split_id,
        "pieces_per_person": {str(k): len(v) for k, v in sorted(by_person.items())},
        "pieces": pieces_meta,
        "layout": layout,
        "layout_filename": layout_filename,
        "layout_local_path": f"/outputs/{layout_filename}",
        "layout_abs_path": layout_path,
    }
