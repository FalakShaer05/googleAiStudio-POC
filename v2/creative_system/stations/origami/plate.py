"""Build a clean color plate from the open crease pattern for spatial wrap."""
from __future__ import annotations

from PIL import Image, ImageFilter

from ...shared.gemini import to_rgb

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    np = None  # type: ignore
    _HAS_NP = False

try:
    from scipy import ndimage as _ndimage
    _HAS_SCIPY = True
except ImportError:
    _ndimage = None  # type: ignore
    _HAS_SCIPY = False

CREAM = (245, 240, 230)


def build_color_plate(artwork_path: str, max_side: int = 1024) -> Image.Image:
    """
    Flat print layout for Gemini:
    - crop to the paper sheet
    - keep colored regions + solid black motifs (star)
    - erase thin crease/dash lines that confuse placement
    - cream page (not black void)
    """
    rgb = to_rgb(Image.open(artwork_path))
    if _HAS_NP:
        plate = _clean_numpy(rgb)
    else:
        plate = _clean_pil(rgb)

    if max(plate.size) > max_side:
        plate = plate.copy()
        plate.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return plate


def _clean_numpy(rgb: Image.Image) -> Image.Image:
    arr = np.asarray(rgb, dtype=np.uint8)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    lum = (0.299 * r + 0.587 * g + 0.114 * b)
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    # Sheet vs void: near-black and low-sat is background OR ink
    near_black = (lum <= 40) & (sat <= 45)
    paper = ~near_black

    # Crop to paper content (include ink inside the sheet bbox)
    ys, xs = np.where(paper)
    if len(xs) < 50:
        return Image.fromarray(arr, mode="RGB")

    pad = max(4, min(arr.shape[:2]) // 80)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(arr.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(arr.shape[1], int(xs.max()) + pad + 1)
    crop = arr[y0:y1, x0:x1].copy()
    c_r = crop[:, :, 0].astype(np.int16)
    c_g = crop[:, :, 1].astype(np.int16)
    c_b = crop[:, :, 2].astype(np.int16)
    c_lum = 0.299 * c_r + 0.587 * c_g + 0.114 * c_b
    c_sat = np.maximum(np.maximum(c_r, c_g), c_b) - np.minimum(np.minimum(c_r, c_g), c_b)
    ink = (c_lum <= 48) & (c_sat <= 50)

    # Keep large black motifs (central star); erase thin crease dashes
    crease = ink.copy()
    if _HAS_SCIPY and ink.any():
        labeled, nlab = _ndimage.label(ink)
        if nlab:
            sizes = np.bincount(labeled.ravel())
            # Largest ink blob(s) that are sizable = printed black art
            keep = np.zeros_like(ink)
            # Sort components by size descending, skip background 0
            order = np.argsort(sizes)[::-1]
            sheet_area = crop.shape[0] * crop.shape[1]
            for idx in order:
                if idx == 0:
                    continue
                sz = int(sizes[idx])
                if sz < 40:
                    continue
                # Thick printed motif: reasonably large and compact-ish
                if sz >= max(120, sheet_area * 0.004):
                    keep |= labeled == idx
                    # keep top few large motifs only
                    if keep.sum() > sheet_area * 0.08:
                        break
            crease = ink & ~keep

    # Inpaint crease pixels from nearby non-ink colors
    out = crop.astype(np.float32)
    if crease.any():
        mask = ~ink
        # Blur a version where ink is replaced by local mean of paper
        filled = out.copy()
        # Seed ink with cream temporarily for blur fill
        filled[ink] = CREAM
        blur = Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=2.5)
        )
        blur_arr = np.asarray(blur, dtype=np.float32)
        # Stronger neighbor fill via box average on mask
        if _HAS_SCIPY:
            for ch in range(3):
                channel = out[:, :, ch]
                weights = mask.astype(np.float32)
                num = _ndimage.uniform_filter(channel * weights, size=9)
                den = _ndimage.uniform_filter(weights, size=9)
                avg = np.divide(num, den, out=np.full_like(num, CREAM[ch]), where=den > 0.05)
                channel = channel.copy()
                channel[crease] = avg[crease]
                # fallback where den weak
                weak = crease & (den <= 0.05)
                channel[weak] = blur_arr[:, :, ch][weak]
                out[:, :, ch] = channel
        else:
            out[crease] = blur_arr[crease]

    # Outside-sheet (still near-black after crop edges) → cream
    edge_void = (c_lum <= 28) & (c_sat <= 40) & ~ink
    # Only clear void that is connected-ish to borders
    if _HAS_SCIPY and edge_void.any():
        structure = np.ones((3, 3), dtype=bool)
        # flood from borders
        border = np.zeros_like(edge_void)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        reachable = edge_void & border
        # dilate into edge_void
        for _ in range(max(edge_void.shape)):
            grown = _ndimage.binary_dilation(reachable, structure=structure) & edge_void
            if np.array_equal(grown, reachable):
                break
            reachable = grown
        out[reachable] = CREAM
    else:
        out[edge_void] = CREAM

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _clean_pil(rgb: Image.Image) -> Image.Image:
    """Fallback: crop non-black bbox onto cream; light blur of dark lines."""
    arr_img = rgb.convert("RGB")
    gray = arr_img.convert("L")
    # bbox of brighter content
    mask = gray.point(lambda p: 255 if p > 42 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return arr_img
    cropped = arr_img.crop(bbox)
    # Soften thin dark dashes by slight median
    try:
        cleaned = cropped.filter(ImageFilter.MedianFilter(size=3))
    except Exception:
        cleaned = cropped
    # Composite: where very dark thin, use median; keep strong black blobs roughly
    base = Image.new("RGB", cleaned.size, CREAM)
    base.paste(cleaned, (0, 0))
    return base


def describe_layout_hint(artwork_path: str) -> str:
    """
    Short spatial hint from the open sheet (center vs corners) for the prompt.
    Helps stop random facet painting when the print is symmetric.
    """
    if not _HAS_NP:
        return ""
    try:
        plate = build_color_plate(artwork_path, max_side=240)
        arr = np.asarray(plate, dtype=np.int16)
    except Exception:
        return ""

    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        return ""

    cy0, cy1 = h // 3, 2 * h // 3
    cx0, cx1 = w // 3, 2 * w // 3
    center = arr[cy0:cy1, cx0:cx1]
    c_lum = 0.299 * center[:, :, 0] + 0.587 * center[:, :, 1] + 0.114 * center[:, :, 2]
    has_black_center = float((c_lum <= 40).mean()) > 0.08

    # Tight corner tips (yellow on this pattern) — avoid inner pink rings
    t = max(4, min(h, w) // 8)
    corners = [
        arr[0:t, 0:t],
        arr[0:t, w - t : w],
        arr[h - t : h, 0:t],
        arr[h - t : h, w - t : w],
    ]
    corner_cols = []
    for tile in corners:
        flat = tile.reshape(-1, 3)
        lum = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
        sat = flat.max(axis=1) - flat.min(axis=1)
        keep = (sat > 40) & (lum > 50)
        if keep.sum() < 5:
            continue
        sample = flat[keep]
        q = (sample // 32) * 32
        keys, counts = np.unique(q, axis=0, return_counts=True)
        best = keys[counts.argmax()]
        corner_cols.append("#{:02X}{:02X}{:02X}".format(int(best[0]), int(best[1]), int(best[2])))

    bits = []
    if has_black_center:
        bits.append(
            "center print is black motif/star — must stay a coherent mark on the "
            "folded body/back, not speckles"
        )
    if corner_cols:
        uniq = list(dict.fromkeys(corner_cols))
        bits.append(
            "sheet corners are "
            + "/".join(uniq[:3])
            + " — those colors belong on outer tips/toes/limb ends after folding, "
            "not sprinkled on the torso"
        )
    bits.append(
        "large cream/off-white areas stay large contiguous body faces — "
        "do not break into random neon triangles"
    )
    return "LAYOUT FROM OPEN SHEET: " + "; ".join(bits) + "."
