"""Post-process origami outputs: true alpha cutout, centered transparent PNG."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PIL import Image

from ...shared.gemini import knockout_cream_card_background, to_rgb
from .palette import RGB, soft_clamp_to_palette

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


def _rembg_cutout(img: Image.Image) -> Optional[Image.Image]:
    """True-alpha subject cutout via rembg (keeps black star/eyes on the model)."""
    try:
        from rembg import remove
        from utils.bg_remover import (
            _decontaminate_cutout_edges,
            _get_rembg_session,
            _has_real_transparency,
            _rembg_model_names,
        )
    except Exception as exc:
        print(f"origami rembg import skip: {exc}")
        return None

    model_name = (_rembg_model_names() or ["u2net"])[0]
    try:
        session = _get_rembg_session(model_name)
        cut = remove(to_rgb(img), session=session)
        if not isinstance(cut, Image.Image):
            cut = Image.open(cut) if hasattr(cut, "read") else None
        if cut is None:
            return None
        cut = cut.convert("RGBA")
        if not _has_real_transparency(cut):
            print("origami rembg: opaque alpha — falling back")
            return None
        return _decontaminate_cutout_edges(cut)
    except Exception as exc:
        print(f"origami rembg failed: {exc}")
        return None


def clear_edge_backdrop(img: Image.Image) -> Image.Image:
    """
    Force true transparency: clear edge-connected cream/white/black studio void.
    Keeps interior black print (star) and eyes — only edge-reachable backdrop dies.
    """
    rgba = img.convert("RGBA")
    if not _HAS_NP:
        try:
            return knockout_cream_card_background(rgba).convert("RGBA")
        except Exception:
            return rgba

    arr = np.asarray(rgba).copy()
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    a = arr[:, :, 3]
    lum = (0.299 * r + 0.587 * g + 0.114 * b)
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    cream = ((lum >= 175) & (sat <= 80)) | ((lum >= 235) & (sat <= 30))
    black_void = (lum <= 42) & (sat <= 40)
    already_clear = a < 8
    walkable = cream | black_void | already_clear

    h, w = walkable.shape
    reach = np.zeros((h, w), dtype=bool)
    reach[0, :] = walkable[0, :]
    reach[-1, :] = walkable[-1, :]
    reach[:, 0] = walkable[:, 0]
    reach[:, -1] = walkable[:, -1]

    if _HAS_SCIPY:
        structure = np.ones((3, 3), dtype=bool)
        for _ in range(max(h, w)):
            grown = _ndimage.binary_dilation(reach, structure=structure) & walkable
            if np.array_equal(grown, reach):
                break
            reach = grown
    else:
        # Simple iterative flood
        changed = True
        while changed:
            changed = False
            up = np.zeros_like(reach)
            up[1:, :] = reach[:-1, :]
            down = np.zeros_like(reach)
            down[:-1, :] = reach[1:, :]
            left = np.zeros_like(reach)
            left[:, 1:] = reach[:, :-1]
            right = np.zeros_like(reach)
            right[:, :-1] = reach[:, 1:]
            grown = (up | down | left | right | reach) & walkable
            if not np.array_equal(grown, reach):
                reach = grown
                changed = True

    arr[reach, 3] = 0
    arr[reach, 0:3] = 0
    return Image.fromarray(arr, mode="RGBA")


def make_transparent(img: Image.Image) -> Image.Image:
    """rembg cutout, then always clear residual edge cream/black void."""
    cut = _rembg_cutout(img)
    if cut is None:
        try:
            cut = knockout_cream_card_background(img).convert("RGBA")
        except Exception as exc:
            print(f"origami knockout fallback failed: {exc}")
            cut = to_rgb(img).convert("RGBA")
    return clear_edge_backdrop(cut)


def clamp_opaque_to_palette(
    img: Image.Image,
    palette: Sequence[RGB],
    solid_color: Optional[RGB] = None,
) -> Image.Image:
    """Soft-clamp opaque pixels to submission gamut; keep alpha intact."""
    if not palette and solid_color is None:
        return img.convert("RGBA") if img.mode == "RGBA" else to_rgb(img).convert("RGBA")

    rgba = img.convert("RGBA")
    clamped_rgb = soft_clamp_to_palette(
        to_rgb(rgba),
        palette,
        solid_color=solid_color,
    )

    if _HAS_NP:
        arr = np.asarray(rgba)
        alpha = arr[:, :, 3]
        out = np.asarray(clamped_rgb).copy()
        rgba_out = np.dstack([out, alpha])
        rgba_out[alpha < 8, :] = 0
        return Image.fromarray(rgba_out, mode="RGBA")

    out = clamped_rgb.convert("RGBA")
    out.putalpha(rgba.getchannel("A"))
    return out


def center_on_transparent(
    img: Image.Image,
    canvas_size: Optional[Tuple[int, int]] = None,
    pad_ratio: float = 0.08,
) -> Image.Image:
    """Crop to opaque bounds and center on a transparent canvas."""
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return rgba

    cropped = rgba.crop(bbox)
    cw, ch = cropped.size

    if canvas_size is None:
        side = max(cw, ch)
        side = max(int(side / (1 - 2 * pad_ratio)), 768)
        canvas_size = (side, side)

    # Never shrink the transparent canvas below a usable preview size
    canvas_size = (max(canvas_size[0], 512), max(canvas_size[1], 512))

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    target_w = int(canvas_size[0] * (1 - 2 * pad_ratio))
    target_h = int(canvas_size[1] * (1 - 2 * pad_ratio))
    scale = min(target_w / max(cw, 1), target_h / max(ch, 1))
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (canvas_size[0] - new_w) // 2
    y = (canvas_size[1] - new_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def finalize_origami(
    output_path: str,
    palette: Sequence[RGB],
    canvas_size: Optional[Tuple[int, int]] = None,
    solid_color: Optional[RGB] = None,
    clamp_colors: bool = True,
) -> None:
    """Cutout → clear edge void → optional solid clamp → centered RGBA PNG."""
    with Image.open(output_path) as raw:
        transparent = make_transparent(raw)
        if clamp_colors and (solid_color is not None or palette):
            palette_list = list(palette)
            if (0, 0, 0) not in palette_list and not any(max(c) <= 40 for c in palette_list):
                palette_list.append((0, 0, 0))
            transparent = clamp_opaque_to_palette(
                transparent,
                palette_list,
                solid_color=solid_color,
            )
            # Re-clear after clamp in case snap refilled void
            transparent = clear_edge_backdrop(transparent)
        if canvas_size is None:
            w, h = transparent.size
            side = max(w, h, 768)
            canvas_size = (side, side)
        final = center_on_transparent(transparent, canvas_size=canvas_size)
        # Guarantee RGBA PNG (no accidental RGB flatten)
        final = final.convert("RGBA")
        final.save(output_path, format="PNG", optimize=True)
        # Sanity log
        alpha = np.asarray(final.split()[3]) if _HAS_NP else None
        if alpha is not None:
            clear_pct = float((alpha < 8).mean() * 100)
            print(f"origami finalize: {final.size} RGBA, {clear_pct:.1f}% transparent")
