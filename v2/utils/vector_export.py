"""
Export helpers for Adobe Illustrator workflows.

- SVG: auto-traced editable vector paths (vtracer). Looks posterized by nature.
- EPS: Encapsulated PostScript with the full raster embedded (correct appearance
  in Illustrator / print). We do NOT convert traced paths to EPS — that path
  was producing scrambled output.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

from PIL import Image


_MAX_TRACE_EDGE = 1600


def _load_rgb(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _prepare_trace_image(image_path: str) -> Tuple[str, Optional[str]]:
    """Downscale for tracing if needed; always write a clean temp RGB PNG."""
    img = _load_rgb(image_path)
    w, h = img.size
    longest = max(w, h)
    if longest > _MAX_TRACE_EDGE:
        scale = _MAX_TRACE_EDGE / float(longest)
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )

    fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(temp_path, "PNG")
    return temp_path, temp_path


def write_eps_from_raster(image_path: str, eps_path: str) -> None:
    """
    Write an EPSF file that embeds the PNG/JPEG as an image.

    Opens correctly in Adobe Illustrator with full visual fidelity.
    (Not editable vector paths — use SVG for that.)
    """
    img = _load_rgb(image_path)
    out_dir = os.path.dirname(eps_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Pillow writes Level-2 EPS with the raster payload
    img.save(eps_path, "EPS")


def write_svg_traced(image_path: str, svg_path: str) -> None:
    """Auto-trace raster → SVG color paths via vtracer."""
    import vtracer

    trace_path, temp_path = _prepare_trace_image(image_path)
    try:
        out_dir = os.path.dirname(svg_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        vtracer.convert_image_to_svg_py(
            trace_path,
            svg_path,
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=8,
            color_precision=7,
            layer_difference=12,
            corner_threshold=60,
            length_threshold=4.5,
            max_iterations=10,
            splice_threshold=45,
            path_precision=2,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def raster_to_vector(
    image_path: str,
    output_path: str,
    fmt: str = "svg",
) -> Tuple[bool, str]:
    """
    Export image for Illustrator.

    fmt:
      - "svg": traced editable vectors (posterized)
      - "eps": full-quality EPS with embedded image (looks correct in AI)
    """
    try:
        if not os.path.exists(image_path):
            return False, f"Image not found: {image_path}"

        fmt = (fmt or "svg").strip().lower()
        if fmt not in {"eps", "svg"}:
            return False, "format must be eps or svg"

        if fmt == "eps":
            write_eps_from_raster(image_path, output_path)
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                return False, "EPS export failed"
            return True, "EPS ready for Adobe Illustrator (full image quality)"

        try:
            import vtracer  # noqa: F401
        except ImportError:
            return (
                False,
                "SVG vector export requires 'vtracer'. Install with: pip install vtracer",
            )

        write_svg_traced(image_path, output_path)
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return False, "SVG tracing produced an empty file"
        return True, "SVG vector ready for Adobe Illustrator (auto-traced paths)"

    except Exception as e:
        print(f"raster_to_vector error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)
