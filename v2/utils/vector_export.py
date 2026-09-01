"""
Export helpers for Adobe Illustrator workflows and print-ready downloads.

- PNG / TIFF / PDF: full-quality raster exports with PPI metadata where supported
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

SUPPORTED_EXPORT_FORMATS = frozenset({"png", "tiff", "pdf", "svg", "eps"})

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


def _load_pil(image_path: str) -> Image.Image:
    return Image.open(image_path)


def _read_dpi(image_path: str) -> int:
    with Image.open(image_path) as img:
        dpi = img.info.get("dpi")
        if dpi and isinstance(dpi, tuple) and dpi[0]:
            return max(72, min(600, int(round(float(dpi[0])))))
    return 300


def write_png(image_path: str, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with _load_pil(image_path) as img:
        dpi = img.info.get("dpi") or (300, 300)
        save_kwargs = {"optimize": True}
        if isinstance(dpi, tuple):
            save_kwargs["dpi"] = dpi
        img.save(output_path, "PNG", **save_kwargs)


def write_tiff(image_path: str, output_path: str, dpi: int = 300) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with _load_pil(image_path) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
        img.save(
            output_path,
            "TIFF",
            compression="tiff_lzw",
            dpi=(dpi, dpi),
        )


def write_pdf(image_path: str, output_path: str, dpi: int = 300) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    img = _load_rgb(image_path)
    img.save(output_path, "PDF", resolution=float(dpi))


def export_image_format(
    image_path: str,
    output_path: str,
    fmt: str = "png",
) -> Tuple[bool, str]:
    """
    Export an output image to a user-selected print format.

    fmt: png | tiff | pdf | svg | eps
    """
    try:
        if not os.path.exists(image_path):
            return False, f"Image not found: {image_path}"

        fmt = (fmt or "png").strip().lower()
        if fmt == "tif":
            fmt = "tiff"
        if fmt not in SUPPORTED_EXPORT_FORMATS:
            return False, f"format must be one of: {', '.join(sorted(SUPPORTED_EXPORT_FORMATS))}"

        dpi = _read_dpi(image_path)

        if fmt == "png":
            write_png(image_path, output_path)
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                return False, "PNG export failed"
            return True, "PNG ready for download"

        if fmt == "tiff":
            write_tiff(image_path, output_path, dpi=dpi)
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                return False, "TIFF export failed"
            return True, f"TIFF ready for print ({dpi} PPI)"

        if fmt == "pdf":
            write_pdf(image_path, output_path, dpi=dpi)
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                return False, "PDF export failed"
            return True, f"PDF ready for print ({dpi} PPI)"

        return raster_to_vector(image_path=image_path, output_path=output_path, fmt=fmt)

    except Exception as e:
        print(f"export_image_format error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


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
