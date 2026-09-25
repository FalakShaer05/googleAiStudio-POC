"""
Background removal utilities using cloud APIs (Freepik, remove.bg) and
local rembg for true PNG alpha (internal fallback).
"""
import os
import time
from typing import List, Optional, Tuple

import requests
from PIL import Image

from .s3_utils import build_public_image_url

# Cache rembg sessions so model weights are loaded once per process.
_REMBG_SESSIONS = {}


def _bg_removed_output_path(image_path: str) -> str:
    base, _ext = os.path.splitext(image_path)
    return f"{base}_bg_removed.png"


def _should_skip_freepik() -> bool:
    return os.getenv("SKIP_FREEPIK", "").lower() in ("1", "true", "yes")


def _is_placeholder_key(value: str) -> bool:
    normalized = value.strip().lower()
    return not normalized or normalized.startswith("your_") or normalized.endswith("_here")


def _save_response_image(image_path: str, content: bytes) -> str:
    output_path = _bg_removed_output_path(image_path)
    with open(output_path, "wb") as output_file:
        output_file.write(content)
    return output_path


def _save_pil_image(image_path: str, image: Image.Image) -> str:
    output_path = _bg_removed_output_path(image_path)
    image.convert("RGBA").save(output_path, "PNG")
    return output_path


def _has_real_transparency(image: Image.Image, min_ratio: float = 0.02) -> bool:
    """True when a meaningful share of pixels are actually transparent."""
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    # Histogram: index i = count of pixels with alpha == i
    hist = alpha.histogram()
    transparent = sum(hist[:16])  # alpha 0..15
    total = rgba.size[0] * rgba.size[1]
    if total <= 0:
        return False
    return (transparent / total) >= min_ratio


def _rembg_model_names() -> List[str]:
    # Default to u2net only — extra models (isnet/birefnet) are huge cold downloads.
    raw = os.getenv("REMBG_MODEL_ORDER", "u2net")
    names = [part.strip() for part in raw.split(",") if part.strip()]
    return names or ["u2net"]


def _rembg_max_side() -> int:
    try:
        return max(512, int(os.getenv("REMBG_MAX_SIDE", "2048")))
    except ValueError:
        return 2048


def _load_rgb_for_rembg(image_path: str) -> Image.Image:
    """Load RGB and downscale large images so ONNX inference stays responsive."""
    source = Image.open(image_path).convert("RGB")
    max_side = _rembg_max_side()
    width, height = source.size
    if max(width, height) > max_side:
        source.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        print(f"📐 rembg input resized {width}x{height} → {source.size[0]}x{source.size[1]}")
    return source


def _get_rembg_session(model_name: str):
    if model_name in _REMBG_SESSIONS:
        return _REMBG_SESSIONS[model_name]
    from rembg import new_session

    session = new_session(model_name)
    _REMBG_SESSIONS[model_name] = session
    return session


def warmup_rembg_sessions() -> None:
    """
    Pre-load the first rembg model so the first selfie / Me Remix request
    does not pay download + ONNX session init latency.
    """
    model_name = (_rembg_model_names() or ["u2net"])[0]
    try:
        print(f"🔧 Warming rembg session model={model_name}...")
        _get_rembg_session(model_name)
        print(f"✅ rembg warmup complete model={model_name}")
    except Exception as exc:
        print(f"⚠️ rembg warmup skipped: {exc}")


def _decontaminate_cutout_edges(image: Image.Image) -> Image.Image:
    """
    Reduce light/white fringe on rembg cutouts by un-premultiplying against a
    light background estimate, then hardening near-transparent fringe pixels.
    """
    try:
        import numpy as np
    except ImportError:
        return image.convert("RGBA")

    arr = np.array(image.convert("RGBA"), dtype=np.float32)
    alpha = arr[:, :, 3] / 255.0
    h, w = alpha.shape
    if h < 2 or w < 2:
        return image.convert("RGBA")

    # Estimate background from mostly-transparent border samples (fallback: white).
    border = np.concatenate(
        [
            arr[0, :, :3].reshape(-1, 3),
            arr[-1, :, :3].reshape(-1, 3),
            arr[:, 0, :3].reshape(-1, 3),
            arr[:, -1, :3].reshape(-1, 3),
        ],
        axis=0,
    )
    border_alpha = np.concatenate(
        [alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]],
        axis=0,
    )
    bg_samples = border[border_alpha < 0.2]
    if bg_samples.size:
        bg = bg_samples.mean(axis=0)
    else:
        bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    # Soft edge band: decontaminate RGB so light bg doesn't bleed into subject.
    soft = (alpha > 0.02) & (alpha < 0.98)
    safe_a = np.maximum(alpha, 1e-4)
    for c in range(3):
        channel = arr[:, :, c]
        restored = (channel - (1.0 - alpha) * bg[c]) / safe_a
        arr[:, :, c] = np.where(soft, np.clip(restored, 0, 255), channel)

    # Drop dust / hairline fringe that is almost fully transparent.
    arr[:, :, 3] = np.where(alpha < 0.06, 0, arr[:, :, 3])

    # Mild alpha harden near edges: shrink very soft fringe without eating hair.
    mid = (alpha >= 0.06) & (alpha < 0.35)
    # Only harden when pixel is still near the estimated background (halo color).
    near_bg = (
        (np.abs(arr[:, :, 0] - bg[0]) < 28)
        & (np.abs(arr[:, :, 1] - bg[1]) < 28)
        & (np.abs(arr[:, :, 2] - bg[2]) < 28)
    )
    arr[mid & near_bg, 3] = 0

    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def remove_background_with_rembg(image_path: str) -> Optional[str]:
    """
    Local background removal with rembg — true PNG alpha (no baked checkerboard).
    Model order comes from REMBG_MODEL_ORDER (comma-separated).
    """
    if not os.path.exists(image_path):
        return None

    try:
        from rembg import remove
    except Exception as e:
        print(f"❌ rembg is not available: {e}")
        return None

    try:
        print("🔧 Requesting background removal from rembg (local)...")
        started = time.perf_counter()
        source = _load_rgb_for_rembg(image_path)
        last_error = None

        for model_name in _rembg_model_names():
            try:
                session = _get_rembg_session(model_name)
                cut = remove(source, session=session)
                if not isinstance(cut, Image.Image):
                    from io import BytesIO

                    cut = Image.open(BytesIO(cut))
                cut = cut.convert("RGBA")
                if not _has_real_transparency(cut):
                    print(f"⚠️ rembg model={model_name} returned opaque alpha — trying next")
                    continue
                cut = _decontaminate_cutout_edges(cut)
                output_path = _save_pil_image(image_path, cut)
                print(
                    f"✅ rembg background removal successful "
                    f"(model={model_name}) in {time.perf_counter() - started:.2f}s"
                )
                return output_path
            except Exception as model_exc:
                last_error = model_exc
                print(f"⚠️ rembg model={model_name} failed: {model_exc}")

        if last_error:
            print(f"❌ rembg exhausted models: {last_error}")
        else:
            print("❌ rembg did not produce a transparent cutout")
        return None
    except Exception as e:
        print(f"❌ Error in rembg background removal: {e}")
        return None


def remove_background_with_gemini_api(image_path: str) -> Optional[str]:
    """
    Deprecated for /remove-bg: Gemini often paints a fake checkerboard into RGB.
    Kept for callers that still import it; prefer rembg for true transparency.
    """
    if _is_placeholder_key(os.getenv("GEMINI_API_KEY", "")):
        print("❌ GEMINI_API_KEY not set — cannot use Gemini fallback")
        return None

    if not os.path.exists(image_path):
        return None

    try:
        from .character_utils import (
            _extract_final_image_from_response,
            _generate_content_image,
            get_gemini_client,
            get_gemini_image_model,
            select_gemini_aspect_ratio,
        )

        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        aspect_ratio = select_gemini_aspect_ratio(width, height)
        prompt = (
            "Remove the background from this image completely. "
            "Output ONLY the main subject with a fully transparent background. "
            "Do NOT draw a checkerboard, grid, dither, gray squares, or any "
            "transparency-preview pattern. Transparency must be real empty pixels, "
            "not a painted pattern. "
            "Do not add a new background, floor shadow, or extra objects. "
            "Preserve the subject exactly as shown."
        )

        print("🔧 Requesting background removal from Gemini...")
        started = time.perf_counter()
        client = get_gemini_client()
        response = _generate_content_image(
            client=client,
            model=get_gemini_image_model(),
            contents=[prompt, image],
            aspect_ratio=aspect_ratio,
            operation="art_generation:bg_removal",
        )
        result = _extract_final_image_from_response(response)
        if result is None:
            print("❌ Gemini did not return an image")
            return None

        result = result.convert("RGBA")
        if not _has_real_transparency(result):
            print(
                "❌ Gemini returned opaque / fake-transparency output "
                "(likely a painted checkerboard) — rejecting"
            )
            return None

        output_path = _save_pil_image(image_path, result)
        print(f"✅ Gemini background removal successful in {time.perf_counter() - started:.2f}s")
        return output_path
    except Exception as e:
        print(f"❌ Error in Gemini background removal: {e}")
        return None


def remove_background_with_removebg_api(image_path: str) -> Optional[str]:
    """
    Remove background using remove.bg API.
    Documentation: https://www.remove.bg/api
    """
    api_key = os.getenv("REMOVE_BG_API_KEY", "").strip()
    if _is_placeholder_key(api_key):
        print("❌ REMOVE_BG_API_KEY not set — add a valid key from https://www.remove.bg/api")
        return None

    if not os.path.exists(image_path):
        return None

    headers = {"X-Api-Key": api_key}
    started = time.perf_counter()

    try:
        print("🔧 Requesting background removal from remove.bg (file upload)...")
        with open(image_path, "rb") as image_file:
            response = requests.post(
                "https://api.remove.bg/v1.0/removebg",
                files={"image_file": (os.path.basename(image_path), image_file, "application/octet-stream")},
                data={"size": "auto", "format": "png"},
                headers=headers,
                timeout=60,
            )

        if response.status_code == 200:
            output_path = _save_response_image(image_path, response.content)
            print(f"✅ remove.bg background removal successful in {time.perf_counter() - started:.2f}s")
            return output_path

        print(f"❌ remove.bg file upload failed: {response.status_code}")
        try:
            print(f"   Error: {response.json().get('errors', response.text)}")
        except Exception:
            print(f"   Response: {response.text}")

        if response.status_code == 403:
            return None

        public_url = build_public_image_url(image_path)
        if not public_url:
            return None

        print("🔧 Retrying remove.bg with public image URL...")
        response = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            data={"image_url": public_url, "size": "auto", "format": "png"},
            headers=headers,
            timeout=60,
        )

        if response.status_code == 200:
            output_path = _save_response_image(image_path, response.content)
            print(f"✅ remove.bg background removal successful in {time.perf_counter() - started:.2f}s")
            return output_path

        print(f"❌ remove.bg URL request failed: {response.status_code}")
        try:
            print(f"   Error: {response.json().get('errors', response.text)}")
        except Exception:
            print(f"   Response: {response.text}")
        return None
    except Exception as e:
        print(f"❌ Error in remove.bg background removal: {e}")
        return None


def remove_background(image_path: str) -> Tuple[Optional[str], str, str]:
    """
    Remove background using providers in order:
      1. rembg local (fast, true PNG alpha — already warmed in Docker)
      2. remove.bg (requires valid REMOVE_BG_API_KEY)
      3. Freepik (unless SKIP_FREEPIK=true; needs public S3 URL)

    Returns:
        (result_path, method, error_summary)
    """
    errors = []

    # Local first: avoids S3 upload + cloud round-trips that often hang /remove-bg.
    print("🔧 Trying rembg (local) first...")
    result = remove_background_with_rembg(image_path)
    if result:
        return result, "rembg", ""
    errors.append("rembg local did not return a transparent image")

    print("⚠️ Falling back to remove.bg...")
    if _is_placeholder_key(os.getenv("REMOVE_BG_API_KEY", "")):
        errors.append("REMOVE_BG_API_KEY is not configured")
    else:
        result = remove_background_with_removebg_api(image_path)
        if result:
            return result, "removebg", ""
        errors.append("remove.bg rejected the request (check REMOVE_BG_API_KEY is valid)")

    if not _should_skip_freepik():
        print("⚠️ Falling back to Freepik...")
        result = remove_background_with_freepik_api(image_path)
        if result:
            return result, "freepik", ""
        errors.append("Freepik unavailable or out of credits")
    else:
        print("⚡ SKIP_FREEPIK enabled — skipping Freepik")

    return None, "none", "; ".join(errors)


def remove_background_with_freepik_api(image_path: str) -> Optional[str]:
    """
    Remove background using Freepik API
    Documentation: https://docs.freepik.com/api-reference/remove-background/post-beta-remove-background
    """
    try:
        freepik_api_key = os.getenv("FREEPIK_API_KEY", "").strip()
        if _is_placeholder_key(freepik_api_key):
            print("❌ FREEPIK_API_KEY not found in environment variables")
            return None

        if not os.path.exists(image_path):
            return None

        public_url = build_public_image_url(image_path)
        if not public_url:
            print("❌ Freepik API requires a publicly accessible image URL")
            return None

        print(f"🌐 Using public image URL for Freepik: {public_url}")

        api_url = "https://api.freepik.com/v1/ai/beta/remove-background"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "x-freepik-api-key": freepik_api_key,
        }

        print("🔧 Requesting background removal from Freepik...")
        response = requests.post(
            api_url,
            headers=headers,
            data={"image_url": public_url},
            timeout=30,
        )

        if response.status_code != 200:
            print(f"❌ Freepik API request failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', response.text)}")
            except Exception:
                print(f"   Response: {response.text}")
            return None

        result = response.json()
        output_url = result.get("high_resolution") or result.get("url")
        if not output_url:
            print("❌ No output URL in Freepik response")
            return None

        output_response = requests.get(output_url, timeout=30)
        if output_response.status_code == 200:
            output_path = _save_response_image(image_path, output_response.content)
            print("✅ Freepik background removal successful")
            return output_path

        print(f"❌ Failed to download Freepik result: {output_response.status_code}")
        return None

    except Exception as e:
        print(f"❌ Error in Freepik background removal: {e}")
        return None
