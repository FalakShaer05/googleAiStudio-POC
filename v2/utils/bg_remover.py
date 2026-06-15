"""
Background removal utilities using cloud APIs (Freepik, remove.bg, Gemini).
"""
import os
import time
from typing import Optional, Tuple

import requests
from PIL import Image

from .s3_utils import build_public_image_url


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


def remove_background_with_gemini_api(image_path: str) -> Optional[str]:
    """Remove background using Gemini image editing (no local model storage)."""
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
        )
        result = _extract_final_image_from_response(response)
        if result is None:
            print("❌ Gemini did not return an image")
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
    Remove background using cloud providers in order:
      1. Freepik (unless SKIP_FREEPIK=true)
      2. remove.bg (requires valid REMOVE_BG_API_KEY)
      3. Gemini (uses GEMINI_API_KEY)

    Returns:
        (result_path, method, error_summary)
    """
    errors = []

    if not _should_skip_freepik():
        result = remove_background_with_freepik_api(image_path)
        if result:
            return result, "freepik", ""
        errors.append("Freepik unavailable or out of credits")
    else:
        print("⚡ SKIP_FREEPIK enabled — skipping Freepik")

    print("⚠️ Falling back to remove.bg...")
    if _is_placeholder_key(os.getenv("REMOVE_BG_API_KEY", "")):
        errors.append("REMOVE_BG_API_KEY is not configured")
    else:
        result = remove_background_with_removebg_api(image_path)
        if result:
            return result, "removebg", ""
        errors.append("remove.bg rejected the request (check REMOVE_BG_API_KEY is valid)")

    print("⚠️ Falling back to Gemini...")
    result = remove_background_with_gemini_api(image_path)
    if result:
        return result, "gemini", ""

    errors.append("Gemini fallback did not return an image")
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
