"""
Character generation utilities using Google Gemini API.
"""
import os
import uuid
import io
import re
import time
import random
from typing import Optional, Tuple, Dict, Any

from PIL import Image, ImageOps
import requests
import google.genai as genai

try:
    # Newer google-genai SDKs expose GenerateContentConfig / ImageConfig here
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover
    types = None  # type: ignore


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def get_gemini_image_model() -> str:
    """
    Returns the Gemini model id used for image generation/editing.

    Defaults to the stable GA model. Override via GEMINI_IMAGE_MODEL to avoid
    code changes when Google retires preview model ids.
    """
    return os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")


def get_gemini_fallback_image_model() -> Optional[str]:
    """
    Optional fallback model id used when the primary model is temporarily unavailable.
    """
    value = (os.getenv("GEMINI_FALLBACK_IMAGE_MODEL") or "").strip()
    return value or None


def _is_transient_gemini_error(exc: Exception) -> bool:
    """
    Best-effort classification of transient Gemini API errors.

    We intentionally keep this heuristic string-based because google-genai exceptions
    differ across SDK versions, and we don't want production to break due to import issues.
    """
    msg = str(exc)
    transient_markers = [
        "503", "UNAVAILABLE",
        "Deadline expired", "DEADLINE_EXCEEDED",
        "429", "RESOURCE_EXHAUSTED",
        "Internal", "500",
    ]
    return any(m in msg for m in transient_markers)


def _iter_gemini_response_parts(response):
    """
    Normalize response parts across google-genai SDK versions.

    Some versions expose `response.parts`; others expose `response.candidates[0].content.parts`.
    """
    if response is None:
        return []
    parts = getattr(response, "parts", None)
    if parts is not None:
        return parts
    candidates = getattr(response, "candidates", None)
    if candidates:
        content = getattr(candidates[0], "content", None)
        if content is not None:
            return getattr(content, "parts", []) or []
    return []


def _extract_final_image_from_response(response) -> Optional[Image.Image]:
    """
    Extract the final rendered image from a Gemini response.

    For Gemini 3 Pro Image, the model may emit interim "thought" images first; we skip those and
    prefer the last non-thought image.
    """
    parts = _iter_gemini_response_parts(response)
    if not parts:
        return None

    non_thought_images = []
    any_images = []

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        # Prefer SDK helper if present
        as_image = getattr(part, "as_image", None)
        try:
            if callable(as_image):
                img = as_image()
                # Ensure it's a PIL Image object - convert if needed
                if not isinstance(img, Image.Image):
                    # If it's not a PIL Image, try to convert from bytes
                    img = Image.open(io.BytesIO(inline.data))
            else:
                img = Image.open(io.BytesIO(inline.data))
            
            # Verify the image has the size attribute (PIL Image should have it)
            if not hasattr(img, 'size'):
                continue
        except Exception:
            continue

        any_images.append(img)
        if not bool(getattr(part, "thought", False)):
            non_thought_images.append(img)

    if non_thought_images:
        return non_thought_images[-1]
    if any_images:
        return any_images[-1]
    return None


def _generate_content_image(client: genai.Client, model: str, contents):
    """
    Call `client.models.generate_content` with an optional config to force image output.

    Falls back to calling without config for older SDK versions.
    """
    # If the SDK supports it, force image-only output for consistency.
    config = None
    if types is not None and hasattr(types, "GenerateContentConfig"):
        try:
            config = types.GenerateContentConfig(response_modalities=["IMAGE"])
        except Exception:
            config = None

    def _call(selected_model: str):
        try:
            if config is not None:
                return client.models.generate_content(model=selected_model, contents=contents, config=config)
            return client.models.generate_content(model=selected_model, contents=contents)
        except TypeError:
            # Some SDK versions don't accept `config=...`
            return client.models.generate_content(model=selected_model, contents=contents)

    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    initial_delay_s = float(os.getenv("GEMINI_RETRY_INITIAL_DELAY_S", "1.0"))
    max_delay_s = float(os.getenv("GEMINI_RETRY_MAX_DELAY_S", "10.0"))

    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"🔁 Gemini retry {attempt}/{max_retries} (model={model})")
            return _call(model)
        except Exception as e:
            last_exc = e
            if not _is_transient_gemini_error(e) or attempt >= max_retries:
                break

            # exponential backoff with jitter
            delay = min(max_delay_s, initial_delay_s * (2 ** (attempt - 0)))
            delay = delay * (0.7 + random.random() * 0.6)  # jitter 0.7x..1.3x
            print(f"⏳ Gemini transient error, backing off {delay:.1f}s: {e}")
            time.sleep(delay)

    # Optional fallback model (useful for gemini-3-pro-image-preview 503s)
    fallback_model = get_gemini_fallback_image_model()
    if fallback_model and fallback_model != model:
        print(f"↩️ Falling back to Gemini model={fallback_model} after error on model={model}: {last_exc}")
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔁 Gemini retry {attempt}/{max_retries} (model={fallback_model})")
                return _call(fallback_model)
            except Exception as e:
                last_exc = e
                if not _is_transient_gemini_error(e) or attempt >= max_retries:
                    break
                delay = min(max_delay_s, initial_delay_s * (2 ** (attempt - 0)))
                delay = delay * (0.7 + random.random() * 0.6)
                print(f"⏳ Gemini transient error (fallback), backing off {delay:.1f}s: {e}")
                time.sleep(delay)

    # Give the original exception context back to the caller.
    if last_exc:
        raise last_exc
    raise RuntimeError("Gemini request failed with unknown error")


def generate_unique_filename(original_name: str, prefix: str = "output") -> str:
    base, ext = os.path.splitext(original_name)
    if not ext:
        ext = ".png"
    return f"{prefix}_{uuid.uuid4().hex}{ext}"


def download_image_from_url(url: str, dest_dir: str) -> Optional[str]:
    try:
        os.makedirs(dest_dir, exist_ok=True)
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        ext = ".jpg"
        content_type = resp.headers.get("Content-Type", "")
        if "png" in content_type:
            ext = ".png"
        filename = f"background_{uuid.uuid4().hex}{ext}"
        path = os.path.join(dest_dir, filename)
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except Exception as e:
        print(f"Failed to download background: {e}")
        return None


def cleanup_file(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def apply_exif_orientation(img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation data to image if present.
    This ensures images are displayed in the correct orientation.
    Uses PIL's ImageOps.exif_transpose for reliable EXIF handling.
    """
    try:
        # Use PIL's built-in method to handle EXIF orientation
        # This automatically applies the correct rotation/flip based on EXIF data
        img = ImageOps.exif_transpose(img)
    except (AttributeError, KeyError, TypeError, Exception) as e:
        # No EXIF data or error reading it, return image as-is
        # This is expected for images without EXIF orientation data
        pass
    
    return img


def get_image_info(path: str) -> Dict[str, Any]:
    try:
        with Image.open(path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
            }
    except Exception as e:
        print(f"get_image_info error: {e}")
        return {}


def generate_character_with_identity(
    selfie_path: str,
    character_prompt: str,
    output_path: str,
    white_background: bool = False,
    position: str = "center",
    scale: float = 1.0,
    background_dimensions: Optional[dict] = None,
    canvas_size: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[bool, str]:
    """
    Generate a character from a selfie only. If white_background=True, we ask
    the model for a simple light background; we do NOT remove it locally.
    
    Detects if the prompt is a style conversion (like pencil sketch) vs character transformation,
    and handles them differently to preserve original composition.
    """
    try:
        client = get_gemini_client()
        if not os.path.exists(selfie_path):
            return False, f"Selfie not found: {selfie_path}"

        # Load image and apply EXIF orientation to preserve correct orientation
        selfie_image = Image.open(selfie_path)
        selfie_image = apply_exif_orientation(selfie_image)

        # Detect if this is a style conversion task (pencil sketch, painting, etc.)
        # vs a character transformation task
        # Only match actual artistic medium conversions, not general style descriptions like "retro 90s style"
        artistic_medium_keywords = [
            "pencil sketch", "sketch", "drawing", "hand-drawn", "graphite",
            "painting", "watercolor", "oil painting", "charcoal", "ink drawing"
        ]
        prompt_lower = character_prompt.lower()
        is_style_conversion = any(keyword in prompt_lower for keyword in artistic_medium_keywords)
        
        # Detect if this is specifically a pencil sketch (to handle background differently)
        pencil_sketch_keywords = ["pencil sketch", "graphite", "hand-drawn graphite"]
        is_pencil_sketch = any(keyword in prompt_lower for keyword in pencil_sketch_keywords)

        # Detect if this is a monochrome/grayscale request
        monochrome_keywords = [
            "monochrome", "grayscale", "black and white", "black-and-white", "b&w",
            "pencil sketch", "graphite", "charcoal", "ink drawing", "strict monochrome",
            "no color", "ignore color", "ignore all colors", "purely black and white"
        ]
        is_monochrome = any(keyword.lower() in character_prompt.lower() for keyword in monochrome_keywords)

        bg_context = ""
        if background_dimensions:
            bg_context = (
                f"\nThe character will later be composited on a background of size "
                f"{background_dimensions.get('width')}x{background_dimensions.get('height')} pixels. "
                f"Target position: {position}, scale: {scale}x."
            )

        canvas_context = ""
        if canvas_size:
            if is_style_conversion:
                canvas_context = (
                    f"\nTarget print size: {canvas_size} at {dpi} DPI. "
                    f"Maintain the same aspect ratio and composition as the input image."
                )
            else:
                canvas_context = (
                    f"\nTarget print size: {canvas_size} at {dpi} DPI. "
                    f"Ensure the full body fits comfortably inside this canvas."
                )

        if white_background:
            bg_req = (
                "Use a plain, uniform light background (white or very light grey) "
                "with no objects or scenery."
            )
        else:
            bg_req = "You may choose an appropriate background; do not crop the subject."

        if is_style_conversion:
            # Build monochrome instructions if needed - place at the very beginning for maximum impact
            monochrome_prefix = ""
            monochrome_suffix = ""
            if is_monochrome:
                monochrome_prefix = """ABSOLUTE COLOR REQUIREMENT - THIS OVERRIDES EVERYTHING ELSE:
The output image MUST be 100% pure monochrome/grayscale. NO colors are allowed - not even subtle tints.
- Use ONLY neutral gray tones: white, light gray, mid gray, dark gray, and black
- IGNORE and discard ALL color information from the input image
- Convert ALL colors (including blue, red, green, yellow, cyan, magenta) to gray tones based ONLY on their brightness/luminance
- Do NOT add any color casts, color temperature, warm tones, cool tones, sepia, or any color tints
- If you see blue in the input, render it as a neutral gray - do NOT preserve any blue hue
- Every pixel must have equal RGB values (R=G=B) - this ensures true grayscale
- Any color in the output is a critical error

"""
                monochrome_suffix = """

FINAL COLOR CHECK:
Before outputting, verify that the image is pure grayscale with no color tints. If any color is present, convert it to neutral gray.
"""
            
            # Get image dimensions for orientation preservation
            img_width, img_height = selfie_image.size
            is_portrait = img_height > img_width
            
            # For style conversions, preserve exact composition and framing
            # Process user's prompt to handle "Negative prompt:" section (common in Stable Diffusion format)
            # Convert it to explicit "DO NOT" instructions for Gemini
            processed_prompt = character_prompt
            negative_prompt_section = ""
            
            if "Negative prompt:" in character_prompt or "negative prompt:" in character_prompt:
                # Split the prompt and negative prompt
                parts = character_prompt.split("Negative prompt:", 1)
                if len(parts) == 1:
                    parts = character_prompt.split("negative prompt:", 1)
                
                if len(parts) == 2:
                    processed_prompt = parts[0].strip()
                    negative_items = parts[1].strip().split(",")
                    negative_items = [item.strip() for item in negative_items if item.strip()]
                    
                    # For pencil sketch, remove background-related items from negative prompt
                    # For other style conversions, keep all negative prompt items (including background removal)
                    if is_pencil_sketch:
                        background_related = ['background', 'scenery', 'room', 'landscape', 'horizon']
                        negative_items = [item for item in negative_items 
                                         if not any(bg_term in item.lower() for bg_term in background_related)]
                    
                    if negative_items:
                        negative_list = '\n'.join([f'- {item}' for item in negative_items])
                        negative_prompt_section = f"""
STRICT PROHIBITIONS - DO NOT INCLUDE:
{negative_list}
"""
            
            # For pencil sketch, remove background removal instructions from the prompt
            if is_pencil_sketch:
                # Remove background removal phrases
                background_removal_patterns = [
                    (r'[^.]*?completely removing the background[^.]*?\.', ''),
                    (r'[^.]*?removing the background[^.]*?\.', ''),
                    (r'[^.]*?remove the background[^.]*?\.', ''),
                    (r'[^.]*?completely removing the background so[^.]*?\.', ''),
                    (r'[^.]*?removing the background so[^.]*?\.', ''),
                    (r'[^.]*?subjects appear alone on a pure white canvas[^.]*?\.', ''),
                    (r'[^.]*?appear alone on a pure white canvas[^.]*?\.', ''),
                    (r'[^.]*?on a pure white canvas[^.]*?\.', ''),
                    (r'[^.]*?pure white canvas[^.]*?\.', ''),
                    (r'[^.]*?white canvas[^.]*?\.', ''),
                    (r'completely removing the background[^,.]*?[,.]', ''),
                    (r'removing the background[^,.]*?[,.]', ''),
                    (r'remove the background[^,.]*?[,.]', ''),
                    (r'pure white canvas[^,.]*?[,.]', ''),
                    (r'white canvas[^,.]*?[,.]', ''),
                ]
                
                for pattern, replacement in background_removal_patterns:
                    processed_prompt = re.sub(pattern, replacement, processed_prompt, flags=re.IGNORECASE)
                
                # Remove standalone phrases
                for phrase in [r'\bno background\b', r'\bwithout background\b', r'\bwhite background\b', r'\bno shadows\b']:
                    processed_prompt = re.sub(phrase, '', processed_prompt, flags=re.IGNORECASE)
                
                # Clean up formatting
                processed_prompt = re.sub(r'\s+', ' ', processed_prompt)
                processed_prompt = re.sub(r'\s*\.\s*\.+', '.', processed_prompt)
                processed_prompt = re.sub(r'\s*,\s*,+', ',', processed_prompt)
                processed_prompt = re.sub(r'^\s*[.,;]\s*', '', processed_prompt)
                processed_prompt = processed_prompt.strip()
            
            # Add background preservation instruction for pencil sketch, or background section for others
            background_preservation_instruction = ""
            background_section = ""
            
            if is_pencil_sketch:
                background_preservation_instruction = """
BACKGROUND PRESERVATION (CRITICAL - HIGHEST PRIORITY):
- You MUST preserve the original background from the reference image exactly as it appears
- Do NOT remove, replace, modify, or change the background in any way
- Keep ALL background elements, colors, textures, and details from the original image
- The background should appear in the same style (pencil sketch) but remain in its original location and composition
- If the original has a background, it must be visible in the output
- Do NOT create a white canvas, transparent background, or remove any background elements

"""
            else:
                # Only add background section if user hasn't specified background removal
                has_background_removal = any(phrase in processed_prompt.lower() for phrase in [
                    'remove background', 'removing background', 'no background', 'without background',
                    'white background', 'transparent background', 'pure white', 'white canvas'
                ])
                
                if not has_background_removal:
                    background_section = f"""
BACKGROUND:
{bg_req}
"""
            
            full_prompt = f"""{monochrome_prefix}Convert the reference image to the requested style while preserving the EXACT composition, pose, framing, and subject matter.
{background_preservation_instruction}
CRITICAL REQUIREMENTS:
- Preserve the EXACT same framing, crop, and composition as the input image
- Keep the same pose, position, and body parts visible (if it's a half picture, keep it as a half picture)
- Maintain the same aspect ratio ({img_width}x{img_height}) and orientation ({"portrait" if is_portrait else "landscape"})
- Do NOT rotate, flip, or change the orientation of the image
- Do NOT add or remove body parts (e.g., if only upper body is shown, do NOT make it full body)
- Do NOT change the subject's position or pose
- Apply the style transformation to the EXISTING image composition
{f"- Preserve the original background exactly as it appears in the reference image" if is_pencil_sketch else ""}

STYLE CONVERSION:
{processed_prompt}
{negative_prompt_section}
{background_section}
{bg_context}
{canvas_context}

OUTPUT:
Return the converted image with the exact same composition and framing as the input, only with the style applied.{monochrome_suffix}
"""
        else:
            # For character transformations, use the original prompt structure
            full_prompt = f"""Transform this person into a character while preserving their identity.

CHARACTER TRANSFORMATION:
{character_prompt}

BACKGROUND:
{bg_req}

FULL-BODY REQUIREMENT:
Show the character from head to feet, entirely inside the frame.

{bg_context}
{canvas_context}
"""
        response = _generate_content_image(
            client=client,
            model=get_gemini_image_model(),
            contents=[full_prompt, selfie_image],
        )

        img = _extract_final_image_from_response(response)
        if img is not None:
            # Ensure img is a valid PIL Image with size attribute
            if not hasattr(img, 'size') or not isinstance(img, Image.Image):
                return False, "Invalid image object returned from Gemini (missing size attribute)"
            
            # Preserve original orientation - check if output orientation matches input
            original_width, original_height = selfie_image.size
            output_width, output_height = img.size
            original_is_portrait = original_height > original_width
            output_is_portrait = output_height > output_width
            
            # If orientation doesn't match, rotate to match original
            if original_is_portrait != output_is_portrait:
                img = img.rotate(-90, expand=True)
            
            # Post-process: If monochrome was requested, ensure output is pure grayscale
            if is_monochrome and img.mode != 'L':
                img = img.convert('L').convert('RGB')
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            return True, "Character generated successfully"

        return False, "No image generated by Gemini"
    except Exception as e:
        print(f"generate_character_with_identity error: {e}")
        return False, str(e)


def generate_character_composited_with_background(
    selfie_path: str,
    background_path: str,
    character_prompt: str,
    output_path: str,
    position: str = "bottom",
    scale: float = 1.0,
    canvas_size: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[bool, str]:
    """
    One-shot: Gemini creates a character AND composites it onto the given background.
    """
    try:
        client = get_gemini_client()
        if not os.path.exists(selfie_path):
            return False, f"Selfie not found: {selfie_path}"
        if not os.path.exists(background_path):
            return False, f"Background not found: {background_path}"

        # Load images and apply EXIF orientation to preserve correct orientation
        selfie_image = Image.open(selfie_path)
        selfie_image = apply_exif_orientation(selfie_image)
        background_image = Image.open(background_path)
        background_image = apply_exif_orientation(background_image)
        bg_w, bg_h = background_image.size

        canvas_context = ""
        if canvas_size:
            canvas_context = (
                f"\nTarget print size: {canvas_size} at {dpi} DPI. Keep the same aspect ratio as "
                f"the provided background ({bg_w}x{bg_h})."
            )

        full_prompt = f"""TASK:
Create a full-body cartoon/caricature of this person and composite them onto this exact background image.

BACKGROUND USAGE (MUST FOLLOW EXACTLY):
1. Use the provided background image AS-IS (no crop, no stretch, no extra elements).
2. Keep the same aspect ratio as the background ({bg_w}x{bg_h}).

CHARACTER POSITIONING:
3. Place the character standing at the {position.upper()} of the background, centered horizontally.
4. The character must be full-body (head to feet), entirely inside the frame.
5. Character height ≈ {int(scale * 75)}% of the total image height.

STYLE & IDENTITY:
6. Preserve the person's identity (face, hair, skin tone).
7. Use clean outlines and vibrant colors suitable for printing.

RESTRICTIONS:
8. Do NOT add text, logos, borders or extra objects.

CHARACTER DESCRIPTION:
{character_prompt}

{canvas_context}

OUTPUT:
Return a SINGLE final composited image ready for printing.
"""

        response = _generate_content_image(
            client=client,
            model=get_gemini_image_model(),
            contents=[full_prompt, selfie_image, background_image],
        )

        img = _extract_final_image_from_response(response)
        if img is not None:
            # Ensure img is a valid PIL Image with size attribute
            if not hasattr(img, 'size') or not isinstance(img, Image.Image):
                return False, "Invalid image object returned from Gemini (missing size attribute)"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            return True, "Gemini composited character successfully"

        return False, "No composited image generated by Gemini"
    except Exception as e:
        print(f"generate_character_composited_with_background error: {e}")
        return False, str(e)

