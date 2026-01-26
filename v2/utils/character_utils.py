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
    character_only: bool = False,
) -> Tuple[bool, str]:
    """
    Generate a character from a selfie only. If white_background=True, we ask
    the model for a simple light background; we do NOT remove it locally.
    If character_only=True, explicitly request character without any background.
    
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
            "painting", "watercolor", "oil painting", "charcoal", "ink drawing",
            "pop art", "warhol", "acrylic", "silkscreen", "portrait"
        ]
        prompt_lower = character_prompt.lower()
        is_style_conversion = any(keyword in prompt_lower for keyword in artistic_medium_keywords)
        
        # Detect if this is specifically a pencil sketch (to handle background differently)
        pencil_sketch_keywords = ["pencil sketch", "graphite", "hand-drawn graphite"]
        is_pencil_sketch = any(keyword in prompt_lower for keyword in pencil_sketch_keywords)
        
        # Detect if this is a Warhol/pop art style that requires grid layout (skip composition preservation)
        warhol_keywords = ["warhol", "pop art", "four-panel", "2×2", "2x2", "grid", "four panel"]
        is_warhol_style = any(keyword in prompt_lower for keyword in warhol_keywords)

        # Detect if this is a monochrome/grayscale request
        monochrome_keywords = [
            "monochrome", "grayscale", "black and white", "black-and-white", "b&w",
            "pencil sketch", "graphite", "charcoal", "ink drawing", "strict monochrome",
            "no color", "ignore color", "ignore all colors", "purely black and white"
        ]
        is_monochrome = any(keyword.lower() in character_prompt.lower() for keyword in monochrome_keywords)

        # Initialize variables to avoid UnboundLocalError in f-strings
        bg_context = ""
        canvas_context = ""
        full_prompt = None
        monochrome_prefix = ""
        monochrome_suffix = ""
        negative_prompt_section = ""
        background_preservation_instruction = ""
        background_section = ""
        prompt_to_use = character_prompt
        character_only_override = ""
        prohibitions_section = ""
        background_preservation_text = ""
        background_removal_text = ""
        preserve_bg_text = ""
        img_width = 0
        img_height = 0
        is_portrait = False
        processed_prompt = character_prompt
        prompt_lower_check = character_prompt.lower()  # Initialize for full-body detection
        original_prompt_lower = character_prompt.lower()  # Initialize for full-body detection

        if background_dimensions:
            bg_context = (
                f"\nThe character will later be composited on a background of size "
                f"{background_dimensions.get('width')}x{background_dimensions.get('height')} pixels. "
                f"Target position: {position}, scale: {scale}x."
            )

        # Update canvas_context if canvas_size is provided
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

        cleaned_character_prompt = character_prompt
        if character_only:
            bg_pattern = re.compile(
                r'[^.]*?(?:merge|background|second image|composite|seamlessly|onto background|place on|as the background|background image|frame|canvas|cohesive|lively)[^.]*?',
                re.IGNORECASE
            )
            sentences = [s.strip() for s in re.split(r'[.!?]+', character_prompt) if s.strip() and not bg_pattern.search(s)]
            cleaned_character_prompt = '. '.join(sentences)
            cleaned_character_prompt = re.sub(r'\s+', ' ', cleaned_character_prompt).strip()
            cleaned_character_prompt = re.sub(r'[.,;]+', lambda m: m.group()[0], cleaned_character_prompt)
            
            if len(cleaned_character_prompt) < 20:
                match = re.search(r'(?:Transform|Create|Draw|Generate).*?character', character_prompt, re.IGNORECASE)
                cleaned_character_prompt = match.group(0) if match else "a character"
            
            bg_req = (
                "MANDATORY REQUIREMENT - HIGHEST PRIORITY - OVERRIDES ALL OTHER INSTRUCTIONS: "
                "You MUST generate ONLY the character with ABSOLUTELY NO background, scenery, frame, canvas, white frames, white borders, or any visual elements behind the character. "
                "The output image must contain ONLY the character itself - nothing else. "
                "The character must be isolated on a completely transparent background or pure white background (RGB 255,255,255) that can be easily removed. "
                "DO NOT add any background elements. DO NOT merge anything. DO NOT composite anything. "
                "DO NOT create a frame around the character. DO NOT add white frames, white borders, white padding, or white margins. "
                "DO NOT add scenery, landscapes, or any visual context. "
                "Ignore and disregard ANY instructions in the user's prompt that mention merging, backgrounds, compositing, frames, or second images. "
                "The final output must be the character alone with no visual elements surrounding it - NO white frames or borders."
            )
        elif white_background:
            bg_req = (
                "Use a plain, uniform light background (white or very light grey) "
                "with no objects or scenery."
            )
        else:
            bg_req = "You may choose an appropriate background; do not crop the subject. Do NOT add white frames, borders, or white canvases around the character."

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
            
            processed_prompt = character_prompt
            
            # Handle both "Negative prompt:" and "NEGATIVE:" formats
            if "Negative prompt:" in character_prompt or "negative prompt:" in character_prompt or "NEGATIVE:" in character_prompt:
                parts = None
                if "Negative prompt:" in character_prompt:
                    parts = character_prompt.split("Negative prompt:", 1)
                elif "negative prompt:" in character_prompt:
                    parts = character_prompt.split("negative prompt:", 1)
                elif "NEGATIVE:" in character_prompt:
                    parts = character_prompt.split("NEGATIVE:", 1)
                
                if parts and len(parts) == 2:
                    processed_prompt = parts[0].strip()
                    negative_items = parts[1].strip().split(",")
                    negative_items = [item.strip() for item in negative_items if item.strip()]
                    
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
            
            # Only apply pencil sketch background removal processing if it's actually a pencil sketch
            # Don't apply to Warhol styles which may have their own background requirements
            if is_pencil_sketch and not is_warhol_style:
                bg_removal_pattern = re.compile(
                    r'[^.]*?(?:completely removing|removing|remove)\s+the\s+background[^.]*?\.|'
                    r'[^.]*?(?:subjects|appear)\s+alone\s+on\s+a\s+pure\s+white\s+canvas[^.]*?\.|'
                    r'[^.]*?pure\s+white\s+canvas[^.]*?\.|'
                    r'[^.]*?white\s+canvas[^.]*?\.|'
                    r'\b(?:no|without)\s+background\b|\bwhite\s+background\b|\bno\s+shadows\b',
                    re.IGNORECASE
                )
                processed_prompt = bg_removal_pattern.sub('', processed_prompt)
                processed_prompt = re.sub(r'\s+', ' ', processed_prompt).strip()
            
            if is_pencil_sketch and not is_warhol_style:
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
                background_section = ""
                if not character_only:
                    has_background_removal = any(phrase in processed_prompt.lower() for phrase in [
                        'remove background', 'removing background', 'no background', 'without background',
                        'white background', 'transparent background', 'pure white', 'white canvas'
                    ])
                    
                    if not has_background_removal:
                        background_section = f"""
BACKGROUND:
{bg_req}
"""
            
            prompt_to_use = cleaned_character_prompt if character_only else processed_prompt
            
            if character_only and not is_pencil_sketch:
                character_only_override = """⚠️ CRITICAL INSTRUCTION - HIGHEST PRIORITY - OVERRIDES ALL OTHER INSTRUCTIONS ⚠️

YOU MUST GENERATE ONLY THE CHARACTER WITH NO BACKGROUND WHATSOEVER.

REQUIREMENTS:
- Output ONLY the character - no background, no scenery, no frame, no canvas
- Character must be on transparent or pure white background (RGB 255,255,255)
- DO NOT merge, composite, or place the character on any background
- DO NOT add any visual elements behind or around the character
- IGNORE any instructions in the user's prompt about merging, backgrounds, compositing, or second images
- The character must be completely isolated with nothing surrounding it

THIS IS THE MOST IMPORTANT REQUIREMENT - ALL OTHER INSTRUCTIONS ARE SECONDARY.

"""
            
            if character_only and not is_pencil_sketch:
                prohibitions_section = """
STRICT PROHIBITIONS (when character_only=True):
- DO NOT add any background, scenery, or visual context
- DO NOT merge or composite the character onto anything
- DO NOT create frames, borders, white frames, white borders, or white canvases
- DO NOT add any white space, white padding, or white margins around the character
- DO NOT add any elements behind or around the character
- The output must be ONLY the character on transparent/white background with NO white frames or borders
"""
            
            background_preservation_text = "" if character_only or is_warhol_style else background_preservation_instruction
            background_removal_text = "- CRITICAL: Remove and ignore the background from the input image. Output ONLY the character/subject with no background." if character_only and not is_pencil_sketch and not is_warhol_style else ""
            preserve_bg_text = "- Preserve the original background exactly as it appears in the reference image" if is_pencil_sketch and not character_only and not is_warhol_style else ""
            
            # Check if user explicitly requests full-body in style conversion
            has_explicit_full_body_style = re.search(r'\bfull[\s-]?body\b', prompt_lower_check) or re.search(r'\bfull[\s-]?body\b', original_prompt_lower)
            
            if is_warhol_style:
                composition_instructions = """COMPOSITION:
- Follow the user's specific composition requirements (grid layouts, panels, etc.)
- Apply the requested layout and framing as specified in the prompt
"""
            else:
                if has_explicit_full_body_style:
                    composition_instructions = f"""CRITICAL REQUIREMENTS:
- The user's prompt explicitly requests "full-body" - you MUST show the complete character from head to feet
- Preserve the EXACT same framing, crop, and composition as the input image
- Keep the same pose, position, and body parts visible
- Follow any pose instructions in the user's prompt (e.g., "standing in playful pose")
- Maintain the same aspect ratio ({img_width}x{img_height}) and orientation ({"portrait" if is_portrait else "landscape"})
- Do NOT rotate, flip, or change the orientation of the image
- Do NOT cut off or hide body parts (hands, feet, arms, legs) - show the complete full body
- Do NOT change the subject's position or pose unless the prompt explicitly requests a different pose
- Show complete body parts (hands, feet, arms, legs) - the character must be full-body as requested
- Apply the style transformation to the EXISTING image composition
"""
                else:
                    composition_instructions = f"""CRITICAL REQUIREMENTS:
- Preserve the EXACT same framing, crop, and composition as the input image
- Keep the same pose, position, and body parts visible (if it's a half picture, keep it as a half picture)
- Follow any pose instructions in the user's prompt (e.g., "standing in playful pose")
- Maintain the same aspect ratio ({img_width}x{img_height}) and orientation ({"portrait" if is_portrait else "landscape"})
- Do NOT rotate, flip, or change the orientation of the image
- Do NOT add or remove body parts (e.g., if only upper body is shown, do NOT make it full body)
- Do NOT change the subject's position or pose unless the prompt explicitly requests a different pose
- Show complete body parts (hands, feet, arms, legs) unless the prompt explicitly requests otherwise
- Apply the style transformation to the EXISTING image composition
"""
            
            # For Warhol styles, skip color preservation (user specifies exact color palettes)
            color_preservation_section = ""
            if not is_warhol_style:
                color_preservation_section = f"""
COLOR PRESERVATION (CRITICAL):
- Preserve the EXACT colors of all clothing, outfits, and garments from the input image
- Keep the same color scheme, color palette, and color combinations as the original
- Do NOT change clothing colors, fabric colors, or accessory colors
- Maintain the exact same hues, shades, and tones of all clothing items
- If the original has a blue shirt, keep it blue (or gray if monochrome is requested)
- If the original has a red dress, keep it red (or gray if monochrome is requested)
- Preserve all color details including patterns, stripes, prints, and color accents
{preserve_bg_text}
{background_removal_text}
"""
            
            full_prompt = f"""{monochrome_prefix}{character_only_override}Convert the reference image to the requested style{"" if is_warhol_style else " while preserving the EXACT composition, pose, framing, and subject matter"}.
{background_preservation_text}
{composition_instructions}
{color_preservation_section}

STYLE CONVERSION:
{prompt_to_use}
{negative_prompt_section}
{background_section}
{bg_context}
{canvas_context}
{prohibitions_section}

OUTPUT:
{"Return the converted image following the user's composition requirements (grid, panels, etc.) with the style applied." if is_warhol_style else "Return the converted image with the exact same composition and framing as the input, only with the style applied."}{monochrome_suffix}
"""
        else:
            prompt_to_use = cleaned_character_prompt if character_only else character_prompt
            
            # Update prompt_lower_check with the processed prompt
            prompt_lower_check = prompt_to_use.lower()
            
            # Check if user explicitly requests NOT to make it full-body
            # Only disable full-body if user EXPLICITLY says not to (not just mentions keywords)
            
            # Only match explicit negative phrases (removed generic keywords that might be in style descriptions)
            explicit_no_full_body_patterns = [
                r"not\s+full\s+body", r"no\s+full\s+body", r"do\s+not\s+full\s+body", r"avoid\s+full\s+body",
                r"not\s+whole\s+body", r"no\s+whole\s+body", r"do\s+not\s+whole\s+body", r"avoid\s+whole\s+body",
                r"not\s+head\s+to\s+toe", r"no\s+head\s+to\s+toe", r"do\s+not\s+head\s+to\s+toe",
                r"no\s+feet", r"without\s+feet", r"exclude\s+feet", r"no\s+legs", r"without\s+legs"
            ]
            
            # Check for "whole body" only in negative section (NEGATIVE: whole body)
            has_negative_whole_body = False
            if "negative:" in original_prompt_lower or "NEGATIVE:" in original_prompt_lower:
                negative_match = re.search(r'(?:negative|NEGATIVE):\s*[^.]*whole\s+body', original_prompt_lower)
                if negative_match:
                    has_negative_whole_body = True
            
            # Check if user explicitly requests full-body in prompt
            has_explicit_full_body = re.search(r'\bfull[\s-]?body\b', prompt_lower_check) or re.search(r'\bfull[\s-]?body\b', original_prompt_lower)
            
            # Default to full-body, only disable if explicit pattern found
            user_wants_full_body = True
            for pattern in explicit_no_full_body_patterns:
                if re.search(pattern, prompt_lower_check) or re.search(pattern, original_prompt_lower):
                    user_wants_full_body = False
                    break
            
            if has_negative_whole_body:
                user_wants_full_body = False
            
            # If user explicitly says "full-body" in prompt, force it to True
            if has_explicit_full_body:
                user_wants_full_body = True
            
            if character_only:
                character_only_override = """⚠️ CRITICAL INSTRUCTION - HIGHEST PRIORITY - OVERRIDES ALL OTHER INSTRUCTIONS ⚠️

YOU MUST GENERATE ONLY THE CHARACTER WITH NO BACKGROUND WHATSOEVER.

REQUIREMENTS:
- Output ONLY the character - no background, no scenery, no frame, no canvas, no white borders
- Character must be on transparent or pure white background (RGB 255,255,255) that can be easily removed
- DO NOT add white frames, white borders, or white canvases around the character
- DO NOT merge, composite, or place the character on any background
- DO NOT add any visual elements behind or around the character
- IGNORE any instructions in the user's prompt about merging, backgrounds, compositing, or second images
- The character must be completely isolated with nothing surrounding it - no frames, no borders, no white space around edges

THIS IS THE MOST IMPORTANT REQUIREMENT - ALL OTHER INSTRUCTIONS ARE SECONDARY.

"""
            
            if character_only:
                prohibitions_section = """
STRICT PROHIBITIONS:
- DO NOT add any background, scenery, or visual context
- DO NOT merge or composite the character onto anything
- DO NOT create frames, borders, white frames, white borders, or white canvases
- DO NOT add any white space, white padding, or white margins around the character
- DO NOT add any elements behind or around the character
- The output must be ONLY the character on transparent/white background with NO white frames or borders
"""
            
            # Build full-body section conditionally (can't use triple quotes in f-string conditional)
            if user_wants_full_body:
                if has_explicit_full_body:
                    full_body_section = """
FRAMING REQUIREMENT (CRITICAL - USER EXPLICITLY REQUESTED):
- The user's prompt explicitly requests "full-body" - you MUST show the complete character from head to feet
- Show the character from head to feet, entirely inside the frame
- Ensure the full body is visible - do NOT cut off any body parts
- Do NOT crop or hide hands, feet, arms, or legs
- The character must be complete and fully visible
"""
                else:
                    full_body_section = """
FRAMING REQUIREMENT:
- Show the character from head to feet, entirely inside the frame.
- Ensure the full body is visible.
"""
            else:
                full_body_section = ""
            
            full_prompt = f"""{character_only_override}Transform this person into a character while preserving their identity.

CHARACTER TRANSFORMATION:
{prompt_to_use}

POSE & BODY PRESERVATION (CRITICAL):
- Follow the user's specific pose instructions from the prompt (e.g., "standing in playful pose", "sitting", etc.)
- Do NOT change the pose described in the prompt
- Show the complete character with all body parts visible unless the prompt explicitly requests otherwise
- Do NOT cut off or hide body parts (hands, feet, arms, legs) unless explicitly requested
- Maintain natural body proportions and positioning as described in the prompt
- If the prompt says "standing", show the character standing
- If the prompt says "playful pose", show a playful pose, not a different pose

COLOR PRESERVATION (CRITICAL):
- Preserve the EXACT colors of all clothing, outfits, and garments from the input image
- Keep the same color scheme, color palette, and color combinations as the original
- Do NOT change clothing colors, fabric colors, or accessory colors
- Maintain the exact same hues, shades, and tones of all clothing items
- Preserve white colors in the character (white clothing, white eyes, etc.) - do NOT make them transparent
- If the original has a blue shirt, keep it blue in the character
- If the original has a red dress, keep it red in the character
- Preserve all color details including patterns, stripes, prints, and color accents
- Only transform the style (cartoon/caricature), NOT the colors

BACKGROUND:
{bg_req}
{full_body_section}

{prohibitions_section}

{bg_context}
{canvas_context}
"""
        
        if full_prompt is None:
            return False, "Internal error: full_prompt not defined"
        
        response = _generate_content_image(
            client=client,
            model=get_gemini_image_model(),
            contents=[full_prompt, selfie_image],
        )

        img = _extract_final_image_from_response(response)
        if img is not None:
            if not hasattr(img, 'size') or not isinstance(img, Image.Image):
                return False, "Invalid image object returned from Gemini (missing size attribute)"
            
            original_width, original_height = selfie_image.size
            output_width, output_height = img.size
            original_is_portrait = original_height > original_width
            output_is_portrait = output_height > output_width
            
            if original_is_portrait != output_is_portrait:
                img = img.rotate(-90, expand=True)
            
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
    use_gemini_compositing: bool = True,
) -> Tuple[bool, str]:
    """
    Generate character and composite onto background.
    
    If use_gemini_compositing=True (default): Gemini creates character AND composites it onto background.
    If use_gemini_compositing=False: Generate character first, then composite background locally using PIL.
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

        if use_gemini_compositing:
            if canvas_size:
                canvas_context = (
                    f"\nTarget print size: {canvas_size} at {dpi} DPI. Keep the same aspect ratio as "
                    f"the provided background ({bg_w}x{bg_h})."
                )

            # Check if user explicitly requests NOT to make it full-body
            # Only disable full-body if user EXPLICITLY says not to (not just mentions keywords)
            prompt_lower_composite = character_prompt.lower()
            
            # Only match explicit negative phrases (removed generic keywords that might be in style descriptions)
            explicit_no_full_body_patterns_composite = [
                r"not\s+full\s+body", r"no\s+full\s+body", r"do\s+not\s+full\s+body", r"avoid\s+full\s+body",
                r"not\s+whole\s+body", r"no\s+whole\s+body", r"do\s+not\s+whole\s+body", r"avoid\s+whole\s+body",
                r"not\s+head\s+to\s+toe", r"no\s+head\s+to\s+toe", r"do\s+not\s+head\s+to\s+toe",
                r"no\s+feet", r"without\s+feet", r"exclude\s+feet", r"no\s+legs", r"without\s+legs"
            ]
            
            # Check for "whole body" only in negative section (NEGATIVE: whole body)
            has_negative_whole_body_composite = False
            if "negative:" in prompt_lower_composite or "NEGATIVE:" in prompt_lower_composite:
                negative_match = re.search(r'(?:negative|NEGATIVE):\s*[^.]*whole\s+body', prompt_lower_composite)
                if negative_match:
                    has_negative_whole_body_composite = True
            
            # Check if user explicitly requests full-body in prompt
            has_explicit_full_body_composite = re.search(r'\bfull[\s-]?body\b', prompt_lower_composite)
            
            # Default to full-body, only disable if explicit pattern found
            user_wants_full_body_composite = True
            for pattern in explicit_no_full_body_patterns_composite:
                if re.search(pattern, prompt_lower_composite):
                    user_wants_full_body_composite = False
                    break
            
            if has_negative_whole_body_composite:
                user_wants_full_body_composite = False
            
            # If user explicitly says "full-body" in prompt, force it to True
            if has_explicit_full_body_composite:
                user_wants_full_body_composite = True
            
            full_body_instruction = ""
            if user_wants_full_body_composite:
                if has_explicit_full_body_composite:
                    full_body_instruction = f"""CHARACTER POSITIONING (CRITICAL - USER EXPLICITLY REQUESTED):
3. The user's prompt explicitly requests "full-body" - you MUST show the complete character from head to feet
4. Place the character standing at the {position.upper()} of the background, centered horizontally.
5. The character must be full-body (head to feet), entirely inside the frame - do NOT cut off any body parts
6. Do NOT crop or hide hands, feet, arms, or legs
7. Character height should be approximately 95% of the total image height to cover the full height while maintaining proper proportions.
"""
                else:
                    full_body_instruction = f"""CHARACTER POSITIONING:
3. Place the character standing at the {position.upper()} of the background, centered horizontally.
4. The character must be full-body (head to feet), entirely inside the frame.
5. Character height should be approximately 95% of the total image height to cover the full height while maintaining proper proportions.
"""
            else:
                full_body_instruction = f"""CHARACTER POSITIONING:
3. Place the character at the {position.upper()} of the background, centered horizontally.
4. Follow the user's specific framing requirements from the prompt (do NOT make it full-body if the user specified otherwise).
5. Character height should be appropriate to the framing requested in the prompt.
"""

            full_prompt = f"""TASK:
Create a {"full-body " if user_wants_full_body_composite else ""}cartoon/caricature of this person and composite them onto this exact background image.

BACKGROUND USAGE (MUST FOLLOW EXACTLY):
1. Use the provided background image AS-IS (no crop, no stretch, no extra elements).
2. Keep the same aspect ratio as the background ({bg_w}x{bg_h}).
3. Do NOT add white frames, borders, or white canvases around the character.

{full_body_instruction}

POSE & BODY PRESERVATION (CRITICAL):
- Follow the user's specific pose instructions from the prompt (e.g., "standing in playful pose", "sitting", etc.)
- Do NOT change the pose described in the prompt
- Show the complete character with all body parts visible unless the prompt explicitly requests otherwise
- Do NOT cut off or hide body parts (hands, feet, arms, legs) unless explicitly requested
- Maintain natural body proportions and positioning as described in the prompt
- If the prompt says "standing", show the character standing
- If the prompt says "playful pose", show a playful pose, not a different pose

STYLE & IDENTITY:
6. Preserve the person's identity (face, hair, skin tone).
7. Use clean outlines and vibrant colors suitable for printing.

COLOR PRESERVATION (CRITICAL):
8. Preserve the EXACT colors of all clothing, outfits, and garments from the input image
9. Keep the same color scheme, color palette, and color combinations as the original
10. Do NOT change clothing colors, fabric colors, or accessory colors
11. Maintain the exact same hues, shades, and tones of all clothing items
12. Preserve white colors in the character (white clothing, white eyes, etc.) - do NOT make them transparent
13. If the original has a blue shirt, keep it blue in the character
14. If the original has a red dress, keep it red in the character
15. Preserve all color details including patterns, stripes, prints, and color accents
16. Only transform the style (cartoon/caricature), NOT the colors

RESTRICTIONS:
17. Do NOT add text, logos, borders or extra objects.
18. Do NOT add white frames, white borders, or white canvases.

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
                if not hasattr(img, 'size') or not isinstance(img, Image.Image):
                    return False, "Invalid image object returned from Gemini (missing size attribute)"
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path)
                return True, "Gemini composited character successfully"

            return False, "No composited image generated by Gemini"
        else:
            temp_char_path = output_path.replace(".png", "_temp_char.png")
            success, message = generate_character_with_identity(
                selfie_path=selfie_path,
                character_prompt=character_prompt,
                output_path=temp_char_path,
                white_background=False,
                position=position,
                scale=scale,
                background_dimensions=None,
                canvas_size=None,
                dpi=dpi,
                character_only=True,
            )
            
            if not success:
                return False, f"Character generation failed: {message}"
            
            try:
                char_image = Image.open(temp_char_path).convert("RGBA")
                
                # Remove ALL white background boxes - smart removal that preserves white in character
                # Strategy: Flood-fill from edges to remove white backgrounds, but preserve white pixels
                # that are surrounded by non-white pixels (indicating they're part of the character)
                w, h = char_image.size
                data = list(char_image.getdata())
                
                # Helper function to check if pixel is white/light
                def is_white_pixel(r, g, b, threshold=235):
                    return r > threshold and g > threshold and b > threshold
                
                # Helper function to check if pixel has non-white neighbors (character detection)
                def has_character_neighbors(x, y, data, w, h):
                    """Check if white pixel is surrounded by non-white pixels (part of character)"""
                    non_white_count = 0
                    total_neighbors = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                total_neighbors += 1
                                idx = ny * w + nx
                                if idx < len(data):
                                    r, g, b, a = data[idx]
                                    if not is_white_pixel(r, g, b, threshold=235) and a > 0:
                                        non_white_count += 1
                    # If most neighbors are non-white, this white pixel is likely part of character
                    return total_neighbors > 0 and non_white_count >= total_neighbors * 0.4
                
                # Create a mask for pixels to remove (white backgrounds)
                to_remove = [False] * len(data)
                visited = set()
                queue = []
                
                # Start flood-fill from ALL edge pixels that are white/light
                for y in range(h):
                    for x in range(w):
                        if x == 0 or x == w-1 or y == 0 or y == h-1:
                            idx = y * w + x
                            if idx < len(data):
                                r, g, b, a = data[idx]
                                # If edge pixel is white/light, start flood-fill from here
                                if is_white_pixel(r, g, b, threshold=235) and a > 0:
                                    queue.append((x, y))
                                    visited.add((x, y))
                
                # Flood-fill from edges to mark all connected white background pixels
                while queue:
                    x, y = queue.pop(0)
                    idx = y * w + x
                    if idx < len(data):
                        r, g, b, a = data[idx]
                        
                        # Check if this white pixel is part of character (surrounded by non-white)
                        is_character_white = has_character_neighbors(x, y, data, w, h)
                        
                        if is_white_pixel(r, g, b, threshold=235) and a > 0:
                            # Only mark for removal if it's NOT part of character
                            if not is_character_white:
                                to_remove[idx] = True
                                
                                # Continue flood-fill to neighbors
                                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                                        visited.add((nx, ny))
                                        nidx = ny * w + nx
                                        if nidx < len(data):
                                            nr, ng, nb, na = data[nidx]
                                            if is_white_pixel(nr, ng, nb, threshold=235) and na > 0:
                                                queue.append((nx, ny))
                
                # Apply removal - make white background pixels transparent
                new_data = []
                for idx, (r, g, b, a) in enumerate(data):
                    if to_remove[idx]:
                        new_data.append((r, g, b, 0))  # Make transparent
                    else:
                        new_data.append((r, g, b, a))  # Keep original
                
                char_image.putdata(new_data)
                
                # Apply canvas size to background if specified
                if canvas_size:
                    size_map = {
                        "8x10": (8, 10),
                        "11x14": (11, 14),
                        "16x20": (16, 20),
                    }
                    if canvas_size in size_map:
                        target_w, target_h = size_map[canvas_size]
                        target_width = int(target_w * dpi)
                        target_height = int(target_h * dpi)
                        bg_aspect = bg_w / bg_h
                        target_aspect = target_width / target_height
                        
                        if bg_aspect > target_aspect:
                            new_h = target_height
                            new_w = int(new_h * bg_aspect)
                        else:
                            new_w = target_width
                            new_h = int(new_w / bg_aspect)
                        
                        background_image = background_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        final_bg = Image.new("RGB", (target_width, target_height), (255, 255, 255))
                        paste_x = (target_width - new_w) // 2
                        paste_y = (target_height - new_h) // 2
                        final_bg.paste(background_image, (paste_x, paste_y))
                        background_image = final_bg
                        bg_w, bg_h = background_image.size
                
                target_char_height = int(bg_h * scale * 0.95)
                
                char_w, char_h = char_image.size
                scale_factor = target_char_height / char_h if char_h > 0 else 1.0
                new_char_w = int(char_w * scale_factor)
                new_char_h = int(char_h * scale_factor)
                
                if new_char_w > bg_w or new_char_h > bg_h:
                    fit_scale = min(bg_w / new_char_w, bg_h / new_char_h)
                    new_char_w = int(new_char_w * fit_scale)
                    new_char_h = int(new_char_h * fit_scale)
                
                char_image = char_image.resize((new_char_w, new_char_h), Image.Resampling.LANCZOS)
                
                y_pos = bg_h - new_char_h - int(bg_h * 0.05) if position == "bottom" else (bg_h - new_char_h) // 2
                x_pos = (bg_w - new_char_w) // 2
                
                composite = background_image.copy().convert("RGBA")
                composite.paste(char_image, (x_pos, y_pos), char_image)
                composite = composite.convert("RGB")
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                composite.save(output_path, "PNG", dpi=(dpi, dpi))
                cleanup_file(temp_char_path)
                
                return True, "Character generated and composited locally"
            except Exception as e:
                cleanup_file(temp_char_path)
                return False, f"Local compositing failed: {str(e)}"
    except Exception as e:
        print(f"generate_character_composited_with_background error: {e}")
        return False, str(e)

