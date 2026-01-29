"""
Character generation utilities using Google Gemini API.
"""
import os
import uuid
import io
import re
import time
import random
import hashlib
from collections import deque
from typing import Optional, Tuple, Dict, Any

from PIL import Image, ImageOps, ImageDraw, ImageFont
import requests
import google.genai as genai

# Try to import numpy for faster image operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

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


def _generate_content_image(client: genai.Client, model: str, contents, seed: Optional[int] = None):
    """
    Call `client.models.generate_content` with an optional config to force image output.

    Falls back to calling without config for older SDK versions.
    
    Args:
        client: Gemini client instance
        model: Model name to use
        contents: Content to generate (prompt + images)
        seed: Optional seed value for deterministic results. If provided, will attempt
              to use it in generation config. Note: Gemini API may not support seed
              for all models/versions.
    """
    # If the SDK supports it, force image-only output for consistency.
    config = None
    if types is not None and hasattr(types, "GenerateContentConfig"):
        try:
            # Try to create config with seed if provided
            config_kwargs = {"response_modalities": ["IMAGE"]}
            if seed is not None:
                # Try to add seed if the API supports it
                try:
                    # Check if seed parameter is supported
                    if hasattr(types.GenerateContentConfig, '__init__'):
                        # Try with seed parameter
                        try:
                            config = types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                seed=seed
                            )
                        except (TypeError, AttributeError):
                            # Seed not supported, use without it
                            config = types.GenerateContentConfig(**config_kwargs)
                    else:
                        config = types.GenerateContentConfig(**config_kwargs)
                except Exception:
                    # If seed fails, try without it
                    config = types.GenerateContentConfig(**config_kwargs)
            else:
                config = types.GenerateContentConfig(**config_kwargs)
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


def generate_seed_from_prompt(normalized_prompt: str) -> int:
    """
    Generate a consistent seed value from a normalized prompt string.
    This ensures the same prompt produces the same seed for deterministic results.
    
    Args:
        normalized_prompt: Already normalized prompt text (to avoid duplicate normalization)
        
    Returns:
        Integer seed value (0-2^31-1)
    """
    # Create hash and convert to integer seed
    hash_obj = hashlib.md5(normalized_prompt.encode('utf-8'))
    # Use first 4 bytes of hash as seed (0 to 2^32-1, but limit to 2^31-1 for safety)
    seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)
    return seed


def normalize_prompt_for_consistency(prompt: str) -> str:
    """
    Normalize prompts to fix conflicting instructions and improve consistency.
    
    Specifically handles:
    - Conflicting framing instructions (ultra-close-up vs full face with neck/shoulders)
    - Duplicate or contradictory requirements
    - Ambiguous specifications
    
    Args:
        prompt: Original prompt text
        
    Returns:
        Normalized prompt with conflicts resolved
    """
    prompt_lower = prompt.lower()
    
    # Detect conflicting framing instructions
    has_ultra_close = any(phrase in prompt_lower for phrase in [
        "ultra-close-up", "ultra close up", "filling the entire frame",
        "no neck", "no shoulders", "no borders"
    ])
    
    has_full_face_neck = any(phrase in prompt_lower for phrase in [
        "full face along with visible neck", "visible neck and shoulders",
        "keep all full face", "neck and shoulders"
    ])
    
    # If both conflicting instructions exist, prioritize the last one mentioned
    # (as it's likely the user's final intent)
    if has_ultra_close and has_full_face_neck:
        # Find which comes last in the prompt
        ultra_close_pos = max([
            prompt_lower.rfind("ultra-close-up"),
            prompt_lower.rfind("ultra close up"),
            prompt_lower.rfind("filling the entire frame"),
            prompt_lower.rfind("no neck"),
            prompt_lower.rfind("no shoulders")
        ])
        
        full_face_pos = max([
            prompt_lower.rfind("full face along with visible neck"),
            prompt_lower.rfind("visible neck and shoulders"),
            prompt_lower.rfind("keep all full face"),
            prompt_lower.rfind("neck and shoulders")
        ])
        
        # Remove the earlier conflicting instruction
        if full_face_pos > ultra_close_pos:
            # User wants full face with neck/shoulders - remove ultra-close-up instructions
            # Remove phrases about ultra-close-up and no neck/shoulders
            prompt = re.sub(
                r'(?:ultra-close-up|ultra close up)[^.]*?(?:no neck|no shoulders|no borders)[^.]*?\.',
                '',
                prompt,
                flags=re.IGNORECASE
            )
            prompt = re.sub(
                r'filling the entire frame[^.]*?(?:no neck|no shoulders)[^.]*?\.',
                '',
                prompt,
                flags=re.IGNORECASE
            )
            # Remove standalone "no neck, shoulders, or borders" phrases
            prompt = re.sub(
                r'[^.]*?no neck[^.]*?no shoulders[^.]*?no borders[^.]*?\.',
                '',
                prompt,
                flags=re.IGNORECASE
            )
            # Ensure the full face instruction is clear and prominent
            if "full face along with visible neck" not in prompt_lower and "visible neck and shoulders" not in prompt_lower:
                prompt = prompt.rstrip('. ') + ". Include the full face with visible neck and shoulders."
        else:
            # User wants ultra-close-up - remove full face with neck instructions
            prompt = re.sub(
                r'keep all full face along with visible neck and shoulders[^.]*?\.',
                '',
                prompt,
                flags=re.IGNORECASE
            )
            prompt = re.sub(
                r'[^.]*?visible neck and shoulders[^.]*?\.',
                '',
                prompt,
                flags=re.IGNORECASE
            )
            prompt = re.sub(
                r'[^.]*?full face along with visible neck[^.]*?\.',
                '',
                prompt,
                flags=re.IGNORECASE
            )
    
    # Remove duplicate instructions
    # Split into sentences and remove exact duplicates
    sentences = [s.strip() for s in re.split(r'[.!?]+', prompt) if s.strip()]
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        # Normalize whitespace for comparison
        sentence_normalized = re.sub(r'\s+', ' ', sentence_lower)
        if sentence_normalized not in seen:
            seen.add(sentence_normalized)
            unique_sentences.append(sentence)
    
    prompt = '. '.join(unique_sentences)
    if prompt and not prompt.endswith(('.', '!', '?')):
        prompt += '.'
    
    # Clean up multiple spaces
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    
    return prompt


def extract_positive_prompt(prompt: str) -> str:
    """
    Extract the positive prompt, excluding negative prompt sections.
    
    Args:
        prompt: Full prompt that may contain negative sections
        
    Returns:
        Only the positive prompt part (without negative sections)
    """
    prompt_lower = prompt.lower()
    
    # Check for negative prompt markers (various formats)
    # Try to find negative section markers
    negative_patterns = [
        r'negative\s+prompt\s*:',  # "negative prompt:" or "NEGATIVE PROMPT:"
        r'negative\s*:',  # "negative:" or "NEGATIVE:"
        r'NEGATIVE\s+PROMPT',  # "NEGATIVE PROMPT" (all caps, no colon)
    ]
    
    for pattern in negative_patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            # Return only the part before the negative marker
            return prompt[:match.start()].strip()
    
    # No negative prompt found, return original
    return prompt.strip()


def get_signature_image_path(style: str = "others") -> Optional[str]:
    """
    Get the path to the signature image based on style.
    
    Args:
        style: Style type - "pencil-sketch", "cartoon", or "others"
    
    Returns:
        Path to signature image if exists, None otherwise
    """
    # Try multiple possible locations
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from utils/ to v2/
    
    # Define file names based on style
    style_files = {
        "pencil-sketch": ["pencil-sketch.png", "pencil-sketch.jpg", "pencil_sketch.png", "pencil_sketch.jpg"],
        "cartoon": ["cartoon.png", "cartoon.jpg"],
        "others": ["others.png", "others.jpg"]
    }
    
    # Get possible filenames for this style
    filenames = style_files.get(style, style_files["others"])
    
    # Build possible paths
    possible_paths = []
    for filename in filenames:
        possible_paths.append(os.path.join(base_dir, "constants", filename))
        possible_paths.append(os.path.join(os.path.dirname(__file__), "..", "constants", filename))
    
    print(f"🔍 Looking for signature image (style: {style}). Base dir: {base_dir}")
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        print(f"   Checking: {abs_path} - {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        if exists:
            print(f"✅ Found signature image at: {abs_path}")
            return abs_path
    
    print(f"❌ Signature image not found for style '{style}'")
    print(f"   Please place '{filenames[0]}' in: {os.path.abspath(os.path.join(base_dir, 'constants'))}")
    return None


def add_signature_image_overlay(img: Image.Image, style: str = "others") -> Image.Image:
    """
    Add the signature image overlay to the bottom right of an image.
    The signature will be 20% width and 20% height of the generated image.
    
    Args:
        img: PIL Image object
        style: Style type - "pencil-sketch", "cartoon", or "others"
        
    Returns:
        PIL Image with signature overlay added
    """
    # Ensure image is in RGB or RGBA mode
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    
    # Get signature image path based on style
    signature_path = get_signature_image_path(style)
    if not signature_path:
        print(f"⚠️ Signature image not found for style '{style}'. Skipping signature overlay.")
        return img
    
    print(f"✅ Found signature image at: {signature_path}")
    
    try:
        # Load signature image
        signature_img = Image.open(signature_path)
        
        # Convert signature to RGBA if needed for transparency support
        if signature_img.mode != 'RGBA':
            signature_img = signature_img.convert('RGBA')
        
        # Get dimensions
        img_width, img_height = img.size
        
        # Calculate signature size: 15% of image dimensions
        sig_width = int(img_width * 0.20)
        sig_height = int(img_height * 0.20)
        
        # Resize signature maintaining aspect ratio
        sig_aspect = signature_img.width / signature_img.height
        target_aspect = sig_width / sig_height
        
        if sig_aspect > target_aspect:
            # Signature is wider, fit to width
            new_sig_width = sig_width
            new_sig_height = int(sig_width / sig_aspect)
        else:
            # Signature is taller, fit to height
            new_sig_height = sig_height
            new_sig_width = int(sig_height * sig_aspect)
        
        # Resize signature image
        signature_resized = signature_img.resize((new_sig_width, new_sig_height), Image.Resampling.LANCZOS)
        
        # Create a copy of the main image
        img_with_signature = img.copy()
        
        # Convert to RGBA if needed for compositing
        if img_with_signature.mode != 'RGBA':
            img_with_signature = img_with_signature.convert('RGBA')
        
        # Calculate position: top right with small padding (1% from edges)
        padding_x = int(img_width * 0.01)
        padding_y = int(img_height * 0.01)
        x = img_width - new_sig_width - padding_x
        y = padding_y  # Top position instead of bottom
        
        # Composite signature onto image
        img_with_signature.paste(signature_resized, (x, y), signature_resized)
        
        print(f"✅ Signature overlay added successfully!")
        print(f"   Position: ({x}, {y}), Size: ({new_sig_width}x{new_sig_height})")
        print(f"   Image size: ({img_width}x{img_height})")
        
        # Convert back to original mode if needed
        if img.mode == 'RGB' and img_with_signature.mode == 'RGBA':
            # Create white background and paste
            final_img = Image.new('RGB', img_with_signature.size, (255, 255, 255))
            final_img.paste(img_with_signature, mask=img_with_signature.split()[3] if img_with_signature.mode == 'RGBA' else None)
            return final_img
        
        return img_with_signature
        
    except Exception as e:
        import traceback
        print(f"⚠️ Failed to add signature overlay: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return img


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

        # Normalize the prompt early to fix conflicts and improve consistency
        character_prompt = normalize_prompt_for_consistency(character_prompt)

        # Load image - DO NOT apply EXIF orientation to preserve user's intended orientation
        # EXIF orientation can incorrectly rotate images, so we use the image as-is
        selfie_image = Image.open(selfie_path)
        original_width, original_height = selfie_image.size
        is_original_portrait = original_height > original_width

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
        background_section = ""
        prompt_to_use = character_prompt
        character_only_override = ""
        prohibitions_section = ""
        background_removal_text = ""
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
                    
                    if negative_items:
                        negative_list = '\n'.join([f'- {item}' for item in negative_items])
                        negative_prompt_section = f"""
STRICT PROHIBITIONS - DO NOT INCLUDE:
{negative_list}
"""
            
            # For style conversions, check if user specified background removal
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
            
            if character_only:
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
            
            if character_only:
                prohibitions_section = """
STRICT PROHIBITIONS (when character_only=True):
- DO NOT add any background, scenery, or visual context
- DO NOT merge or composite the character onto anything
- DO NOT create frames, borders, white frames, white borders, or white canvases
- DO NOT add any white space, white padding, or white margins around the character
- DO NOT add any elements behind or around the character
- The output must be ONLY the character on transparent/white background with NO white frames or borders
"""
            
            background_removal_text = "- CRITICAL: Remove and ignore the background from the input image. Output ONLY the character/subject with no background." if character_only and not is_warhol_style else ""
            
            # Check if user explicitly requests full-body in style conversion
            has_explicit_full_body_style = re.search(r'\bfull[\s-]?body\b', prompt_lower_check) or re.search(r'\bfull[\s-]?body\b', original_prompt_lower)
            
            if is_warhol_style:
                composition_instructions = """COMPOSITION:
- Follow the user's specific composition requirements (grid layouts, panels, etc.)
- Apply the requested layout and framing as specified in the prompt
"""
            else:
                if has_explicit_full_body_style:
                    composition_instructions = f"""CRITICAL REQUIREMENTS - ORIENTATION PRESERVATION (HIGHEST PRIORITY):
- ⚠️ CRITICAL: The input image is {"PORTRAIT" if is_portrait else "LANDSCAPE"} orientation ({img_width}x{img_height})
- ⚠️ CRITICAL: You MUST output the image in the EXACT SAME orientation: {"PORTRAIT" if is_portrait else "LANDSCAPE"}
- ⚠️ CRITICAL: Do NOT rotate the image - if input is tall (portrait), output must be tall (portrait)
- ⚠️ CRITICAL: Do NOT rotate the image - if input is wide (landscape), output must be wide (landscape)
- The user's prompt explicitly requests "full-body" - you MUST show the complete character from head to feet
- Preserve the EXACT same framing, crop, and composition as the input image
- Keep the same pose, position, and body parts visible
- Follow any pose instructions in the user's prompt (e.g., "standing in playful pose")
- Maintain the same aspect ratio ({img_width}x{img_height})
- Do NOT flip, or change the orientation of the image in ANY way
- Do NOT cut off or hide body parts (hands, feet, arms, legs) - show the complete full body
- Do NOT change the subject's position or pose unless the prompt explicitly requests a different pose
- Show complete body parts (hands, feet, arms, legs) - the character must be full-body as requested
- Apply the style transformation to the EXISTING image composition
"""
                else:
                    composition_instructions = f"""CRITICAL REQUIREMENTS - ORIENTATION PRESERVATION (HIGHEST PRIORITY):
- ⚠️ CRITICAL: The input image is {"PORTRAIT" if is_portrait else "LANDSCAPE"} orientation ({img_width}x{img_height})
- ⚠️ CRITICAL: You MUST output the image in the EXACT SAME orientation: {"PORTRAIT" if is_portrait else "LANDSCAPE"}
- ⚠️ CRITICAL: Do NOT rotate the image - if input is tall (portrait), output must be tall (portrait)
- ⚠️ CRITICAL: Do NOT rotate the image - if input is wide (landscape), output must be wide (landscape)
- Preserve the EXACT same framing, crop, and composition as the input image
- Keep the same pose, position, and body parts visible (if it's a half picture, keep it as a half picture)
- Follow any pose instructions in the user's prompt (e.g., "standing in playful pose")
- Maintain the same aspect ratio ({img_width}x{img_height})
- Do NOT flip, or change the orientation of the image in ANY way
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
{background_removal_text}
"""
            
            full_prompt = f"""{monochrome_prefix}{character_only_override}Convert the reference image to the requested style{"" if is_warhol_style else " while preserving the EXACT composition, pose, framing, and subject matter"}.
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
        
        # Normalize prompt and generate seed for consistency
        normalized_prompt = normalize_prompt_for_consistency(full_prompt)
        seed = generate_seed_from_prompt(normalized_prompt)  # Pass already normalized prompt
        
        response = _generate_content_image(
            client=client,
            model=get_gemini_image_model(),
            contents=[normalized_prompt, selfie_image],
            seed=seed,
        )

        img = _extract_final_image_from_response(response)
        if img is not None:
            if not hasattr(img, 'size') or not isinstance(img, Image.Image):
                return False, "Invalid image object returned from Gemini (missing size attribute)"
            
            
            # Optimize: Convert mode only if necessary
            if is_monochrome:
                if img.mode != 'L':
                    img = img.convert('L')
                if img.mode != 'RGB':  # Ensure RGB for saving
                    img = img.convert('RGB')
            elif img.mode not in ('RGB', 'RGBA'):
                # Only convert if not already in a compatible format
                img = img.convert('RGB')
            
            # Detect style and add appropriate signature overlay
            # Priority: caricature/pencil-sketch → cartoon → others
            # Extract positive prompt only (exclude negative sections)
            positive_prompt = extract_positive_prompt(character_prompt)
            prompt_lower_style = positive_prompt.lower()
            
            # Debug: Show what was extracted
            if positive_prompt != character_prompt:
                print(f"📝 Negative prompt section removed. Positive part: {positive_prompt[:200]}...")
            
            # Check for caricature FIRST (highest priority - caricature uses pencil-sketch.png)
            is_caricature = "caricature" in prompt_lower_style
            
            # Pencil sketch keywords (excluding caricature which is checked separately)
            is_pencil_sketch = (
                "pencil sketch" in prompt_lower_style or
                "pencil drawing" in prompt_lower_style or
                "graphite" in prompt_lower_style or
                "charcoal" in prompt_lower_style or
                ("hand-drawn" in prompt_lower_style and "sketch" in prompt_lower_style) or
                ("sketch" in prompt_lower_style and "pencil" in prompt_lower_style)
            )
            
            # Cartoon keywords - only if NOT caricature (caricature takes priority)
            cartoon_keywords = ["cartoon", "comic", "animated", "cartoon-style", "anime"]
            is_cartoon = False
            if not is_caricature:  # Only check cartoon if it's NOT a caricature
                is_cartoon = any(keyword in prompt_lower_style for keyword in cartoon_keywords)
            
            # Determine style for signature (caricature/pencil-sketch takes priority)
            if is_caricature or is_pencil_sketch:
                style = "pencil-sketch"
                if is_caricature:
                    print(f"🎨 Caricature detected! Adding signature overlay (pencil-sketch.png).")
                else:
                    print(f"🎨 Pencil sketch detected! Adding signature overlay.")
            elif is_cartoon:
                style = "cartoon"
                matched_keywords = [k for k in cartoon_keywords if k in prompt_lower_style]
                print(f"🎨 Cartoon style detected! Adding signature overlay.")
                print(f"   Matched keywords: {matched_keywords}")
            else:
                style = "others"
                print(f"🎨 Other style detected (Warhol, Wynwood, etc.)! Adding signature overlay.")
            
            print(f"   Style: {style}, Prompt preview: {character_prompt[:150]}...")
            img = add_signature_image_overlay(img, style)
            
            # Optimize: Create directory only once
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Optimize: Use optimized PNG saving
            img.save(output_path, optimize=True)
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

        # Normalize the prompt early to fix conflicts and improve consistency
        character_prompt = normalize_prompt_for_consistency(character_prompt)

        # Load images - DO NOT apply EXIF orientation to preserve user's intended orientation
        # EXIF orientation can incorrectly rotate images, so we use the images as-is
        selfie_image = Image.open(selfie_path)
        original_selfie_width, original_selfie_height = selfie_image.size
        is_original_selfie_portrait = original_selfie_height > original_selfie_width
        
        background_image = Image.open(background_path)
        bg_w, bg_h = background_image.size

        canvas_context = ""
        if canvas_size:
            canvas_context = (
                f"\nTarget print size: {canvas_size} at {dpi} DPI. Keep the same aspect ratio as "
                f"the provided background ({bg_w}x{bg_h})."
            )

        if use_gemini_compositing:
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

            # Normalize prompt and generate seed for consistency
            normalized_prompt = normalize_prompt_for_consistency(full_prompt)
            seed = generate_seed_from_prompt(normalized_prompt)  # Pass already normalized prompt

            response = _generate_content_image(
                client=client,
                model=get_gemini_image_model(),
                contents=[normalized_prompt, selfie_image, background_image],
                seed=seed,
            )

            img = _extract_final_image_from_response(response)
            if img is not None:
                if not hasattr(img, 'size') or not isinstance(img, Image.Image):
                    return False, "Invalid image object returned from Gemini (missing size attribute)"
                
                # Detect style and add appropriate signature overlay
                # Extract positive prompt only (exclude negative sections)
                positive_prompt_comp = extract_positive_prompt(character_prompt)
                prompt_lower_comp = positive_prompt_comp.lower()
                
                # Check for caricature FIRST (highest priority - caricature uses pencil-sketch.png)
                is_caricature = "caricature" in prompt_lower_comp
                
                # Pencil sketch keywords (excluding caricature which is checked separately)
                is_pencil_sketch = (
                    "pencil sketch" in prompt_lower_comp or
                    "pencil drawing" in prompt_lower_comp or
                    "graphite" in prompt_lower_comp or
                    "charcoal" in prompt_lower_comp or
                    ("hand-drawn" in prompt_lower_comp and "sketch" in prompt_lower_comp) or
                    ("sketch" in prompt_lower_comp and "pencil" in prompt_lower_comp)
                )
                
                # Cartoon keywords - only if NOT caricature (caricature takes priority)
                cartoon_keywords = ["cartoon", "comic", "animated", "cartoon-style", "anime"]
                is_cartoon = False
                if not is_caricature:  # Only check cartoon if it's NOT a caricature
                    is_cartoon = any(keyword in prompt_lower_comp for keyword in cartoon_keywords)
                
                # Determine style (caricature/pencil-sketch takes priority)
                if is_caricature or is_pencil_sketch:
                    style = "pencil-sketch"
                elif is_cartoon:
                    style = "cartoon"
                else:
                    style = "others"
                
                print(f"🎨 Adding signature overlay (style: {style}) to composited image")
                img = add_signature_image_overlay(img, style)
                
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
                
                # Optimized background removal using numpy if available, otherwise optimized PIL
                w, h = char_image.size
                threshold = 220
                
                if NUMPY_AVAILABLE:
                    # Fast numpy-based background removal
                    img_array = np.array(char_image)
                    r, g, b, a = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3]
                    
                    # Create background mask: white/light colors OR uniform gray
                    is_light = (r > threshold) & (g > threshold) & (b > threshold)
                    rgb_avg = (r + g + b) / 3
                    rgb_diff = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
                    is_uniform_gray = (rgb_avg > threshold) & (rgb_diff < 30)
                    background_mask = (is_light | is_uniform_gray) & (a > 0)
                    
                    # Flood-fill from edges using scipy or manual implementation
                    # Mark edge pixels as background candidates
                    edge_mask = np.zeros_like(background_mask, dtype=bool)
                    edge_mask[0, :] = True  # top edge
                    edge_mask[-1, :] = True  # bottom edge
                    edge_mask[:, 0] = True  # left edge
                    edge_mask[:, -1] = True  # right edge
                    
                    # Start flood-fill from edge background pixels
                    to_remove = np.zeros_like(background_mask, dtype=bool)
                    queue = deque()
                    visited = np.zeros_like(background_mask, dtype=bool)
                    
                    # Initialize queue with edge background pixels
                    edge_bg = edge_mask & background_mask
                    edge_coords = np.argwhere(edge_bg)
                    for y, x in edge_coords:
                        queue.append((x, y))
                        visited[y, x] = True
                    
                    # Optimized flood-fill with deque (O(1) operations)
                    while queue:
                        x, y = queue.popleft()  # O(1) instead of O(n) with list.pop(0)
                        
                        if background_mask[y, x]:
                            # Quick neighbor check: if surrounded by non-bg, likely part of character
                            y_min, y_max = max(0, y-1), min(h, y+2)
                            x_min, x_max = max(0, x-1), min(w, x+2)
                            neighbors = background_mask[y_min:y_max, x_min:x_max]
                            non_bg_ratio = 1.0 - np.sum(neighbors) / neighbors.size
                            
                            # Only remove if not surrounded by non-background (likely edge/background)
                            if non_bg_ratio < 0.4:  # Less than 40% non-background neighbors
                                to_remove[y, x] = True
                                
                                # Add neighbors to queue
                                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                                        visited[ny, nx] = True
                                        if background_mask[ny, nx]:
                                            queue.append((nx, ny))
                    
                    # Apply removal - make background pixels transparent
                    img_array[:, :, 3] = np.where(to_remove, 0, img_array[:, :, 3])
                    char_image = Image.fromarray(img_array, 'RGBA')
                else:
                    # Optimized PIL-based background removal (fallback)
                    data = list(char_image.getdata())
                    data_len = len(data)
                    
                    # Pre-compute background pixels for faster lookup
                    is_bg = [False] * data_len
                    for idx in range(data_len):
                        r, g, b, a = data[idx]
                        if a > 0:
                            if r > threshold and g > threshold and b > threshold:
                                is_bg[idx] = True
                            else:
                                rgb_avg = (r + g + b) / 3
                                rgb_diff = max(r, g, b) - min(r, g, b)
                                if rgb_avg > threshold and rgb_diff < 30:
                                    is_bg[idx] = True
                    
                    # Use deque for O(1) queue operations
                    to_remove = [False] * data_len
                    visited = set()
                    queue = deque()
                    
                    # Start flood-fill from edge background pixels
                    for y in range(h):
                        for x in range(w):
                            if x == 0 or x == w-1 or y == 0 or y == h-1:
                                idx = y * w + x
                                if idx < data_len and is_bg[idx]:
                                    queue.append((x, y))
                                    visited.add((x, y))
                    
                    # Optimized flood-fill
                    while queue:
                        x, y = queue.popleft()  # O(1) operation
                        idx = y * w + x
                        
                        if idx < data_len and is_bg[idx]:
                            # Quick neighbor check
                            non_bg_neighbors = 0
                            total_neighbors = 0
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    nidx = ny * w + nx
                                    if nidx < data_len:
                                        total_neighbors += 1
                                        if not is_bg[nidx] and data[nidx][3] > 0:
                                            non_bg_neighbors += 1
                            
                            # Only remove if not surrounded by non-background
                            if total_neighbors == 0 or non_bg_neighbors < total_neighbors * 0.4:
                                to_remove[idx] = True
                                
                                # Add neighbors
                                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < w and 0 <= ny < h:
                                        coord = (nx, ny)
                                        if coord not in visited:
                                            visited.add(coord)
                                            nidx = ny * w + nx
                                            if nidx < data_len and is_bg[nidx]:
                                                queue.append(coord)
                    
                    # Apply removal
                    new_data = []
                    for idx, (r, g, b, a) in enumerate(data):
                        if to_remove[idx]:
                            new_data.append((r, g, b, 0))
                        else:
                            new_data.append((r, g, b, a))
                
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
                
                # Optimize: Only convert background if needed
                if background_image.mode != "RGBA":
                    composite = background_image.convert("RGBA")
                else:
                    composite = background_image.copy()
                composite.paste(char_image, (x_pos, y_pos), char_image)
                
                # Only convert to RGB if needed (PNG supports RGBA, but RGB is smaller)
                if composite.mode != "RGB":
                    composite = composite.convert("RGB")
                
                # Detect style and add appropriate signature overlay
                # Extract positive prompt only (exclude negative sections)
                positive_prompt_comp = extract_positive_prompt(character_prompt)
                prompt_lower_comp = positive_prompt_comp.lower()
                
                # Check for caricature FIRST (highest priority - caricature uses pencil-sketch.png)
                is_caricature = "caricature" in prompt_lower_comp
                
                # Pencil sketch keywords (excluding caricature which is checked separately)
                is_pencil_sketch = (
                    "pencil sketch" in prompt_lower_comp or
                    "pencil drawing" in prompt_lower_comp or
                    "graphite" in prompt_lower_comp or
                    "charcoal" in prompt_lower_comp or
                    ("hand-drawn" in prompt_lower_comp and "sketch" in prompt_lower_comp) or
                    ("sketch" in prompt_lower_comp and "pencil" in prompt_lower_comp)
                )
                
                # Cartoon keywords - only if NOT caricature (caricature takes priority)
                cartoon_keywords = ["cartoon", "comic", "animated", "cartoon-style", "anime"]
                is_cartoon = False
                if not is_caricature:  # Only check cartoon if it's NOT a caricature
                    is_cartoon = any(keyword in prompt_lower_comp for keyword in cartoon_keywords)
                
                # Determine style (caricature/pencil-sketch takes priority)
                if is_caricature or is_pencil_sketch:
                    style = "pencil-sketch"
                elif is_cartoon:
                    style = "cartoon"
                else:
                    style = "others"
                
                print(f"🎨 Adding signature overlay (style: {style}) to composited image")
                composite = add_signature_image_overlay(composite, style)
                
                # Optimize: Create directory only once
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Optimize: Use optimized PNG saving
                composite.save(output_path, "PNG", optimize=True, dpi=(dpi, dpi))
                cleanup_file(temp_char_path)
                
                return True, "Character generated and composited locally"
            except Exception as e:
                cleanup_file(temp_char_path)
                return False, f"Local compositing failed: {str(e)}"
    except Exception as e:
        print(f"generate_character_composited_with_background error: {e}")
        return False, str(e)

