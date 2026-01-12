"""
Character generation utilities using Google Gemini API.
"""
import os
import uuid
import io
from typing import Optional, Tuple, Dict, Any

from PIL import Image
import requests
import google.genai as genai


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


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

        selfie_image = Image.open(selfie_path)

        # Detect if this is a style conversion task (pencil sketch, painting, etc.)
        # vs a character transformation task
        style_keywords = [
            "pencil sketch", "sketch", "drawing", "hand-drawn", "graphite",
            "convert", "transform into", "render as", "style of", "in the style",
            "painting", "watercolor", "oil painting", "charcoal", "ink drawing"
        ]
        is_style_conversion = any(keyword.lower() in character_prompt.lower() for keyword in style_keywords)

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
            
            # For style conversions, preserve exact composition and framing
            full_prompt = f"""{monochrome_prefix}Convert the reference image to the requested style while preserving the EXACT composition, pose, framing, and subject matter.

CRITICAL REQUIREMENTS:
- Preserve the EXACT same framing, crop, and composition as the input image
- Keep the same pose, position, and body parts visible (if it's a half picture, keep it as a half picture)
- Maintain the same aspect ratio
- Do NOT add or remove body parts (e.g., if only upper body is shown, do NOT make it full body)
- Do NOT change the subject's position or pose
- Apply the style transformation to the EXISTING image composition

STYLE CONVERSION:
{character_prompt}

BACKGROUND:
{bg_req}

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
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[full_prompt, selfie_image],
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img = Image.open(io.BytesIO(part.inline_data.data))
                
                # Post-process: If monochrome was requested, ensure output is pure grayscale
                # This prevents any color tints (like blue shades) that Gemini might add
                if is_monochrome:
                    # Convert to grayscale (L mode) then back to RGB to ensure pure grayscale
                    # This removes any color information and ensures R=G=B for all pixels
                    if img.mode != 'L':
                        img = img.convert('L').convert('RGB')
                        print("🖼️ Post-processed output to ensure pure grayscale (removed any color tints)")
                
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

        selfie_image = Image.open(selfie_path)
        background_image = Image.open(background_path)
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

        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[full_prompt, selfie_image, background_image],
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img = Image.open(io.BytesIO(part.inline_data.data))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path)
                return True, "Gemini composited character successfully"

        return False, "No composited image generated by Gemini"
    except Exception as e:
        print(f"generate_character_composited_with_background error: {e}")
        return False, str(e)

