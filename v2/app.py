import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flasgger import Swagger, swag_from
from flask_cors import CORS

# Load environment variables from .env file
# Try to load from parent directory (root) first, then current directory
env_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),  # Root .env
    os.path.join(os.path.dirname(__file__), '.env'),  # v2/.env (fallback)
    '/app/.env',  # Docker path
]
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ Loaded .env from: {env_path}")
        break
else:
    print("⚠️ No .env file found. Using system environment variables only.")

from utils.character_utils import (
    generate_unique_filename,
    download_image_from_url,
    cleanup_file,
    get_image_info,
    generate_character_with_identity,
    normalize_prompt_for_consistency,
    generate_seed_from_prompt,
    generate_character_composited_with_background,
    add_signature_image_overlay,
)
from utils.bg_remover import remove_background_with_freepik_api
from utils.s3_utils import upload_image_to_s3, create_zip_archive, upload_zip_to_s3
from utils.auth import require_api_key
from utils.prompts import HOBBY_PROMPTS, COMPOSITING_PROMPT

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}

# Swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api-docs"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Character Generator API",
        "description": "API for generating AI characters and removing backgrounds from images",
        "version": "1.0.0",
        "contact": {
            "name": "API Support"
        }
    },
    "securityDefinitions": {
        "ApiKeyAuth": {
            "type": "apiKey",
            "name": "X-API-Key",
            "in": "header",
            "description": "API key for authentication. Get your API key from the administrator."
        }
    },
    "security": [
        {
            "ApiKeyAuth": []
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_single_character(item_data, index):
    """
    Process a single character generation request (selfie only, no background).
    Returns a dict with success status and result/error.
    """
    try:
        selfie_file = item_data.get("selfie_file")
        selfie_url = item_data.get("selfie_url", "").strip()
        character_prompt = item_data.get("character_prompt", "").strip()
        station = item_data.get("station", "").strip() or None
        canvas_size = item_data.get("canvas_size") or None
        dpi = int(item_data.get("dpi", "300"))

        if not selfie_file and not selfie_url:
            return {
                "index": index,
                "success": False,
                "error": "Either selfie file or selfie_url is required"
            }

        if selfie_file and selfie_url:
            return {
                "index": index,
                "success": False,
                "error": "Provide either selfie file or selfie_url, not both"
            }

        if not character_prompt:
            return {
                "index": index,
                "success": False,
                "error": "Character prompt is required"
            }

        # Handle selfie
        selfie_path = None
        selfie_filename = None
        if selfie_file:
            if not selfie_file.filename or not allowed_file(selfie_file.filename):
                return {
                    "index": index,
                    "success": False,
                    "error": "Invalid selfie file type"
                }
            selfie_filename = generate_unique_filename(selfie_file.filename, "selfie")
            selfie_path = os.path.join(UPLOAD_FOLDER, selfie_filename)
            selfie_file.save(selfie_path)
        elif selfie_url:
            selfie_path = download_image_from_url(selfie_url, UPLOAD_FOLDER)
            if not selfie_path:
                return {
                    "index": index,
                    "success": False,
                    "error": "Failed to download selfie from URL"
                }
            selfie_filename = os.path.basename(selfie_path)

        # Generate output filename
        output_filename = generate_unique_filename(f"character_{selfie_filename}.png", "output")
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Generate character (no background)
        success, message = generate_character_with_identity(
            selfie_path=selfie_path,
            character_prompt=character_prompt,
            output_path=output_path,
            white_background=False,
            position="center",
            scale=1.0,
            background_dimensions=None,
            canvas_size=canvas_size,
            dpi=dpi,
            station=station,
        )

        if not success:
            cleanup_file(selfie_path)
            return {
                "index": index,
                "success": False,
                "error": f"Character generation failed: {message}"
            }

        # Upload to S3
        cloudfront_url = upload_image_to_s3(output_path)
        
        # Cleanup input
        cleanup_file(selfie_path)

        info = get_image_info(output_path)

        result = {
            "index": index,
            "success": True,
            "message": "Character generated successfully",
            "output_filename": output_filename,
            "local_path": f"/outputs/{output_filename}",
            "metadata": {
                "image_info": info,
                "character_prompt": character_prompt,
            },
        }

        if cloudfront_url:
            result["image_url"] = cloudfront_url

        return result

    except Exception as e:
        import traceback
        print(f"Error processing item {index}: {e}")
        print(traceback.format_exc())
        return {
            "index": index,
            "success": False,
            "error": f"Error: {str(e)}"
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate-character-web", methods=["POST"])
def generate_character_web():
    """
    Main web endpoint for character generation.
    - Selfie is required (file upload or URL).
    - Background file or URL is optional.
      - If provided: Gemini returns final composite.
      - If not: we return exactly what Gemini generates (no local BG removal).
    """
    try:
        selfie_file = request.files.get("selfie")
        selfie_url = request.form.get("selfie_url", "").strip()

        if not selfie_file and not selfie_url:
            return jsonify({"error": "Either selfie file or selfie_url is required"}), 400

        if selfie_file and selfie_url:
            return jsonify({"error": "Provide either selfie file or selfie_url, not both"}), 400

        character_prompt = request.form.get("character_prompt", "").strip()
        if not character_prompt:
            return jsonify({"error": "Character prompt is required"}), 400

        background_file = request.files.get("background")
        background_url = request.form.get("background_url", "").strip()

        if background_file and background_url:
            return jsonify({"error": "Provide either background file or background_url, not both"}), 400

        position = request.form.get("position", "bottom").strip()
        scale = float(request.form.get("scale", "1.0"))
        canvas_size = request.form.get("canvas_size", "").strip() or None
        dpi = int(request.form.get("dpi", "300"))
        use_gemini_compositing = request.form.get("use_gemini_compositing", "true").lower() == "true"
        station = request.form.get("station", "").strip() or None  # Station: pencil-sketch, cartoon, caricature, retro90, wynwood

        if scale < 0.1 or scale > 3.0:
            return jsonify({"error": "Scale must be between 0.1 and 3.0"}), 400

        # Handle selfie (file or URL)
        selfie_path = None
        selfie_filename = None
        if selfie_file:
            if not selfie_file.filename or not allowed_file(selfie_file.filename):
                return jsonify({"error": "Invalid selfie file type"}), 400
            selfie_filename = generate_unique_filename(selfie_file.filename, "selfie")
            selfie_path = os.path.join(UPLOAD_FOLDER, selfie_filename)
            selfie_file.save(selfie_path)
        elif selfie_url:
            selfie_path = download_image_from_url(selfie_url, UPLOAD_FOLDER)
            if not selfie_path:
                return jsonify({"error": "Failed to download selfie from URL"}), 400
            selfie_filename = os.path.basename(selfie_path)

        # Handle optional background
        background_path = None
        if background_file and background_file.filename:
            if not allowed_file(background_file.filename):
                cleanup_file(selfie_path)
                return jsonify({"error": "Invalid background file type"}), 400
            bg_filename = generate_unique_filename(background_file.filename, "background")
            background_path = os.path.join(UPLOAD_FOLDER, bg_filename)
            background_file.save(background_path)
        elif background_url:
            background_path = download_image_from_url(background_url, UPLOAD_FOLDER)
            if not background_path:
                cleanup_file(selfie_path)
                return jsonify({"error": "Failed to download background from URL"}), 400

        # Decide output filename
        if background_path:
            output_filename = generate_unique_filename(f"composited_{selfie_filename}.png", "output")
        else:
            output_filename = generate_unique_filename(f"character_{selfie_filename}.png", "output")
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        print("🎨 Character Generator Flow")
        print(f"  Background provided: {bool(background_path)}")
        print(f"  Position: {position}, scale: {scale}, canvas_size: {canvas_size}, dpi: {dpi}")

        # Call appropriate helper
        if background_path:
            success, message = generate_character_composited_with_background(
                selfie_path=selfie_path,
                background_path=background_path,
                character_prompt=character_prompt,
                output_path=output_path,
                position=position,
                scale=scale,
                canvas_size=canvas_size,
                dpi=dpi,
                use_gemini_compositing=use_gemini_compositing,
                station=station,
            )
        else:
            success, message = generate_character_with_identity(
                selfie_path=selfie_path,
                character_prompt=character_prompt,
                output_path=output_path,
                white_background=False,  # keep Gemini background as-is
                position=position,
                scale=scale,
                background_dimensions=None,
                canvas_size=canvas_size,
                dpi=dpi,
                station=station,
            )

        if not success:
            cleanup_file(selfie_path)
            cleanup_file(background_path)
            return jsonify({"error": f"Character generation failed: {message}"}), 500

        # Upload output image to S3 and get CloudFront URL
        print(f"📤 Attempting to upload to S3: {output_path}")
        cloudfront_url = upload_image_to_s3(output_path)

        # Cleanup inputs
        cleanup_file(selfie_path)
        cleanup_file(background_path)

        info = get_image_info(output_path)

        response_data = {
                "success": True,
                "message": "Character generated successfully",
                "output_filename": output_filename,
            "local_path": f"/outputs/{output_filename}",
                "metadata": {
                    "image_info": info,
                    "character_prompt": character_prompt,
                    "position": position,
                    "scale": scale,
                },
            }
        
        # Add CloudFront URL if upload was successful
        if cloudfront_url:
            response_data["image_url"] = cloudfront_url
            print(f"✅ S3 upload successful: {cloudfront_url}")
        else:
            print(f"⚠️ S3 upload failed or skipped. Check logs above for details.")
            print(f"   Make sure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET are set in .env")

        return jsonify(response_data)
    except Exception as e:
        import traceback

        print("Error in generate-character-web:", e)
        print(traceback.format_exc())
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/upload-batch-background", methods=["POST"])
def upload_batch_background():
    """
    Upload and save background image for batch processing.
    This background will be used later when compositing all characters.
    """
    try:
        background_file = request.files.get("background")
        background_url = request.form.get("background_url", "").strip()

        if not background_file and not background_url:
            return jsonify({"error": "Either background file or background_url is required"}), 400

        if background_file and background_url:
            return jsonify({"error": "Provide either background file or background_url, not both"}), 400

        # Handle background (always save locally first, then upload to S3 for a public URL)
        if background_file:
            if not background_file.filename or not allowed_file(background_file.filename):
                return jsonify({"error": "Invalid background file type"}), 400
            bg_filename = generate_unique_filename(background_file.filename, "batch_bg")
            background_path = os.path.join(UPLOAD_FOLDER, bg_filename)
            background_file.save(background_path)
        elif background_url:
            background_path = download_image_from_url(background_url, UPLOAD_FOLDER)
            if not background_path:
                return jsonify({"error": "Failed to download background from URL"}), 400
            bg_filename = os.path.basename(background_path)

        # Upload background to S3 to get a CloudFront URL
        bg_cloudfront_url = upload_image_to_s3(background_path)

        return jsonify({
            "success": True,
            "message": "Background uploaded successfully",
            "background_filename": bg_filename,
            "local_path": f"/uploads/{bg_filename}",
            "background_url": bg_cloudfront_url,
        })

    except Exception as e:
        import traceback
        print("Error in upload-batch-background:", e)
        print(traceback.format_exc())
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/generate-characters-batch-web", methods=["POST"])
def generate_characters_batch_web():
    """
    Batch character generation endpoint for web UI.
    Accepts multiple images with individual prompts and processes them in parallel.
    """
    try:
        # Parse items from JSON string (form data) or JSON body
        if request.is_json:
            data = request.get_json()
            items = data.get("items", [])
        else:
            items_json = request.form.get("items", "[]")
            items = json.loads(items_json) if items_json else []

        if not items or len(items) == 0:
            return jsonify({"error": "At least one item is required"}), 400

        if len(items) > 20:
            return jsonify({"error": "Maximum 20 items allowed per batch"}), 400

        # Prepare items for processing
        items_to_process = []
        for idx, item in enumerate(items):
            # Get file from form data if file index is specified
            selfie_file = None
            if f"selfie_{idx}" in request.files:
                selfie_file = request.files.get(f"selfie_{idx}")

            item_data = {
                "selfie_file": selfie_file,
                "selfie_url": item.get("selfie_url", "").strip(),
                "character_prompt": item.get("character_prompt", "").strip(),
                "station": item.get("station", "").strip() or None,
                "canvas_size": item.get("canvas_size") or None,
                "dpi": int(item.get("dpi", "300")),
            }
            items_to_process.append((item_data, idx))

        # Process in parallel
        results = []
        max_workers = min(len(items), 5)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_single_character, item_data, idx): idx
                for item_data, idx in items_to_process
            }

            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = future_to_index[future]
                    results.append({
                        "index": idx,
                        "success": False,
                        "error": f"Processing error: {str(e)}"
                    })

        # Sort results by index to maintain order
        results.sort(key=lambda x: x["index"])

        # Count successes and failures
        success_count = sum(1 for r in results if r.get("success", False))
        failure_count = len(results) - success_count

        # Create zip archive of successful results if requested
        zip_url = None
        zip_filename = None
        create_zip = request.form.get("create_zip", "false").lower() == "true"
        
        if create_zip and success_count > 0:
            try:
                # Get paths of successful results
                successful_paths = []
                for result in results:
                    if result.get("success") and result.get("output_filename"):
                        file_path = os.path.join(OUTPUT_FOLDER, secure_filename(result["output_filename"]))
                        if os.path.exists(file_path):
                            successful_paths.append(file_path)
                
                if successful_paths:
                    # Create zip archive
                    zip_path = create_zip_archive(successful_paths)
                    zip_filename = os.path.basename(zip_path)
                    
                    # Upload to S3 if configured
                    zip_url = upload_zip_to_s3(zip_path)
            except Exception as e:
                print(f"⚠️ Failed to create zip archive: {e}")

        response_data = {
            "success": True,
            "message": f"Batch processing completed: {success_count} succeeded, {failure_count} failed",
            "total": len(results),
            "succeeded": success_count,
            "failed": failure_count,
            "results": results
        }
        
        # Add zip info if created
        if zip_filename:
            response_data["zip_filename"] = zip_filename
            response_data["zip_local_path"] = f"/download/{zip_filename}"
            if zip_url:
                response_data["zip_url"] = zip_url

        return jsonify(response_data)

    except Exception as e:
        import traceback
        print("Error in generate-characters-batch-web:", e)
        print(traceback.format_exc())
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/composite-characters-on-background", methods=["POST"])
def composite_characters_on_background():
    """
    Composite multiple generated characters onto a single background using PIL.
    This preserves the exact background image without modification.
    """
    try:
        from PIL import Image, ImageDraw
        import math
        import requests
        from io import BytesIO
        
        # --- Characters: prefer S3/CloudFront URLs; filenames only as fallback ---
        character_filenames_json = request.form.get("character_filenames", "[]")
        character_filenames = json.loads(character_filenames_json) if character_filenames_json else []
        
        character_urls_json = request.form.get("character_urls", "")
        character_urls = json.loads(character_urls_json) if character_urls_json else []
        
        if not character_urls and not character_filenames:
            return jsonify({"error": "At least one character image is required (provide character_urls or character_filenames)."}), 400
        
        # --- Background: prefer S3/CloudFront URL; filename only as fallback ---
        background_filename = request.form.get("background_filename", "").strip()
        background_url = request.form.get("background_url", "").strip()
        
        if not background_url and not background_filename:
            return jsonify({"error": "Background image is required (provide background_url or background_filename)."}), 400

        position = request.form.get("position", "bottom").strip()
        scale = float(request.form.get("scale", "1.0"))
        canvas_size = request.form.get("canvas_size", "").strip() or None
        dpi = int(request.form.get("dpi", "300"))

        if scale < 0.1 or scale > 3.0:
            return jsonify({"error": "Scale must be between 0.1 and 3.0"}), 400
        
        # Load background image (in-memory; S3/URL preferred)
        if background_url:
            try:
                resp = requests.get(background_url, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                return jsonify({"error": f"Failed to load background from URL: {str(e)}"}), 400
            try:
                background_image = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                return jsonify({"error": f"Failed to decode background image from URL: {str(e)}"}), 400
        else:
            # Fallback: load from local uploads folder (legacy behavior)
            background_path = os.path.join(UPLOAD_FOLDER, secure_filename(background_filename))
            if not os.path.exists(background_path):
                return jsonify({"error": "Background file not found"}), 400
            background_image = Image.open(background_path).convert("RGB")
        
        bg_w, bg_h = background_image.size

        # Apply canvas size if specified
        if canvas_size:
            # Parse canvas size (e.g., "8x10", "11x14", "16x20")
            size_map = {
                "8x10": (8, 10),
                "11x14": (11, 14),
                "16x20": (16, 20),
            }
            if canvas_size in size_map:
                target_w, target_h = size_map[canvas_size]
                # Calculate dimensions at specified DPI
                target_width = int(target_w * dpi)
                target_height = int(target_h * dpi)
                # Resize background maintaining aspect ratio, then crop or pad
                bg_aspect = bg_w / bg_h
                target_aspect = target_width / target_height
                
                if bg_aspect > target_aspect:
                    # Background is wider, fit to height
                    new_h = target_height
                    new_w = int(new_h * bg_aspect)
                else:
                    # Background is taller, fit to width
                    new_w = target_width
                    new_h = int(new_w / bg_aspect)
                
                background_image = background_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                # Create new image with target size and paste background
                final_bg = Image.new("RGB", (target_width, target_height), (255, 255, 255))
                paste_x = (target_width - new_w) // 2
                paste_y = (target_height - new_h) // 2
                final_bg.paste(background_image, (paste_x, paste_y))
                background_image = final_bg
                bg_w, bg_h = background_image.size

        # Load all character images (in-memory; URLs preferred)
        character_images = []
        
        if character_urls:
            for url in character_urls:
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                except Exception as e:
                    return jsonify({"error": f"Failed to load character image from URL '{url}': {str(e)}"}), 400
                
                try:
                    char_img = Image.open(BytesIO(resp.content)).convert("RGBA")
                    character_images.append(char_img)
                except Exception as e:
                    return jsonify({"error": f"Failed to decode character image from URL '{url}': {str(e)}"}), 400
        else:
            # Fallback: use local filenames (existing behavior)
            for filename in character_filenames:
                char_path = os.path.join(OUTPUT_FOLDER, secure_filename(filename))
                if not os.path.exists(char_path):
                    return jsonify({"error": f"Character file not found: {filename}"}), 400
                try:
                    char_img = Image.open(char_path).convert("RGBA")
                    character_images.append(char_img)
                except Exception as e:
                    return jsonify({"error": f"Failed to load character image {filename}: {str(e)}"}), 400

        # Calculate character sizes and positions
        num_characters = len(character_images)
        target_char_height = int(bg_h * scale * 0.75)  # 75% of image height scaled
        
        # Resize all characters to consistent height
        resized_characters = []
        for char_img in character_images:
            char_w, char_h = char_img.size
            aspect_ratio = char_w / char_h
            new_w = int(target_char_height * aspect_ratio)
            new_h = target_char_height
            resized_char = char_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_characters.append(resized_char)

        # Use Gemini for artistic backgrounds to better integrate characters
        # This provides better blending and integration with watercolor/artistic backgrounds
        use_gemini_compositing = request.form.get("use_gemini_compositing", "false").lower() == "true"
        
        if use_gemini_compositing:
            # Use Gemini for better artistic integration
            from utils.character_utils import (
                get_gemini_client,
                get_gemini_image_model,
                _extract_final_image_from_response,
                _generate_content_image,
                normalize_prompt_for_consistency,
                generate_seed_from_prompt,
            )
            import io
            
            client = get_gemini_client()
            bg_w, bg_h = background_image.size
            
            num_characters = len(character_images)
            canvas_context = ""
            if canvas_size:
                canvas_context = (
                    f"\nTarget print size: {canvas_size} at {dpi} DPI. Keep the same aspect ratio as "
                    f"the provided background ({bg_w}x{bg_h})."
                )

            canvas_context_str = f"\n{canvas_context}" if canvas_context else ""
            
            # Get custom compositing prompt if provided, otherwise use default
            custom_compositing_prompt = request.form.get("compositing_prompt", "").strip()
            base_prompt = custom_compositing_prompt if custom_compositing_prompt else COMPOSITING_PROMPT
            
            full_prompt = f"""{base_prompt}

ADDITIONAL INSTRUCTIONS:
- Number of characters to merge: {num_characters}
- Position preference: {position.upper()}
- Scale factor: {scale}{canvas_context_str}

IMPORTANT REMINDERS:
- Each character image is provided separately - use each character EXACTLY as shown in their individual image.
- Do NOT combine equipment or objects from different characters.
- Do NOT add any new objects, balls, or equipment that weren't in the original character images.
- If a character has a soccer ball in their image, they keep ONLY that soccer ball.
- If a character has a softball/baseball in their image, they keep ONLY that ball.
- No mixing or adding of sports equipment between characters.

OUTPUT:
Return a SINGLE final composited image with all characters naturally placed on the EXACT background, ready for printing.
The background must be identical to the input background.
Each character must appear exactly as they do in their original image, with no extra objects added. And all of the characters must match the  {num_characters}"""

            # Prepare contents for Gemini (background + all characters)
            # Normalize prompt and generate seed for consistency
            normalized_prompt = normalize_prompt_for_consistency(full_prompt)
            seed = generate_seed_from_prompt(normalized_prompt)  # Pass already normalized prompt
            contents = [normalized_prompt, background_image] + character_images

            response = _generate_content_image(
                client=client,
                model=get_gemini_image_model(),
                contents=contents,
                seed=seed,
            )

            composite = _extract_final_image_from_response(response)
            if composite is None:
                return jsonify({"error": "No composited image generated by Gemini"}), 500
        else:
            # Use PIL for simple compositing (preserves background exactly)
            # Calculate positions for characters
            total_char_width = sum(char.size[0] for char in resized_characters)
            spacing = (bg_w - total_char_width) / (num_characters + 1) if num_characters > 1 else (bg_w - total_char_width) / 2
            
            # Create final composite
            composite = background_image.copy()
            
            # Calculate Y position based on position parameter
            if position == "bottom":
                y_pos = bg_h - target_char_height - int(bg_h * 0.05)  # 5% from bottom
            else:  # center
                y_pos = (bg_h - target_char_height) // 2
            
            # Place characters
            x_offset = spacing if num_characters > 1 else (bg_w - total_char_width) / 2
            for char_img in resized_characters:
                char_w, char_h = char_img.size
                # Create a temporary image to composite with alpha
                temp_bg = Image.new("RGBA", composite.size, (0, 0, 0, 0))
                temp_bg.paste(char_img, (int(x_offset), y_pos), char_img)
                composite = Image.alpha_composite(composite.convert("RGBA"), temp_bg).convert("RGB")
                x_offset += char_w + spacing

        # Always add the station/signature overlay to the final composite
        try:
            # For multi-character composites we default to the cartoon-style signature
            composite = add_signature_image_overlay(composite, style="cartoon")
        except Exception as _e:
            # Non-fatal: if signature fails, continue without blocking the composite
            print(f"⚠️ Failed to add signature overlay to composite: {_e}")

        # Generate output filename
        output_filename = generate_unique_filename(f"composited_batch_{num_characters}chars.png", "output")
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Save the composited result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        composite.save(output_path, "PNG", dpi=(dpi, dpi))
        
        # Upload to S3
        cloudfront_url = upload_image_to_s3(output_path)
        
        info = get_image_info(output_path)
        
        response_data = {
            "success": True,
            "message": f"Successfully composited {num_characters} character(s) onto background",
            "output_filename": output_filename,
            "local_path": f"/outputs/{output_filename}",
            "metadata": {
                "image_info": info,
                "num_characters": num_characters,
                "position": position,
                "scale": scale,
            },
        }
        
        if cloudfront_url:
            response_data["image_url"] = cloudfront_url
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print("Error in composite-characters-on-background:", e)
        print(traceback.format_exc())
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/api/generate-character", methods=["POST"])
@require_api_key
def api_generate_character():
    """
    Generate a character from a selfie using Google AI Studio.
    ---
    tags:
      - Character Generation
    security:
      - ApiKeyAuth: []
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: selfie
        type: file
        required: false
        description: Selfie image file (PNG, JPG, JPEG, GIF, BMP, TIFF)
      - in: formData
        name: selfie_url
        type: string
        required: false
        description: URL of the selfie image (alternative to file upload)
      - in: formData
        name: character_prompt
        type: string
        required: true
        description: Description of the character to generate
        example: "A full-body cartoon caricature with bright colors"
      - in: formData
        name: background
        type: file
        required: false
        description: Background image file (optional)
      - in: formData
        name: background_url
        type: string
        required: false
        description: URL of the background image (optional)
      - in: formData
        name: position
        type: string
        required: false
        default: bottom
        enum: [center, bottom]
        description: Character position on background
      - in: formData
        name: scale
        type: number
        required: false
        default: 1.0
        description: Character scale (0.1 to 3.0)
      - in: formData
        name: canvas_size
        type: string
        required: false
        enum: ["", "8x10", "11x14", "16x20"]
        description: Print size (optional)
      - in: formData
        name: dpi
        type: integer
        required: false
        default: 300
        description: Output DPI
      - in: header
        name: X-API-Key
        type: string
        required: true
        description: API key for authentication
    responses:
      200:
        description: Character generated successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
            output_filename:
              type: string
            local_path:
              type: string
            image_url:
              type: string
              description: CloudFront CDN URL
            metadata:
              type: object
      400:
        description: Bad request (missing parameters or invalid input)
      401:
        description: Unauthorized (missing or invalid API key)
      500:
        description: Server error
    """
    return generate_character_web()


@app.route("/api/generate-characters-batch", methods=["POST"])
@require_api_key
def api_generate_characters_batch():
    """
    Batch character generation API endpoint.
    ---
    tags:
      - Character Generation
    security:
      - ApiKeyAuth: []
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            items:
              type: array
              description: Array of character generation requests
              items:
                type: object
                required:
                  - character_prompt
                properties:
                  selfie_url:
                    type: string
                    description: URL of the selfie image
                  character_prompt:
                    type: string
                    required: true
                    description: Description of the character to generate
                  canvas_size:
                    type: string
                    enum: ["", "8x10", "11x14", "16x20"]
                    description: Print size (optional)
                  dpi:
                    type: integer
                    default: 300
                    description: Output DPI
      - in: header
        name: X-API-Key
        type: string
        required: true
        description: API key for authentication
    responses:
      200:
        description: Batch processing completed
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
            total:
              type: integer
            succeeded:
              type: integer
            failed:
              type: integer
            results:
              type: array
              items:
                type: object
                properties:
                  index:
                    type: integer
                  success:
                    type: boolean
                  message:
                    type: string
                  output_filename:
                    type: string
                  local_path:
                    type: string
                  image_url:
                    type: string
                    description: CloudFront CDN URL
                  error:
                    type: string
                  metadata:
                    type: object
      400:
        description: Bad request (missing parameters or invalid input)
      401:
        description: Unauthorized (missing or invalid API key)
      500:
        description: Server error
    """
    return generate_characters_batch_web()


@app.route("/download/<filename>")
def download_file(filename):
    """Download a file from the outputs folder or temp directory (for zip files)"""
    filename_secure = secure_filename(filename)
    file_path = os.path.join(OUTPUT_FOLDER, filename_secure)
    
    # If it's a zip file, check temp directory too
    if not os.path.exists(file_path) and filename_secure.endswith('.zip'):
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), filename_secure)
        if os.path.exists(temp_path):
            return send_file(temp_path, as_attachment=True, download_name=filename_secure)
    
    return send_from_directory(OUTPUT_FOLDER, filename_secure, as_attachment=True)


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, secure_filename(filename))


@app.route("/download-batch-zip", methods=["POST"])
def download_batch_zip():
    """
    Create and download a zip archive of multiple generated images.
    Accepts a list of filenames in the request.
    """
    try:
        if request.is_json:
            data = request.get_json()
            filenames = data.get("filenames", [])
        else:
            filenames_json = request.form.get("filenames", "[]")
            filenames = json.loads(filenames_json) if filenames_json else []
        
        if not filenames or len(filenames) == 0:
            return jsonify({"error": "No filenames provided"}), 400
        
        # Build full paths for all files
        file_paths = []
        for filename in filenames:
            file_path = os.path.join(OUTPUT_FOLDER, secure_filename(filename))
            if os.path.exists(file_path):
                file_paths.append(file_path)
        
        if not file_paths:
            return jsonify({"error": "None of the specified files exist"}), 404
        
        # Create zip archive
        zip_path = create_zip_archive(file_paths)
        
        # Upload to S3 if configured, otherwise serve locally
        zip_url = upload_zip_to_s3(zip_path)
        
        zip_filename = os.path.basename(zip_path)
        
        response_data = {
            "success": True,
            "message": f"Created zip archive with {len(file_paths)} files",
            "zip_filename": zip_filename,
            "local_path": f"/download/{zip_filename}",
            "file_count": len(file_paths)
        }
        
        if zip_url:
            response_data["zip_url"] = zip_url
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print("Error in download-batch-zip:", e)
        print(traceback.format_exc())
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename))


@app.route("/remove-bg", methods=["POST"])
@require_api_key
def remove_bg():
    """
    Remove background from an image using Freepik API.
    ---
    tags:
      - Background Removal
    security:
      - ApiKeyAuth: []
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: image
        type: file
        required: false
        description: Image file to remove background from (PNG, JPG, JPEG, GIF, BMP, TIFF)
      - in: formData
        name: image_url
        type: string
        required: false
        description: URL of the image (alternative to file upload)
      - in: header
        name: X-API-Key
        type: string
        required: true
        description: API key for authentication
    responses:
      200:
        description: Background removed successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
            output_filename:
              type: string
            local_path:
              type: string
            image_url:
              type: string
              description: CloudFront CDN URL
            metadata:
              type: object
              properties:
                image_info:
                  type: object
      400:
        description: Bad request (missing parameters or invalid input)
      401:
        description: Unauthorized (missing or invalid API key)
      500:
        description: Server error
    """
    try:
        image_file = request.files.get("image")
        image_url = request.form.get("image_url", "").strip()
        
        if not image_file and not image_url:
            return jsonify({"error": "Either image file or image_url is required"}), 400
        
        if image_file and image_url:
            return jsonify({"error": "Provide either image file or image_url, not both"}), 400
        
        # Handle file upload
        if image_file:
            if not image_file.filename:
                return jsonify({"error": "Invalid image file"}), 400
            if not allowed_file(image_file.filename):
                return jsonify({"error": "Invalid image file type"}), 400
            
            # Save uploaded file
            image_filename = generate_unique_filename(image_file.filename, "input")
            image_path = os.path.join(UPLOAD_FOLDER, image_filename)
            image_file.save(image_path)
        else:
            # Download from URL
            image_path = download_image_from_url(image_url, UPLOAD_FOLDER)
            if not image_path:
                return jsonify({"error": "Failed to download image from URL"}), 400
        
        # Remove background using Freepik
        result_path = remove_background_with_freepik_api(image_path)
        
        if not result_path or not os.path.exists(result_path):
            cleanup_file(image_path)
            # Check if it's an API key issue
            freepik_key = os.getenv('FREEPIK_API_KEY')
            if not freepik_key:
                error_msg = "FREEPIK_API_KEY is not set in environment variables. Please add it to your .env file."
            else:
                error_msg = "Background removal failed. Please check server logs for details. Common issues: invalid API key, inaccessible image URL, or rate limiting."
            return jsonify({"error": error_msg}), 500
        
        # Generate output filename
        output_filename = generate_unique_filename(f"bg_removed_{os.path.basename(image_path)}", "output")
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Move result to output folder
        import shutil
        shutil.move(result_path, output_path)
        
        # Upload output image to S3 and get CloudFront URL
        print(f"📤 Attempting to upload to S3: {output_path}")
        cloudfront_url = upload_image_to_s3(output_path)
        
        # Cleanup input file
        cleanup_file(image_path)
        
        info = get_image_info(output_path)
        
        response_data = {
            "success": True,
            "message": "Background removed successfully",
            "output_filename": output_filename,
            "local_path": f"/outputs/{output_filename}",
            "metadata": {
                "image_info": info,
            },
        }
        
        # Add CloudFront URL if upload was successful
        if cloudfront_url:
            response_data["image_url"] = cloudfront_url
            print(f"✅ S3 upload successful: {cloudfront_url}")
        else:
            print(f"⚠️ S3 upload failed or skipped. Check logs above for details.")
            print(f"   Make sure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET are set in .env")
        
        return jsonify(response_data)
    except Exception as e:
        import traceback
        
        print("Error in remove-bg:", e)
        print(traceback.format_exc())
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/health")
def health():
    """
    Health check endpoint.
    ---
    tags:
      - System
    responses:
      200:
        description: Service is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
    """
    return jsonify({"status": "ok"})


@app.route("/api/hobby-prompts", methods=["GET"])
def get_hobby_prompts():
    """
    Get available hobby prompts.
    ---
    tags:
      - Character Generation
    responses:
      200:
        description: List of available hobby prompts
        schema:
          type: object
          properties:
            hobbies:
              type: object
            prompts:
              type: object
    """
    return jsonify({
        "hobbies": {
            "football": "Playing Football",
            "basketball": "Playing Basketball",
            "baseball": "Playing Baseball",
            "cricket": "Playing Cricket",
            "skateboarding": "Playing Skateboarding"
        },
        "prompts": HOBBY_PROMPTS
    })


@app.route("/api-docs")
def api_docs_redirect():
    """Redirect to Swagger UI."""
    return render_template("swagger_redirect.html") if os.path.exists(os.path.join(BASE_DIR, "templates", "swagger_redirect.html")) else jsonify({
        "message": "Swagger documentation available at /api-docs",
        "swagger_ui": "/api-docs",
        "api_spec": "/apispec.json"
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)