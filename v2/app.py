import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

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
    generate_character_composited_with_background,
)
from utils.bg_remover import remove_background_with_freepik_api
from utils.s3_utils import upload_image_to_s3

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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

        if scale < 0.1 or scale > 3.0:
            return jsonify({"error": "Scale must be between 0.1 and 3.0"}), 400

        # Handle selfie (file or URL)
        selfie_path = None
        selfie_filename = None
        if selfie_file and selfie_file.filename:
            if not allowed_file(selfie_file.filename):
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


@app.route("/api/generate-character", methods=["POST"])
def api_generate_character():
    """API alias; same JSON as /generate-character-web."""
    return generate_character_web()


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, secure_filename(filename), as_attachment=True)


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, secure_filename(filename))


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename))


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    """
    Remove background from an image using Freepik API.
    Accepts either a file upload or an image URL.
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
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)