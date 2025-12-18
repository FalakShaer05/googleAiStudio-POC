"""
API Endpoints
Main API routes for image processing service
"""

import os
import time
from flask import Blueprint, request, jsonify, send_file, current_app
from flask_cors import cross_origin
from werkzeug.utils import secure_filename

from .auth import require_api_key, get_api_key_info
from .models import create_error_response, create_success_response, ERROR_CODES
from .utils import (
    allowed_file, generate_unique_filename, format_file_size, 
    format_processing_time, cleanup_file, validate_image_file, 
    get_image_info, create_download_url, remove_background,
    download_image_from_url, upload_image_to_s3,
    generate_character_with_identity, composite_character_on_background_manual
)

# Import existing functions from main app (lazy import to avoid circular dependency)
def get_app_functions():
    """Get functions from main app module to avoid circular imports"""
    from app import convert_image_to_image
    return {
        'convert_image_to_image': convert_image_to_image
    }

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

@api_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if Gemini client is available
        from app import client
        client_status = client is not None
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'services': {
                'gemini_client': client_status,
                'file_uploads': True,
                'file_outputs': True
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@api_bp.route('/validate', methods=['POST'])
@cross_origin()
def validate_request():
    """
    Validate API key and request parameters
    """
    try:
        # Get API key
        api_key = request.headers.get('X-API-Key') or request.form.get('api_key')
        
        if not api_key:
            return jsonify(create_error_response('AUTH_001')), 401
        
        # Validate API key
        from .auth import validate_api_key
        if not validate_api_key(api_key):
            return jsonify(create_error_response('AUTH_002')), 401
        
        # Get API key info
        key_info = get_api_key_info(api_key)
        
        return jsonify({
            'success': True,
            'message': 'API key is valid',
            'api_key_info': key_info
        })
        
    except Exception as e:
        return jsonify(create_error_response('SERVICE_003', str(e))), 500

@api_bp.route('/process', methods=['POST'])
@cross_origin()
@require_api_key
def process_image():
    """
    Main image processing endpoint
    Converts image using prompt and optionally removes background
    """
    start_time = time.time()
    
    try:
        # Validate required parameters
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            return jsonify(create_error_response('VALIDATION_001', 'Prompt is required')), 400
        
        # Check if image is provided as file or URL
        image_file = request.files.get('image')
        # Support both 'image_url' and 'file_url' for compatibility
        image_url = request.form.get('image_url', '').strip() or request.form.get('file_url', '').strip()
        
        if not image_file and not image_url:
            return jsonify(create_error_response('VALIDATION_001', 'Either image file or image_url/file_url is required')), 400
        
        if image_file and image_url:
            return jsonify(create_error_response('VALIDATION_001', 'Provide either image file or image_url/file_url, not both')), 400
        
        # Get optional remove_bg parameter
        remove_bg = request.form.get('remove_bg', 'false').lower() == 'true'
        print(f"📋 Request parameters - prompt: {prompt[:50]}..., remove_bg: {remove_bg}, image_url: {image_url[:50] if image_url else 'N/A'}...")
        
        # Create upload and output directories
        upload_dir = current_app.config['UPLOAD_FOLDER']
        output_dir = current_app.config['OUTPUT_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle input image (file or URL)
        input_path = None
        if image_file:
            # Validate uploaded file
            is_valid, error_msg = validate_image_file(image_file)
            if not is_valid:
                return jsonify(create_error_response('VALIDATION_002', error_msg)), 400
            
            # Save uploaded file
            input_filename = generate_unique_filename(image_file.filename, 'input')
            input_path = os.path.join(upload_dir, input_filename)
            image_file.save(input_path)
        else:
            # Download image from URL
            input_path = download_image_from_url(image_url, upload_dir)
            if not input_path:
                return jsonify(create_error_response('VALIDATION_002', 'Failed to download image from URL. Please check the URL and try again.')), 400
            # Derive a filename from the downloaded path for downstream naming
            input_filename = os.path.basename(input_path)
        
        # Get app functions to avoid circular import
        app_funcs = get_app_functions()
        
        # Use the prompt as-is (no enhancement for white background)
        # Generate output filename (ensure input_filename defined for both file and URL flows)
        if 'input_filename' not in locals() or not input_filename:
            input_filename = os.path.basename(input_path)
        
        # If background removal is requested, use PNG format from the start
        # Otherwise, preserve original format
        if remove_bg:
            # Generate PNG filename for background removal
            base_name = os.path.splitext(input_filename)[0]
            output_filename = generate_unique_filename(f"processed_{base_name}.png", 'output')
        else:
            output_filename = generate_unique_filename(f"processed_{input_filename}", 'output')
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Step 1: Convert image using prompt
        # Check if upscaling should be skipped for faster processing
        skip_upscale = os.getenv('SKIP_PRE_UPSCALE', 'false').lower() == 'true'
        upscale_before = not skip_upscale
        
        print(f"🔄 Step 1: Converting image with prompt...")
        success, message = app_funcs['convert_image_to_image'](
            input_image_path=input_path,
            prompt=prompt,  # Use prompt as-is
            output_path=output_path,
            upscale_before=upscale_before,
            scale_factor=2,
            canvas_size=None,  # No canvas size processing
            dpi=300,
            reference_background_path=None,  # No background reference
            enable_background_compositing=False  # No compositing
        )
        
        if not success:
            # Clean up input files
            cleanup_file(input_path)
            return jsonify(create_error_response('PROCESSING_001', message)), 500
        
        print(f"✅ Step 1 complete: Image converted and saved to {output_path}")
        
        # Step 2: Remove background if requested
        if remove_bg:
            from PIL import Image
            
            print(f"🔧 Step 2: Applying background removal to converted image: {output_path}")
            
            # Optimize image size for faster background removal (max 2000px on longest side)
            original_size = None
            bg_removal_input = output_path
            try:
                img_for_bg = Image.open(output_path)
                original_size = img_for_bg.size
                max_dimension = 2000
                
                if max(img_for_bg.size) > max_dimension:
                    # Calculate new size maintaining aspect ratio
                    ratio = max_dimension / max(img_for_bg.size)
                    new_size = (int(img_for_bg.width * ratio), int(img_for_bg.height * ratio))
                    print(f"📐 Resizing image from {original_size} to {new_size} for faster background removal")
                    img_for_bg = img_for_bg.resize(new_size, Image.Resampling.LANCZOS)
                    # Save optimized version temporarily
                    optimized_path = output_path.rsplit('.', 1)[0] + '_optimized.png'
                    img_for_bg.save(optimized_path, format='PNG', optimize=True)
                    bg_removal_input = optimized_path
                    img_for_bg.close()
                    print(f"✅ Image optimized for background removal: {optimized_path}")
                else:
                    img_for_bg.close()
            except Exception as e:
                print(f"⚠️ Could not optimize image for background removal: {e}")
                if 'img_for_bg' in locals():
                    try:
                        img_for_bg.close()
                    except:
                        pass
            
            # Check if output file exists
            if not os.path.exists(output_path):
                print(f"❌ Converted image not found at {output_path}")
                return jsonify(create_error_response('PROCESSING_002', 'Converted image not found for background removal')), 500
            
            # Verify the file is readable
            try:
                test_img = Image.open(output_path)
                print(f"✅ Converted image verified: {test_img.size}, mode: {test_img.mode}, format: {test_img.format}")
                test_img.close()
            except Exception as e:
                print(f"❌ Cannot read converted image: {e}")
                return jsonify(create_error_response('PROCESSING_002', f'Cannot read converted image: {e}')), 500
            
            # Call background removal API with timeout protection
            bg_removal_start = time.time()
            max_bg_removal_time = 90  # Maximum 90 seconds for background removal
            
            try:
                api_result_path = remove_background(bg_removal_input)
                bg_removal_time = time.time() - bg_removal_start
                
                if bg_removal_time > max_bg_removal_time:
                    print(f"⚠️ Background removal took {bg_removal_time:.1f}s (exceeded {max_bg_removal_time}s limit)")
                
                if api_result_path and os.path.exists(api_result_path):
                    print(f"✅ Background removal API returned result: {api_result_path} (took {bg_removal_time:.1f}s)")
                    try:
                        # Load the background-removed image
                        converted_image = Image.open(api_result_path)
                        print(f"✅ Loaded background-removed image: {converted_image.size}, mode: {converted_image.mode}")
                        
                        # Ensure it's RGBA for transparency
                        if converted_image.mode != 'RGBA':
                            print(f"⚠️ Converting to RGBA mode (was {converted_image.mode})")
                            converted_image = converted_image.convert('RGBA')
                        
                        # If we optimized the image, we need to upscale the result back
                        if bg_removal_input != output_path and original_size and original_size != converted_image.size:
                            print(f"📐 Upscaling background-removed image from {converted_image.size} back to {original_size}")
                            converted_image = converted_image.resize(original_size, Image.Resampling.LANCZOS)
                        
                        # Save as PNG to preserve transparency
                        converted_image.save(output_path, format='PNG', optimize=True)
                        print(f"✅ Background-removed image saved as PNG: {output_path}")
                        
                        # Clean up temporary files
                        cleanup_file(api_result_path)
                        if bg_removal_input != output_path:
                            cleanup_file(bg_removal_input)
                    except Exception as e:
                        print(f"❌ Error saving background-removed image: {e}")
                        return jsonify(create_error_response('PROCESSING_002', f'Error saving background-removed image: {e}')), 500
                else:
                    print(f"❌ Background removal API failed - no result returned (took {bg_removal_time:.1f}s)")
                    print(f"   api_result_path: {api_result_path}")
                    print(f"   exists: {os.path.exists(api_result_path) if api_result_path else 'N/A'}")
                    # Don't fail the request, just return the converted image without background removal
                    print(f"⚠️ Returning converted image without background removal")
            except Exception as e:
                print(f"❌ Exception during background removal: {e}")
                # Don't fail the request, just return the converted image without background removal
                print(f"⚠️ Returning converted image without background removal due to error")
        
        # Clean up input file
        cleanup_file(input_path)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = format_processing_time(start_time, end_time)
        
        # Get file size
        file_size = format_file_size(os.path.getsize(output_path))
        
        # Create download URL
        base_url = request.url_root.rstrip('/')
        download_url = create_download_url(output_filename, base_url)
        
        # Schedule cleanup of output file (after 24 hours)
        cleanup_file(output_path, delay_seconds=24 * 3600)
        
        # Get image metadata
        image_info = get_image_info(output_path)
        
        return jsonify(create_success_response(
            message="Image processed successfully",
            output_url=download_url,
            processing_time=processing_time,
            file_size=file_size,
            output_filename=output_filename,
            metadata={
                'image_info': image_info,
                'prompt_used': prompt,
                'background_removed': remove_bg
            }
        ))
        
    except Exception as e:
        # Clean up any temporary files
        try:
            if 'input_path' in locals() and os.path.exists(input_path):
                cleanup_file(input_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                cleanup_file(output_path)
        except:
            pass
        
        return jsonify(create_error_response('SERVICE_003', str(e))), 500

@api_bp.route('/download/<filename>', methods=['GET'])
@cross_origin()
def download_file(filename):
    """
    Download processed image file
    """
    try:
        # Security: ensure filename is safe
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            return jsonify(create_error_response('VALIDATION_004', 'Invalid filename')), 400
        
        # Check if file exists
        file_path = os.path.join(current_app.config['OUTPUT_FOLDER'], safe_filename)
        if not os.path.exists(file_path):
            return jsonify(create_error_response('FILE_001', 'File not found')), 404
        
        # Return file
        return send_file(file_path, as_attachment=True, download_name=safe_filename)
        
    except Exception as e:
        return jsonify(create_error_response('FILE_002', str(e))), 500

@api_bp.route('/status', methods=['GET'])
@cross_origin()
@require_api_key
def get_status():
    """
    Get API status and usage information
    """
    try:
        api_key = request.headers.get('X-API-Key') or request.form.get('api_key')
        key_info = get_api_key_info(api_key)
        
        return jsonify({
            'success': True,
            'api_status': 'operational',
            'api_key_info': key_info,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify(create_error_response('SERVICE_003', str(e))), 500

@api_bp.route('/generate-character', methods=['POST'])
@cross_origin()
@require_api_key
def generate_character():
    """
    Generate a character from a selfie with identity preservation and composite onto background.
    This endpoint uses Google AI Studio to generate characters and avoids 3rd party background removal services.
    
    Request parameters:
    - selfie: Image file (selfie/reference photo)
    - character_prompt: Description of the character transformation
    - background: Background image file (optional, can use background_url instead)
    - background_url: URL of background image (optional, can use background file instead)
    - position: Position for compositing ('center', 'bottom', 'top', etc.) - default: 'center'
    - scale: Scale factor for character (0.5 to 2.0) - default: 1.0
    - upload_background_to_s3: Whether to upload background to S3 first (default: true)
    """
    start_time = time.time()
    
    try:
        # Validate required parameters
        character_prompt = request.form.get('character_prompt', '').strip()
        if not character_prompt:
            return jsonify(create_error_response('VALIDATION_001', 'character_prompt is required')), 400
        
        # Get selfie image
        selfie_file = request.files.get('selfie')
        if not selfie_file or not selfie_file.filename:
            return jsonify(create_error_response('VALIDATION_001', 'selfie image file is required')), 400
        
        # Validate selfie file
        is_valid, error_msg = validate_image_file(selfie_file)
        if not is_valid:
            return jsonify(create_error_response('VALIDATION_002', f'Selfie: {error_msg}')), 400
        
        # Get background (file or URL)
        background_file = request.files.get('background')
        background_url = request.form.get('background_url', '').strip()
        
        if not background_file and not background_url:
            return jsonify(create_error_response('VALIDATION_001', 'Either background file or background_url is required')), 400
        
        if background_file and background_url:
            return jsonify(create_error_response('VALIDATION_001', 'Provide either background file or background_url, not both')), 400
        
        # Get optional parameters
        position = request.form.get('position', 'center').strip()
        scale = float(request.form.get('scale', '1.0'))
        upload_background_to_s3 = request.form.get('upload_background_to_s3', 'true').lower() == 'true'
        
        # Validate scale
        if scale < 0.1 or scale > 3.0:
            return jsonify(create_error_response('VALIDATION_001', 'scale must be between 0.1 and 3.0')), 400
        
        print(f"📋 Request parameters - character_prompt: {character_prompt[:50]}..., position: {position}, scale: {scale}")
        
        # Create upload and output directories
        upload_dir = current_app.config['UPLOAD_FOLDER']
        output_dir = current_app.config['OUTPUT_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save selfie
        selfie_filename = generate_unique_filename(selfie_file.filename, 'selfie')
        selfie_path = os.path.join(upload_dir, selfie_filename)
        selfie_file.save(selfie_path)
        print(f"✅ Selfie saved: {selfie_path}")
        
        # Handle background
        background_path = None
        background_s3_url = None
        
        if background_file:
            # Validate background file
            is_valid, error_msg = validate_image_file(background_file)
            if not is_valid:
                cleanup_file(selfie_path)
                return jsonify(create_error_response('VALIDATION_002', f'Background: {error_msg}')), 400
            
            # Save background file
            background_filename = generate_unique_filename(background_file.filename, 'background')
            background_path = os.path.join(upload_dir, background_filename)
            background_file.save(background_path)
            print(f"✅ Background saved: {background_path}")
            
            # Upload to S3 if requested
            if upload_background_to_s3:
                print(f"📤 Uploading background to S3...")
                background_s3_url = upload_image_to_s3(background_path)
                if background_s3_url:
                    print(f"✅ Background uploaded to S3: {background_s3_url}")
                else:
                    print(f"⚠️ S3 upload failed, using local file")
        else:
            # Download background from URL
            print(f"📥 Downloading background from URL: {background_url}")
            background_path = download_image_from_url(background_url, upload_dir)
            if not background_path:
                cleanup_file(selfie_path)
                return jsonify(create_error_response('VALIDATION_002', 'Failed to download background from URL')), 400
            print(f"✅ Background downloaded: {background_path}")
        
        # Get background dimensions for context
        background_dimensions = None
        if background_path and os.path.exists(background_path):
            try:
                from PIL import Image
                with Image.open(background_path) as bg_img:
                    background_dimensions = {'width': bg_img.width, 'height': bg_img.height}
                    print(f"📐 Background dimensions: {bg_img.width}x{bg_img.height}")
            except Exception as e:
                print(f"⚠️ Could not read background dimensions: {e}")
        
        # Get canvas size and DPI from form (optional)
        canvas_size = request.form.get('canvas_size', '').strip() or None
        dpi = int(request.form.get('dpi', '300'))
        
        # Step 1: Generate character with identity preservation
        print(f"🎨 Step 1: Generating character with identity preservation...")
        character_filename = generate_unique_filename(f"character_{selfie_filename}.png", 'output')
        character_path = os.path.join(output_dir, character_filename)
        
        success, message = generate_character_with_identity(
            selfie_path=selfie_path,
            character_prompt=character_prompt,
            output_path=character_path,
            white_background=True,  # Use white background for easier compositing
            position=position,
            scale=scale,
            background_dimensions=background_dimensions,
            canvas_size=canvas_size,
            dpi=dpi
        )
        
        if not success:
            cleanup_file(selfie_path)
            if background_path:
                cleanup_file(background_path)
            return jsonify(create_error_response('PROCESSING_001', f'Character generation failed: {message}')), 500
        
        print(f"✅ Step 1 complete: Character generated at {character_path}")
        
        # Step 2: Composite character onto background
        print(f"🎨 Step 2: Compositing character onto background...")
        output_filename = generate_unique_filename(f"composited_{selfie_filename}.png", 'output')
        output_path = os.path.join(output_dir, output_filename)
        
        success, message = composite_character_on_background_manual(
            character_path=character_path,
            background_path=background_path,
            output_path=output_path,
            position=position,
            scale=scale,
            add_shadow=True,
            canvas_size=canvas_size,
            dpi=dpi
        )
        
        if not success:
            cleanup_file(selfie_path)
            cleanup_file(character_path)
            if background_path:
                cleanup_file(background_path)
            return jsonify(create_error_response('PROCESSING_002', f'Compositing failed: {message}')), 500
        
        print(f"✅ Step 2 complete: Character composited at {output_path}")
        
        # Clean up intermediate files
        cleanup_file(selfie_path)
        cleanup_file(character_path)
        if background_path and not upload_background_to_s3:
            # Keep background if uploaded to S3 (might be reused)
            cleanup_file(background_path)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = format_processing_time(start_time, end_time)
        
        # Get file size
        file_size = format_file_size(os.path.getsize(output_path))
        
        # Create download URL
        base_url = request.url_root.rstrip('/')
        download_url = create_download_url(output_filename, base_url)
        
        # Schedule cleanup of output file (after 24 hours)
        cleanup_file(output_path, delay_seconds=24 * 3600)
        
        # Get image metadata
        image_info = get_image_info(output_path)
        
        return jsonify(create_success_response(
            message="Character generated and composited successfully",
            output_url=download_url,
            processing_time=processing_time,
            file_size=file_size,
            output_filename=output_filename,
            metadata={
                'image_info': image_info,
                'character_prompt': character_prompt,
                'position': position,
                'scale': scale,
                'background_s3_url': background_s3_url
            }
        ))
        
    except Exception as e:
        # Clean up any temporary files
        try:
            if 'selfie_path' in locals() and os.path.exists(selfie_path):
                cleanup_file(selfie_path)
            if 'background_path' in locals() and os.path.exists(background_path):
                cleanup_file(background_path)
            if 'character_path' in locals() and os.path.exists(character_path):
                cleanup_file(character_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                cleanup_file(output_path)
        except:
            pass
        
        return jsonify(create_error_response('SERVICE_003', str(e))), 500
