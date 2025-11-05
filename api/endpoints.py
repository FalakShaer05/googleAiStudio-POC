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
    get_image_info, create_download_url, remove_background_with_mosida_api,
    download_image_from_url
)

# Import existing functions from main app (lazy import to avoid circular dependency)
def get_app_functions():
    """Get functions from main app module to avoid circular imports"""
    from app import (
        convert_image_to_image, composite_images, remove_white_background,
        remove_background_with_mosida_api,
        upscale_to_canvas_size, resize_to_canvas_size, enhance_prompt_for_white_background,
        resize_background_with_ai, get_print_dimensions
    )
    return {
        'convert_image_to_image': convert_image_to_image,
        'composite_images': composite_images,
        'remove_white_background': remove_white_background,
        'remove_background_with_mosida_api': remove_background_with_mosida_api,
        'upscale_to_canvas_size': upscale_to_canvas_size,
        'resize_to_canvas_size': resize_to_canvas_size,
        'enhance_prompt_for_white_background': enhance_prompt_for_white_background,
        'resize_background_with_ai': resize_background_with_ai,
        'get_print_dimensions': get_print_dimensions
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
    Converts image to caricature and optionally merges with background
    """
    start_time = time.time()
    
    try:
        # Validate required parameters
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            return jsonify(create_error_response('VALIDATION_001', 'Prompt is required')), 400
        
        # Check if image is provided as file or URL
        image_file = request.files.get('image')
        image_url = request.form.get('image_url', '').strip()
        
        if not image_file and not image_url:
            return jsonify(create_error_response('VALIDATION_001', 'Either image file or image_url is required')), 400
        
        if image_file and image_url:
            return jsonify(create_error_response('VALIDATION_001', 'Provide either image file or image_url, not both')), 400
        
        # Get other parameters
        background_file = request.files.get('background')
        background_url = request.form.get('background_url', '').strip()
        
        # Get optional parameters
        position = request.form.get('position', 'center')
        scale = float(request.form.get('scale', '1.0'))
        opacity = float(request.form.get('opacity', '1.0'))
        canvas_size = request.form.get('canvas_size')
        
        # Validate parameters
        if scale < 0.1 or scale > 1.5:
            return jsonify(create_error_response('VALIDATION_004', 'Scale must be between 0.1 and 1.5')), 400
        
        if opacity < 0.0 or opacity > 1.0:
            return jsonify(create_error_response('VALIDATION_004', 'Opacity must be between 0.0 and 1.0')), 400
        
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
        
        # Handle background image (file or URL)
        background_path = None
        if background_file and background_file.filename:
            # Validate uploaded background file
            is_valid, error_msg = validate_image_file(background_file)
            if not is_valid:
                return jsonify(create_error_response('VALIDATION_002', f'Background {error_msg}')), 400
            
            # Save uploaded background file
            background_filename = generate_unique_filename(background_file.filename, 'bg')
            background_path = os.path.join(upload_dir, background_filename)
            background_file.save(background_path)
        elif background_url:
            # Download background from URL
            background_path = download_image_from_url(background_url, upload_dir)
            if not background_path:
                return jsonify(create_error_response('VALIDATION_002', 'Failed to download background image from URL. Please check the URL and try again.')), 400
        
        
        # Get app functions to avoid circular import
        app_funcs = get_app_functions()
        
        # Enhance prompt for better background removal
        enhanced_prompt = app_funcs['enhance_prompt_for_white_background'](prompt)
        
        # Generate output filename (ensure input_filename defined for both file and URL flows)
        if 'input_filename' not in locals() or not input_filename:
            input_filename = os.path.basename(input_path)
        output_filename = generate_unique_filename(f"processed_{input_filename}", 'output')
        output_path = os.path.join(output_dir, output_filename)
        
        # Step 1: Convert image to caricature
        success, message = app_funcs['convert_image_to_image'](
            input_image_path=input_path,
            prompt=enhanced_prompt,
            output_path=output_path,
            upscale_before=True,
            scale_factor=2,
            canvas_size=canvas_size,
            dpi=300,
            reference_background_path=background_path,
            enable_background_compositing=False  # We'll handle compositing separately
        )
        
        if not success:
            # Clean up input files
            cleanup_file(input_path)
            if background_path:
                cleanup_file(background_path)
            return jsonify(create_error_response('PROCESSING_001', message)), 500
        
        # Step 2: Handle background merging if background provided
        if background_path and os.path.exists(background_path):
            # Load the converted image
            from PIL import Image
            converted_image = Image.open(output_path)
            background_image = Image.open(background_path)
            
            # Resize background to canvas size if specified (using AI to prevent distortion)
            if canvas_size:
                target_width, target_height = app_funcs['get_print_dimensions'](canvas_size, 300)
                print(f"📐 Resizing background to canvas size: {canvas_size} ({target_width}x{target_height}px)")
                background_image = app_funcs['resize_background_with_ai'](background_image, target_width, target_height)
                print(f"✅ Background resized to: {background_image.size}")
            
            # Apply professional background removal using Mosida API (with fallbacks)
            print(f"🔧 DEBUG: Applying Remove.bg API for background removal...")
            api_result_path = app_funcs['remove_background_with_mosida_api'](output_path)
            
            if api_result_path and os.path.exists(api_result_path):
                print(f"✅ Remove.bg background removal successful")
                converted_image = Image.open(api_result_path)
            else:
                print(f"❌ Remove.bg failed, trying fallback...")
                # Try fallback API call
                api_result_path = app_funcs['remove_background_with_mosida_api'](output_path)
                if api_result_path and os.path.exists(api_result_path):
                    converted_image = Image.open(api_result_path)
                    print(f"✅ Fallback API background removal successful")
                else:
                    print(f"❌ All APIs failed - trying local background removal...")
                    # Try local background removal as last resort
                    converted_image = Image.open(output_path)
                    
                    # Try both black and white background removal
                    try:
                        from app import remove_black_background
                        converted_image = remove_black_background(converted_image)
                        print(f"✅ Local black background removal applied")
                    except:
                        pass
                    
                    # Also try white background removal
                    converted_image = app_funcs['remove_white_background'](converted_image)
                    print(f"✅ Local white background removal applied")
            
            # Composite images
            final_image = app_funcs['composite_images'](
                converted_image,
                background_image,
                position=position,
                scale=scale,
                opacity=opacity,
                add_shadow=True
            )
            
            # Note: If canvas_size was specified, background was already resized to canvas size
            # So the final image should already match canvas dimensions
            # Only resize if there's a mismatch (shouldn't happen, but safety check)
            if canvas_size:
                target_width, target_height = app_funcs['get_print_dimensions'](canvas_size, 300)
                if final_image.size != (target_width, target_height):
                    print(f"⚠️ Final image size ({final_image.size}) doesn't match canvas ({target_width}x{target_height}), adjusting...")
                    final_image = app_funcs['resize_to_canvas_size'](final_image, canvas_size, 300)
                else:
                    print(f"✅ Final image already matches canvas size: {final_image.size}")
            
            # Save final merged image
            final_filename = f"merged_{output_filename}"
            final_path = os.path.join(output_dir, final_filename)
            final_image.save(final_path, quality=95, optimize=True)
            
            # Update output filename and path
            output_filename = final_filename
            output_path = final_path
            
            # Clean up temporary files
            if api_result_path and os.path.exists(api_result_path):
                cleanup_file(api_result_path)
            cleanup_file(background_path)
        
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
                'background_merged': background_path is not None,
                'position': position,
                'scale': scale,
                'opacity': opacity,
                'canvas_size': canvas_size
            }
        ))
        
    except Exception as e:
        # Clean up any temporary files
        try:
            if 'input_path' in locals() and os.path.exists(input_path):
                cleanup_file(input_path)
            if 'background_path' in locals() and background_path and os.path.exists(background_path):
                cleanup_file(background_path)
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
