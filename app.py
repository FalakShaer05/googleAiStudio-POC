#!/usr/bin/env python3
"""
Web Application for Image-to-Image Conversion using Google Gemini API
A user-friendly web interface with file upload and text prompts.
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google import genai
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
import uuid
import time

# Load environment variables before importing modules that rely on them
load_dotenv()

# Import API blueprint
from api import api_bp
from api.utils import remove_background

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-API-Key"],
        "supports_credentials": True
    }
})

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Register API blueprint
app.register_blueprint(api_bp)

# Initialize Gemini client
def setup_client():
    """Initialize the Gemini client"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None

client = setup_client()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upscale_image(image, scale_factor=2, method='LANCZOS'):
    """
    Upscale an image using various methods
    
    Args:
        image: PIL Image object
        scale_factor: Factor to scale the image (default: 2x)
        method: Upscaling method ('LANCZOS', 'BICUBIC', 'BILINEAR')
    
    Returns:
        PIL Image object (upscaled)
    """
    try:
        # Get original dimensions
        original_width, original_height = image.size
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Choose resampling method
        resample_methods = {
            'LANCZOS': Image.Resampling.LANCZOS,
            'BICUBIC': Image.Resampling.BICUBIC,
            'BILINEAR': Image.Resampling.BILINEAR
        }
        
        resample = resample_methods.get(method.upper(), Image.Resampling.LANCZOS)
        
        # Upscale the image
        upscaled_image = image.resize((new_width, new_height), resample=resample)
        
        # Enhance the upscaled image
        # Sharpen slightly to compensate for blur from upscaling
        enhancer = ImageEnhance.Sharpness(upscaled_image)
        upscaled_image = enhancer.enhance(1.2)
        
        return upscaled_image
        
    except Exception as e:
        print(f"Error upscaling image: {e}")
        return image  # Return original if upscaling fails

def detect_aspect_ratio(image):
    """
    Detect the aspect ratio of an image and categorize it
    
    Args:
        image: PIL Image object
    
    Returns:
        dict: Aspect ratio information
    """
    try:
        width, height = image.size
        ratio = width / height
        
        # Categorize aspect ratio
        if 0.9 <= ratio <= 1.1:
            aspect_type = 'square'
            description = 'Square'
        elif ratio < 0.9:
            aspect_type = 'portrait'
            description = f'Portrait ({width}:{height})'
        elif ratio > 1.1:
            aspect_type = 'landscape'
            description = f'Landscape ({width}:{height})'
        else:
            aspect_type = 'unknown'
            description = f'Custom ({width}:{height})'
        
        return {
            'type': aspect_type,
            'description': description,
            'ratio': ratio,
            'dimensions': (width, height)
        }
    except Exception as e:
        print(f"Error detecting aspect ratio: {e}")
        return {
            'type': 'unknown',
            'description': 'Unknown',
            'ratio': 1.0,
            'dimensions': (0, 0)
        }

def get_smart_canvas_options(aspect_ratio_info):
    """
    Get smart canvas options based on image aspect ratio
    
    Args:
        aspect_ratio_info: Dict from detect_aspect_ratio()
    
    Returns:
        list: Available canvas options for this image type
    """
    aspect_type = aspect_ratio_info.get('type', 'unknown')
    
    # Smart canvas presets based on aspect ratio
    canvas_options = {
        'square': [
            {'value': '6x6', 'label': '6" × 6" (1800×1800px @ 300 DPI)', 'recommended': True},
            {'value': '8x8', 'label': '8" × 8" (2400×2400px @ 300 DPI)', 'recommended': False},
            {'value': '12x12', 'label': '12" × 12" (3600×3600px @ 300 DPI)', 'recommended': False}
        ],
        'portrait': [
            {'value': '4x6', 'label': '4" × 6" (1200×1800px @ 300 DPI)', 'recommended': True},
            {'value': '8x12', 'label': '8" × 12" (2400×3600px @ 300 DPI)', 'recommended': False},
            {'value': '13x19', 'label': '13" × 19" (3900×5700px @ 300 DPI)', 'recommended': False},
            {'value': '24x36', 'label': '24" × 36" (7200×10800px @ 300 DPI)', 'recommended': False}
        ],
        'landscape': [
            {'value': '6x4', 'label': '6" × 4" (1800×1200px @ 300 DPI)', 'recommended': True},
            {'value': '12x8', 'label': '12" × 8" (3600×2400px @ 300 DPI)', 'recommended': False},
            {'value': '19x13', 'label': '19" × 13" (5700×3900px @ 300 DPI)', 'recommended': False},
            {'value': '36x24', 'label': '36" × 24" (10800×7200px @ 300 DPI)', 'recommended': False}
        ],
        'unknown': [
            {'value': '8x12', 'label': '8" × 12" (2400×3600px @ 300 DPI)', 'recommended': True},
            {'value': '12x8', 'label': '12" × 8" (3600×2400px @ 300 DPI)', 'recommended': False},
            {'value': '8x8', 'label': '8" × 8" (2400×2400px @ 300 DPI)', 'recommended': False}
        ]
    }
    
    return canvas_options.get(aspect_type, canvas_options['unknown'])

def resize_to_canvas_size(image, canvas_size, dpi=300):
    """
    Resize image to specific canvas dimensions while maintaining composition
    IMPORTANT: This function will NOT crop the image - it will scale to fit or scale up to fill
    
    Args:
        image: PIL Image object
        canvas_size: Canvas size string (e.g., '4x6', '8x10', '11x14')
        dpi: Dots per inch (default: 300 for print quality)
    
    Returns:
        PIL Image object resized to canvas size (no cropping - full image visible)
    """
    try:
        # Get target dimensions
        target_width, target_height = get_print_dimensions(canvas_size, dpi)
        
        print(f"Resizing image to canvas size: {canvas_size} ({target_width}x{target_height}px @ {dpi} DPI)")
        
        # Get current image dimensions
        current_width, current_height = image.size
        print(f"Current image size: {current_width}x{current_height}")
        
        # Check if image is already larger than target - if so, scale down to fit
        # If image is smaller, scale up to fill canvas (but don't crop)
        scale_x = target_width / current_width
        scale_y = target_height / current_height
        
        if current_width > target_width or current_height > target_height:
            # Image is larger - scale down to fit (use smaller scale to ensure it fits)
            scale = min(scale_x, scale_y)
            print(f"Image is larger than canvas - scaling DOWN by {scale:.3f}")
        else:
            # Image is smaller - scale up to fill canvas (use larger scale to fill)
            # But ensure we don't exceed target dimensions
            scale = min(scale_x, scale_y)  # Use smaller to ensure it fits
            print(f"Image is smaller than canvas - scaling UP by {scale:.3f}")
        
        # Calculate new dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        print(f"New dimensions: {new_width}x{new_height}")
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # If resized image matches target exactly, return it
        if new_width == target_width and new_height == target_height:
            print(f"Image resized to exact canvas size")
            return resized_image
        
        # Create a new canvas with the target size and paste the resized image
        canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        
        # Center the resized image on the canvas (this adds white space, doesn't crop)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        canvas.paste(resized_image, (x_offset, y_offset))
        
        print(f"Final canvas size: {canvas.size} (image centered, no cropping)")
        return canvas
        
    except Exception as e:
        print(f"Error resizing to canvas size: {e}")
        return image

def get_print_dimensions(canvas_size, dpi=300):
    """
    Get print dimensions for standard canvas sizes
    
    Args:
        canvas_size: Canvas size string (e.g., '4x6', '8x10', '11x14')
        dpi: Dots per inch (default: 300 for print quality)
    
    Returns:
        tuple: (width, height) in pixels
    """
    canvas_sizes = {
        '4x6': (4, 6),
        '5x7': (5, 7),
        '8x10': (8, 10),
        '11x14': (11, 14),
        '16x20': (16, 20),
        '18x24': (18, 24),
        '20x30': (20, 30),
        # Add smart canvas options
        '6x6': (6, 6),
        '8x8': (8, 8),
        '12x12': (12, 12),
        '8x12': (8, 12),
        '13x19': (13, 19),
        '24x36': (24, 36),
        '6x4': (6, 4),
        '12x8': (12, 8),
        '19x13': (19, 13),
        '36x24': (36, 24),
    }
    
    if canvas_size in canvas_sizes:
        width_inches, height_inches = canvas_sizes[canvas_size]
        width_pixels = int(width_inches * dpi)
        height_pixels = int(height_inches * dpi)
        return (width_pixels, height_pixels)
    else:
        # Default to 8x10 if size not recognized
        return (2400, 3000)

def upscale_to_canvas_size(image, canvas_size, dpi=300):
    """
    Upscale an image to specific canvas dimensions for printing
    
    Args:
        image: PIL Image object
        canvas_size: Tuple of (width_inches, height_inches) or string like "4x6"
        dpi: Dots per inch for print quality (default: 300)
    
    Returns:
        PIL Image object (upscaled to canvas size)
    """
    try:
        # Enhanced canvas size presets with both orientations
        canvas_presets = {
            # Square options
            '6x6': (6, 6),
            '8x8': (8, 8),
            '12x12': (12, 12),
            
            # Portrait options
            '4x6': (4, 6),
            '8x12': (8, 12),
            '13x19': (13, 19),
            '24x36': (24, 36),
            
            # Landscape options
            '6x4': (6, 4),
            '12x8': (12, 8),
            '19x13': (19, 13),
            '36x24': (36, 24),
            
            'custom': canvas_size if isinstance(canvas_size, tuple) else (8, 12)
        }
        
        if isinstance(canvas_size, str) and canvas_size in canvas_presets:
            width_inches, height_inches = canvas_presets[canvas_size]
        elif isinstance(canvas_size, tuple):
            width_inches, height_inches = canvas_size
        else:
            width_inches, height_inches = 8, 12  # Default to 8x12
        
        # Calculate target dimensions in pixels
        target_width = int(width_inches * dpi)
        target_height = int(height_inches * dpi)
        
        print(f"Upscaling to canvas size: {width_inches}\" x {height_inches}\" ({target_width}x{target_height}px @ {dpi} DPI)")
        
        # Get current image dimensions
        current_width, current_height = image.size
        
        # Calculate scale factors
        scale_x = target_width / current_width
        scale_y = target_height / current_height
        
        # Use the larger scale factor to ensure the image fills the canvas
        scale_factor = max(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(current_width * scale_factor)
        new_height = int(current_height * scale_factor)
        
        # Upscale the image
        upscaled_image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        
        # If the upscaled image is larger than the target canvas, crop it to fit
        if new_width > target_width or new_height > target_height:
            # Calculate crop box (center crop)
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            
            upscaled_image = upscaled_image.crop((left, top, right, bottom))
        
        # If the upscaled image is smaller than the target canvas, pad it
        elif new_width < target_width or new_height < target_height:
            # Create a new image with the target canvas size
            canvas_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            
            # Calculate position to center the image
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            # Paste the upscaled image onto the canvas
            canvas_image.paste(upscaled_image, (paste_x, paste_y))
            upscaled_image = canvas_image
        
        # Enhance the final image
        enhancer = ImageEnhance.Sharpness(upscaled_image)
        upscaled_image = enhancer.enhance(1.1)
        
        return upscaled_image
        
    except Exception as e:
        print(f"Error upscaling to canvas size: {e}")
        return image  # Return original if upscaling fails

def remove_white_background(image, threshold=240):
    """
    Remove white background from an image using flood fill algorithm
    Only removes background, preserves white elements within the subject
    
    Args:
        image: PIL Image object
        threshold: Threshold for white detection (0-255, default: 240)
    
    Returns:
        PIL Image object with transparent background
    """
    try:
        import numpy as np
        from PIL import Image
        
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to numpy array for easier processing
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Create mask for background pixels
        mask = np.zeros((height, width), dtype=bool)
        visited = np.zeros((height, width), dtype=bool)
        
        # Flood fill from edges to find background
        from collections import deque
        queue = deque()
        
        # Add edge pixels to queue
        for x in range(width):
            queue.append((x, 0))
            queue.append((x, height - 1))
        for y in range(height):
            queue.append((0, y))
            queue.append((width - 1, y))
        
        # Flood fill algorithm
        while queue:
            x, y = queue.popleft()
            
            if x < 0 or x >= width or y < 0 or y >= height or visited[y, x]:
                continue
            
            r, g, b = img_array[y, x, :3]
            
            # Check if pixel is white or near white (background)
            if r > threshold and g > threshold and b > threshold:
                mask[y, x] = True
                visited[y, x] = True
                
                # Add neighbors to queue
                queue.append((x + 1, y))
                queue.append((x - 1, y))
                queue.append((x, y + 1))
                queue.append((x, y - 1))
        
        # Apply mask to make background transparent
        img_array[mask, 3] = 0  # Set alpha to 0 for background pixels
        
        # Convert back to PIL Image
        return Image.fromarray(img_array, 'RGBA')
        
    except ImportError:
        # Fallback to simple method if numpy is not available
        print("Numpy not available, using simple background removal")
        return remove_white_background_simple(image, threshold)
    except Exception as e:
        print(f"Error removing white background: {e}")
        return image  # Return original if processing fails

def remove_white_background_simple(image, threshold=240):
    """
    Simple white background removal using threshold
    """
    try:
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get image data
        data = image.getdata()
        
        # Create new data with transparency
        new_data = []
        for item in data:
            # If pixel is close to white, make it transparent
            if item[0] > threshold and item[1] > threshold and item[2] > threshold:
                new_data.append((0, 0, 0, 0))  # Transparent
            else:
                new_data.append(item)  # Keep original
        
        # Update image data
        image.putdata(new_data)
        return image
        
    except Exception as e:
        print(f"Error in simple white background removal: {e}")
        return image

def remove_black_background(image, threshold=15):
    """
    Remove black background from an image using flood fill algorithm
    Only removes background, preserves black elements within the subject
    
    Args:
        image: PIL Image object
        threshold: Threshold for black detection (0-255, default: 15)
    
    Returns:
        PIL Image object with transparent background
    """
    try:
        import numpy as np
        from PIL import Image
        
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to numpy array for easier processing
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Create mask for background pixels
        mask = np.zeros((height, width), dtype=bool)
        visited = np.zeros((height, width), dtype=bool)
        
        # Flood fill from edges to find background
        from collections import deque
        queue = deque()
        
        # Add edge pixels to queue
        for x in range(width):
            queue.append((x, 0))
            queue.append((x, height - 1))
        for y in range(height):
            queue.append((0, y))
            queue.append((width - 1, y))
        
        # Flood fill algorithm
        while queue:
            x, y = queue.popleft()
            
            if x < 0 or x >= width or y < 0 or y >= height or visited[y, x]:
                continue
            
            r, g, b = img_array[y, x, :3]
            
            # Check if pixel is black or near black (background)
            if r < threshold and g < threshold and b < threshold:
                mask[y, x] = True
                visited[y, x] = True
                
                # Add neighbors to queue
                queue.append((x + 1, y))
                queue.append((x - 1, y))
                queue.append((x, y + 1))
                queue.append((x, y - 1))
        
        # Apply mask to make background transparent
        img_array[mask, 3] = 0  # Set alpha to 0 for background pixels
        
        # Convert back to PIL Image
        return Image.fromarray(img_array, 'RGBA')
        
    except ImportError:
        # Fallback to simple method if numpy is not available
        print("Numpy not available, using simple black background removal")
        return remove_black_background_simple(image, threshold)
    except Exception as e:
        print(f"Error removing black background: {e}")
        return image  # Return original if processing fails

def remove_black_background_simple(image, threshold=15):
    """
    Simple black background removal using threshold
    """
    try:
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get image data
        data = image.getdata()
        
        # Create new data with transparency
        new_data = []
        for item in data:
            # If pixel is close to black, make it transparent
            if item[0] < threshold and item[1] < threshold and item[2] < threshold:
                new_data.append((0, 0, 0, 0))  # Transparent
            else:
                new_data.append(item)  # Keep original
        
        # Update image data
        image.putdata(new_data)
        return image
        
    except Exception as e:
        print(f"Error in simple black background removal: {e}")
        return image


def add_drop_shadow(image, offset=(5, 5), blur_radius=10, shadow_color=(0, 0, 0, 100)):
    """
    Add a drop shadow to an image for more realistic composition
    
    Args:
        image: PIL Image object
        offset: Shadow offset (x, y) in pixels
        blur_radius: Blur radius for shadow
        shadow_color: Shadow color (R, G, B, A)
    
    Returns:
        PIL Image object with shadow
    """
    try:
        from PIL import ImageFilter, ImageDraw
        
        # Create shadow layer
        shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Get image mask (non-transparent areas)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Create shadow by drawing the image shape
        shadow_draw.bitmap((0, 0), image.split()[-1], fill=shadow_color)
        
        # Apply blur to shadow
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Create result canvas with shadow offset
        result_width = image.width + abs(offset[0]) + blur_radius * 2
        result_height = image.height + abs(offset[1]) + blur_radius * 2
        result = Image.new('RGBA', (result_width, result_height), (0, 0, 0, 0))
        
        # Paste shadow with offset
        shadow_x = max(0, offset[0]) + blur_radius
        shadow_y = max(0, offset[1]) + blur_radius
        result.paste(shadow, (shadow_x, shadow_y), shadow)
        
        # Paste original image
        image_x = max(0, -offset[0]) + blur_radius
        image_y = max(0, -offset[1]) + blur_radius
        result.paste(image, (image_x, image_y), image)
        
        return result
        
    except Exception as e:
        print(f"Error adding drop shadow: {e}")
        return image

def composite_images(foreground_image, background_image, position='bottom', scale=1.0, opacity=1.0, add_shadow=True):
    """
    Composite foreground image onto background image with position control
    
    Args:
        foreground_image: PIL Image object (converted image)
        background_image: PIL Image object (reference background)
        position: Position string ('center', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'top', 'bottom', 'left', 'right') or tuple (x, y)
        scale: Scale factor for foreground image (default: 1.0)
        opacity: Opacity of foreground image (0.0 to 1.0, default: 1.0)
    
    Returns:
        PIL Image object (composited result)
    """
    try:
        # Keep background at original size (don't resize to match foreground)
        # This matches the frontend preview behavior where background stays full size
        
        # NOTE: The converted image from Google AI Studio is ALWAYS a full-length character
        # No need for portrait detection or estimated missing height
        
        # Scale foreground image if needed
        if scale != 1.0:
            new_width = int(foreground_image.width * scale)
            new_height = int(foreground_image.height * scale)
            foreground_image = foreground_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Store original size before shadow (for positioning calculations)
        original_fg_height = foreground_image.height
        
        # Add drop shadow for realistic composition
        shadow_offset = None
        shadow_blur = None
        if add_shadow:
            print("Adding drop shadow for realistic composition...")
            shadow_offset = (int(5 * scale), int(5 * scale))  # Scale shadow with image
            shadow_blur = int(10 * scale)
            foreground_image = add_drop_shadow(foreground_image, offset=shadow_offset, blur_radius=shadow_blur)
        
        # Create a copy of background for compositing
        result_image = background_image.copy()
        bg_width, bg_height = result_image.size
        fg_width, fg_height = foreground_image.size
        
        # Calculate position
        if isinstance(position, tuple):
            x, y = position
        else:
            # Calculate position based on string
            # For bottom positioning: Place character few pixels up from bottom
            # The converted image is ALWAYS full-length
            if position == 'bottom' and add_shadow and shadow_offset:
                # Calculate where the original image sits within the composite (with shadow)
                image_y_in_composite = max(0, -shadow_offset[1]) + shadow_blur
                
                # Calculate the actual bottom of the character in the composite
                character_bottom_in_composite = image_y_in_composite + original_fg_height
                
                # Position few pixels up from bottom (adjustable offset)
                pixels_up_from_bottom = max(5, int(original_fg_height * 0.01))  # ~1% of height, min 5px
                
                # Calculate desired y position
                desired_y = bg_height - character_bottom_in_composite - pixels_up_from_bottom
                
                # Check if character is too large for background
                # If so, resize background to fit the character while maintaining aspect ratio
                if desired_y < 0 or fg_height > bg_height:
                    print(f"Character ({fg_height}px) is larger than background ({bg_height}px)")
                    
                    # Calculate minimum background height needed to fit character at bottom
                    min_bg_height_needed = fg_height + pixels_up_from_bottom + 50  # Add padding
                    
                    # Calculate new background dimensions maintaining aspect ratio
                    aspect_ratio = bg_width / bg_height
                    new_bg_height = min_bg_height_needed
                    new_bg_width = int(new_bg_height * aspect_ratio)
                    
                    # Resize background
                    result_image = background_image.resize((new_bg_width, new_bg_height), Image.Resampling.LANCZOS)
                    bg_width, bg_height = result_image.size
                    print(f"Background resized to {bg_width}x{bg_height} to fit character")
                    
                    # Recalculate y position with new background size
                    desired_y = bg_height - character_bottom_in_composite - pixels_up_from_bottom
                
                # Position so character's feet are pixels_up_from_bottom above bg bottom
                y = desired_y
                x = bg_width // 2 - fg_width // 2
                
                print(f"Bottom positioning: bg={bg_width}x{bg_height}, character={fg_width}x{fg_height}, character_bottom_in_composite={character_bottom_in_composite}, pixels_up={pixels_up_from_bottom}, y={y}")
            else:
                position_map = {
                    'center': (bg_width // 2 - fg_width // 2, bg_height // 2 - fg_height // 2),
                    'top-left': (0, 0),
                    'top-right': (bg_width - fg_width, 0),
                    'bottom-left': (0, bg_height - fg_height),
                    'bottom-right': (bg_width - fg_width, bg_height - fg_height),
                    'top': (bg_width // 2 - fg_width // 2, 0),
                    'bottom': (bg_width // 2 - fg_width // 2, bg_height - fg_height),
                    'left': (0, bg_height // 2 - fg_height // 2),
                    'right': (bg_width - fg_width, bg_height // 2 - fg_height // 2)
                }
                
                x, y = position_map.get(position, position_map['center'])
        
        # Ensure position is within bounds
        x = max(0, min(x, result_image.width - foreground_image.width))
        
        # Final bounds check - ensure character is always visible
        # For bottom positioning, we've already resized background if needed, so just verify
        if position == 'bottom' and add_shadow and shadow_offset:
            # Final safety check - ensure character fits
            if y < 0:
                # Still negative after resize - this shouldn't happen, but position at top
                y = 0
                print(f"WARNING: y still negative after resize, adjusting to 0")
            elif (y + foreground_image.height) > result_image.height:
                # Bottom extends beyond - adjust
                y = result_image.height - foreground_image.height
                print(f"Adjusted y to {y} to keep character within bounds")
            
            # Verify final position
            if y < 0 or (y + foreground_image.height) > result_image.height:
                print(f"ERROR: Character still doesn't fit! y={y}, fg_height={foreground_image.height}, bg_height={result_image.height}")
        else:
            # Standard bounds check for other positions
            y = max(0, min(y, result_image.height - foreground_image.height))
        
        # Ensure background is in RGB mode for proper compositing
        if result_image.mode != 'RGB':
            result_image = result_image.convert('RGB')
        
        # Ensure foreground has alpha channel for transparency
        if foreground_image.mode != 'RGBA':
            foreground_image = foreground_image.convert('RGBA')
        
        # Apply opacity if needed
        if opacity < 1.0:
            # Create a copy of foreground with opacity
            foreground_with_alpha = foreground_image.copy()
            
            # Create alpha mask
            alpha = foreground_with_alpha.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            foreground_with_alpha.putalpha(alpha)
            
            # Composite with alpha
            result_image.paste(foreground_with_alpha, (x, y), foreground_with_alpha)
        else:
            # Always use alpha channel for proper transparency handling
            result_image.paste(foreground_image, (x, y), foreground_image)
        
        return result_image
        
    except Exception as e:
        print(f"Error compositing images: {e}")
        return background_image  # Return background if compositing fails

def convert_image_to_image(input_image_path, prompt, output_path, upscale_before=True, scale_factor=2, canvas_size=None, dpi=300, reference_background_path=None, enable_background_compositing=False, position='bottom', scale=1.0, opacity=1.0):
    """
    Convert an image using text prompt with Gemini API and optional background compositing
    
    Args:
        input_image_path (str): Path to the input image
        prompt (str): Text description of the desired transformation
        output_path (str): Path where the generated image will be saved
        upscale_before (bool): Whether to upscale the image before conversion
        scale_factor (float): Factor to scale the image (default: 2x)
        canvas_size (str): Canvas size for post-conversion upscaling (e.g., '4x6', '8x12', '13x19', '24x36')
        dpi (int): DPI for print quality (default: 300)
        reference_background_path (str): Path to reference background image (optional)
        enable_background_compositing (bool): Whether to composite result onto background
        position (str): Position for compositing ('center', 'top-left', etc.)
        scale (float): Scale factor for composited image (default: 1.0)
        opacity (float): Opacity of composited image (0.0 to 1.0, default: 1.0)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    if not client:
        return False, "Gemini client not initialized"
    
    try:
        # Load the input image
        image = Image.open(input_image_path)
        original_size = image.size
        
        # Load reference background if provided
        reference_background = None
        if reference_background_path and os.path.exists(reference_background_path):
            print(f"Loading reference background: {reference_background_path}")
            reference_background = Image.open(reference_background_path)
            print(f"Reference background size: {reference_background.size}")
        
        # Upscale the image before conversion if requested
        # Skip if image is already large enough (saves processing time)
        skip_upscale = os.getenv('SKIP_PRE_UPSCALE', 'false').lower() == 'true'
        if upscale_before and scale_factor > 1 and not skip_upscale:
            # Check if image is already large (skip upscaling if > 1000px on either side)
            if image.width < 1000 and image.height < 1000:
                print(f"Upscaling image from {original_size} to {scale_factor}x size...")
                image = upscale_image(image, scale_factor)
                print(f"Upscaled to: {image.size}")
            else:
                print(f"⏭️ Skipping upscale - image already large enough: {image.size}")
        elif skip_upscale:
            print(f"⏭️ Skipping pre-upscale (SKIP_PRE_UPSCALE=true) for faster processing")
        
        # Prepare content for AI conversion
        contents = [prompt, image]
        if reference_background:
            # Add reference background for style context
            contents.append(reference_background)
            print("Using reference background for style context")
        
        # Generate content with AI
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=contents,
        )
        
        # Process the response
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(f"Generated text: {part.text}")
            elif part.inline_data is not None:
                # Save the generated image
                generated_image = Image.open(io.BytesIO(part.inline_data.data))
                
                # Background compositing if enabled
                print(f"Compositing check: enabled={enable_background_compositing}, has_ref={reference_background is not None}")
                if enable_background_compositing and reference_background:
                    print(f"Removing white background from generated image...")
                    # Remove white background from generated image
                    generated_image = remove_white_background(generated_image)
                    
                    print(f"Compositing generated image onto background at position: {position}")
                    generated_image = composite_images(
                        generated_image, 
                        reference_background, 
                        position=position, 
                        scale=scale, 
                        opacity=opacity
                    )
                    print("Compositing completed successfully")
                else:
                    print("Skipping compositing - not enabled or no reference background")
                
                # Post-conversion upscaling to canvas size if requested
                if canvas_size:
                    print(f"Post-conversion upscaling to canvas size: {canvas_size}")
                    generated_image = upscale_to_canvas_size(generated_image, canvas_size, dpi)
                
                # Save image - use PNG format if output path ends with .png
                if output_path.lower().endswith('.png'):
                    generated_image.save(output_path, format='PNG', optimize=True)
                else:
                    generated_image.save(output_path)
                return True, "Image converted successfully"
        
        return False, "No image was generated in the response"
        
    except Exception as e:
        return False, f"Error during conversion: {e}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/group-photo')
def group_photo():
    """Group photo processing page"""
    return render_template('group_photo.html')


@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze uploaded image and return aspect ratio info"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Load and analyze the image
        image = Image.open(file)
        aspect_ratio_info = detect_aspect_ratio(image)
        smart_canvas_options = get_smart_canvas_options(aspect_ratio_info)
        
        return jsonify({
            'success': True,
            'aspect_ratio': aspect_ratio_info,
            'canvas_options': smart_canvas_options
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and image conversion"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'}), 400
        
        # Get prompt from form
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Please provide a transformation prompt'}), 400
        
        # Get optional remove_bg parameter
        remove_bg = request.form.get('remove_bg', 'false').lower() == 'true'

        # Handle optional reference background
        background_file = request.files.get('reference_background')
        background_filename = None
        background_path = None
        background_preview_filename = None
        background_dimensions = None
        has_background = False

        if background_file and background_file.filename:
            if not allowed_file(background_file.filename):
                return jsonify({'error': 'Invalid background file type. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'}), 400

            has_background = True
            bg_secure = secure_filename(background_file.filename)
            background_filename = f"{uuid.uuid4()}_background_{bg_secure}"
            background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)
            background_file.save(background_path)

            try:
                with Image.open(background_path) as bg_img:
                    background_dimensions = {'width': bg_img.width, 'height': bg_img.height}
                    preview_bg = bg_img.copy()
                    preview_bg.thumbnail((800, 800), Image.Resampling.LANCZOS)
                    background_preview_filename = f"preview_background_{background_filename}"
                    preview_bg_path = os.path.join(app.config['OUTPUT_FOLDER'], background_preview_filename)
                    preview_bg = preview_bg.convert('RGB') if preview_bg.mode not in ('RGB', 'RGBA') else preview_bg
                    preview_bg.save(preview_bg_path, quality=85, optimize=True)
            except Exception as bg_error:
                print(f"⚠️ Failed to generate background preview: {bg_error}")
                background_preview_filename = None

        # Force background removal when background is provided
        if has_background:
            remove_bg = True
        
        # Get upscaling options from form (optional, for faster processing)
        upscale_before = request.form.get('upscale_before', 'true').lower() == 'true'
        scale_factor = float(request.form.get('scale_factor', '2'))
        
        # Validate scale factor
        if scale_factor < 1 or scale_factor > 4:
            return jsonify({'error': 'Scale factor must be between 1 and 4'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Generate output filename
        output_filename = f"converted_{unique_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Convert image using prompt
        success, message = convert_image_to_image(
            input_path, 
            prompt,  # Use prompt as-is
            output_path, 
            upscale_before, 
            scale_factor, 
            canvas_size=None,  # No canvas size processing
            dpi=300,
            reference_background_path=background_path if has_background else None,
            enable_background_compositing=False  # No compositing
        )
        
        if success:
            # Clean up input file
            os.remove(input_path)
            
            # Remove background if requested
            if remove_bg:
                print(f"🔧 Applying background removal...")
                api_result_path = remove_background(output_path)
                
                if api_result_path and os.path.exists(api_result_path):
                    print(f"✅ Background removal successful")
                    # Replace output with background-removed version
                    converted_image = Image.open(api_result_path)
                    
                    # If image has transparency (RGBA), save as PNG and update filename
                    if converted_image.mode == 'RGBA':
                        # Change output path to PNG to preserve transparency
                        output_path_png = output_path.rsplit('.', 1)[0] + '.png'
                        # Remove old file if it exists and is different
                        if output_path != output_path_png and os.path.exists(output_path):
                            os.remove(output_path)
                        output_path = output_path_png
                        output_filename = os.path.basename(output_path)
                        # Save as PNG to preserve transparency
                        converted_image.save(output_path, format='PNG', optimize=True)
                        print(f"✅ Background-removed image saved as PNG: {output_path}")
                    else:
                        # No transparency, can save as JPEG
                        converted_image.save(output_path, quality=95, optimize=True)
                    
                    # Clean up temporary file
                    os.remove(api_result_path)
                else:
                    print(f"⚠️ Background removal API failed, returning original converted image")
            
            # Save converted image preview for display
            converted_preview_filename = f"preview_converted_{output_filename}"
            converted_preview_path = os.path.join(app.config['OUTPUT_FOLDER'], converted_preview_filename)
            with Image.open(output_path) as converted_full:
                converted_width, converted_height = converted_full.size
                preview_converted = converted_full.copy()
                preview_width = min(800, preview_converted.width)
                preview_height = int(preview_converted.height * (preview_width / preview_converted.width))
                preview_converted = preview_converted.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
                
                # Save preview with correct format based on image mode
                if preview_converted.mode == 'RGBA':
                    # Change preview filename to PNG if original is PNG
                    if output_filename.endswith('.png'):
                        converted_preview_path = converted_preview_path.rsplit('.', 1)[0] + '.png'
                        converted_preview_filename = os.path.basename(converted_preview_path)
                    preview_converted.save(converted_preview_path, format='PNG', optimize=True)
                else:
                    preview_converted.save(converted_preview_path, quality=85, optimize=True)

            # Prepare preview dictionary
            preview_images = {
                'converted': converted_preview_filename,
                'final': converted_preview_filename
            }

            if background_preview_filename:
                preview_images['background'] = background_preview_filename

            composition_data = None
            if has_background and background_dimensions:
                composition_data = {
                    'enabled': True,
                    'converted_url': f"/outputs/{output_filename}",
                    'background_url': f"/uploads/{background_filename}",
                    'converted_width': converted_width,
                    'converted_height': converted_height,
                    'background_width': background_dimensions['width'],
                    'background_height': background_dimensions['height'],
                    'converted_preview': converted_preview_filename,
                    'background_preview': background_preview_filename
                }
            
            response_payload = {
                'success': True,
                'message': 'Image converted successfully!',
                'output_filename': output_filename,
                'background_removed': remove_bg,
                'preview_images': preview_images
            }

            if composition_data:
                response_payload['composition'] = composition_data

            return jsonify(response_payload)
        else:
            # Clean up input files on failure
            if os.path.exists(input_path):
                os.remove(input_path)
            
            return jsonify({'error': message}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/backgrounds')
def list_backgrounds():
    """List available professional backgrounds"""
    try:
        backgrounds_dir = os.path.join(app.root_path, 'backgrounds')
        categories = {}
        
        for category in os.listdir(backgrounds_dir):
            category_path = os.path.join(backgrounds_dir, category)
            if os.path.isdir(category_path) and category != '__pycache__':
                backgrounds = []
                for file in os.listdir(category_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        backgrounds.append({
                            'filename': file,
                            'path': f'/backgrounds/{category}/{file}',
                            'category': category
                        })
                if backgrounds:
                    categories[category] = backgrounds
        
        return jsonify({
            'success': True,
            'categories': categories
        })
        
    except Exception as e:
        return jsonify({'error': f'Error listing backgrounds: {str(e)}'}), 500

@app.route('/backgrounds/<category>/<filename>')
def serve_background(category, filename):
    """Serve background images"""
    try:
        background_path = os.path.join(app.root_path, 'backgrounds', category, filename)
        if os.path.exists(background_path):
            return send_file(background_path)
        else:
            return jsonify({'error': 'Background not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error serving background: {str(e)}'}), 500

@app.route('/preview-processed', methods=['POST'])
def preview_processed():
    """Get processed image for preview to match download exactly"""
    try:
        converted_filename = request.form.get('converted_filename')
        preserve_white_background = request.form.get('preserve_white_background', 'false').lower() == 'true'
        
        if not converted_filename:
            return jsonify({'error': 'No converted filename provided'}), 400
        
        # Find the converted image
        converted_path = None
        for file in os.listdir(app.config['OUTPUT_FOLDER']):
            if file.startswith('converted_') and converted_filename in file:
                converted_path = os.path.join(app.config['OUTPUT_FOLDER'], file)
                break
        
        if not converted_path or not os.path.exists(converted_path):
            return jsonify({'error': 'Converted image not found'}), 404
        
        # Load the converted image
        converted_image = Image.open(converted_path)
        
        # Apply background removal API
        api_result_path = remove_background(converted_path)
        
        if api_result_path and os.path.exists(api_result_path):
            processed_image = Image.open(api_result_path)
        else:
            processed_image = converted_image
        
        # Save processed image temporarily for preview
        temp_filename = f"preview_processed_{converted_filename}"
        temp_path = os.path.join(app.config['OUTPUT_FOLDER'], temp_filename)
        processed_image.save(temp_path, quality=95, optimize=True)
        
        return send_file(temp_path, as_attachment=False)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary preview file after a delay
        if 'temp_path' in locals() and os.path.exists(temp_path):
            import threading
            import time
            def cleanup():
                time.sleep(5)  # Wait 5 seconds before cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            threading.Thread(target=cleanup).start()

@app.route('/composite', methods=['POST'])
def handle_composite():
    """Handle image compositing with background"""
    try:
        # Get parameters from form
        converted_filename = request.form.get('converted_filename')
        reference_background_filename = request.form.get('reference_background_filename')
        position = request.form.get('position', 'bottom')
        scale = float(request.form.get('scale', '1.0'))
        opacity = float(request.form.get('opacity', '1.0'))
        
        if not converted_filename or not reference_background_filename:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Validate parameters
        if scale < 0.1 or scale > 1.5:
            return jsonify({'error': 'Scale must be between 0.1 and 1.5'}), 400
        
        if opacity < 0.0 or opacity > 1.0:
            return jsonify({'error': 'Opacity must be between 0.0 and 1.0'}), 400
        
        # Load images
        converted_path = os.path.join(app.config['OUTPUT_FOLDER'], converted_filename)
        background_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_background_filename)
        
        if not os.path.exists(converted_path) or not os.path.exists(background_path):
            return jsonify({'error': 'Image files not found'}), 404
        
        # Load images
        converted_image = Image.open(converted_path)
        background_image = Image.open(background_path)
        
        # Get additional options
        print_size = request.form.get('print_size', '8x10')
        add_shadow = request.form.get('add_shadow', 'true').lower() == 'true'
        
        # Apply background removal API
        api_result_path = remove_background(converted_path)
        
        if api_result_path and os.path.exists(api_result_path):
            converted_image = Image.open(api_result_path)
        else:
            # Keep original image if API fails
            pass
        final_image = composite_images(
            converted_image, 
            background_image, 
            position=position, 
            scale=scale, 
            opacity=opacity,
            add_shadow=add_shadow
        )
        
        # Optimize for print size if specified
        if print_size != 'original':
            final_image = resize_to_canvas_size(final_image, print_size)
        
        # Generate final output filename
        final_filename = f"composited_{print_size}_{converted_filename}"
        final_path = os.path.join(app.config['OUTPUT_FOLDER'], final_filename)
        
        # Save final image with high quality
        final_image.save(final_path, quality=95, optimize=True)
        
        # Clean up reference background
        if os.path.exists(background_path):
            os.remove(background_path)
        
        # Clean up API temporary files
        api_temp_files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if '_bg_removed' in f]
        for temp_file in api_temp_files:
            temp_path = os.path.join(app.config['OUTPUT_FOLDER'], temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_file}")
        
        return jsonify({
            'success': True,
            'message': 'Images composited successfully',
            'output_filename': final_filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Compositing error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download the converted image"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@app.route('/outputs/<filename>')
def serve_output_file(filename):
    """Serve output files for preview (without forcing download)"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'File serve error: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def serve_upload_file(filename):
    """Serve uploaded files (like reference backgrounds)"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'File serve error: {str(e)}'}), 500

@app.route('/detect-characters', methods=['POST'])
def detect_characters():
    """Detect and identify characters in a group photo using Gemini Vision"""
    try:
        if not client:
            return jsonify({'error': 'Gemini client not initialized'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Load image
        image = Image.open(input_path)
        
        # Use Gemini Vision API to detect characters
        img_width, img_height = image.size
        detection_prompt = f"""Analyze this group photo and identify all the people (characters) in it. 
        The image dimensions are {img_width}x{img_height} pixels.
        
        CRITICAL: Number characters from LEFT TO RIGHT based on their horizontal position in the image.
        Character 1 = leftmost person, Character 2 = next person to the right, etc.
        
        IMPORTANT: Each character must be isolated separately. Do NOT include overlapping bounding boxes.
        Ensure bounding boxes do NOT overlap - each character should have their own distinct region.
        
        For each person detected, provide:
        1. Character number (1, 2, 3, etc.) - MUST number from LEFT TO RIGHT based on horizontal position
        2. Approximate position in the image (left, center, right, top, middle, bottom)
        3. Approximate age (child, teenager, adult, elderly)
        4. Gender (if clearly identifiable)
        5. Likely role/relationship (e.g., mother, father, son, daughter, friend, etc.)
        6. Brief visual description (clothing, pose, distinctive features that help identify them)
        7. Bounding box coordinates: x_min, y_min, x_max, y_max (in pixels, where 0,0 is top-left)
           - x_min: left edge of THIS CHARACTER ONLY (not including other characters)
           - y_min: top edge of THIS CHARACTER ONLY
           - x_max: right edge of THIS CHARACTER ONLY
           - y_max: bottom edge of THIS CHARACTER ONLY
           - The bounding box should tightly fit around ONLY this one character
           - Do NOT include other characters in the bounding box
           - Add minimal padding (5-10 pixels) around the character edges
        
        Format your response as a JSON array with objects containing: character_number, position, age, gender, role, description, bbox (with x_min, y_min, x_max, y_max).
        If you cannot determine certain attributes, use "unknown".
        Example format:
        [
          {{
            "character_number": 1,
            "position": "left",
            "age": "adult",
            "gender": "female",
            "role": "mother",
            "description": "Adult female on the left side, wearing a blue dress, standing",
            "bbox": {{"x_min": 50, "y_min": 100, "x_max": 280, "y_max": 600}}
          }},
          {{
            "character_number": 2,
            "position": "center",
            "age": "adult",
            "gender": "male",
            "role": "father",
            "description": "Adult male in center, wearing a white shirt, standing",
            "bbox": {{"x_min": 350, "y_min": 80, "x_max": 580, "y_max": 580}}
          }}
        ]
        
        Return ONLY the JSON array, no additional text."""
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[detection_prompt, image]
            )
            
            # Extract JSON from response
            response_text = ""
            for part in response.candidates[0].content.parts:
                if part.text:
                    response_text += part.text
            
            # Try to extract JSON from response
            import json
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                characters_data = json.loads(json_match.group())
            else:
                # Fallback: try to parse entire response as JSON
                characters_data = json.loads(response_text)
            
            # Extract and save preview images for each character
            img_width, img_height = image.size
            preview_urls = []
            
            # First, validate and fix overlapping bounding boxes
            # Sort characters by x position (left to right)
            characters_data.sort(key=lambda x: x.get('bbox', {}).get('x_min', 0))
            
            # Check for overlaps and adjust if needed
            for i in range(len(characters_data)):
                bbox_i = characters_data[i].get('bbox', {})
                if not bbox_i:
                    continue
                
                x_min_i = bbox_i.get('x_min', 0)
                x_max_i = bbox_i.get('x_max', img_width)
                
                # Check overlap with previous characters
                for j in range(i):
                    bbox_j = characters_data[j].get('bbox', {})
                    if not bbox_j:
                        continue
                    
                    x_min_j = bbox_j.get('x_min', 0)
                    x_max_j = bbox_j.get('x_max', img_width)
                    
                    # If boxes overlap horizontally, adjust them
                    if x_min_i < x_max_j and x_max_i > x_min_j:
                        # Calculate overlap
                        overlap = min(x_max_i, x_max_j) - max(x_min_i, x_min_j)
                        # Split the overlap region
                        midpoint = (max(x_min_i, x_min_j) + min(x_max_i, x_max_j)) / 2
                        # Adjust both boxes
                        bbox_i['x_min'] = int(midpoint)
                        bbox_j['x_max'] = int(midpoint)
                        print(f"⚠️ Adjusted overlapping bounding boxes for characters {j+1} and {i+1}")
            
            for char_data in characters_data:
                char_num = char_data.get('character_number', 1)
                bbox = char_data.get('bbox', {})
                
                # Get bounding box coordinates
                x_min = bbox.get('x_min', 0)
                y_min = bbox.get('y_min', 0)
                x_max = bbox.get('x_max', img_width)
                y_max = bbox.get('y_max', img_height)
                
                # Validate and clamp coordinates
                x_min = max(0, min(int(x_min), img_width - 1))
                y_min = max(0, min(int(y_min), img_height - 1))
                x_max = max(x_min + 1, min(int(x_max), img_width))
                y_max = max(y_min + 1, min(int(y_max), img_height))
                
                # If no valid bbox, try to estimate based on position
                if x_min >= x_max or y_min >= y_max:
                    # Estimate position based on character number and position description
                    position = char_data.get('position', 'center').lower()
                    num_chars = len(characters_data)
                    
                    # Divide image into regions
                    if 'left' in position:
                        x_min = 0
                        x_max = img_width // 2
                    elif 'right' in position:
                        x_min = img_width // 2
                        x_max = img_width
                    else:  # center
                        x_min = img_width // 4
                        x_max = 3 * img_width // 4
                    
                    # Vertical position
                    if 'top' in position:
                        y_min = 0
                        y_max = img_height // 2
                    elif 'bottom' in position:
                        y_min = img_height // 2
                        y_max = img_height
                    else:  # middle
                        y_min = img_height // 4
                        y_max = 3 * img_height // 4
                
                # Crop character region with minimal padding to avoid including other characters
                # Use smaller padding to better isolate individual characters
                padding = min(10, min(x_max - x_min, y_max - y_min) // 20)  # 5% padding, max 10px
                crop_x_min = max(0, x_min - padding)
                crop_y_min = max(0, y_min - padding)
                crop_x_max = min(img_width, x_max + padding)
                crop_y_max = min(img_height, y_max + padding)
                
                # Crop the character
                try:
                    char_crop = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                    
                    # Try to isolate the character better using background removal
                    # First, try simple background removal
                    try:
                        # Check if the crop likely contains multiple characters
                        # If the crop is very wide compared to height, it might have multiple people
                        crop_width = crop_x_max - crop_x_min
                        crop_height = crop_y_max - crop_y_min
                        aspect_ratio = crop_width / crop_height if crop_height > 0 else 1
                        
                        # If aspect ratio suggests multiple characters (very wide), try AI extraction
                        use_ai_extraction = aspect_ratio > 1.5  # Wide crop might have multiple people
                        
                        if use_ai_extraction:
                            # Use Gemini to extract this specific character more precisely
                            extraction_prompt = f"""Extract and isolate character number {char_num} from this image.
                            This is character {char_num}: {char_data.get('description', 'a person')}.
                            
                            CRITICAL: Extract ONLY this one character. Remove all other people and background.
                            Make the background completely transparent.
                            Return an image with ONLY this single character visible."""
                            
                            try:
                                extraction_response = client.models.generate_content(
                                    model="gemini-2.5-flash-image-preview",
                                    contents=[extraction_prompt, char_crop]
                                )
                                
                                # Check if we got an image back
                                for part in extraction_response.candidates[0].content.parts:
                                    if part.inline_data is not None:
                                        extracted_image = Image.open(io.BytesIO(part.inline_data.data))
                                        # Remove white background from extracted image
                                        extracted_image = remove_white_background(extracted_image)
                                        char_crop = extracted_image
                                        print(f"✅ Used AI extraction for character {char_num} (wide crop detected)")
                                        break
                            except Exception as extraction_error:
                                print(f"⚠️ AI extraction failed for character {char_num}: {extraction_error}")
                                # Fall through to background removal
                        
                        # Apply background removal to better isolate the character
                        if char_crop.mode != 'RGBA':
                            char_crop = char_crop.convert('RGBA')
                        
                        # Try removing white/light backgrounds
                        char_crop = remove_white_background(char_crop)
                        
                    except Exception as bg_error:
                        print(f"⚠️ Background removal failed for character {char_num}: {bg_error}")
                        # Keep original crop if background removal fails
                    
                    # Save preview image
                    preview_filename = f"char_{char_num}_preview_{uuid.uuid4().hex[:8]}.png"
                    preview_path = os.path.join(app.config['OUTPUT_FOLDER'], preview_filename)
                    
                    # Ensure RGBA mode for transparency
                    if char_crop.mode != 'RGBA':
                        char_crop = char_crop.convert('RGBA')
                    
                    char_crop.save(preview_path, format='PNG', optimize=True)
                    
                    preview_url = f'/outputs/{preview_filename}'
                    char_data['preview_url'] = preview_url
                    preview_urls.append(preview_url)
                    
                    print(f"✅ Saved isolated preview for character {char_num}: {preview_path}")
                except Exception as crop_error:
                    print(f"⚠️ Error cropping character {char_num}: {crop_error}")
                    import traceback
                    print(traceback.format_exc())
                    char_data['preview_url'] = None
            
            # Keep the original file for transformation (don't delete yet)
            # Store the path for later use
            group_photo_filename = unique_filename
            
            return jsonify({
                'success': True,
                'total_characters': len(characters_data),
                'characters': characters_data,
                'group_photo_filename': group_photo_filename  # Return filename for later use
            })
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response_text}")
            # Fallback: return basic detection with estimated crop
            img_width, img_height = image.size
            try:
                # Create a simple center crop as fallback
                crop_size = min(img_width, img_height) // 2
                x_min = (img_width - crop_size) // 2
                y_min = (img_height - crop_size) // 2
                x_max = x_min + crop_size
                y_max = y_min + crop_size
                
                char_crop = image.crop((x_min, y_min, x_max, y_max))
                preview_filename = f"char_1_preview_{uuid.uuid4().hex[:8]}.png"
                preview_path = os.path.join(app.config['OUTPUT_FOLDER'], preview_filename)
                char_crop.save(preview_path, format='PNG', optimize=True)
                preview_url = f'/outputs/{preview_filename}'
            except:
                preview_url = None
            
            return jsonify({
                'success': True,
                'total_characters': 1,
                'characters': [{
                    'character_number': 1,
                    'position': 'center',
                    'age': 'unknown',
                    'gender': 'unknown',
                    'role': 'unknown',
                    'description': 'Character detected but details could not be parsed. Please provide manual input.',
                    'preview_url': preview_url
                }],
                'group_photo_filename': unique_filename,
                'warning': 'Could not parse detailed character information. Using fallback detection.'
            })
        except Exception as e:
            print(f"Error in character detection: {e}")
            import traceback
            print(traceback.format_exc())
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({'error': f'Character detection failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/transform-character', methods=['POST'])
def transform_character():
    """Transform a single character from a group photo using custom prompt"""
    try:
        if not client:
            return jsonify({'error': 'Gemini client not initialized'}), 500
        
        # Get parameters - support both file upload and file path
        group_photo_file = request.files.get('group_photo')
        group_photo_filename = request.form.get('group_photo_filename')
        group_photo_path = request.form.get('group_photo_path')
        character_number = int(request.form.get('character_number', 1))
        transformation_type = request.form.get('transformation_type', '').strip()
        prompt = request.form.get('prompt', '').strip()  # Fallback to custom prompt if provided
        character_description = request.form.get('character_description', '').strip()  # Character description for identification
        
        # Map transformation type to strict prompt if provided
        if transformation_type and not prompt:
            transformation_prompts = {
                'musician': 'Transform this character into a professional musician. Show them in a FULL-LENGTH photo (head to feet) holding a musical instrument (guitar, piano, violin, or drums). They should be dressed in stylish musician attire from head to toe, with a confident and artistic pose. The character should look like a talented musician performing or ready to perform. The entire body must be visible from head to feet. Maintain the character\'s facial features and body structure, but change their clothing and add appropriate musical elements. The background must be solid white (#FFFFFF).',
                'painter': 'Transform this character into a professional painter or artist. Show them in a FULL-LENGTH photo (head to feet) holding a paintbrush and palette, or standing in front of an easel with a canvas. They should be wearing an artist\'s smock or apron with paint stains. The character should have an artistic, creative appearance with paint on their hands or clothes. The entire body must be visible from head to feet. Maintain the character\'s facial features and body structure, but change their clothing to artist attire. The background must be solid white (#FFFFFF).',
                'footballer': 'Transform this character into a professional footballer (soccer player). Show them in a FULL-LENGTH photo (head to feet) wearing a football jersey, shorts, and cleats. They should be in a dynamic football pose, either kicking a ball or in a ready-to-play stance. The character should look athletic and sporty. The entire body must be visible from head to feet including the cleats. Maintain the character\'s facial features and body structure, but change their clothing to football gear and add a football if appropriate. The background must be solid white (#FFFFFF).',
                'cowboy': 'Transform this character into a classic cowboy. Show them in a FULL-LENGTH photo (head to feet) wearing a cowboy hat, boots, jeans, and a western shirt or jacket. They should have a rugged, western appearance. Optionally include cowboy accessories like a belt buckle or bandana. The entire body must be visible from head to feet including the cowboy boots. The character should look like they belong in the Wild West. Maintain the character\'s facial features and body structure, but change their clothing to cowboy attire. The background must be solid white (#FFFFFF).',
                'artist': 'Transform this character into a creative artist. Show them in a FULL-LENGTH photo (head to feet) in an artistic setting with creative elements like brushes, canvases, or art supplies. They should be wearing creative, bohemian-style clothing that reflects their artistic personality. The character should have an expressive, creative appearance. The entire body must be visible from head to feet. Maintain the character\'s facial features and body structure, but change their clothing to artistic, creative attire. The background must be solid white (#FFFFFF).'
            }
            prompt = transformation_prompts.get(transformation_type, '')
        
        if not prompt:
            return jsonify({'error': 'Missing transformation type or prompt'}), 400
        
        # Handle file upload if provided
        if group_photo_file and group_photo_file.filename:
            filename = secure_filename(group_photo_file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            group_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            group_photo_file.save(group_photo_path)
        elif group_photo_filename:
            # Use stored filename from detection
            group_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], group_photo_filename)
        
        if not group_photo_path or not os.path.exists(group_photo_path):
            return jsonify({'error': 'Group photo file not found'}), 404
        
        # Load group photo
        group_image = Image.open(group_photo_path)
        
        # Create a prompt that focuses on the specific character
        # Include character description to help identify the correct person
        char_identifier = ""
        if character_description:
            char_identifier = f"\n\nCHARACTER TO TRANSFORM:\nCharacter number {character_number}: {character_description}\n\nYou MUST transform ONLY this specific character described above. Do NOT transform any other person in the photo."
        
        character_prompt = f"""You are working with a group photo. I need you to transform ONLY ONE SINGLE character from this image.

        {char_identifier}
        
        CRITICAL REQUIREMENTS - READ CAREFULLY:
        
        SINGLE CHARACTER OUTPUT - MANDATORY:
        1. The output image MUST contain EXACTLY ONE person - NO EXCEPTIONS
        2. Identify character {character_number} in the group photo based on the description: "{character_description}"
        3. Extract and transform ONLY this single character - completely remove and ignore ALL other people in the photo
        4. The output must show ONLY ONE isolated person - the transformed character {character_number}
        5. Do NOT include any other people, characters, or figures in the output image
        6. Do NOT create a group image or show multiple characters
        7. The character must be alone, isolated, and centered - no other people visible
        
        TRANSFORMATION:
        8. Transform the character according to this description: {prompt}
        9. Maintain the character's facial features, body structure, and pose from the original
        10. The character should be clearly visible, well-lit, and take up a significant portion of the image
        
        FULL-LENGTH PHOTO REQUIREMENT - MANDATORY:
        11. The image MUST be a FULL-LENGTH photo showing the character from HEAD TO FEET
        12. The entire body must be visible - from the top of the head to the bottom of the feet
        13. Do NOT crop the image to show only upper body, torso, or headshot
        14. The character should be standing or in a pose that shows their complete body
        15. Ensure both head and feet are clearly visible in the frame
        16. The character should be positioned to fit the full body within the image frame
        
        BACKGROUND:
        17. The background must be SOLID WHITE (#FFFFFF) - not transparent, not any other color, but pure white
        18. Do NOT include any other objects, background elements, or people - just the transformed character on white background
        
        VALIDATION:
        19. Before generating, verify: 
            - Will the output show EXACTLY ONE person? If not, do not generate.
            - Will the output show the FULL BODY from head to feet? If not, do not generate.
        20. If you cannot clearly identify character {character_number} from the description, focus on the person in the position described, but still output ONLY that one person in full-length
        
        Return a FULL-LENGTH image (head to feet) with EXACTLY ONE person - character {character_number} transformed, on a solid white background. The image must show a single isolated character in full-body view, not a group, not a cropped portrait."""
        
        # Generate transformed character
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[character_prompt, group_image]
        )
        
                # Process response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Save transformed character
                transformed_image = Image.open(io.BytesIO(part.inline_data.data))
                
                # Save temporarily to use background removal service
                temp_output_filename = f"character_{character_number}_temp_{uuid.uuid4().hex[:8]}.png"
                temp_output_path = os.path.join(app.config['OUTPUT_FOLDER'], temp_output_filename)
                
                # Ensure RGB mode for background removal API (most APIs expect RGB)
                if transformed_image.mode == 'RGBA':
                    # Create white background first
                    white_bg = Image.new('RGB', transformed_image.size, (255, 255, 255))
                    white_bg.paste(transformed_image, mask=transformed_image.split()[3])
                    transformed_image = white_bg
                elif transformed_image.mode != 'RGB':
                    transformed_image = transformed_image.convert('RGB')
                
                # Save temporary file for background removal
                transformed_image.save(temp_output_path, format='PNG', optimize=True)
                
                # Remove white background using configured background removal service
                print(f"🔧 Removing white background from transformed character {character_number} using configured service...")
                bg_removed_path = None
                try:
                    bg_removed_path = remove_background(temp_output_path)
                except Exception as bg_error:
                    print(f"⚠️ Background removal service error: {bg_error}")
                
                if bg_removed_path and os.path.exists(bg_removed_path):
                    # Verify the background-removed image is valid
                    try:
                        test_img = Image.open(bg_removed_path)
                        if test_img.size[0] > 0 and test_img.size[1] > 0:
                            transformed_image = test_img
                            print(f"✅ Background removed successfully for character {character_number} using API service")
                            # Clean up temp file
                            try:
                                if temp_output_path != bg_removed_path and os.path.exists(temp_output_path):
                                    os.remove(temp_output_path)
                            except:
                                pass
                        else:
                            raise ValueError("Invalid image dimensions")
                    except Exception as img_error:
                        print(f"⚠️ Background-removed image invalid: {img_error}, trying manual removal...")
                        transformed_image = Image.open(temp_output_path)
                        bg_removed_path = None  # Force manual removal
                else:
                    # Background removal failed, try manual removal
                    print(f"⚠️ Background removal API failed or returned no result, trying manual white background removal...")
                    transformed_image = Image.open(temp_output_path)
                    bg_removed_path = None
                
                # If API removal failed, use manual removal
                if not bg_removed_path:
                    try:
                        transformed_image = remove_white_background(transformed_image)
                        print(f"✅ Manual white background removal successful")
                    except Exception as manual_error:
                        print(f"⚠️ Manual removal also failed: {manual_error}, trying simple white pixel removal...")
                        # Last resort: simple white pixel removal
                        try:
                            if transformed_image.mode != 'RGBA':
                                transformed_image = transformed_image.convert('RGBA')
                            import numpy as np
                            img_array = np.array(transformed_image)
                            # Remove white/near-white pixels (threshold: all channels > 230)
                            white_mask = (
                                (img_array[:, :, 0] > 230) & (img_array[:, :, 1] > 230) & (img_array[:, :, 2] > 230)
                            )
                            # Set white pixels to transparent
                            img_array[white_mask, 3] = 0
                            transformed_image = Image.fromarray(img_array, 'RGBA')
                            print(f"✅ Applied simple white-to-transparent removal")
                        except Exception as simple_error:
                            print(f"⚠️ Simple removal also failed: {simple_error}")
                            # Convert to RGBA at minimum
                            if transformed_image.mode != 'RGBA':
                                transformed_image = transformed_image.convert('RGBA')
                    
                    # Clean up temp file
                    try:
                        if os.path.exists(temp_output_path):
                            os.remove(temp_output_path)
                    except:
                        pass
                
                # Ensure RGBA mode for transparency
                if transformed_image.mode != 'RGBA':
                    transformed_image = transformed_image.convert('RGBA')
                
                # Additional white pixel cleanup for any remaining white areas (always run this)
                try:
                    import numpy as np
                    img_array = np.array(transformed_image)
                    # Remove white/near-white pixels (threshold: all channels > 230)
                    white_mask = (
                        (img_array[:, :, 0] > 230) & (img_array[:, :, 1] > 230) & (img_array[:, :, 2] > 230)
                    )
                    # Only remove if alpha is not already 0 (preserve existing transparency)
                    white_mask = white_mask & (img_array[:, :, 3] > 0)
                    # Set white pixels to transparent
                    img_array[white_mask, 3] = 0
                    white_pixels_removed = np.sum(white_mask)
                    if white_pixels_removed > 0:
                        transformed_image = Image.fromarray(img_array, 'RGBA')
                        print(f"✅ Applied additional white-to-transparent cleanup (removed {white_pixels_removed} white pixels)")
                except Exception as np_error:
                    print(f"⚠️ Could not apply numpy white removal: {np_error}")
                    # Continue with image as is
                
                # Save final output
                output_filename = f"character_{character_number}_{uuid.uuid4().hex[:8]}.png"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                transformed_image.save(output_path, format='PNG', optimize=True)
                
                # Clean up background-removed temp file if different from output
                if bg_removed_path and bg_removed_path != output_path and os.path.exists(bg_removed_path):
                    try:
                        os.remove(bg_removed_path)
                    except:
                        pass
                
                return jsonify({
                    'success': True,
                    'character_number': character_number,
                    'output_filename': output_filename,
                    'output_url': f'/outputs/{output_filename}'
                })
        
        return jsonify({'error': 'No image generated in response'}), 500
        
    except Exception as e:
        import traceback
        print(f"Error transforming character: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error transforming character: {str(e)}'}), 500

@app.route('/composite-characters', methods=['POST'])
def composite_characters():
    """Composite all transformed characters onto a background"""
    try:
        # Get parameters
        character_files = request.form.getlist('character_files[]')  # List of filenames
        background_file = request.files.get('background')
        positions_json = request.form.get('positions', '[]')  # JSON array of positions
        
        if not character_files:
            return jsonify({'error': 'No character files provided'}), 400
        
        import json
        try:
            positions = json.loads(positions_json)
        except:
            positions = []
        
        # Handle background upload if provided
        if background_file and background_file.filename:
            bg_filename = secure_filename(background_file.filename)
            bg_unique_filename = f"{uuid.uuid4()}_background_{bg_filename}"
            background_path = os.path.join(app.config['UPLOAD_FOLDER'], bg_unique_filename)
            background_file.save(background_path)
            background_image = Image.open(background_path)
        else:
            # Create a default background if none provided
            background_image = Image.new('RGB', (1920, 1080), color='lightblue')
        
        # Load all character images and remove white backgrounds
        character_images = []
        temp_files_to_cleanup = []  # Track temp files for cleanup after compositing
        
        for char_file in character_files:
            char_path = os.path.join(app.config['OUTPUT_FOLDER'], char_file)
            if os.path.exists(char_path):
                try:
                    # Remove white background using configured background removal service
                    print(f"🔧 Removing white background from {char_file}...")
                    bg_removed_path = remove_background(char_path)
                    
                    # Use background-removed image if available, otherwise use original
                    if bg_removed_path and os.path.exists(bg_removed_path):
                        char_img = Image.open(bg_removed_path)
                        print(f"✅ Background removed for {char_file} using API")
                        # Track temp file for cleanup later
                        if bg_removed_path != char_path:
                            temp_files_to_cleanup.append(bg_removed_path)
                    else:
                        char_img = Image.open(char_path)
                        print(f"⚠️ Background removal API failed for {char_file}, trying manual removal...")
                        # Try to remove white background manually as fallback
                        try:
                            char_img = remove_white_background(char_img)
                            print(f"✅ Manual white background removal successful")
                        except Exception as manual_bg_error:
                            print(f"⚠️ Manual background removal also failed: {manual_bg_error}")
                            # Continue with original image
                    
                    # Validate image was loaded properly
                    if not hasattr(char_img, 'size') or not char_img.size:
                        print(f"⚠️ Invalid image: {char_file} - no size attribute")
                        continue
                    
                    char_width, char_height = char_img.size
                    if char_width is None or char_height is None or char_width <= 0 or char_height <= 0:
                        print(f"⚠️ Invalid image dimensions: {char_file} - width: {char_width}, height: {char_height}")
                        continue
                    
                    # Ensure RGBA for transparency
                    if char_img.mode != 'RGBA':
                        char_img = char_img.convert('RGBA')
                    
                    # Additional white background removal: convert white/near-white pixels to transparent
                    try:
                        import numpy as np
                        img_array = np.array(char_img)
                        # Create mask for white/near-white pixels (threshold: all channels > 230 for better removal)
                        # Also check for pixels that are very light (high brightness)
                        white_mask = (
                            (img_array[:, :, 0] > 230) & (img_array[:, :, 1] > 230) & (img_array[:, :, 2] > 230)
                        ) | (
                            # Also remove very bright pixels (high average brightness)
                            ((img_array[:, :, 0] + img_array[:, :, 1] + img_array[:, :, 2]) / 3 > 240)
                        )
                        # Set white/near-white pixels to transparent
                        img_array[white_mask, 3] = 0
                        char_img = Image.fromarray(img_array, 'RGBA')
                        print(f"✅ Applied additional white-to-transparent conversion (removed {np.sum(white_mask)} white pixels)")
                    except Exception as np_error:
                        print(f"⚠️ Could not apply numpy white removal: {np_error}")
                        # Try simple PIL-based white removal as fallback
                        try:
                            # Convert white pixels to transparent using PIL
                            pixels = char_img.load()
                            width, height = char_img.size
                            for y in range(height):
                                for x in range(width):
                                    r, g, b, a = pixels[x, y]
                                    # If pixel is white or near-white, make it transparent
                                    if r > 230 and g > 230 and b > 230:
                                        pixels[x, y] = (r, g, b, 0)
                            print(f"✅ Applied PIL-based white removal")
                        except Exception as pil_error:
                            print(f"⚠️ PIL white removal also failed: {pil_error}")
                            # Continue with image as is
                    
                    character_images.append(char_img)
                            
                except Exception as img_error:
                    print(f"⚠️ Error loading image {char_file}: {img_error}")
                    import traceback
                    print(traceback.format_exc())
                    continue
        
        # Clean up temporary files after compositing is complete
        def cleanup_temp_files():
            for temp_file in temp_files_to_cleanup:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"🧹 Cleaned up temp file: {temp_file}")
                except:
                    pass
        
        if not character_images:
            return jsonify({'error': 'No valid character images found'}), 404
        
        # Composite characters onto background
        result_image = background_image.copy()
        if result_image.mode != 'RGB':
            result_image = result_image.convert('RGB')
        
        bg_width, bg_height = result_image.size
        
        # Position characters (use provided positions or default layout)
        for i, char_img in enumerate(character_images):
            # Get character dimensions using .size tuple
            original_char_width, original_char_height = char_img.size
            
            # Validate dimensions
            if original_char_width is None or original_char_height is None:
                print(f"⚠️ Skipping character {i+1} - invalid dimensions")
                continue
            
            # Get position and scale from positions array if available
            scale = 1.0
            if i < len(positions) and isinstance(positions[i], dict):
                pos_data = positions[i]
                x = pos_data.get('x')
                y = pos_data.get('y')
                scale = float(pos_data.get('scale', 1.0))
            else:
                # Default: distribute characters horizontally
                spacing = bg_width // (len(character_images) + 1)
                x = spacing * (i + 1)
                y = bg_height - 100  # Position at bottom with offset
            
            # Apply scale to character dimensions
            char_width = int(original_char_width * scale)
            char_height = int(original_char_height * scale)
            
            # Resize character image if scale is not 1.0
            if scale != 1.0:
                char_img = char_img.resize((char_width, char_height), Image.Resampling.LANCZOS)
            
            # If x or y is None, use default centering
            if x is None:
                x = bg_width // 2 - char_width // 2
            if y is None:
                y = bg_height // 2 - char_height // 2
            
            # Allow free positioning - characters can be positioned anywhere
            # Convert to int
            x = int(x) if x is not None else bg_width // 2 - char_width // 2
            y = int(y) if y is not None else bg_height // 2 - char_height // 2
            
            # Composite character onto background
            # Handle characters that may extend beyond background bounds
            # Calculate the visible region
            paste_x = max(0, x)
            paste_y = max(0, y)
            crop_x = max(0, -x)
            crop_y = max(0, -y)
            crop_width = min(char_width - crop_x, bg_width - paste_x)
            crop_height = min(char_height - crop_y, bg_height - paste_y)
            
            if crop_width > 0 and crop_height > 0:
                # Crop character image if it extends beyond bounds
                if crop_x > 0 or crop_y > 0 or crop_width < char_width or crop_height < char_height:
                    char_cropped = char_img.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
                    result_image.paste(char_cropped, (paste_x, paste_y), char_cropped)
                else:
                    # Character fits completely - paste as is
                    result_image.paste(char_img, (paste_x, paste_y), char_img)
        
        # Save final composite
        output_filename = f"group_composite_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        result_image.save(output_path, format='PNG', optimize=True)
        
        # Clean up temporary background-removed files
        cleanup_temp_files()
        
        return jsonify({
            'success': True,
            'output_filename': output_filename,
            'output_url': f'/outputs/{output_filename}'
        })
        
    except Exception as e:
        import traceback
        print(f"Error compositing characters: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error compositing characters: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'client_initialized': client is not None
    })

if __name__ == '__main__':
    if not client:
        print("Warning: Gemini client not initialized. Please check your API key.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
