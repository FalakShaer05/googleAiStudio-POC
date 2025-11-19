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

def composite_images(foreground_image, background_image, position='center', scale=1.0, opacity=1.0, add_shadow=True):
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

def convert_image_to_image(input_image_path, prompt, output_path, upscale_before=True, scale_factor=2, canvas_size=None, dpi=300, reference_background_path=None, enable_background_compositing=False, position='center', scale=1.0, opacity=1.0):
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
        position = request.form.get('position', 'center')
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
