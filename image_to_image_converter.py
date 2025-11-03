#!/usr/bin/env python3
"""
Image-to-Image Conversion using Google Gemini API
Based on the Google AI Studio documentation for image editing functionality.
"""

import os
import sys
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image, ImageEnhance
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_client():
    """Initialize the Gemini client"""
    try:
        # Check if API key is available
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            print("Please add your API key to the .env file:")
            print("GOOGLE_API_KEY=your-api-key-here")
            return None
        
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        print("Make sure you have set up your API key properly in the .env file.")
        return None

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

def get_smart_canvas_recommendation(aspect_ratio_info):
    """
    Get smart canvas recommendation based on image aspect ratio
    
    Args:
        aspect_ratio_info: Dict from detect_aspect_ratio()
    
    Returns:
        str: Recommended canvas size
    """
    aspect_type = aspect_ratio_info.get('type', 'unknown')
    
    # Smart canvas recommendations based on aspect ratio
    recommendations = {
        'square': '6x6',
        'portrait': '4x6',
        'landscape': '6x4',
        'unknown': '8x12'
    }
    
    return recommendations.get(aspect_type, '8x12')

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
    Remove white background from an image to make it suitable for compositing
    
    Args:
        image: PIL Image object
        threshold: Threshold for white detection (0-255, default: 240)
    
    Returns:
        PIL Image object with transparent background
    """
    try:
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get image data
        data = image.getdata()
        new_data = []
        
        for item in data:
            # Check if pixel is white (or near white)
            if item[0] > threshold and item[1] > threshold and item[2] > threshold:
                # Make transparent
                new_data.append((item[0], item[1], item[2], 0))
            else:
                # Keep original
                new_data.append(item)
        
        # Update image data
        image.putdata(new_data)
        return image
        
    except Exception as e:
        print(f"Error removing white background: {e}")
        return image  # Return original if processing fails

def composite_images(foreground_image, background_image, position='center', scale=1.0, opacity=1.0):
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
        
        # Scale foreground image if needed
        if scale != 1.0:
            new_width = int(foreground_image.width * scale)
            new_height = int(foreground_image.height * scale)
            foreground_image = foreground_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a copy of background for compositing
        result_image = background_image.copy()
        
        # Calculate position
        if isinstance(position, tuple):
            x, y = position
        else:
            # Calculate position based on string
            bg_width, bg_height = result_image.size
            fg_width, fg_height = foreground_image.size
            
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
        return foreground_image  # Return original if compositing fails

def convert_image_to_image(input_image_path, prompt, output_path="generated_image.png", upscale_before=True, scale_factor=2, canvas_size=None, dpi=300, reference_background_path=None, enable_background_compositing=False, position='center', scale=1.0, opacity=1.0):
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
        bool: True if successful, False otherwise
    """
    
    # Validate input image path
    if not os.path.exists(input_image_path):
        print(f"Error: Input image '{input_image_path}' not found.")
        return False
    
    # Initialize client
    client = setup_client()
    if client is None:
        return False
    
    try:
        # Load the input image
        print(f"Loading image: {input_image_path}")
        image = Image.open(input_image_path)
        original_size = image.size
        
        # Load reference background if provided
        reference_background = None
        if reference_background_path and os.path.exists(reference_background_path):
            print(f"Loading reference background: {reference_background_path}")
            reference_background = Image.open(reference_background_path)
            print(f"Reference background size: {reference_background.size}")
        
        # Detect aspect ratio and provide smart recommendations
        aspect_ratio_info = detect_aspect_ratio(image)
        print(f"Image aspect ratio: {aspect_ratio_info['description']}")
        
        # Display image info
        print(f"Original image size: {original_size}")
        print(f"Image mode: {image.mode}")
        
        # Provide smart canvas recommendation if no canvas size specified
        if not canvas_size:
            recommended_canvas = get_smart_canvas_recommendation(aspect_ratio_info)
            print(f"💡 Smart recommendation: {recommended_canvas} canvas would work well for this {aspect_ratio_info['type']} image")
        
        # Upscale the image before conversion if requested
        if upscale_before and scale_factor > 1:
            print(f"Upscaling image from {original_size} to {scale_factor}x size...")
            image = upscale_image(image, scale_factor)
            print(f"Upscaled to: {image.size}")
        
        print(f"Processing with prompt: '{prompt}'")
        
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
                generated_image = Image.open(BytesIO(part.inline_data.data))
                
                # Background compositing if enabled
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
                
                # Post-conversion upscaling to canvas size if requested
                if canvas_size:
                    print(f"Post-conversion upscaling to canvas size: {canvas_size}")
                    generated_image = upscale_to_canvas_size(generated_image, canvas_size, dpi)
                
                generated_image.save(output_path)
                print(f"✅ Generated image saved as: {output_path}")
                print(f"Generated image size: {generated_image.size}")
                return True
        
        print("No image was generated in the response.")
        return False
        
    except Exception as e:
        print(f"Error during image conversion: {e}")
        return False

def main():
    """Main function with example usage"""
    
    # Example usage
    print("=== Image-to-Image Conversion with Gemini API ===\n")
    
    # You can modify these parameters
    input_image_path = "sample_image.jpg"  # Change this to your input image path
    prompt = "Create a picture of my cat eating a nano-banana in a fancy restaurant under the Gemini constellation"
    output_path = "converted_image.png"
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"Please provide a valid input image path.")
        print(f"Current path: {input_image_path}")
        print("\nExample usage:")
        print("python image_to_image_converter.py /path/to/your/image.jpg")
        return
    
    # Perform the conversion
    success = convert_image_to_image(input_image_path, prompt, output_path)
    
    if success:
        print(f"\n🎉 Image conversion completed successfully!")
        print(f"Input: {input_image_path}")
        print(f"Output: {output_path}")
    else:
        print("\n❌ Image conversion failed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Image-to-Image Conversion using Google Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_to_image_converter.py input.jpg
  python image_to_image_converter.py input.jpg "Transform this into a watercolor painting"
  python image_to_image_converter.py input.jpg "Make this look like a vintage photograph"
        """
    )
    
    parser.add_argument("input_image", help="Path to the input image file")
    parser.add_argument("prompt", nargs="?", default="Transform this image with a creative artistic style", 
                       help="Text prompt describing the desired transformation (optional)")
    parser.add_argument("-o", "--output", default="generated_image.png", 
                       help="Output path for the generated image (default: generated_image.png)")
    parser.add_argument("--no-upscale", action="store_true", 
                       help="Disable upscaling before conversion")
    parser.add_argument("--scale-factor", type=float, default=2.0, 
                       help="Scale factor for upscaling (default: 2.0, range: 1.0-4.0)")
    parser.add_argument("--canvas-size", type=str, 
                       choices=['6x6', '8x8', '12x12', '4x6', '8x12', '13x19', '24x36', '6x4', '12x8', '19x13', '36x24'], 
                       help="Canvas size for post-conversion upscaling. Smart options: square (6x6,8x8,12x12), portrait (4x6,8x12,13x19,24x36), landscape (6x4,12x8,19x13,36x24)")
    parser.add_argument("--dpi", type=int, default=300, choices=[150, 300, 600],
                       help="DPI for print quality (default: 300)")
    parser.add_argument("--smart-canvas", action="store_true",
                       help="Auto-detect best canvas size based on image aspect ratio")
    parser.add_argument("--reference-background", type=str, 
                       help="Path to reference background image for style context")
    parser.add_argument("--enable-compositing", action="store_true", 
                       help="Enable background compositing (requires --reference-background)")
    parser.add_argument("--position", type=str, default="center", 
                       choices=['center', 'top-left', 'top', 'top-right', 'left', 'right', 'bottom-left', 'bottom', 'bottom-right'],
                       help="Position for background compositing (default: center)")
    parser.add_argument("--scale", type=float, default=1.0, 
                       help="Scale factor for composited image (default: 1.0)")
    parser.add_argument("--opacity", type=float, default=1.0, 
                       help="Opacity of composited image (0.0 to 1.0, default: 1.0)")
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found.")
        print("Please provide a valid image file path.")
        sys.exit(1)
    
    # Validate scale factor
    if args.scale_factor < 1.0 or args.scale_factor > 4.0:
        print("Error: Scale factor must be between 1.0 and 4.0")
        sys.exit(1)
    
    # Validate compositing parameters
    if args.scale < 0.1 or args.scale > 1.5:
        print("Error: Scale must be between 0.1 and 1.5")
        sys.exit(1)
    
    if args.opacity < 0.0 or args.opacity > 1.0:
        print("Error: Opacity must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Validate reference background if provided
    if args.reference_background and not os.path.exists(args.reference_background):
        print(f"Error: Reference background '{args.reference_background}' not found.")
        sys.exit(1)
    
    # Validate compositing requirements
    if args.enable_compositing and not args.reference_background:
        print("Error: --enable-compositing requires --reference-background")
        sys.exit(1)
    
    # Determine upscaling options
    upscale_before = not args.no_upscale
    
    # Smart canvas detection
    canvas_size = args.canvas_size
    if args.smart_canvas and not args.canvas_size:
        # Load image to detect aspect ratio
        try:
            temp_image = Image.open(args.input_image)
            aspect_ratio_info = detect_aspect_ratio(temp_image)
            canvas_size = get_smart_canvas_recommendation(aspect_ratio_info)
            print(f"🎯 Smart canvas selected: {canvas_size} for {aspect_ratio_info['type']} image")
        except Exception as e:
            print(f"Warning: Could not detect aspect ratio: {e}")
            canvas_size = None
    
    # Perform the conversion
    success = convert_image_to_image(
        args.input_image, 
        args.prompt, 
        args.output,
        upscale_before=upscale_before,
        scale_factor=args.scale_factor,
        canvas_size=canvas_size,
        dpi=args.dpi,
        reference_background_path=args.reference_background,
        enable_background_compositing=args.enable_compositing,
        position=args.position,
        scale=args.scale,
        opacity=args.opacity
    )
    
    if success:
        print(f"\n🎉 Image conversion completed successfully!")
        print(f"Input: {args.input_image}")
        print(f"Output: {args.output}")
    else:
        print("\n❌ Image conversion failed.")
        sys.exit(1)
