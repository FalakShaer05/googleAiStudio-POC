"""
API Utility Functions
Helper functions for API operations
"""

import os
import time
import uuid
import threading
from typing import Optional, Tuple
from PIL import Image
from werkzeug.utils import secure_filename

# Allowed file extensions for API
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed for API uploads
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file type is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename: str, prefix: str = 'api') -> str:
    """
    Generate unique filename for uploaded files
    
    Args:
        original_filename: Original filename
        prefix: Prefix for the generated filename
        
    Returns:
        str: Unique filename
    """
    secure_name = secure_filename(original_filename)
    unique_id = str(uuid.uuid4())
    name, ext = os.path.splitext(secure_name)
    return f"{prefix}_{unique_id}_{name}{ext}"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted file size
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def format_processing_time(start_time: float, end_time: float) -> str:
    """
    Format processing time in human readable format
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        str: Formatted processing time
    """
    duration = end_time - start_time
    if duration < 1:
        return f"{duration * 1000:.0f}ms"
    elif duration < 60:
        return f"{duration:.1f}s"
    else:
        minutes = int(duration // 60)
        seconds = duration % 60
        return f"{minutes}m {seconds:.1f}s"

def cleanup_file(file_path: str, delay_seconds: int = 0) -> None:
    """
    Clean up file after specified delay
    
    Args:
        file_path: Path to file to delete
        delay_seconds: Delay before deletion (0 for immediate)
    """
    def _cleanup():
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            pass  # Silent cleanup
    
    if delay_seconds > 0:
        thread = threading.Thread(target=_cleanup)
        thread.daemon = True
        thread.start()
    else:
        _cleanup()

def cleanup_directory_files(directory: str, pattern: str, max_age_hours: int = 24) -> None:
    """
    Clean up old files in directory matching pattern
    
    Args:
        directory: Directory to clean
        pattern: Pattern to match filenames
        max_age_hours: Maximum age of files to keep
    """
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            if pattern in filename:
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
    except Exception as e:
        pass  # Silent cleanup

def validate_image_file(file) -> Tuple[bool, str]:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded file object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file or not file.filename:
        return False, "No file provided"
    
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size (16MB max)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > 16 * 1024 * 1024:  # 16MB
        return False, "File too large. Maximum size: 16MB"
    
    if file_size == 0:
        return False, "Empty file provided"
    
    return True, ""

def get_image_info(image_path: str) -> dict:
    """
    Get information about an image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Image information
    """
    try:
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'size_bytes': os.path.getsize(image_path)
            }
    except Exception as e:
        return {
            'error': f"Could not read image: {e}",
            'size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else 0
        }

def create_download_url(filename: str, base_url: str = None) -> str:
    """
    Create download URL for processed file
    
    Args:
        filename: Name of the file
        base_url: Base URL for the API (if None, uses request context)
        
    Returns:
        str: Full download URL
    """
    if base_url:
        return f"{base_url.rstrip('/')}/api/v1/download/{filename}"
    else:
        # This will be set in the endpoint using request context
        return f"/api/v1/download/{filename}"

def download_image_from_url(image_url: str, download_dir: str = 'uploads') -> str:
    """
    Download image from URL and save to local directory
    
    Args:
        image_url: URL of the image to download
        download_dir: Directory to save the downloaded image
        
    Returns:
        str: Path to the downloaded image, or None if failed
    """
    try:
        import requests
        from urllib.parse import urlparse
        import mimetypes
        
        # Validate URL
        if not image_url.startswith(('http://', 'https://')):
            return None
        
        # Get filename from URL or generate one
        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename or '.' not in filename:
            # Generate filename with proper extension
            response = requests.head(image_url, timeout=10)
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                ext = mimetypes.guess_extension(content_type) or '.jpg'
                filename = f"downloaded_image_{uuid.uuid4().hex[:8]}{ext}"
            else:
                filename = f"downloaded_image_{uuid.uuid4().hex[:8]}.jpg"
        
        # Ensure filename is safe
        filename = secure_filename(filename)
        if not filename:
            filename = f"downloaded_image_{uuid.uuid4().hex[:8]}.jpg"
        
        # Create unique filename
        unique_filename = generate_unique_filename(filename, 'url_download')
        file_path = os.path.join(download_dir, unique_filename)
        
        # Download the image
        response = requests.get(image_url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return None
        
        # Save the image
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify the file was saved and is a valid image
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            return file_path
        except Exception:
            # If it's not a valid image, clean up and return None
            if os.path.exists(file_path):
                os.remove(file_path)
            return None
                
    except Exception as e:
        return None

def remove_background_with_mosida_api(image_path: str) -> str:
    """
    Remove background using Mosida API
    
    Args:
        image_path: Path to the input image
        
    Returns:
        str: Path to the result image, or None if failed
    """
    try:
        import requests
        
        # Mosida API endpoint
        api_url = "https://dev-lightsail-sam.mosida.com/remove-bg"
        
        # Check if file exists
        if not os.path.exists(image_path):
            return None
        
        # Prepare the file for upload
        with open(image_path, 'rb') as file:
            files = {'file': file}
            
            # Make request to Mosida API
            response = requests.post(api_url, files=files, timeout=30)
            
            if response.status_code == 200:
                # Save the result to a temporary file
                temp_path = image_path.replace('.', '_bg_removed.')
                with open(temp_path, 'wb') as result_file:
                    result_file.write(response.content)
                
                return temp_path
            else:
                return None
                
    except Exception as e:
        return None
