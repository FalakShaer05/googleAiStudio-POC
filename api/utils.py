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
        
        # If no filename or extension, generate one
        if not filename or '.' not in filename:
            filename = f"downloaded_image_{uuid.uuid4().hex[:8]}.jpg"
        else:
            # Ensure filename is safe
            filename = secure_filename(filename)
            if not filename:
                filename = f"downloaded_image_{uuid.uuid4().hex[:8]}.jpg"
        
        # Create unique filename
        unique_filename = generate_unique_filename(filename, 'url_download')
        file_path = os.path.join(download_dir, unique_filename)
        
        # Download the image with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(image_url, timeout=30, stream=True, headers=headers)
        response.raise_for_status()
        
        # Save the image first
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        
        # Verify the file was saved and is a valid image using PIL
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            
            # If we get here, it's a valid image
            return file_path
            
        except Exception as verify_error:
            # If PIL verification fails, check if it might still be an image
            # Some CDNs return application/octet-stream for images
            is_likely_image = (
                (content_type and 'image' in content_type) or 
                (content_type == 'application/octet-stream' and 
                 any(ext in filename.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']))
            )
            
            if is_likely_image:
                # Trust the file extension or content type
                return file_path
            else:
                # Clean up the file and return None
                if os.path.exists(file_path):
                    os.remove(file_path)
                return None
                
    except requests.exceptions.RequestException as e:
        # Network-related errors
        return None
    except Exception as e:
        # Other errors
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

def remove_background_with_lightx_api(image_path: str) -> str:
    """
    Remove background using LightX API
    Documentation: https://docs.lightxeditor.com/api/remove-background
    
    Args:
        image_path: Path to the input image
        
    Returns:
        str: Path to the result image, or None if failed
    """
    try:
        import requests
        import json
        
        # Get LightX API key from environment
        lightx_api_key = os.getenv('LIGHTX_API_KEY')
        if not lightx_api_key:
            print("❌ LIGHTX_API_KEY not found in environment variables")
            return None
        
        # Check if file exists
        if not os.path.exists(image_path):
            return None
        
        # Step 1: Get image info for upload
        file_size = os.path.getsize(image_path)
        file_ext = os.path.splitext(image_path)[1].lower()
        content_type = "image/jpeg" if file_ext in ['.jpg', '.jpeg'] else "image/png"
        
        # Step 2: Get upload URL
        upload_url_endpoint = "https://api.lightxeditor.com/external/api/v2/uploadImageUrl"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": lightx_api_key
        }
        upload_data = {
            "uploadType": "imageUrl",
            "size": file_size,
            "contentType": content_type
        }
        
        print(f"📤 Step 1: Requesting upload URL from LightX...")
        upload_response = requests.post(upload_url_endpoint, headers=headers, json=upload_data, timeout=30)
        
        if upload_response.status_code != 200:
            print(f"❌ Failed to get upload URL: {upload_response.status_code}")
            return None
        
        upload_result = upload_response.json()
        if upload_result.get('statusCode') != 2000:
            print(f"❌ LightX upload URL error: {upload_result.get('message')}")
            return None
        
        upload_image_url = upload_result['body']['uploadImage']
        image_url = upload_result['body']['imageUrl']
        
        # Step 3: Upload image using PUT request
        print(f"📤 Step 2: Uploading image to LightX...")
        with open(image_path, 'rb') as file:
            put_headers = {
                "Content-Type": content_type,
                "Content-Length": str(file_size)
            }
            put_response = requests.put(upload_image_url, data=file, headers=put_headers, timeout=60)
            
            if put_response.status_code not in [200, 204]:
                print(f"❌ Failed to upload image: {put_response.status_code}")
                return None
        
        # Step 4: Call remove background API
        print(f"🔧 Step 3: Requesting background removal from LightX...")
        remove_bg_endpoint = "https://api.lightxeditor.com/external/api/v2/remove-background"
        remove_bg_data = {
            "imageUrl": image_url,
            "background": "transparent"  # Remove background (transparent)
        }
        
        remove_bg_response = requests.post(
            remove_bg_endpoint, 
            headers=headers, 
            json=remove_bg_data, 
            timeout=30
        )
        
        if remove_bg_response.status_code != 200:
            print(f"❌ Failed to request background removal: {remove_bg_response.status_code}")
            return None
        
        remove_bg_result = remove_bg_response.json()
        if remove_bg_result.get('statusCode') != 2000:
            print(f"❌ LightX remove background error: {remove_bg_result.get('message')}")
            return None
        
        order_id = remove_bg_result['body']['orderId']
        max_retries = remove_bg_result['body'].get('maxRetriesAllowed', 5)
        avg_response_time = remove_bg_result['body'].get('avgResponseTimeInSec', 15)
        
        print(f"⏳ Step 4: Polling for result (orderId: {order_id}, max retries: {max_retries})...")
        
        # Step 5: Poll for status (optimized - check immediately first, then wait adaptively)
        status_endpoint = "https://api.lightxeditor.com/external/api/v2/order-status"
        status_data = {"orderId": order_id}
        
        # Optimized polling: Check immediately first, then wait adaptively
        for attempt in range(max_retries):
            # Don't wait before first check - check immediately
            if attempt > 0:
                # Adaptive wait: shorter waits for early attempts, longer for later
                wait_time = min(2 + attempt, 3)  # 2s, 3s, 3s, 3s, 3s
                time.sleep(wait_time)
            
            status_response = requests.post(
                status_endpoint,
                headers=headers,
                json=status_data,
                timeout=30
            )
            
            if status_response.status_code != 200:
                print(f"❌ Failed to check status: {status_response.status_code}")
                continue
            
            status_result = status_response.json()
            if status_result.get('statusCode') != 2000:
                print(f"❌ Status check error: {status_result.get('message')}")
                continue
            
            status = status_result['body'].get('status')
            
            if status == 'active':
                # Success! Download the result
                output_url = status_result['body'].get('output')
                if not output_url:
                    print(f"❌ No output URL in response")
                    return None
                
                print(f"✅ Step 5: Downloading result from LightX...")
                output_response = requests.get(output_url, timeout=60)
                
                if output_response.status_code == 200:
                    # Save the result to a temporary file
                    temp_path = image_path.replace('.', '_bg_removed.')
                    with open(temp_path, 'wb') as result_file:
                        result_file.write(output_response.content)
                    
                    print(f"✅ LightX background removal successful")
                    return temp_path
                else:
                    print(f"❌ Failed to download result: {output_response.status_code}")
                    return None
                    
            elif status == 'failed':
                print(f"❌ LightX background removal failed")
                return None
            # else status is 'init', continue polling
        
        print(f"❌ LightX background removal timed out after {max_retries} attempts")
        return None
                
    except Exception as e:
        print(f"❌ Error in LightX background removal: {e}")
        return None

def remove_background(image_path: str) -> str:
    """
    Remove background using the service specified in BACKGROUND_REMOVAL_SERVICE env variable.
    Options: 'mosida' or 'lightx'
    Falls back to 'mosida' if not specified or invalid.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        str: Path to the result image, or None if failed
    """
    service = os.getenv('BACKGROUND_REMOVAL_SERVICE', 'mosida').lower()
    
    if service == 'lightx':
        print(f"🔧 Using LightX API for background removal")
        return remove_background_with_lightx_api(image_path)
    else:
        print(f"🔧 Using Mosida API for background removal")
        return remove_background_with_mosida_api(image_path)
