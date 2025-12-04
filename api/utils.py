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
from dotenv import load_dotenv

# Ensure .env variables are loaded even when this module is imported
load_dotenv()


def upload_image_to_s3(image_path: str) -> Optional[str]:
    """
    Upload an image to S3 and return the CloudFront URL.
    
    Args:
        image_path: Path to the local image file
        
    Returns:
        str: CloudFront URL of the uploaded image, or None if upload failed
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Check if S3 is configured
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        s3_bucket = os.getenv('LIGHTX_S3_BUCKET')
        cloudfront_url = os.getenv('LIGHTX_IMAGE_BASE_URL')  # CloudFront distribution URL
        s3_prefix = os.getenv('LIGHTX_S3_PREFIX', 'converted/')  # Optional prefix/folder
        
        missing_envs = []
        if not aws_access_key:
            missing_envs.append("AWS_ACCESS_KEY_ID")
        if not aws_secret_key:
            missing_envs.append("AWS_SECRET_ACCESS_KEY")
        if not s3_bucket:
            missing_envs.append("LIGHTX_S3_BUCKET")
        if not cloudfront_url:
            missing_envs.append("LIGHTX_IMAGE_BASE_URL")
        
        if missing_envs:
            print(f"⚠️ S3 upload skipped: missing env vars {', '.join(missing_envs)}")
            return None
        
        print(f"⚙️ S3 upload config detected (bucket={s3_bucket}, region={os.getenv('AWS_DEFAULT_REGION', 'us-east-1')}, prefix='{s3_prefix}')")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        
        # Generate S3 key (filename with optional prefix)
        filename = os.path.basename(image_path)
        # Ensure unique filename to avoid conflicts
        unique_id = str(uuid.uuid4())
        name, ext = os.path.splitext(filename)
        s3_key = f"{s3_prefix.rstrip('/')}/{unique_id}_{name}{ext}".lstrip('/')
        
        # Determine content type
        ext_lower = ext.lower()
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        content_type = content_type_map.get(ext_lower, 'image/jpeg')
        
        # Upload to S3
        print(f"📤 Uploading image to S3: s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(
            image_path,
            s3_bucket,
            s3_key,
            ExtraArgs={'ContentType': content_type}
        )
        
        # Construct CloudFront URL
        cloudfront_url_clean = cloudfront_url.rstrip('/')
        s3_key_clean = s3_key.lstrip('/')
        public_url = f"{cloudfront_url_clean}/{s3_key_clean}"
        
        print(f"✅ Image uploaded to S3. CloudFront URL: {public_url}")
        return public_url
        
    except ClientError as e:
        print(f"❌ S3 upload error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None

def build_public_image_url(image_path: str) -> Optional[str]:
    """
    Builds a public URL for an image if LIGHTX_IMAGE_BASE_URL is configured.
    First tries S3 upload if configured, otherwise falls back to local path matching.
    """
    print(f"🔍 build_public_image_url called for {image_path}")
    # Try S3 upload first if configured
    s3_url = upload_image_to_s3(image_path)
    if s3_url:
        print(f"🌐 build_public_image_url returning S3 URL: {s3_url}")
        return s3_url
    else:
        print("ℹ️ S3 upload not used; falling back to local path resolution")
    
    # Fall back to local path matching (original behavior)
    base_url = os.getenv("LIGHTX_IMAGE_BASE_URL")
    if not base_url:
        return None

    raw_base_path = os.getenv("LIGHTX_IMAGE_BASE_PATH", "outputs")
    if not raw_base_path:
        return None

    if os.path.isabs(raw_base_path):
        base_path = raw_base_path
    else:
        base_path = os.path.abspath(os.path.join(os.getcwd(), raw_base_path))

    abs_image_path = os.path.abspath(image_path)
    if not abs_image_path.startswith(os.path.abspath(base_path)):
        return None

    relative_path = os.path.relpath(abs_image_path, base_path)
    relative_url = relative_path.replace(os.sep, "/")
    return f"{base_url.rstrip('/')}/{relative_url.lstrip('/')}"

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
            
            # Make request to Mosida API with longer timeout for large images
            response = requests.post(api_url, files=files, timeout=120)
            
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
        
        temp_jpeg_path = None
        image_url = None

        # Define headers early - needed for both S3 URL path and LightX upload path
        headers = {
            "Content-Type": "application/json",
            "x-api-key": lightx_api_key
        }

        public_url = build_public_image_url(image_path)
        print(f"🌐 build_public_image_url result: {public_url}")
        if public_url:
            print(f"🌐 Using public image URL for LightX: {public_url}")
            image_url = public_url
        else:
            from PIL import Image

            try:
                with Image.open(image_path) as img:
                    actual_format = img.format
                    img_size = img.size
                    img_mode = img.mode
                    print(f"📸 Image info: {img_size}, mode: {img_mode}, format: {actual_format}")
                    
                    if img_mode in ('RGBA', 'LA', 'P'):
                        print(f"🔄 Converting from {img_mode} to RGB for JPEG compatibility")
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img_mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img_mode != 'RGB':
                        img = img.convert('RGB')
                    
                    temp_jpeg_path = image_path.rsplit('.', 1)[0] + '_lightx_temp.jpg'
                    img.save(temp_jpeg_path, format='JPEG', quality=95, optimize=True)
                    print(f"✅ Converted to JPEG: {temp_jpeg_path}")
                    content_type = "image/jpeg"
                    file_size = os.path.getsize(temp_jpeg_path)
                    print(f"📦 File size: {file_size} bytes, Content-Type: {content_type}")
                    image_path_to_upload = temp_jpeg_path
            except Exception as e:
                print(f"⚠️ Could not convert image: {e}, using original")
                file_ext = os.path.splitext(image_path)[1].lower()
                content_type = "image/jpeg" if file_ext in ['.jpg', '.jpeg'] else "image/png"
                file_size = os.path.getsize(image_path)
                image_path_to_upload = image_path
                print(f"📦 File size: {file_size} bytes, Content-Type: {content_type}")
            
            upload_url_endpoint = "https://api.lightxeditor.com/external/api/v2/uploadImageUrl"
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
                error_msg = upload_result.get('message', 'Unknown error')
                error_code = upload_result.get('statusCode', 'Unknown')
                print(f"❌ LightX upload URL error:")
                print(f"   Status Code: {error_code}")
                print(f"   Message: {error_msg}")
                print(f"   Full response: {upload_result}")
                return None
            
            upload_image_url = upload_result['body']['uploadImage']
            image_url = upload_result['body']['imageUrl']
            
            print(f"📤 Step 2: Uploading image to LightX...")
            with open(image_path_to_upload, 'rb') as file:
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
        # LightX support: keep the prompt box present but empty, and don't set it to "transparent"
        # Background set to empty string to request transparency
        remove_bg_data = {
            "imageUrl": image_url,
            "background": ""
        }
        
        print(f"📦 LightX payload: {remove_bg_data}")
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
            error_msg = remove_bg_result.get('message', 'Unknown error')
            error_code = remove_bg_result.get('statusCode', 'Unknown')
            print(f"❌ LightX remove background error:")
            print(f"   Status Code: {error_code}")
            print(f"   Message: {error_msg}")
            print(f"   Full response: {remove_bg_result}")
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
                print(f"❌ Failed to check status (attempt {attempt + 1}/{max_retries}): {status_response.status_code}")
                continue
            
            status_result = status_response.json()
            current_status = status_result.get('body', {}).get('status')
            print(f"⏱️ LightX status poll {attempt + 1}/{max_retries}: {current_status or 'UNKNOWN'}")
            if status_result.get('statusCode') != 2000:
                error_msg = status_result.get('message', 'Unknown error')
                error_code = status_result.get('statusCode', 'Unknown')
                print(f"❌ Status check error (attempt {attempt + 1}/{max_retries}):")
                print(f"   Status Code: {error_code}")
                print(f"   Message: {error_msg}")
                print(f"   Full response: {status_result}")
                # If we get a generic AI art error, fail immediately instead of retrying
                if 'GENERIC_AI_ART_ERR' in error_msg or 'AI_ART' in error_msg or error_code == 55044:
                    print(f"❌ LightX API returned AI art error (Code: {error_code})")
                    print(f"   LightX does not support AI-generated images - aborting immediately")
                    return None
                continue
            
            status = status_result['body'].get('status')
            
            if status == 'active':
                # Success! Download the result
                # Check for output URL first, then fall back to mask URL
                output_url = status_result['body'].get('output')
                mask_url = status_result['body'].get('mask')
                
                # Use mask URL if output is null/empty
                if not output_url and mask_url:
                    print(f"📥 LightX output is null, using mask URL instead")
                    output_url = mask_url
                
                if not output_url:
                    print(f"❌ No output or mask URL in response")
                    print(f"   Response body: {status_result.get('body', {})}")
                    return None
                
                print(f"📥 LightX result URL: {output_url}")
                
                print(f"✅ Step 5: Downloading result from LightX...")
                output_response = requests.get(output_url, timeout=60)
                
                if output_response.status_code == 200:
                    # Save the result to a temporary file
                    temp_path = image_path.replace('.', '_bg_removed.')
                    with open(temp_path, 'wb') as result_file:
                        result_file.write(output_response.content)
                    
                    # Clean up temporary JPEG if we created one
                    if temp_jpeg_path and os.path.exists(temp_jpeg_path):
                        try:
                            os.remove(temp_jpeg_path)
                            print(f"🧹 Cleaned up temporary JPEG file")
                        except:
                            pass
                    
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
        # Clean up temporary JPEG if we created one
        if 'temp_jpeg_path' in locals() and temp_jpeg_path and os.path.exists(temp_jpeg_path):
            try:
                os.remove(temp_jpeg_path)
                print(f"🧹 Cleaned up temporary JPEG file")
            except:
                pass
        return None
                
    except Exception as e:
        print(f"❌ Error in LightX background removal: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        # Clean up temporary JPEG if we created one
        if 'temp_jpeg_path' in locals() and temp_jpeg_path and os.path.exists(temp_jpeg_path):
            try:
                os.remove(temp_jpeg_path)
                print(f"🧹 Cleaned up temporary JPEG file")
            except:
                pass
        return None

def remove_background_with_freepik_api(image_path: str) -> str:
    """
    Remove background using Freepik API
    Documentation: https://docs.freepik.com/api-reference/remove-background/post-beta-remove-background
    
    Args:
        image_path: Path to the input image
        
    Returns:
        str: Path to the result image, or None if failed
    """
    try:
        import requests
        
        # Get Freepik API key from environment
        freepik_api_key = os.getenv('FREEPIK_API_KEY')
        if not freepik_api_key:
            print("❌ FREEPIK_API_KEY not found in environment variables")
            return None
        
        # Check if file exists
        if not os.path.exists(image_path):
            return None
        
        # Freepik API requires a publicly accessible image URL
        # Try to get a public URL (S3 upload if configured)
        public_url = build_public_image_url(image_path)
        
        if not public_url:
            print("❌ Freepik API requires a publicly accessible image URL")
            print("   Please configure S3 upload (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, LIGHTX_S3_BUCKET, LIGHTX_IMAGE_BASE_URL)")
            print("   or use a different background removal service")
            return None
        
        print(f"🌐 Using public image URL for Freepik: {public_url}")
        
        # Verify the URL is accessible before sending to Freepik
        # CloudFront/S3 might need a moment to propagate, so we'll retry a few times
        print(f"🔍 Verifying image URL is accessible...")
        url_accessible = False
        max_retries = 3
        wait_time = 1  # Start with 1 second wait
        
        for attempt in range(max_retries):
            try:
                # Use HEAD request first (lighter), fall back to GET if needed
                verify_response = requests.head(public_url, timeout=10, allow_redirects=True)
                status_code = verify_response.status_code
                
                if status_code == 200:
                    # Also verify content-type is an image
                    content_type = verify_response.headers.get('Content-Type', '').lower()
                    if 'image' in content_type:
                        url_accessible = True
                        print(f"✅ Image URL verified as accessible (Content-Type: {content_type})")
                        break
                    else:
                        print(f"⚠️ URL accessible but Content-Type is not an image: {content_type}")
                        # Try GET request to verify it's actually an image
                        get_response = requests.get(public_url, timeout=10, stream=True)
                        if get_response.status_code == 200:
                            # Check if response looks like an image by checking first bytes
                            content = get_response.raw.read(4)
                            get_response.close()
                            # Check for common image magic bytes
                            if content.startswith(b'\xff\xd8') or content.startswith(b'\x89PNG') or content.startswith(b'GIF8'):
                                url_accessible = True
                                print(f"✅ Image URL verified as accessible (detected image format)")
                                break
                elif status_code == 403:
                    print(f"⚠️ URL returned 403 Forbidden (attempt {attempt + 1}/{max_retries})")
                    print(f"   This might indicate S3 bucket permissions or CloudFront access restrictions")
                elif status_code == 404:
                    print(f"⚠️ URL returned 404 Not Found (attempt {attempt + 1}/{max_retries})")
                    print(f"   File might not be uploaded yet or path is incorrect")
                else:
                    print(f"⚠️ URL returned status {status_code} (attempt {attempt + 1}/{max_retries})")
                
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff: 1s, 2s, 4s
                    
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Error verifying URL (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    wait_time *= 2
        
        if not url_accessible:
            print(f"❌ Image URL is not accessible or not a valid image")
            print(f"   URL: {public_url}")
            print(f"   Please verify:")
            print(f"   1. The S3 bucket allows public reads (or CloudFront is properly configured)")
            print(f"   2. The CloudFront distribution is active and serving content")
            print(f"   3. The URL is accessible from external networks")
            print(f"   4. The file was successfully uploaded to S3")
            return None
        
        # Additional verification: Try to actually download a small portion to ensure it works
        # This simulates what Freepik will do
        try:
            print(f"🔍 Performing final verification by downloading image header...")
            test_response = requests.get(public_url, timeout=10, stream=True, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; FreepikBot/1.0)'
            })
            if test_response.status_code != 200:
                print(f"❌ Final verification failed: HTTP {test_response.status_code}")
                return None
            # Read first few bytes to verify it's an image
            first_bytes = test_response.raw.read(10)
            test_response.close()
            if not (first_bytes.startswith(b'\xff\xd8') or first_bytes.startswith(b'\x89PNG') or 
                    first_bytes.startswith(b'GIF8') or first_bytes.startswith(b'RIFF')):
                print(f"❌ Final verification failed: File does not appear to be a valid image")
                return None
            print(f"✅ Final verification passed: Image is downloadable and valid")
        except Exception as e:
            print(f"⚠️ Final verification error (but continuing): {e}")
        
        # Freepik API endpoint
        api_url = "https://api.freepik.com/v1/ai/beta/remove-background"
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'x-freepik-api-key': freepik_api_key
        }
        
        # Prepare form data
        data = {
            'image_url': public_url
        }
        
        print(f"🔧 Requesting background removal from Freepik...")
        response = requests.post(api_url, headers=headers, data=data, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Freepik API request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        result = response.json()
        
        # Get the high resolution URL (or fall back to url field)
        output_url = result.get('high_resolution') or result.get('url')
        
        if not output_url:
            print(f"❌ No output URL in Freepik response")
            print(f"   Response: {result}")
            return None
        
        print(f"📥 Freepik result URL: {output_url}")
        print(f"✅ Downloading result from Freepik...")
        
        # Download the result (URLs are valid for 5 minutes)
        output_response = requests.get(output_url, timeout=60)
        
        if output_response.status_code == 200:
            # Save the result to a temporary file
            temp_path = image_path.replace('.', '_bg_removed.')
            with open(temp_path, 'wb') as result_file:
                result_file.write(output_response.content)
            
            print(f"✅ Freepik background removal successful")
            return temp_path
        else:
            print(f"❌ Failed to download result: {output_response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error in Freepik background removal: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None

def remove_background(image_path: str) -> str:
    """
    Remove background using the service specified in BACKGROUND_REMOVAL_SERVICE env variable.
    Options: 'mosida', 'lightx', or 'freepik'
    Falls back to 'mosida' if not specified or invalid.
    If primary service fails, automatically tries the other services as fallback.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        str: Path to the result image, or None if failed
    """
    service = os.getenv('BACKGROUND_REMOVAL_SERVICE', 'mosida').lower()
    
    # Try primary service first
    if service == 'lightx':
        print(f"🔧 Using LightX API for background removal (primary)")
        result = remove_background_with_lightx_api(image_path)
        if result and os.path.exists(result):
            return result
        else:
            print(f"⚠️ LightX API failed, falling back to Mosida API...")
            return remove_background_with_mosida_api(image_path)
    elif service == 'freepik':
        print(f"🔧 Using Freepik API for background removal (primary)")
        result = remove_background_with_freepik_api(image_path)
        if result and os.path.exists(result):
            return result
        else:
            print(f"⚠️ Freepik API failed, falling back to Mosida API...")
            fallback_result = remove_background_with_mosida_api(image_path)
            if fallback_result and os.path.exists(fallback_result):
                return fallback_result
            else:
                print(f"⚠️ Mosida API also failed, trying LightX API...")
                return remove_background_with_lightx_api(image_path)
    else:
        print(f"🔧 Using Mosida API for background removal (primary)")
        result = remove_background_with_mosida_api(image_path)
        if result and os.path.exists(result):
            return result
        else:
            print(f"⚠️ Mosida API failed, falling back to LightX API...")
            fallback_result = remove_background_with_lightx_api(image_path)
            if fallback_result and os.path.exists(fallback_result):
                return fallback_result
            else:
                print(f"⚠️ LightX API also failed, trying Freepik API...")
                return remove_background_with_freepik_api(image_path)
