"""
S3 and CloudFront utilities for uploading and serving images.
"""
import os
import uuid
import zipfile
import tempfile
from typing import Optional, List


def upload_image_to_s3(image_path: str, prefix: Optional[str] = None) -> Optional[str]:
    """
    Upload an image to S3 and return the CloudFront URL.
    
    Args:
        image_path: Path to the local image file
        prefix: Optional S3 prefix/folder (defaults to S3_PREFIX env var or 'converted/')
        
    Returns:
        str: CloudFront URL of the uploaded image, or None if upload failed
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Check if S3 is configured
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        s3_bucket = os.getenv('S3_BUCKET')
        cloudfront_url = os.getenv('CLOUDFRONT_URL', 'https://d2s4ngnid78ki4.cloudfront.net')
        s3_prefix = prefix or os.getenv('S3_PREFIX', 'converted/')
        
        missing_envs = []
        if not aws_access_key:
            missing_envs.append("AWS_ACCESS_KEY_ID")
        if not aws_secret_key:
            missing_envs.append("AWS_SECRET_ACCESS_KEY")
        if not s3_bucket:
            missing_envs.append("S3_BUCKET")
        
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
        
    except ImportError:
        print("⚠️ boto3 not installed. S3 upload skipped. Install with: pip install boto3")
        return None
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
    Builds a public URL for an image.
    First tries S3 upload if configured, otherwise falls back to local path matching.
    
    Args:
        image_path: Path to the local image file
        
    Returns:
        str: Public URL (CloudFront or local), or None if failed
    """
    print(f"🔍 build_public_image_url called for {image_path}")
    
    # Try S3 upload first if configured
    s3_url = upload_image_to_s3(image_path)
    if s3_url:
        print(f"🌐 build_public_image_url returning S3 URL: {s3_url}")
        return s3_url
    
    # Fall back to local path matching (if CLOUDFRONT_URL is set but S3 upload failed)
    cloudfront_url = os.getenv('CLOUDFRONT_URL', 'https://d2s4ngnid78ki4.cloudfront.net')
    base_path = os.getenv("LIGHTX_IMAGE_BASE_PATH", "outputs")
    
    if not base_path:
        return None

    if os.path.isabs(base_path):
        abs_base_path = base_path
    else:
        abs_base_path = os.path.abspath(os.path.join(os.getcwd(), base_path))

    abs_image_path = os.path.abspath(image_path)
    if not abs_image_path.startswith(os.path.abspath(abs_base_path)):
        return None

    relative_path = os.path.relpath(abs_image_path, abs_base_path)
    relative_url = relative_path.replace(os.sep, "/")
    return f"{cloudfront_url.rstrip('/')}/{relative_url.lstrip('/')}"


def create_zip_archive(file_paths: List[str], output_zip_path: Optional[str] = None) -> str:
    """
    Create a zip archive containing multiple files.
    
    Args:
        file_paths: List of file paths to include in the zip
        output_zip_path: Optional path for the zip file. If None, creates a temp file.
        
    Returns:
        str: Path to the created zip file
    """
    if output_zip_path is None:
        # Create temporary zip file
        temp_dir = tempfile.gettempdir()
        zip_filename = f"batch_images_{uuid.uuid4().hex[:8]}.zip"
        output_zip_path = os.path.join(temp_dir, zip_filename)
    
    # Create zip file with compression
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file_path in file_paths:
            if os.path.exists(file_path):
                # Add file to zip with just the filename (no path)
                filename = os.path.basename(file_path)
                zipf.write(file_path, filename)
    
    return output_zip_path


def upload_zip_to_s3(zip_path: str, prefix: Optional[str] = None) -> Optional[str]:
    """
    Upload a zip file to S3 and return the CloudFront URL.
    
    Args:
        zip_path: Path to the local zip file
        prefix: Optional S3 prefix/folder (defaults to S3_PREFIX env var or 'converted/')
        
    Returns:
        str: CloudFront URL of the uploaded zip file, or None if upload failed
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Check if S3 is configured
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        s3_bucket = os.getenv('S3_BUCKET')
        cloudfront_url = os.getenv('CLOUDFRONT_URL', 'https://d2s4ngnid78ki4.cloudfront.net')
        s3_prefix = prefix or os.getenv('S3_PREFIX', 'converted/')
        
        missing_envs = []
        if not aws_access_key:
            missing_envs.append("AWS_ACCESS_KEY_ID")
        if not aws_secret_key:
            missing_envs.append("AWS_SECRET_ACCESS_KEY")
        if not s3_bucket:
            missing_envs.append("S3_BUCKET")
        
        if missing_envs:
            print(f"⚠️ S3 upload skipped: missing env vars {', '.join(missing_envs)}")
            return None
        
        if not os.path.exists(zip_path):
            print(f"❌ Zip file not found: {zip_path}")
            return None
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        
        # Generate S3 key
        filename = os.path.basename(zip_path)
        unique_id = str(uuid.uuid4())
        name, ext = os.path.splitext(filename)
        s3_key = f"{s3_prefix.rstrip('/')}/{unique_id}_{name}{ext}".lstrip('/')
        
        # Upload to S3 with zip content type
        print(f"📤 Uploading zip to S3: s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(
            zip_path,
            s3_bucket,
            s3_key,
            ExtraArgs={'ContentType': 'application/zip'}
        )
        
        # Construct CloudFront URL
        cloudfront_url_clean = cloudfront_url.rstrip('/')
        s3_key_clean = s3_key.lstrip('/')
        public_url = f"{cloudfront_url_clean}/{s3_key_clean}"
        
        print(f"✅ Zip uploaded to S3. CloudFront URL: {public_url}")
        return public_url
        
    except ImportError:
        print("⚠️ boto3 not installed. S3 upload skipped. Install with: pip install boto3")
        return None
    except ClientError as e:
        print(f"❌ S3 upload error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error uploading zip to S3: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None

