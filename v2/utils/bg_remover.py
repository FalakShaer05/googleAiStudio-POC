"""
Background removal utilities using Freepik API.
"""
import os
import time
from typing import Optional

import requests
from .s3_utils import build_public_image_url


def remove_background_with_freepik_api(image_path: str) -> Optional[str]:
    """
    Remove background using Freepik API
    Documentation: https://docs.freepik.com/api-reference/remove-background/post-beta-remove-background
    
    Args:
        image_path: Path to the input image
        
    Returns:
        str: Path to the result image, or None if failed
    """
    try:
        # Get Freepik API key from environment
        freepik_api_key = os.getenv('FREEPIK_API_KEY')
        if not freepik_api_key:
            print("❌ FREEPIK_API_KEY not found in environment variables")
            print("   Please add FREEPIK_API_KEY to your .env file")
            print("   Get your API key at: https://www.freepik.com/developers/dashboard/api-key")
            return None
        
        if not freepik_api_key.strip():
            print("❌ FREEPIK_API_KEY is empty")
            print("   Please set a valid API key in your .env file")
            return None
        
        # Check if file exists
        if not os.path.exists(image_path):
            return None
        
        # Freepik API requires a publicly accessible image URL
        # Try to get a public URL (S3 upload if configured)
        public_url = build_public_image_url(image_path)
        
        if not public_url:
            print("❌ Freepik API requires a publicly accessible image URL")
            print("   Please configure S3 upload (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET, CLOUDFRONT_URL)")
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
            try:
                error_data = response.json()
                error_message = error_data.get('message', response.text)
                print(f"   Error: {error_message}")
                
                # Provide specific guidance for common errors
                if response.status_code == 401:
                    print(f"   ⚠️ Invalid API key. Please check your FREEPIK_API_KEY in .env file")
                    print(f"   Get your API key at: https://www.freepik.com/developers/dashboard/api-key")
                elif response.status_code == 400:
                    print(f"   ⚠️ Bad request. Check if the image URL is accessible and valid")
                elif response.status_code == 429:
                    print(f"   ⚠️ Rate limit exceeded. Please try again later")
            except:
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

