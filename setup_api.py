#!/usr/bin/env python3
"""
API Setup Script
Helps configure API keys and test the API endpoints
"""

import os
import secrets
import string
from pathlib import Path

def generate_api_key(length=32):
    """Generate a secure random API key"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def setup_api_keys():
    """Setup API keys in environment file"""
    env_file = Path('.env')
    api_env_file = Path('env.api.example')
    
    print("🔑 API Key Setup")
    print("=" * 50)
    
    # Check if .env exists
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please copy env.example to .env and add your GOOGLE_API_KEY first.")
        return False
    
    # Read current .env content
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Check if API_KEYS already exists
    if 'API_KEYS=' in content:
        print("✅ API_KEYS already configured in .env")
        return True
    
    # Generate API keys
    dev_key = generate_api_key()
    prod_key = generate_api_key()
    
    print(f"Generated API Keys:")
    print(f"  Development: {dev_key}")
    print(f"  Production:  {prod_key}")
    print()
    print("⚠️  IMPORTANT: Save these keys securely!")
    print()
    
    # Add API keys to .env
    api_config = f"""
# API Configuration
API_KEYS={dev_key},{prod_key}
API_RATE_LIMIT_REQUESTS=100
API_RATE_LIMIT_WINDOW_MINUTES=60
API_FILE_CLEANUP_HOURS=24
API_MAX_FILE_SIZE_MB=16
"""
    
    with open(env_file, 'a') as f:
        f.write(api_config)
    
    print("✅ API keys added to .env file")
    print()
    print("🔒 Security Notes:")
    print("- Keep these API keys secure and private")
    print("- Use different keys for development and production")
    print("- Rotate keys regularly")
    print("- Never commit API keys to version control")
    
    return True

def test_api_connection():
    """Test API connection"""
    print("\n🧪 Testing API Connection")
    print("=" * 50)
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://localhost:5000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("   Start the server with: python app.py")
        return False
    except ImportError:
        print("❌ requests library not found. Install with: pip install requests")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples")
    print("=" * 50)
    
    print("1. Test API with cURL:")
    print("""
curl -X POST "http://localhost:5000/api/v1/process" \\
  -H "X-API-Key: your-api-key" \\
  -F "image=@test_image.jpg" \\
  -F "prompt=Transform this into a cartoon caricature"
""")
    
    print("\n2. Test with Python:")
    print("""
import requests

response = requests.post(
    "http://localhost:5000/api/v1/process",
    headers={"X-API-Key": "your-api-key"},
    files={"image": open("test_image.jpg", "rb")},
    data={"prompt": "Transform this into a cartoon caricature"}
)

print(response.json())
""")
    
    print("\n3. Run the test script:")
    print("   python test_api.py")

def main():
    """Main setup function"""
    print("🚀 Image Processing API Setup")
    print("=" * 50)
    
    # Setup API keys
    if not setup_api_keys():
        return
    
    # Test connection
    if test_api_connection():
        print("\n🎉 Setup complete! API is ready to use.")
        show_usage_examples()
    else:
        print("\n⚠️  Setup complete, but API server is not running.")
        print("   Start the server with: python app.py")
        print("   Then run: python test_api.py")

if __name__ == "__main__":
    main()
