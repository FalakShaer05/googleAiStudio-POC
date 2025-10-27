#!/usr/bin/env python3
"""
Startup script for the image processing server with API
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment...")
    
    # Check if .env exists
    if not Path('.env').exists():
        print("❌ .env file not found!")
        print("   Please copy env.example to .env and add your GOOGLE_API_KEY")
        return False
    
    # Check if API keys are configured
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = os.getenv('API_KEYS')
    if not api_keys:
        print("⚠️  API_KEYS not configured in .env")
        print("   Run: python setup_api.py")
        print("   Or add API_KEYS to your .env file")
    else:
        print("✅ API keys configured")
    
    # Check Google API key
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        print("❌ GOOGLE_API_KEY not found in .env")
        return False
    else:
        print("✅ Google API key configured")
    
    return True

def start_server():
    """Start the Flask server"""
    print("\n🚀 Starting Image Processing Server...")
    print("=" * 50)
    
    try:
        # Import and run the app
        from app import app
        
        print("✅ Server starting successfully!")
        print("\n📡 Available endpoints:")
        print("   Web Interface: http://localhost:5000")
        print("   API Health:    http://localhost:5000/api/v1/health")
        print("   API Process:   http://localhost:5000/api/v1/process")
        print("\n📚 Documentation: API_DOCUMENTATION.md")
        print("🧪 Test Script:   python test_api.py")
        print("\n" + "=" * 50)
        
        # Start the server
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    """Main startup function"""
    print("🎨 Image Processing Server with API")
    print("=" * 50)
    
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    start_server()

if __name__ == "__main__":
    main()
