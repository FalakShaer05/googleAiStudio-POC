"""
API Authentication utilities for securing endpoints.
"""
import os
from functools import wraps
from flask import request, jsonify


def get_api_key():
    """Get API key from environment variables."""
    return os.getenv('API_KEY', '')


def require_api_key(f):
    """
    Decorator to require API key authentication.
    API key can be provided via:
    - Header: X-API-Key
    - Query parameter: api_key
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = get_api_key()
        
        # If no API key is configured, allow access (for development)
        if not api_key:
            return f(*args, **kwargs)
        
        # Get API key from request
        provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not provided_key:
            return jsonify({
                "error": "API key required",
                "message": "Please provide an API key via X-API-Key header or api_key query parameter"
            }), 401
        
        if provided_key != api_key:
            return jsonify({
                "error": "Invalid API key",
                "message": "The provided API key is invalid"
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated_function

