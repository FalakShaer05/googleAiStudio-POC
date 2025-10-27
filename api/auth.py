"""
API Authentication Module
Handles API key validation and rate limiting
"""

import os
import time
from functools import wraps
from flask import request, jsonify
from typing import Optional, Dict, Any

# In-memory storage for rate limiting (in production, use Redis or database)
rate_limit_storage: Dict[str, Dict[str, Any]] = {}

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key against configured keys
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not api_key:
        return False
    
    # Get valid API keys from environment variables
    # Format: API_KEY_1,API_KEY_2,API_KEY_3
    valid_keys = os.getenv('API_KEYS', '').split(',')
    valid_keys = [key.strip() for key in valid_keys if key.strip()]
    
    # If no API keys configured, use a default for development
    if not valid_keys:
        valid_keys = ['dev-api-key-12345']
    
    return api_key in valid_keys

def check_rate_limit(api_key: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
    """
    Check if API key has exceeded rate limit
    
    Args:
        api_key: The API key to check
        max_requests: Maximum requests allowed in time window
        window_minutes: Time window in minutes
        
    Returns:
        bool: True if within limits, False if exceeded
    """
    current_time = time.time()
    window_seconds = window_minutes * 60
    
    # Initialize or get existing data for this API key
    if api_key not in rate_limit_storage:
        rate_limit_storage[api_key] = {
            'requests': [],
            'last_cleanup': current_time
        }
    
    key_data = rate_limit_storage[api_key]
    
    # Clean up old requests (older than window)
    cutoff_time = current_time - window_seconds
    key_data['requests'] = [req_time for req_time in key_data['requests'] if req_time > cutoff_time]
    
    # Check if under limit
    if len(key_data['requests']) >= max_requests:
        return False
    
    # Add current request
    key_data['requests'].append(current_time)
    return True

def require_api_key(f):
    """
    Decorator to require valid API key for endpoint access
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from header or form data
        api_key = request.headers.get('X-API-Key') or request.form.get('api_key')
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API key required',
                'error_code': 'AUTH_001',
                'details': 'Provide API key via X-API-Key header or api_key parameter'
            }), 401
        
        if not validate_api_key(api_key):
            return jsonify({
                'success': False,
                'error': 'Invalid API key',
                'error_code': 'AUTH_002',
                'details': 'API key not found or invalid'
            }), 401
        
        # Check rate limit
        if not check_rate_limit(api_key):
            return jsonify({
                'success': False,
                'error': 'Rate limit exceeded',
                'error_code': 'RATE_001',
                'details': 'Too many requests. Please try again later.'
            }), 429
        
        return f(*args, **kwargs)
    
    return decorated_function

def get_api_key_info(api_key: str) -> Dict[str, Any]:
    """
    Get information about API key usage
    
    Args:
        api_key: The API key to check
        
    Returns:
        dict: API key usage information
    """
    if api_key not in rate_limit_storage:
        return {
            'requests_count': 0,
            'last_request': None,
            'rate_limit_status': 'OK'
        }
    
    key_data = rate_limit_storage[api_key]
    current_time = time.time()
    
    # Count requests in last hour
    hour_ago = current_time - 3600
    recent_requests = [req_time for req_time in key_data['requests'] if req_time > hour_ago]
    
    return {
        'requests_count': len(recent_requests),
        'last_request': max(key_data['requests']) if key_data['requests'] else None,
        'rate_limit_status': 'OK' if len(recent_requests) < 100 else 'EXCEEDED'
    }
