"""
API Data Models
Defines request/response models for API endpoints
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class ProcessRequest:
    """Request model for image processing"""
    image: Any  # File object
    prompt: str
    background: Optional[Any] = None  # File object
    position: str = 'bottom'
    scale: float = 1.0
    opacity: float = 1.0
    canvas_size: Optional[str] = None
    api_key: Optional[str] = None

@dataclass
class ProcessResponse:
    """Response model for successful image processing"""
    success: bool = True
    message: str = "Image processed successfully"
    output_url: Optional[str] = None
    processing_time: Optional[str] = None
    file_size: Optional[str] = None
    output_filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ErrorResponse:
    """Response model for API errors"""
    success: bool = False
    error: str = "An error occurred"
    error_code: str = "UNKNOWN_001"
    details: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

# Error codes for different types of failures
ERROR_CODES = {
    'AUTH_001': 'API key required',
    'AUTH_002': 'Invalid API key',
    'RATE_001': 'Rate limit exceeded',
    'VALIDATION_001': 'Missing required parameter',
    'VALIDATION_002': 'Invalid file type',
    'VALIDATION_003': 'File too large',
    'VALIDATION_004': 'Invalid parameter value',
    'PROCESSING_001': 'Image conversion failed',
    'PROCESSING_002': 'Background removal failed',
    'PROCESSING_003': 'Image compositing failed',
    'PROCESSING_004': 'File save failed',
    'SERVICE_001': 'Gemini API unavailable',
    'SERVICE_002': 'Remove.bg API unavailable',
    'SERVICE_003': 'Internal processing error',
    'FILE_001': 'File not found',
    'FILE_002': 'File access denied',
    'FILE_003': 'File format not supported'
}

def create_error_response(error_code: str, details: str = None) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_code: Error code from ERROR_CODES
        details: Additional error details
        
    Returns:
        dict: Error response dictionary
    """
    return {
        'success': False,
        'error': ERROR_CODES.get(error_code, 'Unknown error'),
        'error_code': error_code,
        'details': details,
        'timestamp': datetime.utcnow().isoformat()
    }

def create_success_response(message: str, output_url: str = None, 
                          processing_time: str = None, file_size: str = None,
                          output_filename: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        message: Success message
        output_url: URL to download processed image
        processing_time: Time taken for processing
        file_size: Size of output file
        output_filename: Name of output file
        metadata: Additional metadata
        
    Returns:
        dict: Success response dictionary
    """
    response = {
        'success': True,
        'message': message,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if output_url:
        response['output_url'] = output_url
    if processing_time:
        response['processing_time'] = processing_time
    if file_size:
        response['file_size'] = file_size
    if output_filename:
        response['output_filename'] = output_filename
    if metadata:
        response['metadata'] = metadata
    
    return response
