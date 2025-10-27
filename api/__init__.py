"""
API Module for Image Conversion Service
Provides REST API endpoints for caricature generation and background merging
"""

from .endpoints import api_bp
from .auth import validate_api_key
from .models import ProcessRequest, ProcessResponse, ErrorResponse

__all__ = ['api_bp', 'validate_api_key', 'ProcessRequest', 'ProcessResponse', 'ErrorResponse']
