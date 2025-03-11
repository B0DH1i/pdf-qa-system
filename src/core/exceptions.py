"""
Error management.
This module contains custom error classes used throughout the application.
"""

from typing import Any, Dict, Optional
from datetime import datetime
import logging
import traceback

from loguru import logger

class BaseError(Exception):
    """Base error class."""
    
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR"):
        """Initializes the error object."""
        super().__init__(message)
        self.message = message
        self.code = code
        self.timestamp = datetime.now()
        self.details: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the error to dictionary format."""
        return {
            "code": self.code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }

class ProcessingError(BaseError):
    """Processing error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PROCESSING_ERROR")
        if details:
            self.details.update(details)

class StorageError(BaseError):
    """Storage error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "STORAGE_ERROR")
        if details:
            self.details.update(details)

class ResourceError(BaseError):
    """Resource error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RESOURCE_ERROR")
        if details:
            self.details.update(details)

class ValidationError(BaseError):
    """Validation error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR")
        if details:
            self.details.update(details)

class ConfigurationError(BaseError):
    """Configuration error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        if details:
            self.details.update(details)

class ModelError(BaseError):
    """Model error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR")
        if details:
            self.details.update(details)

class APIError(BaseError):
    """API error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "API_ERROR")
        if details:
            self.details.update(details)

class MemoryError(BaseError):
    """Memory error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MEMORY_ERROR")
        if details:
            self.details.update(details)

class CacheError(BaseError):
    """Cache error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CACHE_ERROR")
        if details:
            self.details.update(details)

class CleanupError(BaseError):
    """Cleanup error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CLEANUP_ERROR")
        if details:
            self.details.update(details)

def handle_error(error: Exception) -> Dict[str, Any]:
    """Handles the error and returns it in structured format."""
    if isinstance(error, BaseError):
        return error.to_dict()
    
    # For unknown errors
    error_dict = {
        "code": "UNKNOWN_ERROR",
        "message": str(error),
        "timestamp": datetime.now().isoformat(),
        "details": {
            "type": type(error).__name__,
            "traceback": traceback.format_exc()
        }
    }
    
    logger.error(f"Unhandled error: {error_dict}")
    return error_dict 