"""
Centralized error handling for the Yahoo Finance API client.

This module contains a comprehensive exception hierarchy for all types of errors
that can occur in the application. Using specific exception types allows for
more precise error handling and better diagnostics.
"""

from typing import Optional, Dict, Any, List


class YFinanceError(Exception):
    """Base exception for all Yahoo Finance client errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional details.
        
        Args:
            message: Error message
            details: Additional error details (API response, etc.)
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ValidationError(YFinanceError):
    """Raised when data validation fails."""
    pass


class APIError(YFinanceError):
    """Base class for API-related errors."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is reached."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize with message and recommended retry time.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying (if provided by API)
            details: Additional error details
        """
        self.retry_after = retry_after
        super().__init__(message, details)


class ConnectionError(APIError):
    """Raised when network connection fails."""
    pass


class TimeoutError(APIError):
    """Raised when API request times out."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class UnexpectedResponseError(APIError):
    """Raised when API response format is unexpected."""
    pass


class ResourceNotFoundError(APIError):
    """Raised when requested resource (e.g., ticker) is not found."""
    pass


class DataError(YFinanceError):
    """Base class for data processing errors."""
    pass


class DataQualityError(DataError):
    """
    Raised when data quality is insufficient for analysis.
    
    This error indicates that while data was retrieved successfully,
    its quality or completeness is insufficient for meaningful analysis.
    """
    pass


class MissingDataError(DataError):
    """
    Raised when required data fields are missing.
    
    This error is more specific than DataQualityError and indicates
    that specific required data points are missing for analysis.
    """
    
    def __init__(self, message: str, missing_fields: List[str], details: Optional[Dict[str, Any]] = None):
        """
        Initialize with message and list of missing fields.
        
        Args:
            message: Error message
            missing_fields: List of field names that are missing
            details: Additional error details
        """
        self.missing_fields = missing_fields
        details = details or {}
        details['missing_fields'] = missing_fields
        super().__init__(message, details)


class CacheError(YFinanceError):
    """Raised when cache operations fail."""
    pass


class ConfigError(YFinanceError):
    """Raised when configuration is invalid or missing."""
    pass


class PermissionError(YFinanceError):
    """Raised when operation lacks necessary permissions."""
    pass


def format_error_details(error: Exception) -> str:
    """
    Format error details for logging and display.
    
    Args:
        error: The exception to format
        
    Returns:
        Formatted error message with details
    """
    if isinstance(error, YFinanceError) and error.details:
        details_str = "\n".join(f"  {k}: {v}" for k, v in error.details.items())
        return f"{error.message}\nDetails:\n{details_str}"
    
    return str(error)


def classify_api_error(status_code: int, response_text: str) -> APIError:
    """
    Classify API error based on status code and response.
    
    Args:
        status_code: HTTP status code
        response_text: Response text from API
        
    Returns:
        Appropriate APIError subclass
    """
    error_map = {
        400: UnexpectedResponseError(f"Bad Request: {response_text}", {"status_code": status_code}),
        401: AuthenticationError(f"Authentication failed: {response_text}", {"status_code": status_code}),
        403: PermissionError(f"Permission denied: {response_text}", {"status_code": status_code}),
        404: ResourceNotFoundError(f"Resource not found: {response_text}", {"status_code": status_code}),
        429: RateLimitError(f"Rate limit exceeded: {response_text}", None, {"status_code": status_code}),
        500: APIError(f"Server error: {response_text}", {"status_code": status_code}),
        502: APIError(f"Bad gateway: {response_text}", {"status_code": status_code}),
        503: APIError(f"Service unavailable: {response_text}", {"status_code": status_code}),
        504: TimeoutError(f"Gateway timeout: {response_text}", {"status_code": status_code}),
    }
    
    return error_map.get(status_code, APIError(f"API error ({status_code}): {response_text}", {"status_code": status_code}))