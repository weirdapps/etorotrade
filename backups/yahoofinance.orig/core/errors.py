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
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize with message and optional status code.
        
        Args:
            message: Error message
            status_code: HTTP status code if available
            details: Additional error details
        """
        details = details or {}
        if status_code:
            details['status_code'] = status_code
        super().__init__(message, details)
        
    @property
    def status_code(self) -> Optional[int]:
        """Get the HTTP status code if available."""
        return self.details.get('status_code')


class RateLimitError(APIError):
    """Raised when API rate limit is reached."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, 
                 status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize with message and recommended retry time.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying (if provided by API)
            status_code: HTTP status code if available
            details: Additional error details
        """
        details = details or {}
        if retry_after:
            details['retry_after'] = retry_after
        self.retry_after = retry_after
        super().__init__(message, status_code, details)
        
    def should_retry(self) -> bool:
        """Determine if the request should be retried."""
        return True
        
    def get_retry_delay(self, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """
        Calculate appropriate retry delay.
        
        Args:
            base_delay: Base delay to use if no retry_after is provided
            max_delay: Maximum delay to enforce
            
        Returns:
            Recommended delay in seconds
        """
        if self.retry_after:
            return min(float(self.retry_after), max_delay)
        return min(base_delay * 2, max_delay)  # Default exponential backoff


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
        400: UnexpectedResponseError(f"Bad Request: {response_text}", status_code),
        401: AuthenticationError(f"Authentication failed: {response_text}", status_code),
        403: PermissionError(f"Permission denied: {response_text}", status_code),
        404: ResourceNotFoundError(f"Resource not found: {response_text}", status_code),
        429: RateLimitError(f"Rate limit exceeded: {response_text}", None, status_code),
        500: APIError(f"Server error: {response_text}", status_code),
        502: APIError(f"Bad gateway: {response_text}", status_code),
        503: APIError(f"Service unavailable: {response_text}", status_code),
        504: TimeoutError(f"Gateway timeout: {response_text}", status_code),
    }
    
    return error_map.get(status_code, APIError(f"API error ({status_code}): {response_text}", status_code))


def error_requires_retry(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error indicates a retry should be attempted
    """
    # Rate limit errors should always be retried
    if isinstance(error, RateLimitError):
        return True
        
    # Connection and timeout errors typically warrant a retry
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
        
    # Server errors (500s) may recover on retry
    if isinstance(error, APIError) and error.status_code and 500 <= error.status_code < 600:
        return True
        
    return False


def get_retry_delay(error: Exception, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate retry delay based on error type and attempt number.
    
    Args:
        error: The exception that occurred
        attempt: The current attempt number (1-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Recommended delay in seconds
    """
    # RateLimitError has its own delay logic
    if isinstance(error, RateLimitError):
        return error.get_retry_delay(base_delay, max_delay)
        
    # Exponential backoff with jitter for other retryable errors
    import random
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    jitter = random.uniform(0.8, 1.2)  # Add +/- 20% jitter
    return delay * jitter