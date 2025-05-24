"""
Error handling module for Yahoo Finance data access.

This module defines a comprehensive exception hierarchy for error handling,
ensuring consistent error handling throughout the package.
"""

from typing import Any, Dict, Optional


class YFinanceError(Exception):
    """
    Base class for all Yahoo Finance related errors.

    This class provides a consistent interface for error handling and logging.
    All exceptions raised by the package should inherit from this class.

    Attributes:
        message: Error message
        details: Additional error details
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a YFinanceError.

        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """
        Get string representation of the error.

        Returns:
            String representation of the error
        """
        if not self.details:
            return self.message

        detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
        return f"{self.message} ({detail_str})"


class APIError(YFinanceError):
    """
    Error related to API requests.

    This class represents errors that occur during API requests,
    including network errors, rate limiting, and server errors.
    """

    pass


class ValidationError(YFinanceError):
    """
    Error related to validation.

    This class represents errors that occur during validation,
    such as invalid ticker symbols or invalid parameters.
    """

    pass


class RateLimitError(APIError):
    """
    Error related to rate limiting.

    This class represents errors that occur when an API rate limit is reached.

    Attributes:
        message: Error message
        retry_after: Suggested retry delay in seconds
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a RateLimitError.

        Args:
            message: Error message
            retry_after: Suggested retry delay in seconds
            details: Additional error details
        """
        self.retry_after = retry_after
        super().__init__(message, details)

    def __str__(self) -> str:
        """
        Get string representation of the error.

        Returns:
            String representation of the error
        """
        base_str = super().__str__()
        if self.retry_after is not None:
            return f"{base_str} (retry after {self.retry_after} seconds)"
        return base_str


class NetworkError(APIError):
    """Error related to network issues."""

    pass


class ConnectionError(NetworkError):
    """Error related to network connection issues."""

    pass


class TimeoutError(NetworkError):
    """Error related to request timeouts."""

    pass


class ResourceNotFoundError(APIError):
    """Error related to resources not found (404)."""

    pass


class DataError(YFinanceError):
    """
    Error related to data handling.

    This class represents errors that occur during data processing,
    such as parsing errors or missing data.
    """

    pass


class DataQualityError(DataError):
    """Error related to data quality issues."""

    pass


class MissingDataError(DataError):
    """Error related to missing data."""

    pass


class CacheError(YFinanceError):
    """Error related to caching operations."""

    pass


class ConfigError(YFinanceError):
    """Error related to configuration issues."""

    pass


class MonitoringError(YFinanceError):
    """Error related to monitoring system operations."""

    pass


def format_error_details(error: Exception) -> str:
    """
    Format error details for logging.

    Args:
        error: Exception object

    Returns:
        Formatted error details string
    """
    if isinstance(error, YFinanceError):
        # For YFinanceError, include details in the formatted string
        if hasattr(error, "details") and error.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in error.details.items())
            return f"{error.__class__.__name__}: {error.message} ({detail_str})"
        return f"{error.__class__.__name__}: {error.message}"

    # For other exceptions, return the string representation
    return f"{error.__class__.__name__}: {str(error)}"


def classify_api_error(status_code: int, response_text: str) -> APIError:
    """
    Classify API error based on status code and response text.

    Args:
        status_code: HTTP status code
        response_text: Response text

    Returns:
        Appropriate APIError subclass instance
    """
    details = {
        "status_code": status_code,
        "response_text": response_text[:100] + ("..." if len(response_text) > 100 else ""),
    }

    if status_code == 404:
        return ResourceNotFoundError(f"Resource not found (status code: {status_code})", details)
    elif status_code == 429:
        return RateLimitError(f"Rate limit exceeded (status code: {status_code})", None, details)
    elif status_code >= 500:
        return APIError(f"Server error (status code: {status_code})", details)
    elif status_code >= 400:
        return APIError(f"Client error (status code: {status_code})", details)
    else:
        return APIError(f"Unexpected API error (status code: {status_code})", details)
