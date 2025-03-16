"""
Centralized error handling for the Yahoo Finance API client.

This module is maintained for backward compatibility.
The implementation has moved to yahoofinance.core.errors
"""

import warnings
from yahoofinance.core.errors import (
    YFinanceError,
    ValidationError,
    APIError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    AuthenticationError,
    UnexpectedResponseError,
    ResourceNotFoundError,
    DataError,
    DataQualityError,
    MissingDataError,
    CacheError,
    ConfigError,
    PermissionError,
    format_error_details,
    classify_api_error
)

__all__ = [
    'YFinanceError',
    'ValidationError',
    'APIError',
    'RateLimitError',
    'ConnectionError',
    'TimeoutError',
    'AuthenticationError',
    'UnexpectedResponseError',
    'ResourceNotFoundError',
    'DataError',
    'DataQualityError',
    'MissingDataError',
    'CacheError',
    'ConfigError',
    'PermissionError',
    'format_error_details',
    'classify_api_error'
]

warnings.warn(
    "Importing from yahoofinance.errors is deprecated. "
    "Please import from yahoofinance.core.errors instead.",
    DeprecationWarning,
    stacklevel=2
)