"""Common types and data structures used across the package

This module is maintained for backward compatibility.
The implementation has moved to yahoofinance.core.types
"""

import warnings
from yahoofinance.core.types import StockData

__all__ = [
    'StockData'
]

# Re-import errors to maintain backward compatibility
from yahoofinance.core.errors import (
    YFinanceError,
    APIError,
    ValidationError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    ResourceNotFoundError,
    DataError,
    DataQualityError,
    MissingDataError
)

__all__ += [
    'YFinanceError',
    'APIError',
    'ValidationError',
    'RateLimitError',
    'ConnectionError',
    'TimeoutError',
    'ResourceNotFoundError',
    'DataError',
    'DataQualityError',
    'MissingDataError'
]

warnings.warn(
    "Importing from yahoofinance.types is deprecated. "
    "Please import from yahoofinance.core.types instead.",
    DeprecationWarning,
    stacklevel=2
)