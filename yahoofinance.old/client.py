"""
Yahoo Finance API client with rate limiting and caching.

This module is maintained for backward compatibility.
The implementation has moved to yahoofinance.core.client
"""

import warnings
import yfinance as yf
from yahoofinance.core.client import YFinanceClient
from yahoofinance.core.errors import YFinanceError, ValidationError, APIError, RateLimitError, ConnectionError, TimeoutError, ResourceNotFoundError, DataError, DataQualityError, MissingDataError
from yahoofinance.core.types import StockData

__all__ = [
    'YFinanceClient',
    'YFinanceError',
    'ValidationError',
    'APIError',
    'RateLimitError',
    'ConnectionError',
    'TimeoutError',
    'ResourceNotFoundError',
    'DataError',
    'DataQualityError',
    'MissingDataError',
    'StockData',
    'yf'  # Add yf to __all__
]

warnings.warn(
    "Importing from yahoofinance.client is deprecated. "
    "Please import from yahoofinance.core.client or yahoofinance.core.errors instead.",
    DeprecationWarning,
    stacklevel=2
)