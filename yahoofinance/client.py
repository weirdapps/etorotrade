"""
Yahoo Finance API client with rate limiting and caching.

This module is maintained for backward compatibility.
The implementation has moved to yahoofinance.core.client
"""

import warnings
from yahoofinance.core.client import YFinanceClient

__all__ = [
    'YFinanceClient'
]

warnings.warn(
    "Importing from yahoofinance.client is deprecated. "
    "Please import from yahoofinance.core.client instead.",
    DeprecationWarning,
    stacklevel=2
)