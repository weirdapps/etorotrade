"""
LRU cache implementation with expiration and size limiting.

This module is maintained for backward compatibility.
The implementation has moved to yahoofinance.core.cache
"""

import warnings
from yahoofinance.core.cache import Cache, market_cache, news_cache, earnings_cache

__all__ = [
    'Cache',
    'market_cache',
    'news_cache',
    'earnings_cache'
]

warnings.warn(
    "Importing from yahoofinance.cache is deprecated. "
    "Please import from yahoofinance.core.cache instead.",
    DeprecationWarning,
    stacklevel=2
)