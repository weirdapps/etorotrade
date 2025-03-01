"""
Utility modules for Yahoo Finance API client.

This package contains various utility modules that support the main functionality:
- market_utils: Ticker-related utilities like validation and normalization
- rate_limiter: Advanced rate limiting for API calls
- pagination: Utilities for handling paginated API results
- async_helpers: Asynchronous utilities with rate limiting
"""

from .market_utils import (
    is_us_ticker,
    normalize_hk_ticker
)

from .rate_limiter import (
    AdaptiveRateLimiter,
    global_rate_limiter,
    rate_limited,
    batch_process
)

from .pagination import (
    PaginatedResults,
    paginated_request,
    bulk_fetch
)

from .async_helpers import (
    async_rate_limited,
    gather_with_rate_limit,
    process_batch_async,
    retry_async
)

__all__ = [
    # Market utils
    'is_us_ticker',
    'normalize_hk_ticker',
    
    # Rate limiter
    'AdaptiveRateLimiter',
    'global_rate_limiter',
    'rate_limited',
    'batch_process',
    
    # Pagination
    'PaginatedResults',
    'paginated_request',
    'bulk_fetch',
    
    # Async helpers
    'async_rate_limited',
    'gather_with_rate_limit',
    'process_batch_async',
    'retry_async'
]