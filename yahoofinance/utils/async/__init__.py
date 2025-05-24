"""
Asynchronous utilities for Yahoo Finance data.

This module provides utilities for asynchronous operations including rate limiting,
batch processing, and safe alternatives to standard asyncio functions.

DEPRECATED: This module is deprecated. Use yahoofinance.utils.async_utils instead.
"""

import warnings


warnings.warn(
    "The yahoofinance.utils.async module is deprecated. "
    "Use yahoofinance.utils.async_utils instead, which provides improved async utilities.",
    DeprecationWarning,
    stacklevel=2,
)

from .helpers import (
    AsyncRateLimiter,
    async_bulk_fetch,
    async_rate_limited,
    async_retry,
    gather_with_concurrency,
    gather_with_semaphore,
)


__all__ = [
    # Rate limiting
    "AsyncRateLimiter",
    "async_rate_limited",
    # Helper functions
    "gather_with_concurrency",
    "gather_with_semaphore",
    "async_bulk_fetch",
    "async_retry",
]
