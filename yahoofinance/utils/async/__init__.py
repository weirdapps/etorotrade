"""
Asynchronous utilities for Yahoo Finance API.

This package provides asynchronous helpers for working with the Yahoo Finance API,
including rate limiting, batching, and retry mechanisms for async operations.

Note: This package is deprecated and will be removed in a future version.
      Please use yahoofinance.utils.async_helpers instead.
"""

from ..async_helpers import (
    AsyncRateLimiter,
    global_async_limiter,
    async_rate_limited,
    gather_with_rate_limit,
    process_batch_async,
    retry_async
)

__all__ = [
    'AsyncRateLimiter',
    'global_async_limiter',
    'async_rate_limited',
    'gather_with_rate_limit',
    'process_batch_async',
    'retry_async'
]