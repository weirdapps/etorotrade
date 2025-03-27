"""
Asynchronous utilities for Yahoo Finance data.

This module provides utilities for asynchronous operations including rate limiting,
batch processing, and safe alternatives to standard asyncio functions.

This is the canonical location for async utilities. Code in yahoofinance.utils.async
is deprecated and will be removed in a future version.
"""

from .helpers import (
    gather_with_concurrency, 
    gather_with_semaphore,
    async_bulk_fetch,
    async_retry,
    AsyncRateLimiter,
    async_rate_limited
)

__all__ = [
    # Rate limiting
    'AsyncRateLimiter',
    'async_rate_limited',
    
    # Helper functions
    'gather_with_concurrency',
    'gather_with_semaphore',
    'async_bulk_fetch',
    'async_retry',
]