"""
Asynchronous utilities for network operations.

This package provides standardized asynchronous utilities for network operations,
including rate limiting, batching, and retry mechanisms for async operations.

WARNING: This is a deprecated compatibility layer. Import directly from the async_utils module.
"""

# Re-export from async_utils
from ..async_utils.rate_limiter import (
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