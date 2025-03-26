"""
Async utilities for Yahoo Finance API client.

This module is a compatibility layer that re-exports functionality from the 
yahoofinance.utils.network.async_utils.rate_limiter module for backward compatibility.

WARNING: This module is deprecated. Import from yahoofinance.utils.network.async_utils.rate_limiter instead.
"""

# Re-export from network.async module
from ...utils.network.async_utils.rate_limiter import (
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