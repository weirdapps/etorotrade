"""
Async utilities for Yahoo Finance API client.

This module is a compatibility layer that re-exports functionality from the 
yahoofinance.utils.network.async_utils module for backward compatibility.

WARNING: This module is deprecated. Import from yahoofinance.utils.network.async_utils instead.
"""

# Re-export from network.async_utils module
from .network.async_utils import (
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