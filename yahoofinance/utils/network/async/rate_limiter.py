"""
Async rate limiting utilities for network operations.

This module is a compatibility layer that re-exports functionality from 
yahoofinance.utils.network.async_utils.rate_limiter for backward compatibility.

WARNING: This module is deprecated. Import from yahoofinance.utils.network.async_utils.rate_limiter instead.
"""

# Re-export from async_utils
from ..async_utils.rate_limiter import (
    AsyncRateLimiter,
    global_async_limiter,
    async_rate_limited,
    gather_with_rate_limit,
    process_batch_async,
    retry_async,
    _should_retry_exception,
    _calculate_backoff_delay
)

__all__ = [
    'AsyncRateLimiter',
    'global_async_limiter',
    'async_rate_limited',
    'gather_with_rate_limit',
    'process_batch_async',
    'retry_async'
]