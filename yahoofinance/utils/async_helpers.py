"""
Async utilities for Yahoo Finance API client.

This module is a compatibility layer that re-exports functionality from the yahoofinance.utils.async_utils
module for backward compatibility. Use the async_utils module directly in new code.

WARNING: This module is deprecated. Import from yahoofinance.utils.async_utils instead.
"""

# Re-export from async_utils module - using absolute imports to avoid circular imports
# pylint: disable=import-error,no-name-in-module
from yahoofinance.utils.async_utils import (
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