"""
Async utilities for Yahoo Finance API client.

This module is a compatibility layer that re-exports functionality from the yahoofinance.utils.async.helpers
module for backward compatibility. Use the helpers module directly in new code.

WARNING: This module is deprecated. Import from yahoofinance.utils.async.helpers instead.
"""

# Re-export from helpers module - using absolute imports to avoid circular imports
# pylint: disable=import-error,no-name-in-module
from yahoofinance.utils.async.helpers import (
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
    'retry_async',
    '_should_retry_exception',
    '_calculate_backoff_delay'
]