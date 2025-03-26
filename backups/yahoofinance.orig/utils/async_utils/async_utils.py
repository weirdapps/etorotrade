"""
Async utilities for Yahoo Finance API client.

This module is a compatibility layer that re-exports async utilities
from ./helpers.py to maintain backward compatibility.

WARNING: This module is deprecated. Please use `yahoofinance.utils.async_utils.helpers` instead.
"""

# Re-export from helpers.py module - keeping for backward compatibility
from .helpers import (
    AsyncRateLimiter,
    global_async_limiter,
    async_rate_limited,
    gather_with_rate_limit,
    process_batch_async,
    retry_async,
    _should_retry_exception,
    _calculate_backoff_delay
)

# For documentation purposes
"""
This module provides backward compatibility for:
- AsyncRateLimiter class
- global_async_limiter instance
- async_rate_limited decorator
- gather_with_rate_limit function
- process_batch_async function
- retry_async function

These are now maintained in utils.async_utils.helpers module.
"""