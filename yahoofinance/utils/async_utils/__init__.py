"""
Asynchronous utilities for Yahoo Finance data.

This module provides utilities for asynchronous operations including rate limiting,
batch processing, and safe alternatives to standard asyncio functions.

This is the canonical location for async utilities. Code in yahoofinance.utils.async
is deprecated and will be removed in a future version.
"""

# Import from enhanced module - these are the canonical implementations
from .enhanced import (
    AsyncRateLimiter,
    PriorityAsyncRateLimiter,
    async_rate_limited,
    enhanced_async_rate_limited,
    gather_with_concurrency,
    global_async_rate_limiter,
    global_priority_rate_limiter,
    process_batch_async,
    retry_async_with_backoff,
)

# Import helper functions
from .helpers import (
    adaptive_fetch,
    async_bulk_fetch,
    async_retry,
    gather_with_semaphore,
    prioritized_batch_process,
)


# For backward compatibility, alias some functions
gather_with_rate_limit = gather_with_concurrency
retry_async = async_retry

__all__ = [
    # Rate limiting
    "AsyncRateLimiter",
    "PriorityAsyncRateLimiter",
    "async_rate_limited",
    "global_async_rate_limiter",
    "global_priority_rate_limiter",
    "enhanced_async_rate_limited",
    # Helper functions
    "gather_with_concurrency",
    "gather_with_rate_limit",  # Alias for compatibility
    "gather_with_semaphore",
    "async_bulk_fetch",
    "process_batch_async",
    "retry_async",  # Alias for compatibility
    "retry_async_with_backoff",
    "async_retry",
    "prioritized_batch_process",
    "adaptive_fetch",
]
