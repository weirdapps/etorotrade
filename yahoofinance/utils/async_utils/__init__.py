"""
Asynchronous utilities for Yahoo Finance data.

This module provides utilities for asynchronous operations including rate limiting,
batch processing, and safe alternatives to standard asyncio functions.

This is the canonical location for async utilities. Code in yahoofinance.utils.async
is deprecated and will be removed in a future version.
"""

# Import from split modules - these are the canonical implementations
from .retry import retry_async_with_backoff
from .batch import (
    gather_with_concurrency,
    process_batch_async,
    display_processing_stats,
    get_processing_stats,
)
from .semaphore import (
    AsyncRateLimiter,
    PriorityAsyncRateLimiter,
    global_async_rate_limiter,
    global_priority_rate_limiter,
    async_rate_limited,
    enhanced_async_rate_limited,
    RATE_LIMIT_ERROR_MESSAGE,
    TOO_MANY_REQUESTS_ERROR_MESSAGE,
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
    "RATE_LIMIT_ERROR_MESSAGE",
    "TOO_MANY_REQUESTS_ERROR_MESSAGE",
    # Batch processing
    "gather_with_concurrency",
    "gather_with_rate_limit",  # Alias for compatibility
    "process_batch_async",
    "display_processing_stats",
    "get_processing_stats",
    # Retry
    "retry_async",  # Alias for compatibility
    "retry_async_with_backoff",
    # Helper functions
    "gather_with_semaphore",
    "async_bulk_fetch",
    "async_retry",
    "prioritized_batch_process",
    "adaptive_fetch",
]
