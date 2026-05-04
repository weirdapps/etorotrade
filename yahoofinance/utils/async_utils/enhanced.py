"""
Enhanced async utilities - Backward Compatibility Layer

This module provides backward compatibility for existing code that imports
from yahoofinance.utils.async_utils.enhanced. All core functionality has been
moved to focused sub-modules.

New code should import from yahoofinance.utils.async_utils directly.
"""

from typing import Any

# Module-level variable for storing processing stats (set by progress.py)
_last_processing_stats: dict[str, Any] | None = None

# Import everything from the new split modules for backward compatibility
from .batch import (
    display_processing_stats,
    gather_with_concurrency,
    get_processing_stats,
    process_batch_async,
)
from .retry import retry_async_with_backoff
from .semaphore import (
    RATE_LIMIT_ERROR_MESSAGE,
    TOO_MANY_REQUESTS_ERROR_MESSAGE,
    AsyncRateLimiter,
    PriorityAsyncRateLimiter,
    async_rate_limited,
    enhanced_async_rate_limited,
    global_async_rate_limiter,
    global_priority_rate_limiter,
)

# Re-export all for backward compatibility
__all__ = [
    # Retry
    "retry_async_with_backoff",
    # Batch
    "gather_with_concurrency",
    "process_batch_async",
    "display_processing_stats",
    "get_processing_stats",
    # Semaphore (Rate Limiting)
    "AsyncRateLimiter",
    "PriorityAsyncRateLimiter",
    "global_async_rate_limiter",
    "global_priority_rate_limiter",
    "async_rate_limited",
    "enhanced_async_rate_limited",
    "RATE_LIMIT_ERROR_MESSAGE",
    "TOO_MANY_REQUESTS_ERROR_MESSAGE",
]
