"""
Async utilities for Yahoo Finance API client.

This module is a compatibility layer that re-exports async utilities
from the structured 'async' package to maintain backward compatibility.
"""

# Import components from the async submodule
from .async.async_utils import (
    AsyncRateLimiter,
    global_async_limiter,
    async_rate_limited,
    gather_with_rate_limit,
    process_batch_async,
    retry_async
)

# Explicitly expose all components for backward compatibility
__all__ = [
    'AsyncRateLimiter',
    'global_async_limiter',
    'async_rate_limited',
    'gather_with_rate_limit',
    'process_batch_async',
    'retry_async'
]