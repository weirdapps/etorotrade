"""
Asynchronous rate limiting utilities for Yahoo Finance API.

This module provides utilities for rate limiting async API requests,
allowing for efficient handling of concurrent requests while respecting
rate limits.

DEPRECATED: This module is deprecated. The canonical implementation is now in
yahoofinance.utils.async_utils.enhanced.
"""

import warnings
from typing import Any, Callable, Coroutine, Dict, Optional, TypeVar, cast

# Import from the canonical source
from ..async_utils.enhanced import (
    AsyncRateLimiter,
    async_rate_limited,
)
from ..async_utils.enhanced import (
    global_async_rate_limiter as enhanced_global_rate_limiter,  # Also import the global instance
)
from ..core.logging_config import get_logger


logger = get_logger(__name__)

# Issue deprecation warning
warnings.warn(
    "yahoofinance.utils.async.rate_limiter is deprecated. "
    "Use yahoofinance.utils.async_utils.enhanced instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Define a generic type variable for the return type
T = TypeVar("T")

# Re-export the global rate limiter for backward compatibility
global_async_rate_limiter = enhanced_global_rate_limiter
