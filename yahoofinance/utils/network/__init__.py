"""
Network utilities for Yahoo Finance data.

This module provides utilities for network communication, including
rate limiting and circuit breaking.
"""

from .circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    async_circuit_protected,
    circuit_protected,
    get_async_circuit_breaker,
    get_circuit_breaker,
    reset_all_circuits,
)
from .rate_limiter import RateLimiter, global_rate_limiter, rate_limited


__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "AsyncCircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "get_circuit_breaker",
    "get_async_circuit_breaker",
    "reset_all_circuits",
    "circuit_protected",
    "async_circuit_protected",
    # Rate limiting
    "RateLimiter",
    "rate_limited",
    "global_rate_limiter",
]
