"""
Network utilities for Yahoo Finance data.

This module provides utilities for network communication, including
rate limiting, pagination, batch processing, and circuit breaking.
"""

from .circuit_breaker import (
    CircuitBreaker, 
    AsyncCircuitBreaker, 
    CircuitState, 
    CircuitOpenError,
    get_circuit_breaker,
    get_async_circuit_breaker,
    reset_all_circuits,
    circuit_protected,
    async_circuit_protected
)

from .rate_limiter import (
    RateLimiter,
    rate_limited,
    global_rate_limiter
)

from .pagination import (
    paginated_request,
    PaginatedResults
)

from .batch import (
    batch_process
)

# Import global rate limiter from module
from .rate_limiter import global_rate_limiter

__all__ = [
    # Circuit breaker
    'CircuitBreaker',
    'AsyncCircuitBreaker',
    'CircuitState',
    'CircuitOpenError',
    'get_circuit_breaker',
    'get_async_circuit_breaker',
    'reset_all_circuits',
    'circuit_protected',
    'async_circuit_protected',
    
    # Rate limiting
    'RateLimiter',
    'rate_limited',
    'global_rate_limiter',
    
    # Pagination
    'paginated_request',
    'PaginatedResults',
    
    # Batch processing
    'batch_process',
]