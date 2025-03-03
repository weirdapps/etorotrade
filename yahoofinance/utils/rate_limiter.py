"""
Advanced rate limiting utilities to prevent API throttling.

This module is a compatibility layer that re-exports rate limiting utilities
from the structured 'network' package to maintain backward compatibility.
"""

# Import components directly so they can be patched properly in tests
from .network.rate_limiter import AdaptiveRateLimiter
from .network.rate_limiter import global_rate_limiter
from .network.rate_limiter import batch_process

# Needed for the rate_limited decorator
import functools
import inspect
from typing import TypeVar, Callable, Optional, Any, List

# Type variable for generic function wrapping
T = TypeVar('T')

# Re-implement rate_limited to use the global_rate_limiter from this module
def rate_limited(ticker_param: Optional[str] = None):
    """
    Decorator for rate limiting functions that make API calls.
    This implementation uses the global_rate_limiter from this module.
    
    Args:
        ticker_param: Name of the parameter containing ticker symbol
        
    Returns:
        Decorated function with rate limiting
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            # Extract ticker from the specified parameter
            ticker = None
            if ticker_param:
                if ticker_param in kwargs:
                    ticker = kwargs[ticker_param]
                elif args and len(args) > 0:
                    # Extract parameter position from function signature
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if ticker_param in params:
                        idx = params.index(ticker_param)
                        if idx < len(args):
                            ticker = args[idx]
            
            return global_rate_limiter.execute_with_rate_limit(func, *args, ticker=ticker, **kwargs)
        
        # Preserve function metadata
        return functools.wraps(func)(wrapper)
    
    return decorator

# For documentation purposes
"""
This module provides backward compatibility for:
- AdaptiveRateLimiter class
- global_rate_limiter instance
- rate_limited decorator
- batch_process function

These are now maintained in network.rate_limiter module.
"""