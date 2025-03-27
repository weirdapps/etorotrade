"""
Asynchronous helpers for Yahoo Finance API.

This module provides helper functions for async operations, including
controlled concurrency, safe gathering, and error handling.

NOTE: This module has been reorganized to reduce code duplication.
The AsyncRateLimiter class has been moved to enhanced.py, and this module
now imports it from there to maintain backward compatibility.
"""

import logging
import asyncio
import time
from typing import List, TypeVar, Any, Callable, Coroutine, Dict, Optional, Set, Union
from functools import wraps

from ...core.config import RATE_LIMIT
from ..network.circuit_breaker import CircuitOpenError
from .enhanced import (
    AsyncRateLimiter, 
    gather_with_concurrency, 
    async_rate_limited,
    process_batch_async as enhanced_process_batch_async
)

logger = logging.getLogger(__name__)

# Define a generic type variable for the return type
T = TypeVar('T')

# Re-export AsyncRateLimiter for backward compatibility
# AsyncRateLimiter is now defined in enhanced.py

# Re-export gather_with_concurrency
# This function is now defined in enhanced.py


async def gather_with_semaphore(
    semaphore: asyncio.Semaphore,
    *tasks: Coroutine[Any, Any, T],
    return_exceptions: bool = False
) -> List[T]:
    """
    Run tasks with a semaphore.
    
    This function is similar to asyncio.gather, but uses an existing
    semaphore to limit concurrency.
    
    Args:
        semaphore: Semaphore to use for limiting concurrency
        *tasks: Tasks to run
        return_exceptions: Whether to return exceptions instead of raising them
        
    Returns:
        List of results from the tasks
    """
    async def task_with_semaphore(task: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *(task_with_semaphore(task) for task in tasks),
        return_exceptions=return_exceptions
    )


async def async_bulk_fetch(
    items: List[Any],
    fetch_func: Callable[[Any], Coroutine[Any, Any, T]],
    max_concurrency: int = None,
    batch_size: int = None,
    batch_delay: float = None
) -> Dict[Any, T]:
    """
    Fetch data for multiple items concurrently with rate limiting.
    
    This function is a wrapper around process_batch_async with slightly different 
    parameters for backward compatibility.
    
    Args:
        items: List of items to fetch data for
        fetch_func: Async function to fetch data for a single item
        max_concurrency: Maximum number of concurrent fetches
        batch_size: Size of batches for processing
        batch_delay: Delay between batches in seconds
        
    Returns:
        Dictionary mapping items to their results
    """
    # Set default values from configuration
    if max_concurrency is None:
        max_concurrency = RATE_LIMIT["MAX_CONCURRENT_CALLS"]
    if batch_size is None:
        batch_size = RATE_LIMIT["BATCH_SIZE"]
    if batch_delay is None:
        batch_delay = RATE_LIMIT["BATCH_DELAY"]
    
    # Use the enhanced implementation
    return await enhanced_process_batch_async(
        items=items,
        processor=fetch_func,
        batch_size=batch_size,
        concurrency=max_concurrency,
        delay_between_batches=batch_delay
    )


async def async_retry(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs: Any
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply the delay by after each retry
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function
        
    Raises:
        Exception: Last exception raised by the function
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                # Calculate delay for next retry
                delay = retry_delay * (backoff_factor ** attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds."
                )
                
                # Wait before retrying
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed. "
                    f"Last error: {str(e)}"
                )
    
    # If we get here, all retries failed
    assert last_exception is not None
    raise last_exception

# For backward compatibility
def _make_async_rate_limited_decorator(rate_limiter=None):
    """Create the actual decorator function"""
    from .enhanced import _make_async_rate_limited_decorator as make_decorator
    return make_decorator(rate_limiter)