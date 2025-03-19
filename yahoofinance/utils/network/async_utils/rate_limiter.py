"""
Async rate limiting utilities for network operations.

This module provides async-compatible rate limiting utilities for network operations,
including rate limiting decorators, controlled concurrency, and retry mechanisms.

CANONICAL SOURCE:
This is the canonical source for async rate limiting functionality. Other modules
that provide similar functionality are compatibility layers that import from 
this module. Always prefer to import directly from this module in new code:

    from yahoofinance.utils.network.async_utils.rate_limiter import (
        AsyncRateLimiter, async_rate_limited, gather_with_rate_limit
    )

Key Components:
- AsyncRateLimiter: Class for controlling async operation concurrency
- async_rate_limited: Decorator for rate-limiting async functions
- gather_with_rate_limit: Async gather with concurrency control
- process_batch_async: Process items in controlled batches asynchronously
- retry_async: Retry async operations with exponential backoff

Example usage:
    # Rate-limited async function
    @async_rate_limited(ticker_param='ticker')
    async def fetch_data(ticker: str):
        # Fetch data with proper rate limiting
        
    # Process multiple operations with concurrency control
    results = await gather_with_rate_limit([
        fetch_data('AAPL'),
        fetch_data('MSFT'),
        fetch_data('GOOG')
    ], max_concurrent=5)
"""

import asyncio
import logging
import time
from typing import TypeVar, List, Dict, Any, Callable, Awaitable, Optional, Union, Tuple
from functools import wraps

from ....core.errors import RateLimitError, APIError
from ..rate_limiter import global_rate_limiter

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Return type for async functions


class AsyncRateLimiter:
    """
    Async-compatible rate limiter for API operations.
    
    This class ensures that async operations respect API rate limits
    by controlling concurrency and introducing appropriate delays.
    """
    
    def __init__(self, max_concurrency: int = 4):
        """
        Initialize async rate limiter.
        
        Args:
            max_concurrency: Maximum number of concurrent requests
        """
        self.semaphore = asyncio.Semaphore(max_concurrency)
        # Use the global rate limiter for tracking and delays
        self.rate_limiter = global_rate_limiter
    
    async def execute(self, 
                     func: Callable[..., Awaitable[T]], 
                     *args, 
                     ticker: Optional[str] = None,
                     **kwargs) -> T:
        """
        Execute an async function with rate limiting.
        
        Args:
            func: Async function to execute
            ticker: Optional ticker symbol for rate limiting
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the async function
        """
        # Get appropriate delay from rate limiter
        delay = self.rate_limiter.get_delay(ticker)
        
        # Apply delay before acquiring semaphore to stagger requests
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Limit concurrency with semaphore
        async with self.semaphore:
            try:
                result = await func(*args, **kwargs)
                # Record successful call
                self.rate_limiter.add_call(ticker=ticker)
                return result
            except Exception as e:
                # Record error and re-raise
                self.rate_limiter.add_error(e, ticker=ticker)
                raise


# Create a global async rate limiter instance
global_async_limiter = AsyncRateLimiter()


def async_rate_limited(ticker_param: Optional[str] = None):
    """
    Decorator for rate limiting async functions.
    
    Args:
        ticker_param: Name of the parameter containing ticker symbol
        
    Returns:
        Decorated async function with rate limiting
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract ticker from the specified parameter
            ticker = None
            if ticker_param:
                if ticker_param in kwargs:
                    ticker = kwargs[ticker_param]
                elif args and len(args) > 0:
                    # Extract parameter position from function signature
                    import inspect
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if ticker_param in params:
                        idx = params.index(ticker_param)
                        if idx < len(args):
                            ticker = args[idx]
            
            return await global_async_limiter.execute(func, *args, ticker=ticker, **kwargs)
        
        return wrapper
    
    return decorator


async def gather_with_rate_limit(
        tasks: List[Awaitable[T]], 
        max_concurrent: int = 3,
        delay_between_tasks: float = 1.0,
        return_exceptions: bool = False
) -> List[Union[T, Exception]]:
    """
    Gather async tasks with rate limiting.
    
    This is a safer alternative to asyncio.gather() that prevents
    overwhelming the API with too many concurrent requests.
    
    Args:
        tasks: List of awaitable tasks
        max_concurrent: Maximum number of concurrent tasks
        delay_between_tasks: Delay between starting tasks in seconds
        return_exceptions: If True, exceptions are returned rather than raised
        
    Returns:
        List of results in the same order as the tasks
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(tasks)
    
    async def run_task_with_semaphore(i, task):
        try:
            async with semaphore:
                # Add delay before task to stagger requests
                if i > 0:
                    await asyncio.sleep(delay_between_tasks)
                
                result = await task
                results[i] = result
                return result
        except Exception as e:
            if return_exceptions:
                results[i] = e
                return e
            raise
    
    # Create task wrappers but don't await them yet
    task_wrappers = [
        run_task_with_semaphore(i, task) for i, task in enumerate(tasks)
    ]
    
    # Execute tasks with controlled concurrency
    await asyncio.gather(*task_wrappers, return_exceptions=return_exceptions)
    
    return results


async def process_batch_async(
        items: List[Any],
        processor: Callable[[Any], Awaitable[T]],
        batch_size: int = 10,
        max_concurrency: int = 3
) -> List[Tuple[Any, Optional[T]]]:
    """
    Process a batch of items asynchronously with rate limiting.
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Number of items per batch
        max_concurrency: Maximum concurrent tasks
        
    Returns:
        List of tuples (item, result) where result may be None on error
    """
    all_results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = []
        
        # Process batch with controlled concurrency
        async def process_item(item):
            try:
                result = await global_async_limiter.execute(processor, item)
                return item, result
            except Exception as e:
                logger.warning(f"Error processing {item}: {str(e)}")
                return item, None
        
        # Create tasks for batch
        tasks = [process_item(item) for item in batch]
        
        # Execute batch with controlled concurrency
        batch_results = await gather_with_rate_limit(
            tasks, 
            max_concurrent=max_concurrency,
            delay_between_tasks=global_rate_limiter.get_delay(),
            return_exceptions=False
        )
        
        all_results.extend(batch_results)
        
        # Add delay between batches
        if i + batch_size < len(items):
            batch_delay = global_rate_limiter.get_batch_delay()
            logger.debug(f"Batch complete. Waiting {batch_delay:.1f}s before next batch")
            await asyncio.sleep(batch_delay)
    
    return all_results


def _should_retry_exception(error: Exception, retry_on: List[type]) -> bool:
    """
    Check if an exception should trigger a retry.
    
    Args:
        error: The exception that occurred
        retry_on: List of exception types to retry on
        
    Returns:
        True if the exception should be retried, False otherwise
    """
    for exc_type in retry_on:
        if isinstance(error, exc_type):
            return True
    return False


def _calculate_backoff_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Current attempt number (1-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay time in seconds
    """
    return min(max_delay, base_delay * (2 ** (attempt - 1)))


async def retry_async(
        func: Callable[..., Awaitable[T]],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on: List[type] = None,
        *args,
        **kwargs
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        retry_on: List of exception types to retry on
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the successful function call
        
    Raises:
        The last exception if all retries fail
    """
    if retry_on is None:
        retry_on = [RateLimitError, APIError]
    
    attempt = 0
    last_error = None
    
    while attempt <= max_retries:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            last_error = e
            
            if not _should_retry_exception(e, retry_on) or attempt > max_retries:
                raise
            
            # Calculate backoff delay
            delay = _calculate_backoff_delay(attempt, base_delay, max_delay)
            
            logger.warning(
                f"Retry {attempt}/{max_retries} after error: {str(e)}. "
                f"Waiting {delay:.1f}s"
            )
            
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    raise last_error