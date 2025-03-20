"""
Asynchronous helpers for Yahoo Finance API.

This module provides helper functions for async operations, including
controlled concurrency, safe gathering, and error handling.
"""

import logging
import asyncio
import time
import random
from typing import List, TypeVar, Any, Callable, Coroutine, Dict, Optional, Set, Union
from functools import wraps

from ...core.config import RATE_LIMIT
from ..network.circuit_breaker import CircuitOpenError

logger = logging.getLogger(__name__)

# Define a generic type variable for the return type
T = TypeVar('T')

async def gather_with_concurrency(
    n: int,
    *tasks: Coroutine[Any, Any, T],
    return_exceptions: bool = False
) -> List[T]:
    """
    Run tasks with a limit on concurrent executions.
    
    This function is similar to asyncio.gather, but limits the number of
    tasks that can run concurrently.
    
    Args:
        n: Maximum number of concurrent tasks
        *tasks: Tasks to run
        return_exceptions: Whether to return exceptions instead of raising them
        
    Returns:
        List of results from the tasks
    """
    semaphore = asyncio.Semaphore(n)
    
    async def task_with_semaphore(task: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *(task_with_semaphore(task) for task in tasks),
        return_exceptions=return_exceptions
    )


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
    
    This function fetches data for multiple items concurrently, with
    appropriate rate limiting and error handling.
    
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
    
    # Create batches
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]
    logger.info(f"Processing {len(items)} items in {len(batches)} batches "
               f"with max_concurrency={max_concurrency}")
    
    # Create result dictionary
    results: Dict[Any, T] = {}
    
    # Create semaphore for limiting concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # Process batches
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} items)")
        
        # Create tasks for the batch
        tasks = [fetch_func(item) for item in batch]
        
        # Run tasks with limited concurrency
        batch_results = await gather_with_semaphore(
            semaphore, *tasks, return_exceptions=True
        )
        
        # Process results
        for item, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {item}: {str(result)}")
                results[item] = None  # type: ignore
            else:
                results[item] = result
        
        # Wait before processing the next batch
        if i < len(batches) - 1:
            logger.debug(f"Waiting {batch_delay} seconds before next batch")
            await asyncio.sleep(batch_delay)
    
    return results


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


class AsyncRateLimiter:
    """
    True async implementation of rate limiting with adaptive delay.
    
    This class implements proper async/await patterns for rate limiting,
    rather than using thread pools to simulate async behavior.
    """
    
    def __init__(self, 
                 window_size: int = None,
                 max_calls: int = None,
                 base_delay: float = None,
                 min_delay: float = None,
                 max_delay: float = None):
        """
        Initialize async rate limiter.
        
        Args:
            window_size: Time window in seconds
            max_calls: Maximum calls per window
            base_delay: Base delay between calls
            min_delay: Minimum delay after successful calls
            max_delay: Maximum delay after failures
        """
        self.window_size = window_size or RATE_LIMIT["WINDOW_SIZE"]
        self.max_calls = max_calls or RATE_LIMIT["MAX_CALLS"]
        self.base_delay = base_delay or RATE_LIMIT["BASE_DELAY"]
        self.min_delay = min_delay or RATE_LIMIT["MIN_DELAY"]
        self.max_delay = max_delay or RATE_LIMIT["MAX_DELAY"]
        
        # State tracking
        self.call_times: List[float] = []
        self.current_delay = self.base_delay
        self.success_streak = 0
        self.failure_streak = 0
        self.last_call_time = 0
        
        # Lock for concurrency safety
        self.lock = asyncio.Lock()
        
        logger.debug(
            f"Initialized AsyncRateLimiter: window={self.window_size}s, "
            f"max_calls={self.max_calls}, base_delay={self.base_delay}s"
        )
    
    async def wait(self) -> float:
        """
        Wait for appropriate delay before making a call.
        
        Returns:
            Actual delay in seconds
        """
        async with self.lock:
            now = time.time()
            
            # Clean up old call times outside the window
            window_start = now - self.window_size
            self.call_times = [t for t in self.call_times if t >= window_start]
            
            # Calculate time until we can make another call
            if len(self.call_times) >= self.max_calls:
                # We've hit the limit, need to wait until oldest call exits the window
                oldest_call = min(self.call_times)
                wait_time = oldest_call + self.window_size - now
                
                if wait_time > 0:
                    logger.debug(f"Rate limit reached. Waiting {wait_time:.2f}s")
                    # Release lock during wait
                    self.lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        # Re-acquire lock
                        await self.lock.acquire()
                    
                    # Recalculate now after waiting
                    now = time.time()
            
            # Add delay between calls
            time_since_last_call = now - self.last_call_time if self.last_call_time > 0 else self.current_delay
            
            # If we need to add additional delay
            additional_delay = max(0, self.current_delay - time_since_last_call)
            
            if additional_delay > 0:
                logger.debug(f"Adding delay of {additional_delay:.2f}s between calls")
                # Release lock during wait
                self.lock.release()
                try:
                    await asyncio.sleep(additional_delay)
                finally:
                    # Re-acquire lock
                    await self.lock.acquire()
            
            # Record this call
            call_time = time.time()
            self.call_times.append(call_time)
            self.last_call_time = call_time
            
            # Calculate actual delay that was enforced
            actual_delay = time_since_last_call if additional_delay <= 0 else self.current_delay
            
            return actual_delay
    
    async def record_success(self) -> None:
        """Record a successful API call and adjust delay"""
        async with self.lock:
            self.success_streak += 1
            self.failure_streak = 0
            
            # Reduce delay after consecutive successes, but not below minimum
            if self.success_streak >= 5:
                self.current_delay = max(self.min_delay, self.current_delay * 0.9)
                logger.debug(f"Reduced delay to {self.current_delay:.2f}s after {self.success_streak} successes")
    
    async def record_failure(self, is_rate_limit: bool = False) -> None:
        """
        Record a failed API call and adjust delay.
        
        Args:
            is_rate_limit: Whether the failure was due to rate limiting
        """
        async with self.lock:
            self.failure_streak += 1
            self.success_streak = 0
            
            # Increase delay after failures, with larger increase for rate limit errors
            if is_rate_limit:
                # Double delay for rate limit errors
                self.current_delay = min(self.max_delay, self.current_delay * 2.0)
                logger.warning(f"Rate limit detected. Increased delay to {self.current_delay:.2f}s")
            else:
                # Smaller increase for other errors
                self.current_delay = min(self.max_delay, self.current_delay * 1.5)
                logger.debug(f"Increased delay to {self.current_delay:.2f}s after failure")


def async_rate_limited(rate_limiter: Optional[AsyncRateLimiter] = None):
    """
    Decorator for rate-limiting async functions.
    
    Args:
        rate_limiter: Rate limiter to use (creates new one if None)
        
    Returns:
        Decorated async function
    """
    # Create rate limiter if not provided
    if rate_limiter is None:
        rate_limiter = AsyncRateLimiter()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Wait for rate limiting
            await rate_limiter.wait()
            
            try:
                # Call function
                result = await func(*args, **kwargs)
                
                # Record success
                await rate_limiter.record_success()
                
                return result
                
            except Exception as e:
                # Record failure (check if it's a rate limit error)
                is_rate_limit = any(
                    err_text in str(e).lower() 
                    for err_text in ["rate limit", "too many requests", "429"]
                )
                await rate_limiter.record_failure(is_rate_limit=is_rate_limit)
                
                # Re-raise the exception
                raise e
                
        return wrapper
    return decorator