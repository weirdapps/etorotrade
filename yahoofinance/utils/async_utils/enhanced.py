"""
Enhanced async utilities with proper I/O and concurrency control.

This module provides true async I/O implementations of async helpers,
replacing the thread-pool based async with proper async/await patterns.
It includes rate limiting, circuit breaking, and retry mechanisms
designed specifically for asynchronous operations.

CANONICAL SOURCE: This module is the canonical source for AsyncRateLimiter and
related async rate limiting functionality.
"""

import asyncio
import logging
import time
import random
from typing import TypeVar, List, Dict, Any, Callable, Optional, Tuple, Set, Union, Coroutine
from functools import wraps

from ...core.config import RATE_LIMIT
from ..network.circuit_breaker import (
    get_async_circuit_breaker, 
    CircuitOpenError, 
    async_circuit_protected,
    CircuitState
)

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')

# Set up logging
logger = logging.getLogger(__name__)

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

async def retry_async_with_backoff(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    circuit_name: Optional[str] = None,
    retry_exceptions: Optional[Tuple[type, ...]] = None,
    **kwargs
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Positional arguments for function
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        circuit_name: Name of circuit breaker to use
        retry_exceptions: Exceptions to retry on (default: all except CircuitOpenError)
        **kwargs: Keyword arguments for function
        
    Returns:
        Result of the function
        
    Raises:
        CircuitOpenError: If circuit breaker is open
        Exception: If all retries fail
    """
    if retry_exceptions is None:
        # Retry all exceptions except CircuitOpenError
        retry_exceptions = (Exception,)
    
    # Create retry exclusion set - never retry these
    no_retry_exceptions = (CircuitOpenError, asyncio.CancelledError, KeyboardInterrupt)
    
    attempt = 0
    last_exception = None
    
    # For test integration, we need direct access to the circuit
    circuit = get_async_circuit_breaker(circuit_name) if circuit_name else None
    
    while attempt <= max_retries:
        # If we have a circuit breaker, check its state first
        if circuit:
            # Directly check if we should allow the request using the internal method
            if not circuit._should_allow_request():
                current_state = circuit.state  # Get the current state
                raise CircuitOpenError(
                    f"{circuit.name} is {current_state.value} - request rejected",
                    circuit_name=circuit.name,
                    circuit_state=current_state.value,
                    metrics=circuit.get_metrics()
                )
        
        try:
            # Execute the function
            if circuit:
                try:
                    # Execute with circuit recording
                    result = await func(*args, **kwargs)
                    # Explicitly record success
                    circuit.record_success()
                    return result
                except Exception as e:
                    # Explicitly record failure
                    circuit.record_failure()
                    # Re-raise for retry logic
                    raise e
            else:
                # No circuit breaker, just call the function
                return await func(*args, **kwargs)
                
        except no_retry_exceptions as e:
            # Never retry these exceptions
            raise e
            
        except retry_exceptions as e:
            attempt += 1
            last_exception = e
            
            if attempt > max_retries:
                logger.warning(f"Max retries ({max_retries}) exceeded")
                # If using circuit breaker, ensure failure is recorded
                # This is redundant but ensures consistency
                if circuit and not isinstance(e, CircuitOpenError):
                    circuit.record_failure()
                raise e
            
            # Calculate delay with jitter
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0.75, 1.25)  # Add 25% jitter
            delay = delay * jitter
            
            logger.debug(f"Retry {attempt}/{max_retries} after error: {str(e)}. Waiting {delay:.2f}s")
            await asyncio.sleep(delay)
    
    # This should never happen due to the raise in the loop
    raise last_exception if last_exception else RuntimeError("Unexpected error in retry_async")

def async_rate_limited(rate_limiter: Optional[AsyncRateLimiter] = None):
    """
    Decorator for rate-limiting async functions.
    
    Args:
        rate_limiter: Rate limiter to use (uses global_async_rate_limiter if None)
        
    Returns:
        Decorated async function
    """
    # Use global rate limiter if not provided
    if rate_limiter is None:
        rate_limiter = global_async_rate_limiter
    
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

async def gather_with_concurrency(
    coros: List[Coroutine[Any, Any, T]], 
    limit: int = 5
) -> List[T]:
    """
    Run coroutines with a limit on concurrency.
    
    Args:
        coros: List of coroutines to execute
        limit: Maximum number of concurrent coroutines
        
    Returns:
        List of results in the same order as the input coroutines
    """
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(limit)
    
    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro
    
    # Wrap all coroutines with the semaphore
    tasks = [run_with_semaphore(coro) for coro in coros]
    
    # Run all tasks and return results
    return await asyncio.gather(*tasks)

async def process_batch_async(
    items: List[T],
    processor: Callable[[T], Coroutine[Any, Any, R]],
    batch_size: int = None,
    concurrency: int = None,
    delay_between_batches: float = None,
    description: str = "Processing items",
    show_progress: bool = True
) -> Dict[T, R]:
    """
    Process a batch of items asynchronously with rate limiting and enhanced progress reporting.
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Number of items per batch (default from config)
        concurrency: Maximum concurrent tasks (default from config)
        delay_between_batches: Delay between batches in seconds (default from config)
        description: Description for the progress bar
        show_progress: Whether to show a progress bar
    
    Returns:
        Dictionary mapping input items to their results
    """
    # Use defaults from config if not specified
    batch_size = batch_size or RATE_LIMIT["BATCH_SIZE"]
    concurrency = concurrency or RATE_LIMIT["MAX_CONCURRENT_CALLS"]
    delay_between_batches = delay_between_batches or RATE_LIMIT["BATCH_DELAY"]
    
    results: Dict[T, R] = {}
    total_items = len(items)
    total_batches = (total_items + batch_size - 1) // batch_size
    start_time = time.time()
    success_count = 0
    error_count = 0
    cache_hits = 0
    
    # Initialize progress bar if enabled
    if show_progress:
        from tqdm import tqdm
        progress_bar = tqdm(
            total=total_items,
            desc=description,
            unit="item",
            bar_format="{desc} {percentage:3.0f}% |{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100
        )
    
    # Process items in batches
    for i in range(0, total_items, batch_size):
        batch = items[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        # Update progress bar description
        if show_progress:
            elapsed = time.time() - start_time
            items_per_second = max((success_count + error_count) / max(elapsed, 0.1), 0.01)
            remaining_items = total_items - (success_count + error_count)
            estimated_remaining = remaining_items / items_per_second
            
            # Show the last item being processed in the description
            current_item = batch[-1] if batch else ""
            item_str = f"{current_item:<10}" if current_item else ""
            progress_bar.set_description(f"⚡ {item_str} Batch {batch_num:2d}/{total_batches:2d}")
            
            # Also update postfix with rate and ETA
            progress_bar.set_postfix_str(
                f"{items_per_second:.2f} item/s, ETA: {time.strftime('%M:%S', time.gmtime(estimated_remaining))}"
            )
        
        logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
        
        # Process batch with concurrency limit
        batch_coroutines = [processor(item) for item in batch]
        batch_results = await gather_with_concurrency(batch_coroutines, limit=concurrency)
        
        # Map results back to original items
        for item, result in zip(batch, batch_results):
            if result is not None:
                results[item] = result
                success_count += 1
                
                # Check if this was a cache hit
                if isinstance(result, dict) and result.get('_cache_hit') is True:
                    cache_hits += 1
            else:
                error_count += 1
        
        # Update progress bar
        if show_progress:
            progress_bar.update(len(batch))
        
        # Delay between batches (except for the last batch)
        if i + batch_size < total_items and delay_between_batches > 0:
            if show_progress:
                progress_bar.set_description(f"⏳ Waiting {delay_between_batches:.1f}s")
            
            logger.debug(f"Waiting {delay_between_batches}s between batches")
            await asyncio.sleep(delay_between_batches)
    
    # Close progress bar if used
    if show_progress:
        progress_bar.close()
        
        # Print final summary
        elapsed = time.time() - start_time
        items_per_second = total_items / max(elapsed, 0.1)
        print(f"Processed {total_items} items in {elapsed:.1f}s ({items_per_second:.2f}/s) - "
              f"Success: {success_count}, Errors: {error_count}, Cache hits: {cache_hits}")
    
    return results

# Create a global async rate limiter instance for use across the application
global_async_rate_limiter = AsyncRateLimiter()

def enhanced_async_rate_limited(
    circuit_name: Optional[str] = None,
    max_retries: int = 3,
    rate_limiter: Optional[AsyncRateLimiter] = None,
):
    """
    Enhanced decorator combining rate limiting, circuit breaking, and retries.
    
    Args:
        circuit_name: Name of circuit breaker (disables circuit breaking if None)
        max_retries: Maximum retry attempts (disables retries if 0)
        rate_limiter: Rate limiter to use (creates new one if None)
        
    Returns:
        Decorated async function with all protections
    """
    # Create rate limiter if not provided
    if rate_limiter is None:
        rate_limiter = global_async_rate_limiter
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Apply retries with circuit breaking
            if max_retries > 0:
                retry_func = retry_async_with_backoff(
                    # Apply rate limiting to the original function
                    async_rate_limited(rate_limiter)(func),
                    *args,
                    max_retries=max_retries,
                    circuit_name=circuit_name,
                    **kwargs
                )
                return await retry_func
            else:
                # Just apply rate limiting and circuit breaking without retries
                rate_limited_func = async_rate_limited(rate_limiter)(func)
                
                if circuit_name:
                    circuit = get_async_circuit_breaker(circuit_name)
                    return await circuit.execute_async(rate_limited_func, *args, **kwargs)
                else:
                    return await rate_limited_func(*args, **kwargs)
                    
        return wrapper
    return decorator