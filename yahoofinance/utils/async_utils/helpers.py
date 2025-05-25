"""
Asynchronous helpers for Yahoo Finance API.

This module provides helper functions for async operations, including
controlled concurrency, safe gathering, and error handling.

NOTE: This module has been reorganized to reduce code duplication.
Most functionality has been moved to enhanced.py, and this module
now re-exports it for backward compatibility.
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, TypeVar, Union

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ...utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ...core.config import RATE_LIMIT
from ...core.logging import get_logger
from ..network.circuit_breaker import CircuitOpenError
from .enhanced import (
    AsyncRateLimiter,
    PriorityAsyncRateLimiter,
    async_rate_limited,
    enhanced_async_rate_limited,
    gather_with_concurrency,
    global_async_rate_limiter,
    global_priority_rate_limiter,
)
from .enhanced import process_batch_async as enhanced_process_batch_async


logger = get_logger(__name__)

# Define a generic type variable for the return type
T = TypeVar("T")
R = TypeVar("R")

# Re-export enhanced implementations for backward compatibility


async def gather_with_semaphore(
    semaphore: asyncio.Semaphore, *tasks: Coroutine[Any, Any, T], return_exceptions: bool = False
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
        *(task_with_semaphore(task) for task in tasks), return_exceptions=return_exceptions
    )


@with_retry
async def async_bulk_fetch(
    items: List[Any],
    fetch_func: Callable[[Any], Coroutine[Any, Any, T]],
    max_concurrency: int = None,
    batch_size: int = None,
    batch_delay: float = None,
    priority_items: List[Any] = None,
    timeout_per_batch: float = None,
) -> Dict[Any, T]:
    """
    Fetch data for multiple items concurrently with rate limiting.

    This function is a wrapper around process_batch_async with slightly different
    parameters for backward compatibility. It now includes optimizations like
    item prioritization and timeout controls.

    Args:
        items: List of items to fetch data for
        fetch_func: Async function to fetch data for a single item
        max_concurrency: Maximum number of concurrent fetches
        batch_size: Size of batches for processing
        batch_delay: Delay between batches in seconds
        priority_items: Optional list of items that should be processed first
        timeout_per_batch: Optional timeout in seconds for each batch

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
        delay_between_batches=batch_delay,
        priority_items=priority_items,
        timeout_per_batch=timeout_per_batch,
    )


async def async_retry(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    **kwargs: Any,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply the delay by after each retry
        jitter: Whether to add jitter to the delay
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
        except YFinanceError as e:
            last_exception = e

            if attempt < max_retries:
                # Calculate delay for next retry
                delay = retry_delay * (backoff_factor**attempt)

                # Add jitter to avoid thundering herd problem
                if jitter:
                    import secrets

                    # Generate secure random value between 0.75 and 1.25
                    random_factor = 0.75 + (secrets.randbits(32) / (2**32 - 1)) * 0.5
                    delay = delay * random_factor

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                    f"Retrying in {delay:.2f} seconds."
                )

                # Wait before retrying
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. " f"Last error: {str(e)}")

    # If we get here, all retries failed
    assert last_exception is not None
    raise last_exception


async def prioritized_batch_process(
    items: List[T],
    processor: Callable[[T], Coroutine[Any, Any, R]],
    high_priority_items: Optional[List[T]] = None,
    medium_priority_items: Optional[List[T]] = None,
    batch_size: int = None,
    concurrency: int = None,
    delay_between_batches: float = None,
    show_progress: bool = True,
) -> Dict[T, R]:
    """
    Process items in batches with priority-based ordering.

    This function processes items in batches, prioritizing high-priority items first,
    then medium-priority, and finally low-priority items.

    Args:
        items: List of items to process
        processor: Async function to process a single item
        high_priority_items: Optional list of high-priority items
        medium_priority_items: Optional list of medium-priority items
        batch_size: Size of batches for processing
        concurrency: Maximum concurrent operations
        delay_between_batches: Delay between batches in seconds
        show_progress: Whether to show a progress bar

    Returns:
        Dictionary mapping items to their results
    """
    # Set default values
    batch_size = batch_size or RATE_LIMIT["BATCH_SIZE"]
    concurrency = concurrency or RATE_LIMIT["MAX_CONCURRENT_CALLS"]
    delay_between_batches = delay_between_batches or RATE_LIMIT["BATCH_DELAY"]

    # Create priority categories
    high_priority = set(high_priority_items or [])
    medium_priority = set(medium_priority_items or [])

    # Create ordered list of items
    ordered_items = []

    # Add high-priority items first
    for item in items:
        if item in high_priority:
            ordered_items.append(item)

    # Add medium-priority items next
    for item in items:
        if item in medium_priority and item not in high_priority:
            ordered_items.append(item)

    # Add remaining low-priority items
    for item in items:
        if item not in high_priority and item not in medium_priority:
            ordered_items.append(item)

    # Process items with priority
    return await enhanced_process_batch_async(
        items=ordered_items,
        processor=processor,
        batch_size=batch_size,
        concurrency=concurrency,
        delay_between_batches=delay_between_batches,
        show_progress=show_progress,
    )


async def adaptive_fetch(
    items: List[T],
    fetch_func: Callable[[T], Coroutine[Any, Any, R]],
    initial_concurrency: int = 5,
    max_concurrency: int = 15,
    performance_monitor_interval: int = 10,
    batch_size: int = None,
    priority_items: Optional[List[T]] = None,
) -> Dict[T, R]:
    """
    Fetch data with adaptive concurrency based on performance.

    This function dynamically adjusts concurrency based on success rates
    and response times to optimize throughput.

    Args:
        items: List of items to process
        fetch_func: Async function to process a single item
        initial_concurrency: Starting concurrency level
        max_concurrency: Maximum concurrency level
        performance_monitor_interval: Items to process before adjusting concurrency
        batch_size: Size of batches
        priority_items: Items to prioritize

    Returns:
        Dictionary mapping items to their results
    """
    if not items:
        return {}

    batch_size = batch_size or RATE_LIMIT["BATCH_SIZE"]
    results: Dict[T, R] = {}

    # Performance tracking
    concurrency = initial_concurrency
    success_count = 0
    start_time = time.time()
    last_adjustment_time = start_time
    items_since_adjustment = 0

    # Process items with adaptive concurrency
    all_items = list(items)
    # Prioritize items if specified
    if priority_items:
        # Move priority items to the beginning
        for item in reversed(priority_items):
            if item in all_items:
                all_items.remove(item)
                all_items.insert(0, item)

    # Process in batches
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i : i + batch_size]
        logger.debug(
            f"Processing batch {i//batch_size+1}/{(len(all_items)+batch_size-1)//batch_size} "
            f"with concurrency {concurrency}"
        )

        # Process batch with current concurrency
        batch_results = await enhanced_process_batch_async(
            items=batch,
            processor=fetch_func,
            batch_size=batch_size,
            concurrency=concurrency,
            show_progress=False,
        )

        # Update results
        results.update(batch_results)

        # Track performance
        batch_success = sum(1 for r in batch_results.values() if r is not None)
        success_count += batch_success
        items_since_adjustment += len(batch)

        # Check if we should adjust concurrency
        if items_since_adjustment >= performance_monitor_interval:
            current_time = time.time()
            time_elapsed = current_time - last_adjustment_time
            items_per_second = items_since_adjustment / time_elapsed
            success_rate = batch_success / len(batch) if batch else 0

            # Adjust concurrency based on performance
            if success_rate > 0.95 and concurrency < max_concurrency:
                # High success rate, increase concurrency
                concurrency = min(max_concurrency, concurrency + 1)
                logger.info(
                    f"Increased concurrency to {concurrency} "
                    f"(success rate: {success_rate:.2f}, items/s: {items_per_second:.2f})"
                )
            elif success_rate < 0.7 and concurrency > 1:
                # Low success rate, decrease concurrency
                concurrency = max(1, concurrency - 1)
                logger.info(
                    f"Decreased concurrency to {concurrency} "
                    f"(success rate: {success_rate:.2f}, items/s: {items_per_second:.2f})"
                )

            # Reset tracking
            last_adjustment_time = current_time
            items_since_adjustment = 0

    # Log final performance
    total_time = time.time() - start_time
    items_per_second = len(items) / total_time
    overall_success_rate = success_count / len(items) if items else 0

    logger.info(
        f"Processed {len(items)} items in {total_time:.2f}s "
        f"({items_per_second:.2f} items/s, success rate: {overall_success_rate:.2f})"
    )

    return results
