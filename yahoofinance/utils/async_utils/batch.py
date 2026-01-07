"""
Batch processing utilities for async operations.

This module provides utilities for processing batches of items asynchronously
with concurrency control, progress reporting, and performance metrics.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

from yahoofinance.core.config import RATE_LIMIT
from yahoofinance.core.logging import get_logger

# Type variables for generics
T = TypeVar("T")
R = TypeVar("R")

# Set up logging
logger = get_logger(__name__)

# Global variable to store processing statistics for display after table
_last_processing_stats = None


async def gather_with_concurrency(coros: List[Coroutine[Any, Any, T]], limit: int = 5) -> List[T]:
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

    async def run_with_semaphore(coro, index):
        async with semaphore:
            try:
                return await coro, index, None
            except (asyncio.CancelledError, asyncio.TimeoutError, ValueError, TypeError, KeyError, RuntimeError) as e:
                # Catch and return the exception instead of letting it propagate
                logger.error(f"Task {index} failed with exception: {type(e).__name__}: {e}")
                return None, index, e

    # Wrap all coroutines with the semaphore and include their original index
    tasks = [run_with_semaphore(coro, i) for i, coro in enumerate(coros)]

    # Run all tasks and get results with their indices
    results_with_indices = await asyncio.gather(*tasks)

    # Create a list of the correct size to hold results
    results = [None] * len(coros)

    # Process results using their original indices
    for result, index, error in results_with_indices:
        results[index] = result  # Place result at correct position

    return results  # type: ignore[return-value]


async def process_batch_async(
    items: List[T],
    processor: Callable[[T], Coroutine[Any, Any, R]],
    batch_size: int = None,
    concurrency: int = None,
    delay_between_batches: float = None,
    description: str = "Processing items",
    show_progress: bool = True,
    priority_items: Optional[List[T]] = None,
    timeout_per_batch: Optional[float] = None,
) -> Dict[T, R]:
    """
    Process a batch of items asynchronously with rate limiting, prioritization, and enhanced progress reporting.

    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Number of items per batch (default from config)
        concurrency: Maximum concurrent tasks (default from config)
        delay_between_batches: Delay between batches in seconds (default from config)
        description: Description for the progress bar
        show_progress: Whether to show a progress bar
        priority_items: Optional list of high-priority items to process first
        timeout_per_batch: Optional timeout in seconds for each batch

    Returns:
        Dictionary mapping input items to their results
    """
    # Use defaults from config if not specified
    batch_size = batch_size or RATE_LIMIT["BATCH_SIZE"]
    concurrency = concurrency or RATE_LIMIT["MAX_CONCURRENT_CALLS"]
    delay_between_batches = delay_between_batches or RATE_LIMIT["BATCH_DELAY"]

    if not items:
        return {}

    # Sort items by priority if specified
    processed_items = list(items)
    if priority_items:
        # Move priority items to the beginning
        for item in reversed(priority_items):
            if item in processed_items:
                processed_items.remove(item)
                processed_items.insert(0, item)

    results: Dict[T, R] = {}
    total_items = len(processed_items)
    total_batches = (total_items + batch_size - 1) // batch_size
    start_time = time.time()
    success_count = 0
    error_count = 0
    cache_hits = 0

    # Initialize progress bar if enabled
    if show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=total_items,
                desc=description,
                unit="ticker",
                bar_format="{desc:15} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ncols=80,
                colour="green",
            )
        except ImportError:
            show_progress = False
            logger.warning("tqdm not installed, progress bar disabled")

    # Process items in batches
    for i in range(0, total_items, batch_size):
        batch = processed_items[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Update progress bar description
        if show_progress:
            elapsed = time.time() - start_time
            items_per_second = max((success_count + error_count) / max(elapsed, 0.1), 0.01)
            remaining_items = total_items - (success_count + error_count)
            estimated_remaining = remaining_items / items_per_second

            # Show the last item being processed in the description with cleaner format
            current_item = batch[-1] if batch else ""
            item_str = f"{current_item:<12}" if current_item else ""
            progress_bar.set_description(f"Processing {item_str}")

            # Update postfix with cleaner rate display
            progress_bar.set_postfix_str(
                f"{items_per_second:.1f}/s"
            )

        logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")

        if timeout_per_batch:
            # Process batch with concurrency limit, and optional timeout
            batch_coroutines = [processor(item) for item in batch]
            # Use asyncio.wait_for with timeout for the entire batch
            try:
                batch_results = await asyncio.wait_for(  # type: ignore[assignment]
                    gather_with_concurrency(batch_coroutines, limit=concurrency),
                    timeout=timeout_per_batch,
                )
                # Map results back to original items
                for item, result in zip(batch, batch_results):
                    if result is not None:
                        results[item] = result
                        success_count += 1

                        # Check if this was a cache hit
                        if isinstance(result, dict) and result.get("_cache_hit") is True:
                            cache_hits += 1
                    else:
                        error_count += 1

                    # Smooth progress update: Update after each item
                    if show_progress:
                        progress_bar.update(1)
                        progress_bar.refresh()  # Force immediate display update

            except asyncio.TimeoutError:
                logger.warning(f"Batch {batch_num} timed out after {timeout_per_batch}s")
                # Update progress even for timed out batch
                if show_progress:
                    progress_bar.update(len(batch))
        else:
            # SMOOTH PROGRESS: Process items as they complete for real-time updates
            # Use enumerate to track items instead of task mapping to avoid KeyError
            batch_results = await asyncio.gather(  # type: ignore[assignment]
                *[processor(item) for item in batch],
                return_exceptions=True
            )

            # Process results and update progress as we go
            for i, (item, result) in enumerate(zip(batch, batch_results)):
                try:
                    if isinstance(result, Exception):
                        logger.warning(f"Error processing {item}: {result}")
                        error_count += 1
                    elif result is not None:
                        results[item] = result
                        success_count += 1

                        # Check if this was a cache hit
                        if isinstance(result, dict) and result.get("_cache_hit") is True:
                            cache_hits += 1
                    else:
                        error_count += 1

                except (asyncio.CancelledError, asyncio.TimeoutError, ValueError, TypeError, KeyError, RuntimeError) as e:
                    # Handle individual item errors
                    logger.warning(f"Error processing {item}: {e}")
                    error_count += 1

                # Smooth progress update: Update immediately after each item completes
                if show_progress:
                    progress_bar.update(1)
                    progress_bar.refresh()  # Force immediate display update

        # Delay between batches (except for the last batch) - OPTIMIZED: No delay for maximum performance
        if i + batch_size < total_items and delay_between_batches > 0:
            if show_progress:
                progress_bar.set_description(f"Throttling...")

            logger.debug(f"Batch delay disabled for optimal performance (was {delay_between_batches}s)")
            # await asyncio.sleep(delay_between_batches)  # Disabled for performance optimization

    # Close progress bar if used
    if show_progress:
        progress_bar.close()

    # Calculate final statistics
    elapsed = time.time() - start_time
    items_per_second = total_items / max(elapsed, 0.1)
    seconds_per_item = elapsed / max(total_items, 1)

    # Store statistics globally for later display (after table output)
    global _last_processing_stats
    _last_processing_stats = {
        'total_items': total_items,
        'elapsed': elapsed,
        'items_per_second': items_per_second,
        'seconds_per_item': seconds_per_item,
        'success_count': success_count,
        'error_count': error_count,
        'cache_hits': cache_hits,
        'show_progress': show_progress
    }

    # Store statistics in results metadata as well
    if hasattr(results, '__dict__'):
        results._processing_stats = _last_processing_stats  # type: ignore[attr-defined]

    return results


def display_processing_stats():
    """Display the stored processing statistics after table output."""
    global _last_processing_stats
    if _last_processing_stats and _last_processing_stats.get('show_progress'):
        stats = _last_processing_stats
        total = stats['total_items']
        if total > 0:
            print(f"\n✅ Processing complete: {stats['success_count']}/{total} succeeded, "
                  f"{stats['error_count']} failed | {stats['elapsed']:.1f}s total ({stats['seconds_per_item']:.1f}s/ticker)")
            if stats['cache_hits'] > 0:
                print(f"   Cache hits: {stats['cache_hits']} ({stats['cache_hits']*100//total}% from cache)")
        _last_processing_stats = None  # Clear after displaying


def get_processing_stats():
    """Get the stored processing statistics without clearing them."""
    # No need for global declaration since we're not reassigning _last_processing_stats
    return _last_processing_stats
