"""
Batch processing utilities for Yahoo Finance API.

This module provides utilities for processing API requests in batches
to improve performance while respecting rate limits.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Generic, Iterable, List, TypeVar

from ...core.config import RATE_LIMIT
from ...core.errors import APIError, DataError, ValidationError, YFinanceError
from ...core.logging import get_logger
from ..error_handling import enrich_error_context, safe_operation, translate_error, with_retry
from .rate_limiter import global_rate_limiter


logger = get_logger(__name__)

# Define generic type variables
T = TypeVar("T")  # Type of input items
R = TypeVar("R")  # Type of result items


class BatchProcessor(Generic[T, R]):
    """
    Process items in batches with rate limiting.

    This class handles the processing of items in batches, with
    appropriate rate limiting and error handling.

    Attributes:
        process_func: Function to process a single item
        batch_size: Number of items per batch
        batch_delay: Delay between batches in seconds
        max_workers: Maximum number of worker threads
        rate_limiter: Rate limiter to use
    """

    def __init__(
        self,
        process_func: Callable[[T], R],
        batch_size: int = RATE_LIMIT["BATCH_SIZE"],
        batch_delay: float = RATE_LIMIT["BATCH_DELAY"],
        max_workers: int = RATE_LIMIT["MAX_CONCURRENT_CALLS"],
        rate_limiter=global_rate_limiter,
    ):
        """
        Initialize the batch processor.

        Args:
            process_func: Function to process a single item
            batch_size: Number of items per batch
            batch_delay: Delay between batches in seconds
            max_workers: Maximum number of worker threads
            rate_limiter: Rate limiter to use
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.max_workers = max_workers
        self.rate_limiter = rate_limiter

        # Initialize threading event for cancellation
        self.cancel_event = threading.Event()

        logger.debug(
            f"Initialized batch processor with batch_size={batch_size}, "
            f"batch_delay={batch_delay}, max_workers={max_workers}"
        )

    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """
        Create batches from a list of items.

        Args:
            items: List of items to process

        Returns:
            List of batches, where each batch is a list of items
        """
        return [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]

    def _process_batch(self, batch: List[T], results: Dict[int, R], offset: int) -> None:
        """
        Process a batch of items.

        Args:
            batch: List of items to process
            results: Dictionary to store results
            offset: Offset of the batch in the original list
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Define a worker function that adds results to the dictionary
            def worker(index: int, item: T) -> None:
                if self.cancel_event.is_set():
                    return

                try:
                    # Process the item
                    result = self.process_func(item)
                    # Store the result
                    results[offset + index] = result
                except YFinanceError as e:
                    logger.error(f"Error processing item {item}: {str(e)}")
                    # Store the error in the results
                    results[offset + index] = None

            # Submit tasks to the executor
            futures = [executor.submit(worker, i, item) for i, item in enumerate(batch)]

            # Wait for all tasks to complete
            for future in futures:
                if self.cancel_event.is_set():
                    break
                future.result()

    def process(self, items: List[T]) -> List[R]:
        """
        Process a list of items in batches.

        Args:
            items: List of items to process

        Returns:
            List of results in the same order as the input items
        """
        if not items:
            return []

        # Reset cancellation event
        self.cancel_event.clear()

        # Create batches
        batches = self._create_batches(items)
        logger.info(f"Processing {len(items)} items in {len(batches)} batches")

        # Process batches
        results: Dict[int, R] = {}
        for i, batch in enumerate(batches):
            if self.cancel_event.is_set():
                break

            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} items)")

            # Calculate offset in the original list
            offset = i * self.batch_size

            # Process the batch
            self._process_batch(batch, results, offset)

            # Wait before processing the next batch
            if i < len(batches) - 1 and not self.cancel_event.is_set():
                logger.debug(f"Waiting {self.batch_delay} seconds before next batch")
                time.sleep(self.batch_delay)

        # Convert results to a list in the same order as the input items
        return [results.get(i) for i in range(len(items))]

    def cancel(self) -> None:
        """
        Cancel the batch processing.

        This method can be called from another thread to cancel the
        batch processing.
        """
        self.cancel_event.set()
        logger.info("Batch processing cancelled")


def batch_process(
    items: List[T],
    process_func: Callable[[T], R],
    batch_size: int = RATE_LIMIT["BATCH_SIZE"],
    batch_delay: float = RATE_LIMIT["BATCH_DELAY"],
    max_workers: int = RATE_LIMIT["MAX_CONCURRENT_CALLS"],
) -> List[R]:
    """
    Process a list of items in batches.

    This is a convenience function that creates a BatchProcessor and
    processes the items.

    Args:
        items: List of items to process
        process_func: Function to process a single item
        batch_size: Number of items per batch
        batch_delay: Delay between batches in seconds
        max_workers: Maximum number of worker threads

    Returns:
        List of results in the same order as the input items
    """
    processor = BatchProcessor(
        process_func=process_func,
        batch_size=batch_size,
        batch_delay=batch_delay,
        max_workers=max_workers,
    )
    return processor.process(items)


@with_retry
def bulk_fetch(
    items: Iterable[T],
    fetch_func: Callable[[T], R],
    transform_func: Callable[[Dict[T, R]], Dict[str, Any]] = None,
    batch_size: int = RATE_LIMIT["BATCH_SIZE"],
    batch_delay: float = RATE_LIMIT["BATCH_DELAY"],
    max_workers: int = RATE_LIMIT["MAX_CONCURRENT_CALLS"],
) -> Dict[T, R]:
    """
    Fetch data for multiple items in batches.

    This function fetches data for multiple items in batches, with
    appropriate rate limiting and error handling. It returns a dictionary
    mapping items to their results.

    Args:
        items: Iterable of items to fetch data for
        fetch_func: Function to fetch data for a single item
        transform_func: Optional function to transform the results
        batch_size: Number of items per batch
        batch_delay: Delay between batches in seconds
        max_workers: Maximum number of worker threads

    Returns:
        Dictionary mapping items to their results
    """
    # Convert items to a list if not already
    items_list = list(items)
    if not items_list:
        return {}

    # Create a dictionary to store results
    results: Dict[T, R] = {}

    # Define a worker function that adds results to the dictionary
    def worker(item: T) -> None:
        try:
            # Fetch data for the item
            result = fetch_func(item)
            # Store the result
            results[item] = result
        except YFinanceError as e:
            logger.error(f"Error fetching data for {item}: {str(e)}")
            # Store None as the result
            results[item] = None

    # Process items in batches
    batch_process(
        items=items_list,
        process_func=worker,
        batch_size=batch_size,
        batch_delay=batch_delay,
        max_workers=max_workers,
    )

    # Apply transform function if provided
    if transform_func is not None:
        return transform_func(results)

    return results
