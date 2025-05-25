#!/usr/bin/env python3
"""
Tests for asynchronous utilities and pagination

This test file verifies:
- Pagination utilities for handling API results
- Async helpers with proper rate limiting
- Bulk processing of data with safety controls
"""

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Import test fixtures to reduce duplication
from tests.fixtures.async_fixtures import (
    create_async_processor_mock,
    create_bulk_fetch_mocks,
    create_flaky_function,
    create_mock_fetcher,
    create_paginated_data,
)
from yahoofinance.core.errors import APIError, RateLimitError, YFinanceError
from yahoofinance.core.logging import get_logger
from yahoofinance.utils.async_utils.enhanced import (
    AsyncRateLimiter,
    async_rate_limited,
    process_batch_async,
)
from yahoofinance.utils.async_utils.helpers import async_retry, gather_with_concurrency
from yahoofinance.utils.network.batch import batch_process
from yahoofinance.utils.network.pagination import PaginatedResults, paginated_request


# Set up logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)


class TestPagination(unittest.TestCase):
    """Test the pagination utilities."""

    def test_paginated_results(self):
        """Test iterating through paginated results."""
        # Create mock paginated data
        pages = create_paginated_data(num_pages=3, items_per_page=3)
        mock_fetcher = create_mock_fetcher(pages)

        # Create functions for PaginatedResults
        def process_results(page_data):
            return page_data["items"]

        def get_next_token(page_data):
            return page_data["next_page_token"]

        # Create paginated results iterator
        paginator = PaginatedResults(
            fetch_page_func=mock_fetcher,
            process_results_func=process_results,
            get_next_page_token_func=get_next_token,
        )

        # Fetch all results
        all_items = paginator.fetch_all()

        # Should have all items (1-9)
        expected_items = list(range(1, 10))
        self.assertEqual(all_items, expected_items)

    def test_paginated_request_function(self):
        """Test the paginated_request convenience function."""
        # Create mock paginated data with 2 pages
        pages = create_paginated_data(num_pages=2, items_per_page=3)
        mock_fetcher = create_mock_fetcher(pages)

        # Create functions for paginated_request
        def process_results(page_data):
            return page_data["items"]

        def get_next_token(page_data):
            return page_data["next_page_token"]

        # Get all results using the alias function
        all_items = paginated_request(
            fetch_page_func=mock_fetcher,
            process_results_func=process_results,
            get_next_page_token_func=get_next_token,
            max_pages=2,
        )

        # Should have all items (1-6)
        expected_items = list(range(1, 7))
        self.assertEqual(all_items, expected_items)

    @patch("yahoofinance.utils.network.batch.global_rate_limiter")  # Mock the rate limiter
    @patch("yahoofinance.utils.network.rate_limiter.RateLimiter")  # Mock the rate limiter class
    def test_batch_process(self, mock_rate_limiter_class, mock_rate_limiter):
        """Test batch processing with rate limiting."""
        # Set up rate limiter mock
        mock_rate_limiter.return_value = 0.0  # No delay for tests

        # Set up a mock rate limiter instance to return from the class
        mock_instance = MagicMock()
        mock_instance.get_delay.return_value = 0.0
        mock_rate_limiter_class.return_value = mock_instance

        # Create mock executor that doesn't actually run in threads
        with patch("yahoofinance.utils.network.batch.ThreadPoolExecutor") as mock_executor_class:
            # Mock the executor to actually run the function synchronously
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            def submit_effect(fn, *args, **kwargs):
                # Make the submit function actually run the function with its args
                mock_future = MagicMock()
                mock_future.result = lambda: fn(*args, **kwargs)
                return mock_future

            mock_executor.submit.side_effect = submit_effect

            # Create test items and functions
            items = [1, 2, 3, 4, 5]

            def mock_processor(item):
                if item == 3:
                    # Simulate an error
                    raise YFinanceError("Test error")
                return item * 2

            # Fetch results for multiple items
            results = batch_process(
                items=items,
                process_func=mock_processor,
                batch_size=2,
                max_workers=2,  # Use a small worker pool for testing
            )

            # Check results - None for item 3 (error), others should be item * 2
            expected = [2, 4, None, 8, 10]
            self.assertEqual(results, expected)


class TestAsyncHelpers(unittest.IsolatedAsyncioTestCase):
    """Test async helper functions."""

    async def test_async_rate_limited_decorator(self):
        """Test the async_rate_limited decorator."""
        # Use a local AsyncRateLimiter instance for testing
        test_limiter = AsyncRateLimiter(
            window_size=60,
            max_calls=60,
            base_delay=0.01,  # Very small delay for testing
            min_delay=0.01,
            max_delay=0.01,
        )

        # Mock the wait method to return immediately
        original_wait = test_limiter.wait
        test_limiter.wait = AsyncMock(return_value=0.0)

        try:
            calls = []

            @async_rate_limited(rate_limiter=test_limiter)
            async def test_func(ticker, value):
                calls.append((ticker, value))
                return f"{ticker}:{value}"

            # Call the function a few times
            for i in range(3):
                result = await test_func("AAPL", i)
                self.assertEqual(result, f"AAPL:{i}")

            # All calls should be in the calls list
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls, [("AAPL", 0), ("AAPL", 1), ("AAPL", 2)])

            # Verify wait was called the correct number of times
            self.assertEqual(test_limiter.wait.call_count, 3)

            # Verify that wait was called with the correct ticker keyword argument
            for call in test_limiter.wait.mock_calls:
                # Check that the 'ticker' keyword argument was passed
                self.assertIn("ticker", call.kwargs)
                self.assertEqual(call.kwargs["ticker"], "AAPL")
        finally:
            # Restore original wait method
            test_limiter.wait = original_wait

    async def test_gather_with_concurrency(self):
        """Test gathering async tasks with rate limiting."""

        # Create simple async functions that resolve immediately
        async def coro(i):
            return i * 2

        # Create a list of coroutines
        coroutines = [coro(i) for i in range(5)]

        # Use gather_with_concurrency with a concurrency limit of 2
        results = await gather_with_concurrency(limit=2, coros=coroutines)

        # Verify results are in the expected order
        self.assertEqual(results, [0, 2, 4, 6, 8])

    async def test_process_batch_async(self):
        """Test async batch processing."""
        # Get mock async processor
        mock_processor = create_async_processor_mock(error_item=3)

        # Process batch of items
        items = [1, 2, 3, 4, 5]
        results = await process_batch_async(
            items=items,
            processor=mock_processor,
            batch_size=2,
            concurrency=2,  # Parameter name is concurrency not max_concurrency
        )

        # process_batch_async now returns a dictionary
        # Remove None values for comparison
        cleaned_results = {k: v for k, v in results.items() if v is not None}
        self.assertEqual(cleaned_results, {1: 2, 2: 4, 4: 8, 5: 10})

    async def test_retry_async(self):
        """Test async retry with exponential backoff."""
        # Create flaky function that succeeds on third try
        flaky_function = await create_flaky_function(fail_count=2)

        # Retry the flaky function
        result = await async_retry(
            flaky_function, max_retries=3, retry_delay=0.1, backoff_factor=2.0
        )

        # Check results
        self.assertEqual(result, "success")


if __name__ == "__main__":
    unittest.main()
