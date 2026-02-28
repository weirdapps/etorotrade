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
)
from yahoofinance.core.errors import APIError, RateLimitError, YFinanceError
from yahoofinance.core.logging import get_logger
from yahoofinance.utils.async_utils.enhanced import (
    AsyncRateLimiter,
    async_rate_limited,
    process_batch_async,
)
from yahoofinance.utils.async_utils.helpers import async_retry, gather_with_concurrency


# Set up logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)


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
