#!/usr/bin/env python3
"""
Integration tests for advanced utility modules: rate limiting, pagination and async helpers.

These tests verify the proper integration between different utility modules.
"""

import logging
import unittest
from unittest.mock import patch

import pytest

# Import test fixtures
from tests.fixtures import create_flaky_function, create_mock_fetcher, create_paginated_data
from yahoofinance.core.errors import RateLimitError, YFinanceError
from yahoofinance.utils.async_utils.helpers import async_retry
from yahoofinance.utils.network.pagination import PaginatedResults
from yahoofinance.utils.network.rate_limiter import RateLimiter


# Set up logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_advanced_utils")


@pytest.mark.integration
class TestAdvancedIntegration(unittest.TestCase):
    """Test the integration between rate limiting and pagination."""

    def test_paginator_with_retry_mechanism(self):
        """Test that pagination properly uses retry mechanisms."""
        # Create mock paginated data
        pages = create_paginated_data(num_pages=3, items_per_page=2)
        mock_fetcher = create_mock_fetcher(pages)

        # Create a custom fetch function that simulates rate limiting
        def fetch_page_with_rate_limiting(page_token=None):
            # Return the page data from the mock fetcher
            return mock_fetcher(page_token)

        # Create a custom process function
        def process_results(response):
            return response.get("items", [])

        # Create a custom token function
        def get_next_token(response):
            return response.get("next_page_token")

        # Initialize a PaginatedResults object with our functions
        paginator = PaginatedResults(
            fetch_page_func=fetch_page_with_rate_limiting,
            process_results_func=process_results,
            get_next_page_token_func=get_next_token,
        )

        # Fetch all pages
        results = paginator.fetch_all()

        # Verify we got all the expected items from all pages
        expected_items = []
        for page in pages:
            expected_items.extend(page["items"])

        self.assertEqual(len(results), len(expected_items))
        self.assertEqual(results, expected_items)


@pytest.mark.integration
@pytest.mark.asyncio
class TestRateLimiterErrorRecovery(unittest.IsolatedAsyncioTestCase):
    """Test the error recovery strategies with rate limiting."""

    async def test_retry_with_rate_limited_errors(self):
        """Test retry mechanism handling rate limited errors."""
        # Create a flaky function that fails due to rate limits
        flaky_func = await create_flaky_function(fail_count=2)

        # Verify retry mechanism
        result = await async_retry(
            flaky_func,
            max_retries=5,  # More than needed
            retry_delay=0.01,
            backoff_factor=2.0,
            jitter=False,
        )

        # Check results
        self.assertEqual(result, "success")

        # Test with insufficient retries
        with self.assertRaises(RateLimitError):
            flaky_func_harder = await create_flaky_function(fail_count=4)
            await async_retry(
                flaky_func_harder,
                max_retries=2,  # Not enough
                retry_delay=0.01,
                backoff_factor=2.0,
                jitter=False,
            )


if __name__ == "__main__":
    logger.info("Running advanced utility integration tests...")
    unittest.main()
