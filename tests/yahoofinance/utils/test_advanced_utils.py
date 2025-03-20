#!/usr/bin/env python3
"""
Integration tests for advanced utility modules: rate limiting, pagination and async helpers.

These tests verify the proper integration between different utility modules.
"""

import unittest
import pytest
import logging
from unittest.mock import patch

from yahoofinance.utils.network.rate_limiter import global_rate_limiter
from yahoofinance.utils.network.pagination import PaginatedResults
from yahoofinance.utils.async_utils import retry_async
from yahoofinance.core.errors import RateLimitError

# Import test fixtures
from ...fixtures import (
    create_paginated_data, create_mock_fetcher, create_flaky_function
)

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_advanced_utils")


@pytest.mark.integration
class TestAdvancedIntegration(unittest.TestCase):
    """Test the integration between rate limiting and pagination."""
    
    @patch('yahoofinance.utils.network.pagination.global_rate_limiter')
    def test_paginator_with_rate_limiter(self, mock_rate_limiter):
        """Test that pagination properly uses rate limiting."""
        # Setup mock rate limiter
        mock_rate_limiter.get_delay.return_value = 0.01
        mock_rate_limiter.add_call.return_value = None
        
        # Create mock paginated data
        pages = create_paginated_data(num_pages=3, items_per_page=2)
        mock_fetcher = create_mock_fetcher(pages)
        
        # Create paginated results iterator
        paginator = PaginatedResults(
            fetcher=mock_fetcher,
            items_key="items",
            token_key="next_page_token",
            ticker="AAPL"
        )
        
        # Iterate through all pages to trigger rate limiting
        list(paginator)
        
        # Verify rate limiter was used
        self.assertTrue(mock_rate_limiter.get_delay.called)
        self.assertEqual(mock_rate_limiter.get_delay.call_count, 3)  # Three pages
        
        # Verify add_call was called for each page
        self.assertEqual(mock_rate_limiter.add_call.call_count, 3)  # Three pages
        
        # Verify correct ticker was passed
        mock_rate_limiter.get_delay.assert_called_with("AAPL")
        mock_rate_limiter.add_call.assert_called_with(ticker="AAPL")


@pytest.mark.integration
@pytest.mark.asyncio
class TestRateLimiterErrorRecovery(unittest.IsolatedAsyncioTestCase):
    """Test the error recovery strategies with rate limiting."""
    
    async def test_retry_with_rate_limited_errors(self):
        """Test retry mechanism handling rate limited errors."""
        # Create a flaky function that fails due to rate limits
        flaky_func = await create_flaky_function(fail_count=2)
        
        # Verify retry mechanism
        result = await retry_async(
            flaky_func,
            max_retries=5,  # More than needed
            base_delay=0.01,
            max_delay=0.1
        )
        
        # Check results
        self.assertEqual(result, "success")
        
        # Test with insufficient retries
        with self.assertRaises(RateLimitError):
            await retry_async(
                await create_flaky_function(fail_count=4),
                max_retries=2,  # Not enough
                base_delay=0.01,
                max_delay=0.1
            )


if __name__ == "__main__":
    logger.info("Running advanced utility integration tests...")
    unittest.main()