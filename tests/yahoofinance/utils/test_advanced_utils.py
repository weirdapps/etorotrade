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
from tests.fixtures import create_flaky_function
from yahoofinance.core.errors import RateLimitError, YFinanceError
from yahoofinance.utils.async_utils.helpers import async_retry


# Set up logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_advanced_utils")


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
