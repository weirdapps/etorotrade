"""
Tests for the yahoofinance.data.cache module.

This is an example of a well-organized test file in the new structure.
"""

import os
import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from yahoofinance.core.errors import DataError, YFinanceError

from .cache_test_base import BaseCacheTest


@pytest.mark.unit
class TestCache(BaseCacheTest):
    """Test cases for the Cache class."""

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        self.assertTrue(os.path.exists(self.test_cache_dir))

    def test_basic_cache_operations(self):
        """Test basic cache set and get operations."""
        test_data = {"key": "value"}
        self.cache.set("test_key", test_data)

        # Test retrieval
        cached_data = self.cache.get("test_key")
        self.assertEqual(cached_data, test_data)

        # Test non-existent key
        self.assertIsNone(self.cache.get("nonexistent_key"))

    def test_cache_expiration(self):
        """Test that expired cache entries are not returned."""
        test_data = {"key": "value"}

        # Set cache with expired timestamp using the helper method
        cache_path = self.create_expired_cache_entry("test_key", test_data)

        # Verify expired data is not returned
        self.assertIsNone(self.cache.get("test_key"))

        # Verify cache file is cleaned up
        self.assertFalse(os.path.exists(cache_path))

    def test_cache_key_handling(self):
        """Test that cache keys are properly handled."""
        test_cases = [
            ("simple_key", "value1"),
            ("key/with/slashes", "value2"),
            ("key with spaces", "value3"),
            ("key_with_symbols!@#$%", "value4"),
        ]

        for test_key, test_value in test_cases:
            self.cache.set(test_key, test_value)
            self.assertEqual(self.cache.get(test_key), test_value)

    @pytest.mark.slow
    def test_large_cache_performance(self):
        """Test performance with large number of items."""
        # Add many items to cache
        for i in range(100):
            self.cache.set(f"perf_key_{i}", {"data": f"value_{i}"})

        # Verify retrieval is still fast
        start_time = datetime.now()
        for i in range(100):
            self.cache.get(f"perf_key_{i}")
        elapsed = (datetime.now() - start_time).total_seconds()

        # Should complete quickly
        self.assertLess(elapsed, 1.0, "Cache retrieval is too slow")

    @pytest.mark.integration
    def test_cache_integration_with_client(self):
        """Test cache integration with API client."""
        mock_client = MagicMock()
        mock_client.get_ticker_info.return_value = {"price": 150.0}

        # Function that uses cache + client
        def get_cached_data(ticker):
            cached = self.cache.get(f"ticker_{ticker}")
            if cached:
                return cached
            data = mock_client.get_ticker_info(ticker)
            self.cache.set(f"ticker_{ticker}", data)
            return data

        # First call should hit the API
        result1 = get_cached_data("AAPL")
        self.assertEqual(result1, {"price": 150.0})
        self.assertEqual(mock_client.get_ticker_info.call_count, 1)

        # Second call should use cache
        result2 = get_cached_data("AAPL")
        self.assertEqual(result2, {"price": 150.0})
        self.assertEqual(mock_client.get_ticker_info.call_count, 1, "Should not call API twice")


if __name__ == "__main__":
    unittest.main()
