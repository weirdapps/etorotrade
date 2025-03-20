"""
Tests for the yahoofinance.core.cache module.

This is an example of a well-organized test file in the new structure.
"""

import unittest
import os
import shutil
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from yahoofinance.core.cache import Cache
from yahoofinance.core.errors import CacheError


@pytest.mark.unit
class TestCache(unittest.TestCase):
    """Test cases for the Cache class."""
    
    def setUp(self):
        """Set up test cache directory."""
        self.test_cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'fixtures', 'cache', 'test_cache')
        self.cache = Cache(cache_dir=self.test_cache_dir, expiration_minutes=15)

    def tearDown(self):
        """Clean up test cache directory."""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

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
        
        # Set cache with expired timestamp
        cache_path = self.cache._get_cache_path("test_key")
        expired_time = datetime.now() - timedelta(minutes=20)
        cache_data = {
            'timestamp': expired_time.isoformat(),
            'value': test_data
        }
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        
        # Verify expired data is not returned
        self.assertIsNone(self.cache.get("test_key"))
        
        # Verify cache file is cleaned up
        self.assertFalse(os.path.exists(cache_path))

    @pytest.mark.parametrize("key,value", [
        ("simple_key", "value1"),
        ("key/with/slashes", "value2"),
        ("key with spaces", "value3"),
        ("key_with_symbols!@#$%", "value4")
    ])
    def test_cache_key_handling(self, key, value):
        """Test that cache keys are properly handled."""
        self.cache.set(key, value)
        self.assertEqual(self.cache.get(key), value)

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


if __name__ == '__main__':
    unittest.main()