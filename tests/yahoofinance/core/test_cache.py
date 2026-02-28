import os
import shutil
import time
import unittest

from yahoofinance.data.cache import CacheManager


class TestCache(unittest.TestCase):
    def setUp(self):
        """Set up test cache directory"""
        self.test_cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
        self.cache = CacheManager(disk_cache_dir=self.test_cache_dir)

    def tearDown(self):
        """Clean up test cache directory"""
        del self.cache
        time.sleep(0.1)
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist"""
        self.assertIsNotNone(self.cache)
        self.assertTrue(os.path.exists(self.test_cache_dir))

    def test_basic_cache_operations(self):
        """Test basic cache set and get operations"""
        test_data = {"key": "value"}

        # Set using the new API with data_type parameter
        self.cache.set("test_key", test_data, "default", True)

        # Test retrieval
        cached_data = self.cache.get("test_key")
        self.assertEqual(cached_data, test_data)

        # Test non-existent key
        self.assertIsNone(self.cache.get("nonexistent_key"))

    def test_cache_expiration(self):
        """Test that expired cache entries are not returned"""
        test_data = {"key": "value"}
        test_key = "test_expiration_key"

        # Set with a very short TTL that will expire right away
        self.cache.set(test_key, test_data, 0.1)

        # Sleep to ensure it expires
        time.sleep(0.2)

        # Verify expired data is not returned
        self.assertIsNone(self.cache.get(test_key))

    def test_cache_clear(self):
        """Test clearing all cache entries"""
        # Set multiple cache entries
        test_data = [
            ("key1", {"data": "value1"}),
            ("key2", {"data": "value2"}),
            ("key3", {"data": "value3"}),
        ]

        for key, value in test_data:
            # Use the new API with data_type parameter
            self.cache.set(key, value, "default", True)

        # Verify we can get the values before clearing
        for key, value in test_data:
            cached_value = self.cache.get(key)
            self.assertEqual(cached_value, value)

        # Clear cache
        self.cache.clear()

        # Verify all entries are cleared
        for key, _ in test_data:
            self.assertIsNone(self.cache.get(key))

    def test_invalid_cache_data(self):
        """Test handling of missing cache keys returns None"""
        cache_key = "nonexistent_test_key_" + str(time.time())
        self.assertIsNone(self.cache.get(cache_key))

    def test_cache_key_handling(self):
        """Test that cache keys are properly handled"""
        # Test with various key types
        test_cases = [
            ("simple_key", "value1"),
            ("key/with/slashes", "value2"),
            ("key with spaces", "value3"),
            ("key_with_symbols!@#$%", "value4"),
        ]

        for key, value in test_cases:
            # Use the new API with data_type parameter
            self.cache.set(key, value, "default", True)
            self.assertEqual(self.cache.get(key), value)


if __name__ == "__main__":
    unittest.main()
