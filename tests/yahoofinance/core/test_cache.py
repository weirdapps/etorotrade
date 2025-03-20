import unittest
import os
import shutil
import json
from datetime import datetime, timedelta
from unittest.mock import patch
from yahoofinance.core.cache import Cache

class TestCache(unittest.TestCase):
    def setUp(self):
        """Set up test cache directory"""
        self.test_cache_dir = os.path.join(os.path.dirname(__file__), 'test_cache')
        self.cache = Cache(cache_dir=self.test_cache_dir, expiration_minutes=15)

    def tearDown(self):
        """Clean up test cache directory"""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist"""
        self.assertTrue(os.path.exists(self.test_cache_dir))

    def test_basic_cache_operations(self):
        """Test basic cache set and get operations"""
        test_data = {"key": "value"}
        self.cache.set("test_key", test_data)
        
        # Test retrieval
        cached_data = self.cache.get("test_key")
        self.assertEqual(cached_data, test_data)
        
        # Test non-existent key
        self.assertIsNone(self.cache.get("nonexistent_key"))

    def test_cache_expiration(self):
        """Test that expired cache entries are not returned"""
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

    def test_cache_clear(self):
        """Test clearing all cache entries"""
        # Set multiple cache entries
        test_data = [
            ("key1", {"data": "value1"}),
            ("key2", {"data": "value2"}),
            ("key3", {"data": "value3"})
        ]
        
        for key, value in test_data:
            self.cache.set(key, value)
        
        # Clear cache
        self.cache.clear()
        
        # Verify all entries are cleared
        for key, _ in test_data:
            self.assertIsNone(self.cache.get(key))

    def test_invalid_cache_data(self):
        """Test handling of corrupted cache files"""
        cache_path = self.cache._get_cache_path("test_key")
        
        # Create corrupted cache file
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            f.write("invalid json data")
        
        # Verify corrupted data is handled gracefully
        self.assertIsNone(self.cache.get("test_key"))
        
        # Verify corrupted file is cleaned up
        self.assertFalse(os.path.exists(cache_path))

    def test_cache_key_handling(self):
        """Test that cache keys are properly handled"""
        # Test with various key types
        test_cases = [
            ("simple_key", "value1"),
            ("key/with/slashes", "value2"),
            ("key with spaces", "value3"),
            ("key_with_symbols!@#$%", "value4")
        ]
        
        for key, value in test_cases:
            self.cache.set(key, value)
            self.assertEqual(self.cache.get(key), value)

if __name__ == '__main__':
    unittest.main()