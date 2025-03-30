"""
Base test class for cache-related tests.

This module provides a common base test class for all cache-related test files
to reduce duplication and ensure consistent test setups.
"""

import unittest
import os
import shutil
import json
from datetime import datetime, timedelta
import pytest

from yahoofinance.core.cache import Cache


class BaseCacheTest(unittest.TestCase):
    """
    Base test class for cache-related tests.
    
    This class provides common setup and teardown methods for cache tests,
    reducing duplication across test files.
    """
    
    def setUp(self):
        """Set up test cache directory."""
        # Create a unique cache directory for each test subclass
        test_class_name = self.__class__.__name__.lower()
        self.test_cache_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'fixtures', 
            'cache', 
            f'test_cache_{test_class_name}'
        )
        
        # Create cache instance with default settings
        self.cache = Cache(cache_dir=self.test_cache_dir, expiration_minutes=15)

    def tearDown(self):
        """Clean up test cache directory."""
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def create_expired_cache_entry(self, key, value, minutes_expired=20):
        """
        Create an explicitly expired cache entry for testing.
        
        Args:
            key: Cache key
            value: Value to cache
            minutes_expired: How many minutes in the past to set timestamp
        """
        cache_path = self.cache._get_cache_path(key)
        expired_time = datetime.now() - timedelta(minutes=minutes_expired)
        cache_data = {
            'timestamp': expired_time.isoformat(),
            'value': value
        }
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        
        return cache_path