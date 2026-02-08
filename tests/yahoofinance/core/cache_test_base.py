"""
Base test class for cache-related tests.

This module provides a common base test class for all cache-related test files
to reduce duplication and ensure consistent test setups.
"""

import json
import os
import shutil
import time
import unittest
from datetime import datetime, timedelta

import pytest

from yahoofinance.data.cache import CacheManager


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
            "..",
            "..",
            "fixtures",
            "cache",
            f"test_cache_{test_class_name}",
        )

        # Create the directory if it doesn't exist
        os.makedirs(self.test_cache_dir, exist_ok=True)

        # Create cache instance with default settings
        self.cache = CacheManager(
            disk_cache_dir=self.test_cache_dir,
            enable_memory_cache=True,
            enable_disk_cache=False,  # Disable disk cache for faster tests
            memory_cache_ttl=30,  # Use a shorter TTL for testing
        )

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
        # First set the value normally
        self.cache.set(key, value)

        # Handle unified CacheService (new implementation)
        # The unified cache stores entries as (value, expiry_timestamp) tuples
        if hasattr(self.cache, "_memory_cache") and hasattr(self.cache, "_cache_lock"):
            # This is the unified CacheService
            with self.cache._cache_lock:
                if key in self.cache._memory_cache:
                    current_value, _ = self.cache._memory_cache[key]
                    # Set expiry to past timestamp (expired)
                    expired_time = time.time() - (minutes_expired * 60) - 1
                    self.cache._memory_cache[key] = (current_value, expired_time)
            # Create a mock path for memory-only mode
            mock_path = os.path.join(self.test_cache_dir, f"{key}.cache")
            return mock_path

        # Handle old CacheManager wrapper (backward compatibility layer)
        if hasattr(self.cache, "memory_cache") and self.cache.memory_cache:
            # Check if memory_cache is the unified cache
            if hasattr(self.cache.memory_cache, "_memory_cache") and hasattr(self.cache.memory_cache, "_cache_lock"):
                # This is the unified CacheService wrapped
                with self.cache.memory_cache._cache_lock:
                    if key in self.cache.memory_cache._memory_cache:
                        current_value, _ = self.cache.memory_cache._memory_cache[key]
                        # Set expiry to past timestamp (expired)
                        expired_time = time.time() - (minutes_expired * 60) - 1
                        self.cache.memory_cache._memory_cache[key] = (current_value, expired_time)
            else:
                # Old-style cache with different structure
                # Manipulate the cache entry to make it expired
                if hasattr(self.cache.memory_cache, '_lock'):
                    with self.cache.memory_cache._lock:
                        # Check the actual attribute name - could be 'cache' or '_memory_cache'
                        cache_dict = getattr(self.cache.memory_cache, 'cache', None)
                        if cache_dict is None:
                            cache_dict = getattr(self.cache.memory_cache, '_memory_cache', None)

                        if cache_dict and key in cache_dict:
                            # Get current tuple - handle both 2-tuple and 3-tuple formats
                            cache_entry = cache_dict[key]
                            if len(cache_entry) == 2:
                                # Format: (value, expiry)
                                current_value, expiry = cache_entry
                                # Set with expired timestamp (way in the past)
                                expired_time = time.time() - (minutes_expired * 60) - 300 - 1
                                cache_dict[key] = (current_value, expired_time)
                            elif len(cache_entry) == 3:
                                # Format: (value, timestamp, expiry)
                                current_value, _, expiry = cache_entry
                                # Set with expired timestamp (way in the past)
                                expired_time = time.time() - (minutes_expired * 60) - expiry - 1
                                cache_dict[key] = (current_value, expired_time, expiry)

        # Also manually clear thread-local cache if it exists
        if (hasattr(self.cache, "memory_cache") and
            hasattr(self.cache.memory_cache, "_local") and
            hasattr(self.cache.memory_cache._local, "recent_hits") and
            key in self.cache.memory_cache._local.recent_hits):
            del self.cache.memory_cache._local.recent_hits[key]

        # Also handle disk cache if enabled (check for old-style disk cache with index)
        if hasattr(self.cache, "disk_cache") and self.cache.disk_cache:
            # Check if disk cache has the old-style _get_file_path and index attributes
            if hasattr(self.cache.disk_cache, "_get_file_path") and hasattr(self.cache.disk_cache, "index"):
                # If disk cache is available, get the file path
                cache_path = self.cache.disk_cache._get_file_path(key)

                # Ensure the file exists
                if os.path.exists(cache_path):
                    # Manually manipulate the index to make it expired
                    with self.cache.disk_cache._lock:
                        if key in self.cache.disk_cache.index:
                            entry = self.cache.disk_cache.index[key].copy()
                            expired_time = (
                                time.time() - (minutes_expired * 60) - entry.get("expiry", 3600) - 1
                            )
                            entry["timestamp"] = expired_time
                            self.cache.disk_cache.index[key] = entry

                return str(cache_path)

        # Create a mock path for memory-only mode
        mock_path = os.path.join(self.test_cache_dir, f"{key}.cache")
        return mock_path
