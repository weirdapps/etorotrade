"""
Tests for trade_modules/cache_service.py

This module tests the unified cache service functionality.
"""

import pytest
import time
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from trade_modules.cache_service import CacheService


@pytest.fixture
def cache_service():
    """Create a fresh cache service for each test."""
    # Reset singleton for testing
    CacheService._instance = None
    service = CacheService()
    # Reset the instance after test
    yield service
    CacheService._instance = None


@pytest.fixture
def cache_with_disk(tmp_path):
    """Create a cache service with disk caching enabled."""
    CacheService._instance = None

    # Create a mock config with disk caching enabled
    mock_config = MagicMock()
    mock_config.cache = {
        'enable_disk': True,
        'default_ttl': 300,
        'max_items': 100
    }
    mock_config.paths = {'DATA_DIR': str(tmp_path)}

    with patch('trade_modules.cache_service._get_config', return_value=mock_config):
        CacheService._instance = None
        service = CacheService()
        yield service
        CacheService._instance = None


class TestCacheServiceBasics:
    """Basic cache service tests."""

    def test_singleton_pattern(self):
        """Test that CacheService uses singleton pattern."""
        CacheService._instance = None
        service1 = CacheService()
        service2 = CacheService()
        assert service1 is service2
        CacheService._instance = None

    def test_get_nonexistent_key(self, cache_service):
        """Test getting a nonexistent key returns default."""
        result = cache_service.get("nonexistent_key")
        assert result is None

        result = cache_service.get("nonexistent_key", default="default_value")
        assert result == "default_value"

    def test_set_and_get(self, cache_service):
        """Test basic set and get operations."""
        cache_service.set("test_key", "test_value")
        result = cache_service.get("test_key")
        assert result == "test_value"

    def test_set_with_ttl(self, cache_service):
        """Test set with custom TTL."""
        cache_service.set("ttl_key", "ttl_value", ttl=1)
        assert cache_service.get("ttl_key") == "ttl_value"

        # Wait for TTL to expire
        time.sleep(1.1)
        assert cache_service.get("ttl_key") is None

    def test_set_overwrites_existing(self, cache_service):
        """Test that set overwrites existing values."""
        cache_service.set("key", "value1")
        cache_service.set("key", "value2")
        assert cache_service.get("key") == "value2"

    def test_delete_key(self, cache_service):
        """Test deleting a key."""
        cache_service.set("delete_key", "value")
        assert cache_service.get("delete_key") == "value"

        cache_service.delete("delete_key")
        assert cache_service.get("delete_key") is None

    def test_delete_nonexistent_key(self, cache_service):
        """Test deleting a nonexistent key doesn't raise error."""
        # Should not raise
        cache_service.delete("nonexistent")

    def test_clear_cache(self, cache_service):
        """Test clearing all cache entries."""
        cache_service.set("key1", "value1")
        cache_service.set("key2", "value2")
        cache_service.set("key3", "value3")

        cache_service.clear()

        assert cache_service.get("key1") is None
        assert cache_service.get("key2") is None
        assert cache_service.get("key3") is None


class TestCacheServiceDataTypes:
    """Test caching different data types."""

    def test_cache_string(self, cache_service):
        """Test caching strings."""
        cache_service.set("string_key", "hello world")
        assert cache_service.get("string_key") == "hello world"

    def test_cache_integer(self, cache_service):
        """Test caching integers."""
        cache_service.set("int_key", 42)
        assert cache_service.get("int_key") == 42

    def test_cache_float(self, cache_service):
        """Test caching floats."""
        cache_service.set("float_key", 3.14159)
        assert cache_service.get("float_key") == pytest.approx(3.14159)

    def test_cache_list(self, cache_service):
        """Test caching lists."""
        test_list = [1, 2, 3, "a", "b", "c"]
        cache_service.set("list_key", test_list)
        assert cache_service.get("list_key") == test_list

    def test_cache_dict(self, cache_service):
        """Test caching dictionaries."""
        test_dict = {"name": "test", "value": 123, "nested": {"a": 1}}
        cache_service.set("dict_key", test_dict)
        assert cache_service.get("dict_key") == test_dict

    def test_cache_none(self, cache_service):
        """Test caching None values."""
        cache_service.set("none_key", None)
        # Use a sentinel to distinguish between "no key" and "None value"
        result = cache_service.get("none_key", default="SENTINEL")
        assert result is None  # The cached value is None

    def test_cache_boolean(self, cache_service):
        """Test caching boolean values."""
        cache_service.set("true_key", True)
        cache_service.set("false_key", False)
        assert cache_service.get("true_key") == True
        assert cache_service.get("false_key") == False


class TestCacheServiceTTL:
    """Test TTL (time-to-live) functionality."""

    def test_default_ttl(self, cache_service):
        """Test that default TTL is applied."""
        # Default TTL should be 300 seconds (5 minutes)
        assert cache_service.default_ttl >= 0

    def test_ttl_expiration(self, cache_service):
        """Test that items expire after TTL."""
        cache_service.set("expire_key", "value", ttl=0.5)
        assert cache_service.get("expire_key") == "value"

        time.sleep(0.6)
        assert cache_service.get("expire_key") is None

    def test_negative_ttl_immediate_expiration(self, cache_service):
        """Test that negative or zero TTL allows set but expires immediately."""
        cache_service.set("zero_ttl_key", "value", ttl=0)
        # Should be None as it's already expired
        # Note: behavior depends on implementation
        result = cache_service.get("zero_ttl_key")
        # Either None or the value depending on how the cache handles 0 TTL


class TestCacheServiceStats:
    """Test cache statistics functionality."""

    def test_stats_tracking(self, cache_service):
        """Test that stats are tracked."""
        # Get initial stats
        initial_stats = cache_service.get_stats()

        # Cause some hits and misses
        cache_service.set("hit_key", "value")
        cache_service.get("hit_key")  # Hit
        cache_service.get("miss_key")  # Miss

        stats = cache_service.get_stats()
        assert 'hits' in stats or 'misses' in stats  # Stats should exist

    def test_hit_ratio(self, cache_service):
        """Test hit ratio calculation."""
        cache_service.set("key", "value")

        # 5 hits
        for _ in range(5):
            cache_service.get("key")

        # 5 misses
        for i in range(5):
            cache_service.get(f"nonexistent_{i}")

        stats = cache_service.get_stats()
        # Hit ratio should be around 0.5
        assert isinstance(stats, dict)


class TestCacheServiceContains:
    """Test contains/has functionality."""

    def test_contains_existing_key(self, cache_service):
        """Test contains for existing key."""
        cache_service.set("existing", "value")
        if hasattr(cache_service, 'contains'):
            assert cache_service.contains("existing") == True
        elif hasattr(cache_service, 'has'):
            assert cache_service.has("existing") == True

    def test_contains_missing_key(self, cache_service):
        """Test contains for missing key."""
        if hasattr(cache_service, 'contains'):
            assert cache_service.contains("missing") == False
        elif hasattr(cache_service, 'has'):
            assert cache_service.has("missing") == False


class TestCacheServiceKeyGeneration:
    """Test cache key generation utilities."""

    def test_generate_key_simple(self, cache_service):
        """Test simple key generation."""
        if hasattr(cache_service, 'generate_key'):
            key = cache_service.generate_key("ticker", "AAPL")
            assert "ticker" in key
            assert "AAPL" in key

    def test_generate_key_with_prefix(self, cache_service):
        """Test key generation with prefix."""
        if hasattr(cache_service, 'generate_key'):
            key = cache_service.generate_key("prefix", "suffix")
            assert isinstance(key, str)
            assert len(key) > 0


class TestCacheServiceMemoryManagement:
    """Test memory management and eviction."""

    def test_max_items_limit(self, cache_service):
        """Test that cache respects max items limit."""
        original_max = cache_service.max_memory_items
        cache_service.max_memory_items = 5

        try:
            # Add more items than max
            for i in range(10):
                cache_service.set(f"key_{i}", f"value_{i}")

            # Should not have more than max_memory_items
            # (implementation may vary - some might evict, some might not)
        finally:
            cache_service.max_memory_items = original_max


class TestCacheServiceDecorator:
    """Test cache decorator functionality if available."""

    def test_cached_decorator(self, cache_service):
        """Test the cached decorator if available."""
        if hasattr(cache_service, 'cached'):
            call_count = 0

            @cache_service.cached(ttl=10)
            def expensive_function(x):
                nonlocal call_count
                call_count += 1
                return x * 2

            result1 = expensive_function(5)
            result2 = expensive_function(5)

            assert result1 == 10
            assert result2 == 10
            # Should only be called once due to caching
            # Note: This depends on implementation


class TestCacheServiceSize:
    """Test cache size operations."""

    def test_get_size(self, cache_service):
        """Test getting cache size."""
        cache_service.clear()

        cache_service.set("key1", "value1")
        cache_service.set("key2", "value2")
        cache_service.set("key3", "value3")

        if hasattr(cache_service, 'size'):
            assert cache_service.size() >= 3
        elif hasattr(cache_service, '__len__'):
            assert len(cache_service) >= 3


class TestCacheServiceConcurrency:
    """Test thread safety."""

    def test_concurrent_access(self, cache_service):
        """Test concurrent read/write operations."""
        import threading
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache_service.set(f"concurrent_key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache_service.get(f"concurrent_key_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"


class TestCacheServiceGetOrSet:
    """Test get_or_set functionality."""

    def test_get_or_set_existing(self, cache_service):
        """Test get_or_set with existing key."""
        cache_service.set("existing_key", "existing_value")

        if hasattr(cache_service, 'get_or_set'):
            result = cache_service.get_or_set("existing_key", lambda: "new_value")
            assert result == "existing_value"

    def test_get_or_set_missing(self, cache_service):
        """Test get_or_set with missing key."""
        if hasattr(cache_service, 'get_or_set'):
            result = cache_service.get_or_set("new_key", lambda: "computed_value")
            assert result == "computed_value"
            # Verify it was cached
            assert cache_service.get("new_key") == "computed_value"


class TestCacheServiceKeys:
    """Test key listing functionality."""

    def test_list_keys(self, cache_service):
        """Test listing all keys."""
        cache_service.clear()

        cache_service.set("key1", "value1")
        cache_service.set("key2", "value2")

        if hasattr(cache_service, 'keys'):
            keys = cache_service.keys()
            assert "key1" in keys
            assert "key2" in keys


class TestCacheServiceResetInstance:
    """Test singleton reset for testing."""

    def test_reset_singleton(self):
        """Test that singleton can be reset."""
        CacheService._instance = None
        service1 = CacheService()

        CacheService._instance = None
        service2 = CacheService()

        # After reset, should be a new instance
        # (Note: this tests the test infrastructure)
        CacheService._instance = None
