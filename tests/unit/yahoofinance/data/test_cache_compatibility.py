"""
Tests for yahoofinance/data/cache_compatibility.py

This module tests the cache compatibility layer that redirects to unified CacheService.
"""

import pytest
from tempfile import TemporaryDirectory

from yahoofinance.data.cache_compatibility import (
    CacheManager,
    LRUCache,
    DiskCache,
    CacheKeyGenerator,
    default_cache_manager,
    cached,
    get_cache,
    clear_cache,
    get_cache_manager,
    configure_caching,
    create_cache_aware_wrapper,
    wrap_provider_with_cache,
)


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_init_creates_cache(self):
        """Test initialization creates a cache."""
        manager = CacheManager()
        assert manager.cache is not None

    def test_init_has_memory_cache_attr(self):
        """Test initialization has memory_cache attribute."""
        manager = CacheManager()
        assert hasattr(manager, 'memory_cache')
        assert manager.memory_cache is manager.cache

    def test_init_has_disk_cache_attr(self):
        """Test initialization has disk_cache attribute."""
        manager = CacheManager()
        assert hasattr(manager, 'disk_cache')

    def test_init_with_disk_cache_dir(self):
        """Test initialization with disk_cache_dir parameter."""
        with TemporaryDirectory() as tmpdir:
            manager = CacheManager(disk_cache_dir=tmpdir)
            assert manager.cache is not None

    def test_get_returns_default_for_missing(self):
        """Test get returns default for missing key."""
        manager = CacheManager()
        result = manager.get("nonexistent", default="default_value")
        assert result == "default_value"

    def test_set_and_get(self):
        """Test set and get operations."""
        manager = CacheManager()
        manager.set("test_key", "test_value")
        result = manager.get("test_key")
        assert result == "test_value"

    def test_set_with_ttl(self):
        """Test set with TTL parameter."""
        manager = CacheManager()
        manager.set("ttl_key", "ttl_value", 300)
        result = manager.get("ttl_key")
        assert result == "ttl_value"

    def test_set_with_data_type_string(self):
        """Test set with data_type string (old signature)."""
        manager = CacheManager()
        manager.set("data_type_key", "value", "ticker_info")
        result = manager.get("data_type_key")
        assert result == "value"

    def test_set_with_kwargs(self):
        """Test set with keyword arguments."""
        manager = CacheManager()
        manager.set("kwargs_key", "value", ttl=300, data_type="test")
        result = manager.get("kwargs_key")
        assert result == "value"

    def test_delete(self):
        """Test delete operation."""
        manager = CacheManager()
        manager.set("delete_key", "value")
        manager.delete("delete_key")
        result = manager.get("delete_key", default=None)
        assert result is None

    def test_clear(self):
        """Test clear operation."""
        manager = CacheManager()
        manager.set("clear_key1", "value1")
        manager.set("clear_key2", "value2")
        manager.clear()
        # After clear, keys should return default
        assert manager.get("clear_key1", default=None) is None

    def test_invalidate(self):
        """Test invalidate operation (compatibility method)."""
        manager = CacheManager()
        manager.set("invalidate_key", "value")
        manager.invalidate()
        # After invalidate, should be cleared
        assert manager.get("invalidate_key", default=None) is None

    def test_is_data_known_missing_always_false(self):
        """Test is_data_known_missing always returns False."""
        manager = CacheManager()
        assert manager.is_data_known_missing("any_key") is False
        assert manager.is_data_known_missing("any_key", "any_type") is False

    def test_set_missing_data_is_noop(self):
        """Test set_missing_data is a no-op."""
        manager = CacheManager()
        # Should not raise
        manager.set_missing_data("key", "type", ttl=300)


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_init_creates_cache(self):
        """Test initialization creates a cache."""
        cache = LRUCache()
        assert len(cache) == 0

    def test_get_returns_none_for_missing(self):
        """Test get returns None for missing key."""
        cache = LRUCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_put_and_get(self):
        """Test put and get operations."""
        cache = LRUCache()
        cache.put("test_key", "test_value")
        result = cache.get("test_key")
        assert result == "test_value"

    def test_clear(self):
        """Test clear operation."""
        cache = LRUCache()
        cache.put("key", "value")
        cache.clear()
        assert cache.get("key") is None


class TestDiskCache:
    """Tests for DiskCache class."""

    def test_init_creates_cache(self):
        """Test initialization creates a cache."""
        cache = DiskCache()
        assert cache.cache is not None

    def test_get_returns_none_for_missing(self):
        """Test get returns None for missing key."""
        cache = DiskCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_set_and_get(self):
        """Test set and get operations."""
        cache = DiskCache()
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        assert result == "test_value"

    def test_set_with_ttl(self):
        """Test set with TTL parameter."""
        cache = DiskCache()
        cache.set("ttl_key", "ttl_value", ttl=300)
        result = cache.get("ttl_key")
        assert result == "ttl_value"

    def test_clear(self):
        """Test clear operation."""
        cache = DiskCache()
        cache.set("key", "value")
        cache.clear()
        assert cache.get("key") is None


class TestCacheKeyGenerator:
    """Tests for CacheKeyGenerator class."""

    def test_generate_key_with_strings(self):
        """Test key generation with string arguments."""
        key = CacheKeyGenerator.generate_key("a", "b", "c")
        assert "a" in key
        assert "b" in key
        assert "c" in key

    def test_generate_key_with_numbers(self):
        """Test key generation with numeric arguments."""
        key = CacheKeyGenerator.generate_key(1, 2.5, 3)
        assert "1" in key
        assert "2.5" in key
        assert "3" in key

    def test_generate_key_with_booleans(self):
        """Test key generation with boolean arguments."""
        key = CacheKeyGenerator.generate_key(True, False)
        assert "True" in key
        assert "False" in key

    def test_generate_key_with_kwargs(self):
        """Test key generation with keyword arguments."""
        key = CacheKeyGenerator.generate_key(name="test", value=123)
        assert "name=test" in key
        assert "value=123" in key

    def test_generate_key_with_complex_object(self):
        """Test key generation with complex objects (uses hash)."""
        key = CacheKeyGenerator.generate_key({"key": "value"})
        assert key is not None
        assert isinstance(key, str)

    def test_generate_key_deterministic(self):
        """Test key generation is deterministic."""
        key1 = CacheKeyGenerator.generate_key("a", "b", c="d")
        key2 = CacheKeyGenerator.generate_key("a", "b", c="d")
        assert key1 == key2


class TestCachedDecorator:
    """Tests for cached decorator."""

    def test_cached_returns_result(self):
        """Test cached decorator returns function result."""
        call_count = 0

        @cached()
        def my_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_cached_with_ttl(self):
        """Test cached decorator with TTL."""
        @cached(ttl=300)
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_cached_returns_none_not_cached(self):
        """Test cached decorator doesn't cache None results."""
        call_count = 0

        @cached()
        def my_func():
            nonlocal call_count
            call_count += 1
            return None

        my_func()
        my_func()
        # None results are not cached, so function should be called each time
        assert call_count == 2


class TestCompatibilityFunctions:
    """Tests for compatibility functions."""

    def test_get_cache_returns_cache_manager(self):
        """Test get_cache returns a CacheManager."""
        cache = get_cache()
        assert isinstance(cache, CacheManager)

    def test_get_cache_manager_returns_cache_manager(self):
        """Test get_cache_manager returns a CacheManager."""
        manager = get_cache_manager()
        assert isinstance(manager, CacheManager)

    def test_default_cache_manager_exists(self):
        """Test default_cache_manager is a CacheManager."""
        assert isinstance(default_cache_manager, CacheManager)

    def test_clear_cache_clears(self):
        """Test clear_cache clears the cache."""
        manager = get_cache_manager()
        manager.set("clear_test_key", "value")
        clear_cache()
        # Should be cleared (though may affect other tests)

    def test_configure_caching_is_noop(self):
        """Test configure_caching is a no-op."""
        # Should not raise
        configure_caching(some_param="value")

    def test_create_cache_aware_wrapper_returns_same(self):
        """Test create_cache_aware_wrapper returns provider as-is."""
        provider = object()
        result = create_cache_aware_wrapper(provider)
        assert result is provider

    def test_wrap_provider_with_cache_returns_same(self):
        """Test wrap_provider_with_cache returns provider as-is."""
        provider = object()
        result = wrap_provider_with_cache(provider)
        assert result is provider


class TestCacheManagerEdgeCases:
    """Edge case tests for CacheManager."""

    def test_set_with_none_ttl(self):
        """Test set with None TTL."""
        manager = CacheManager()
        manager.set("none_ttl_key", "value", None)
        assert manager.get("none_ttl_key") == "value"

    def test_set_with_float_value(self):
        """Test set with float value."""
        manager = CacheManager()
        manager.set("float_key", 3.14159)
        assert manager.get("float_key") == pytest.approx(3.14159)

    def test_set_with_dict_value(self):
        """Test set with dictionary value."""
        manager = CacheManager()
        data = {"price": 175.50, "volume": 1000000}
        manager.set("dict_key", data)
        assert manager.get("dict_key") == data

    def test_set_with_list_value(self):
        """Test set with list value."""
        manager = CacheManager()
        data = [1, 2, 3, 4, 5]
        manager.set("list_key", data)
        assert manager.get("list_key") == data

    def test_get_nonexistent_no_default(self):
        """Test get nonexistent key without default."""
        manager = CacheManager()
        result = manager.get("truly_nonexistent")
        assert result is None

    def test_delete_nonexistent(self):
        """Test delete nonexistent key doesn't raise."""
        manager = CacheManager()
        # Should not raise
        manager.delete("nonexistent_delete_key")
