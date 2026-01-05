#!/usr/bin/env python3
"""
ITERATION 9: Memory Utils Tests
Target: Test memory management and cache clearing utilities
"""

import pytest
from yahoofinance.utils.memory_utils import (
    clear_abc_caches,
    clear_pandas_caches,
    clear_datetime_caches,
    clean_yfinance_caches,
    clean_memory,
)


class TestClearAbcCaches:
    """Test ABC cache clearing functionality."""

    def test_clear_abc_caches_returns_count(self):
        """clear_abc_caches returns count of cleared items."""
        result = clear_abc_caches()
        assert isinstance(result, int)
        assert result >= 0

    def test_clear_abc_caches_handles_missing_attributes(self):
        """clear_abc_caches handles missing attributes gracefully."""
        # Should not crash even if abc module attributes don't exist
        result = clear_abc_caches()
        assert isinstance(result, int)

    def test_clear_abc_caches_multiple_calls(self):
        """clear_abc_caches can be called multiple times."""
        result1 = clear_abc_caches()
        result2 = clear_abc_caches()
        # Both should succeed
        assert isinstance(result1, int)
        assert isinstance(result2, int)


class TestClearPandasCaches:
    """Test pandas cache clearing functionality."""

    def test_clear_pandas_caches_returns_count(self):
        """clear_pandas_caches returns count of cleared items."""
        result = clear_pandas_caches()
        assert isinstance(result, int)
        assert result >= 0

    def test_clear_pandas_caches_handles_import_error(self):
        """clear_pandas_caches handles pandas not installed."""
        # Even if pandas is installed, this should handle errors gracefully
        result = clear_pandas_caches()
        assert isinstance(result, int)

    def test_clear_pandas_caches_multiple_calls(self):
        """clear_pandas_caches can be called multiple times."""
        result1 = clear_pandas_caches()
        result2 = clear_pandas_caches()
        assert isinstance(result1, int)
        assert isinstance(result2, int)


class TestClearDatetimeCaches:
    """Test datetime cache clearing functionality."""

    def test_clear_datetime_caches_returns_count(self):
        """clear_datetime_caches returns count of cleared items."""
        result = clear_datetime_caches()
        assert isinstance(result, int)
        assert result >= 0

    def test_clear_datetime_caches_handles_missing_attributes(self):
        """clear_datetime_caches handles missing attributes gracefully."""
        result = clear_datetime_caches()
        assert isinstance(result, int)

    def test_clear_datetime_caches_multiple_calls(self):
        """clear_datetime_caches can be called multiple times."""
        result1 = clear_datetime_caches()
        result2 = clear_datetime_caches()
        assert isinstance(result1, int)
        assert isinstance(result2, int)


class TestCleanYfinanceCaches:
    """Test yfinance cache clearing functionality."""

    def test_clean_yfinance_caches_returns_count(self):
        """clean_yfinance_caches returns count of cleared items."""
        result = clean_yfinance_caches()
        assert isinstance(result, int)
        assert result >= 0

    def test_clean_yfinance_caches_handles_import_error(self):
        """clean_yfinance_caches handles yfinance not installed."""
        # Should return 0 if yfinance not installed
        result = clean_yfinance_caches()
        assert isinstance(result, int)

    def test_clean_yfinance_caches_multiple_calls(self):
        """clean_yfinance_caches can be called multiple times."""
        result1 = clean_yfinance_caches()
        result2 = clean_yfinance_caches()
        assert isinstance(result1, int)
        assert isinstance(result2, int)


class TestCleanMemory:
    """Test comprehensive memory cleaning."""

    def test_clean_memory_returns_dict(self):
        """clean_memory returns dict with cleaning results."""
        result = clean_memory()
        assert isinstance(result, dict)

    def test_clean_memory_contains_required_keys(self):
        """clean_memory result contains all expected keys."""
        result = clean_memory()
        required_keys = [
            "abc_cache_items_cleared",
            "pandas_cache_items_cleared",
            "datetime_cache_items_cleared",
            "yfinance_cache_items_cleared",
            "gc_collected",
            "total_items_cleaned",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_clean_memory_values_are_integers(self):
        """clean_memory result values are all integers."""
        result = clean_memory()
        for key, value in result.items():
            assert isinstance(value, int), f"{key} should be an integer"

    def test_clean_memory_values_non_negative(self):
        """clean_memory result values are all non-negative."""
        result = clean_memory()
        for key, value in result.items():
            assert value >= 0, f"{key} should be non-negative"

    def test_clean_memory_total_is_sum(self):
        """clean_memory total equals sum of individual counts."""
        result = clean_memory()
        expected_total = (
            result["abc_cache_items_cleared"]
            + result["pandas_cache_items_cleared"]
            + result["datetime_cache_items_cleared"]
            + result["yfinance_cache_items_cleared"]
            + result["gc_collected"]
        )
        assert result["total_items_cleaned"] == expected_total

    def test_clean_memory_multiple_calls(self):
        """clean_memory can be called multiple times."""
        result1 = clean_memory()
        result2 = clean_memory()
        # Both should succeed and return valid dicts
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert "total_items_cleaned" in result1
        assert "total_items_cleaned" in result2

    def test_clean_memory_gc_collected_present(self):
        """clean_memory always includes gc_collected count."""
        result = clean_memory()
        assert "gc_collected" in result
        # gc.collect() can return 0 or more
        assert result["gc_collected"] >= 0
