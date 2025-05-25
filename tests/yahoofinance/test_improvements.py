#!/usr/bin/env python3
"""
Test script to verify the improvements made to the codebase.
This script tests the centralized utility functions, config values,
and improved cache implementation without hitting rate limits.
"""

import logging
import os
import time

from yahoofinance.core.config import CACHE_CONFIG, POSITIVE_GRADES, RATE_LIMIT, RISK_METRICS
from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from yahoofinance.core.logging import get_logger
from yahoofinance.data.cache import CacheManager, default_cache_manager
from yahoofinance.utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)
from yahoofinance.utils.market.ticker_utils import is_us_ticker, normalize_hk_ticker


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_improvements")


def test_market_utils():
    """Test the centralized market utilities."""
    logger.info("Testing market utilities...")

    # Test US ticker detection
    assert is_us_ticker("AAPL") == True
    assert is_us_ticker("MSFT") == True
    assert is_us_ticker("BRK.B") == True  # Special case
    assert is_us_ticker("9988.HK") == False

    # Test HK ticker normalization
    assert normalize_hk_ticker("03690.HK") == "3690.HK"
    assert normalize_hk_ticker("0700.HK") == "700.HK"  # Current behavior removes leading zeros
    assert normalize_hk_ticker("AAPL") == "AAPL"  # Non-HK, no change

    logger.info("✅ Market utilities test passed")


def test_config_values():
    """Test config values are properly loaded."""
    logger.info("Testing config values...")

    # Test rate limiting config
    assert "WINDOW_SIZE" in RATE_LIMIT
    assert "MAX_CALLS" in RATE_LIMIT
    assert "BASE_DELAY" in RATE_LIMIT

    # Test cache config
    assert "MEMORY_CACHE_TTL" in CACHE_CONFIG
    assert "DISK_CACHE_TTL" in CACHE_CONFIG
    assert "MEMORY_CACHE_SIZE" in CACHE_CONFIG

    # Test analyst ratings config
    assert "Buy" in POSITIVE_GRADES
    assert "Strong Buy" in POSITIVE_GRADES

    logger.info("✅ Config values test passed")


def test_cache_implementation():
    """Test the improved cache implementation."""
    logger.info("Testing improved cache implementation...")

    # Create test cache with specific configurations
    market_cache = CacheManager(
        enable_memory_cache=True,
        memory_cache_ttl=CACHE_CONFIG.get("MARKET_DATA_MEMORY_TTL", 60),
        enable_disk_cache=False,
    )

    news_cache = CacheManager(
        enable_memory_cache=True,
        memory_cache_ttl=CACHE_CONFIG.get("NEWS_MEMORY_TTL", 600),
        enable_disk_cache=False,
    )

    earnings_cache = CacheManager(
        enable_memory_cache=True,
        memory_cache_ttl=CACHE_CONFIG.get("EARNINGS_DATA_MEMORY_TTL", 600),
        enable_disk_cache=False,
    )

    # Test cache operations
    test_key = "test_key"
    test_value = {"data": "test_value", "timestamp": time.time()}

    # Set a value
    market_cache.set(test_key, test_value)

    # Retrieve the value
    retrieved_value = market_cache.get(test_key)
    assert retrieved_value == test_value

    # Test LRU behavior by adding many entries
    for i in range(10):
        market_cache.set(f"test_key_{i}", f"test_value_{i}")

    # Original entry should still be available (max entries is much larger)
    assert market_cache.get(test_key) == test_value

    # Test a fresh custom cache manager instead of the default
    custom_cache = CacheManager(
        enable_memory_cache=True, memory_cache_ttl=30, enable_disk_cache=False
    )

    custom_cache.set("custom_test_key", "custom_test_value")
    cached_value = custom_cache.get("custom_test_key")
    assert cached_value == "custom_test_value"

    # Clear the caches for cleanup
    market_cache.clear()
    news_cache.clear()
    earnings_cache.clear()
    custom_cache.clear()

    # Verify cleared
    assert market_cache.get(test_key) is None
    assert custom_cache.get("custom_test_key") is None

    logger.info("✅ Cache implementation test passed")


def main():
    """Run all tests."""
    try:
        test_market_utils()
        test_config_values()
        test_cache_implementation()
        logger.info("🎉 All tests passed successfully! The improvements are working as expected.")
    except AssertionError as e:
        logger.error(f"❌ Test failed: {str(e)}")
    except YFinanceError as e:
        logger.error(f"❌ Unexpected error: {str(e)}")


if __name__ == "__main__":
    # Run from the project root directory
    main()
