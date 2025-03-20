#!/usr/bin/env python3
"""
Test script to verify the improvements made to the codebase.
This script tests the centralized utility functions, config values,
and improved cache implementation without hitting rate limits.
"""

import os
import time
import logging
from yahoofinance.utils.market_utils import is_us_ticker, normalize_hk_ticker
from yahoofinance.core.config import CACHE, RISK_METRICS, POSITIVE_GRADES, RATE_LIMIT
from yahoofinance.core.cache import market_cache, news_cache, earnings_cache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    assert normalize_hk_ticker("0700.HK") == "0700.HK"  # 4 digits, no change
    assert normalize_hk_ticker("AAPL") == "AAPL"  # Non-HK, no change
    
    logger.info("✅ Market utilities test passed")

def test_config_values():
    """Test config values are properly loaded."""
    logger.info("Testing config values...")
    
    # Test rate limiting config
    assert "WINDOW_SIZE" in RATE_LIMIT
    assert "MAX_CALLS" in RATE_LIMIT
    assert "BATCH_SIZE" in RATE_LIMIT
    
    # Test cache config
    assert "MARKET_DATA_TTL" in CACHE
    assert "NEWS_DATA_TTL" in CACHE
    assert "DEFAULT_TTL" in CACHE
    
    # Test analyst ratings config
    assert "Buy" in POSITIVE_GRADES
    assert "Strong Buy" in POSITIVE_GRADES
    
    logger.info("✅ Config values test passed")

def test_cache_implementation():
    """Test the improved cache implementation."""
    logger.info("Testing improved cache implementation...")
    
    # Test different cache instances
    assert market_cache.expiration_minutes == CACHE["MARKET_DATA_TTL"]
    assert news_cache.expiration_minutes == CACHE["NEWS_DATA_TTL"]
    assert earnings_cache.expiration_minutes == CACHE["EARNINGS_DATA_TTL"]
    
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
    
    # Clear the cache for cleanup
    market_cache.clear()
    assert market_cache.get(test_key) is None
    
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
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Run from the project root directory
    main()