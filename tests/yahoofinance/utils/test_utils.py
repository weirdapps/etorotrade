#!/usr/bin/env python3
"""
Tests for utility modules (market_utils, format_utils)

This test file verifies the functionality of utility modules including:
- Market utilities like ticker normalization and validation
- Formatting utilities for HTML and output
- General utility improvements to the codebase
"""

import logging
import time
import unittest
from unittest.mock import MagicMock, patch

from yahoofinance.core.config import CACHE_CONFIG, POSITIVE_GRADES, RATE_LIMIT, RISK_METRICS
from yahoofinance.core.logging import get_logger
from yahoofinance.data.cache import CacheManager
from yahoofinance.presentation.html import FormatUtils
from yahoofinance.utils.market.ticker_utils import is_us_ticker, normalize_hk_ticker


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)


class TestMarketUtils(unittest.TestCase):
    """Test market utilities for ticker validation and normalization."""

    def test_us_ticker_detection(self):
        """Test detection of US vs non-US tickers."""
        # US tickers
        self.assertTrue(is_us_ticker("AAPL"))
        self.assertTrue(is_us_ticker("MSFT"))
        self.assertTrue(is_us_ticker("BRK.B"))  # Special case with dot
        self.assertTrue(is_us_ticker("BF.A"))  # Special case with dot

        # Non-US tickers
        self.assertFalse(is_us_ticker("9988.HK"))
        self.assertFalse(is_us_ticker("BP.L"))
        self.assertFalse(is_us_ticker("MAERSK-A.CO"))

    def test_hk_ticker_normalization(self):
        """Test normalization of Hong Kong tickers."""
        # 5-digit ticker with leading zeros
        self.assertEqual(normalize_hk_ticker("03690.HK"), "3690.HK")

        # 4-digit ticker (behavior has changed, now it also strips zeros)
        self.assertEqual(normalize_hk_ticker("0700.HK"), "700.HK")

        # Non-HK ticker (should remain unchanged)
        self.assertEqual(normalize_hk_ticker("AAPL"), "AAPL")
        self.assertEqual(normalize_hk_ticker("MSFT.US"), "MSFT.US")


class TestFormatUtils(unittest.TestCase):
    """Test formatting utilities for output formatting."""

    def test_format_market_metrics(self):
        """Test formatting of market metrics."""
        # Test metrics
        test_metrics = {
            "AAPL": {"value": 5.25, "label": "Apple Inc", "is_percentage": True},
            "MSFT": {"value": -2.1, "label": "Microsoft", "is_percentage": True},
            "RATE": {"value": 3.75, "label": "Interest Rate", "is_percentage": False},
        }

        # Format metrics
        formatted = FormatUtils.format_market_metrics(test_metrics)

        # Verify results
        self.assertEqual(len(formatted), 3)

        # Check Apple formatting
        apple = next((m for m in formatted if m["key"] == "AAPL"), None)
        self.assertIsNotNone(apple)
        self.assertEqual(apple["formatted_value"], "5.2%")
        self.assertEqual(apple["color"], "positive")

        # Check Microsoft formatting
        msft = next((m for m in formatted if m["key"] == "MSFT"), None)
        self.assertIsNotNone(msft)
        self.assertEqual(msft["formatted_value"], "-2.1%")
        self.assertEqual(msft["color"], "negative")

        # Check non-percentage formatting
        rate = next((m for m in formatted if m["key"] == "RATE"), None)
        self.assertIsNotNone(rate)
        self.assertEqual(rate["formatted_value"], "3.75")

    def test_generate_market_html(self):
        """Test generation of market HTML."""
        # Test data
        title = "Market Dashboard"
        sections = [
            {
                "title": "Test Section",
                "metrics": [
                    {
                        "key": "AAPL",
                        "label": "Apple",
                        "formatted_value": "5.3%",
                        "color": "positive",
                    }
                ],
                "columns": 2,
                "width": "500px",
            }
        ]

        # Patch FILE_PATHS temporarily during the test (use safe test directory)
        with patch("yahoofinance.presentation.html.FILE_PATHS", {"OUTPUT_DIR": "test_output"}):
            # Create HTML generator instance
            from yahoofinance.presentation.html import HTMLGenerator

            html_generator = HTMLGenerator()

            # Generate HTML
            html = html_generator.generate_market_html(title, sections)

            # Verify output
            self.assertIsInstance(html, str)
            self.assertIn(title, html)
            self.assertIn("Test Section", html)
            self.assertIn("Apple", html)
            self.assertIn("5.3%", html)
            self.assertIn("positive", html)


class TestConfigAndCache(unittest.TestCase):
    """Test configuration values and cache improvements."""

    def test_config_values(self):
        """Test that config values are properly loaded."""
        # Test rate limiting config
        self.assertIn("WINDOW_SIZE", RATE_LIMIT)
        self.assertIn("MAX_CALLS", RATE_LIMIT)
        self.assertIn("BATCH_SIZE", RATE_LIMIT)

        # Test cache config
        self.assertIn("MEMORY_CACHE_TTL", CACHE_CONFIG)
        self.assertIn("MARKET_DATA_MEMORY_TTL", CACHE_CONFIG)
        self.assertIn("NEWS_MEMORY_TTL", CACHE_CONFIG)

        # Test analyst ratings config
        self.assertIn("Buy", POSITIVE_GRADES)
        self.assertIn("Strong Buy", POSITIVE_GRADES)

    def test_cache_implementation(self):
        """Test the improved cache implementation."""
        # Create test cache instances with specific configurations
        market_cache = CacheManager(
            enable_memory_cache=True,
            memory_cache_ttl=CACHE_CONFIG.get("MARKET_DATA_MEMORY_TTL", 60),
            enable_disk_cache=False,
        )
        news_cache = CacheManager(
            enable_memory_cache=True,
            memory_cache_ttl=CACHE_CONFIG.get("NEWS_DATA_MEMORY_TTL", 30),
            enable_disk_cache=False,
        )

        # Test cache operations
        test_key = "test_key"
        test_value = {"data": "test_value", "timestamp": time.time()}

        # Set a value
        market_cache.set(test_key, test_value)

        # Retrieve the value
        retrieved_value = market_cache.get(test_key)
        self.assertEqual(retrieved_value, test_value)

        # Test LRU behavior by adding many entries
        for i in range(10):
            market_cache.set(f"test_key_{i}", f"test_value_{i}")

        # Original entry should still be available (max entries is much larger)
        self.assertEqual(market_cache.get(test_key), test_value)

        # Test a fresh custom cache manager instead of the default
        custom_cache = CacheManager(
            enable_memory_cache=True, memory_cache_ttl=30, enable_disk_cache=False
        )

        # Set and retrieve a value in the custom cache
        custom_key = "custom_key"
        custom_value = {"custom": "value"}
        custom_cache.set(custom_key, custom_value)
        self.assertEqual(custom_cache.get(custom_key), custom_value)

        # Clear the caches for cleanup
        market_cache.clear()
        news_cache.clear()
        custom_cache.clear()

        # Verify cache was cleared
        self.assertIsNone(market_cache.get(test_key))


if __name__ == "__main__":
    unittest.main()
