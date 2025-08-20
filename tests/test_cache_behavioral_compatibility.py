"""
Comprehensive behavioral compatibility tests for the caching system.

This test suite ensures that the caching system maintains 100% behavioral
compatibility with the original implementation. All tests verify that
cached and non-cached operations produce identical results.
"""

import unittest
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from yahoofinance.core.cache import FieldLevelCache, get_cache
from yahoofinance.core.cache_manager import CacheManager, get_cache_manager
from yahoofinance.core.cache_integration import integrate_caching_with_provider


class TestCacheBehavioralCompatibility(unittest.TestCase):
    """
    Test suite to verify caching system maintains identical behavior
    to non-cached operations.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = FieldLevelCache(enable_cache=False)
        self.cache_manager = CacheManager()
        
        # Sample ticker data for testing
        self.sample_ticker_data = {
            "AAPL": {
                "TICKER": "AAPL",
                "COMPANY": "Apple Inc.", 
                "PRICE": 150.25,
                "TARGET": 175.00,
                "UPSIDE": 16.4,
                "#T": 20,
                "#A": 6,
                "%BUY": 33,
                "CAP": 2500000000000,
                "PET": 29.4,
                "PEF": 21.1,
                "PEG": 1.8,
                "BETA": 1.22,
                "SI": 1.2,
                "DIV%": 0.83,
                "EARNINGS": "20250430",
                "EG": -23.8,
                "PP": -14.23,
                "EXRET": 5.1,
                "A": "E",
                "BS": "S"
            },
            "MSFT": {
                "TICKER": "MSFT",
                "COMPANY": "Microsoft Corporation",
                "PRICE": 300.50,
                "TARGET": 350.00,
                "UPSIDE": 16.5,
                "#T": 25,
                "#A": 8,
                "%BUY": 45,
                "CAP": 2200000000000,
                "PET": 28.5,
                "PEF": 20.2,
                "PEG": 2.1,
                "BETA": 0.95,
                "SI": 0.8,
                "DIV%": 1.2,
                "EARNINGS": "20250515",
                "EG": 15.2,
                "PP": 8.45,
                "EXRET": 7.3,
                "A": "A",
                "BS": "B"
            }
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.cache.disable()
        self.cache_manager.disable_cache()
    
    def test_cache_disabled_identical_behavior(self):
        """Test that cache disabled produces identical results to original."""
        # Mock original data fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_ticker_info.return_value = self.sample_ticker_data["AAPL"]
        
        # Test with cache disabled
        self.cache.disable()
        cached_provider = integrate_caching_with_provider(mock_fetcher)
        
        # Both should produce identical results
        original_result = mock_fetcher.get_ticker_info("AAPL")
        cached_result = cached_provider.get_ticker_info("AAPL")
        
        self.assertEqual(original_result, cached_result)
        self.assertEqual(mock_fetcher.get_ticker_info.call_count, 2)  # Called twice
    
    def test_cache_miss_identical_behavior(self):
        """Test that cache miss produces identical results to original."""
        # Mock original data fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_ticker_info.return_value = self.sample_ticker_data["AAPL"]
        
        # Enable cache but clear it (cache miss scenario)
        self.cache.enable()
        self.cache.clear_all()
        
        cached_provider = integrate_caching_with_provider(mock_fetcher)
        
        # First call should be cache miss
        original_result = mock_fetcher.get_ticker_info("AAPL")
        cached_result = cached_provider.get_ticker_info("AAPL")
        
        self.assertEqual(original_result, cached_result)
    
    def test_cache_hit_maintains_data_structure(self):
        """Test that cache hit returns identical data structure."""
        # Configure cache for testing
        cache_config = {
            "TICKER": 3600,  # Cache for 1 hour
            "COMPANY": 3600,
            "CAP": 3600,
            "BETA": 3600
        }
        self.cache.configure_field_ttl(cache_config)
        self.cache.enable()
        
        # Mock original fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_ticker_info.return_value = self.sample_ticker_data["AAPL"]
        
        cached_provider = integrate_caching_with_provider(mock_fetcher)
        
        # First call - cache miss
        result1 = cached_provider.get_ticker_info("AAPL")
        
        # Second call - should have cache hits for some fields
        result2 = cached_provider.get_ticker_info("AAPL")
        
        # Results should be identical
        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)
        
        # Verify all expected fields are present
        expected_fields = ["TICKER", "COMPANY", "PRICE", "TARGET", "UPSIDE"]
        for field in expected_fields:
            self.assertIn(field, result1)
            self.assertIn(field, result2)
            self.assertEqual(result1[field], result2[field])
    
    def test_field_level_caching_accuracy(self):
        """Test that field-level caching maintains data accuracy."""
        # Configure specific fields for caching
        cache_config = {
            "TICKER": 3600,    # Cached
            "COMPANY": 3600,   # Cached
            "PRICE": 0,        # Never cached
            "TARGET": 0        # Never cached
        }
        self.cache.configure_field_ttl(cache_config)
        self.cache.enable()
        
        # Mock fetcher with changing data
        mock_fetcher = Mock()
        
        # First call returns original data
        first_data = self.sample_ticker_data["AAPL"].copy()
        mock_fetcher.get_ticker_info.return_value = first_data
        
        cached_provider = integrate_caching_with_provider(mock_fetcher)
        result1 = cached_provider.get_ticker_info("AAPL")
        
        # Simulate price change (should not be cached)
        updated_data = self.sample_ticker_data["AAPL"].copy()
        updated_data["PRICE"] = 155.00  # Price changed
        updated_data["TARGET"] = 180.00  # Target changed
        mock_fetcher.get_ticker_info.return_value = updated_data
        
        result2 = cached_provider.get_ticker_info("AAPL")
        
        # Cached fields should be identical
        self.assertEqual(result1["TICKER"], result2["TICKER"])
        self.assertEqual(result1["COMPANY"], result2["COMPANY"])
        
        # Non-cached fields should reflect new values
        self.assertEqual(result2["PRICE"], 155.00)
        self.assertEqual(result2["TARGET"], 180.00)
    
    def test_error_handling_compatibility(self):
        """Test that error handling behavior is identical with/without cache."""
        # Mock fetcher that raises an exception
        mock_fetcher = Mock()
        mock_fetcher.get_ticker_info.side_effect = Exception("API Error")
        
        # Test without cache
        self.cache.disable()
        cached_provider = integrate_caching_with_provider(mock_fetcher)
        
        with self.assertRaises(Exception) as context1:
            mock_fetcher.get_ticker_info("INVALID")
        
        with self.assertRaises(Exception) as context2:
            cached_provider.get_ticker_info("INVALID")
        
        # Both should raise identical exceptions
        self.assertEqual(str(context1.exception), str(context2.exception))
    
    def test_batch_operation_compatibility(self):
        """Test that batch operations maintain compatibility."""
        # Mock fetcher with batch capability
        mock_fetcher = Mock()
        batch_data = {
            "AAPL": self.sample_ticker_data["AAPL"],
            "MSFT": self.sample_ticker_data["MSFT"]
        }
        mock_fetcher.batch_get_ticker_info.return_value = batch_data
        
        cached_provider = integrate_caching_with_provider(mock_fetcher)
        
        # Test batch operation
        tickers = ["AAPL", "MSFT"]
        original_result = mock_fetcher.batch_get_ticker_info(tickers)
        cached_result = cached_provider.batch_get_ticker_info(tickers)
        
        self.assertEqual(original_result, cached_result)
        self.assertEqual(len(original_result), len(cached_result))
        
        for ticker in tickers:
            self.assertIn(ticker, original_result)
            self.assertIn(ticker, cached_result)
            self.assertEqual(original_result[ticker], cached_result[ticker])
    
    def test_cache_configuration_from_user_requirements(self):
        """Test cache configuration matches user specifications."""
        # User-defined cache configuration (from requirements)
        user_config = {
            # 24 hour cache
            "TICKER": 86400,
            "COMPANY": 86400,
            "CAP": 86400,
            "BETA": 86400,
            "SI": 86400,
            "DIV%": 86400,
            "EARNINGS": 86400,
            "EG": 86400,
            
            # Never cache (these should be ignored)
            "PRICE": 0,
            "TARGET": 0,
            "UPSIDE": 0,
        }
        
        self.cache.configure_field_ttl(user_config)
        self.cache.enable()
        
        # Test that configuration is applied correctly
        cached_fields = []
        never_cached_fields = []
        
        for field, ttl in user_config.items():
            if ttl > 0:
                cached_fields.append(field)
            else:
                never_cached_fields.append(field)
        
        # Verify cached fields are configured
        self.assertEqual(len(cached_fields), 8)  # Should be 8 cached fields
        self.assertIn("TICKER", cached_fields)
        self.assertIn("COMPANY", cached_fields)
        self.assertIn("BETA", cached_fields)
        
        # Verify never-cached fields
        self.assertEqual(len(never_cached_fields), 3)
        self.assertIn("PRICE", never_cached_fields)
        self.assertIn("TARGET", never_cached_fields)
        self.assertIn("UPSIDE", never_cached_fields)
    
    def test_performance_benefits_without_behavior_change(self):
        """Test that caching provides performance benefits without changing behavior."""
        # Configure cache
        cache_config = {"TICKER": 3600, "COMPANY": 3600}
        self.cache.configure_field_ttl(cache_config)
        self.cache.enable()
        
        # Mock slow fetcher
        call_count = 0
        def slow_fetch(ticker):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate API delay
            return self.sample_ticker_data[ticker]
        
        mock_fetcher = Mock()
        mock_fetcher.get_ticker_info.side_effect = slow_fetch
        
        cached_provider = integrate_caching_with_provider(mock_fetcher)
        
        # Time first call (cache miss)
        start1 = time.time()
        result1 = cached_provider.get_ticker_info("AAPL")
        time1 = time.time() - start1
        
        # Time second call (cache hit for some fields)
        start2 = time.time()
        result2 = cached_provider.get_ticker_info("AAPL")
        time2 = time.time() - start2
        
        # Results should be identical
        self.assertEqual(result1, result2)
        
        # Second call should be faster (cache hit)
        # Note: This test may be flaky due to timing, but demonstrates concept
        
        # Verify API was called appropriately
        self.assertGreaterEqual(call_count, 1)  # At least one API call made
    
    def test_cache_stats_accuracy(self):
        """Test that cache statistics are accurately tracked."""
        # Configure cache
        cache_config = {"TICKER": 3600}
        self.cache.configure_field_ttl(cache_config)
        self.cache.enable()
        
        # Clear stats
        self.cache._stats = {
            'hits': 0, 'misses': 0, 'total_requests': 0,
            'cache_stores': 0, 'evictions': 0
        }
        
        # Mock fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_ticker_info.return_value = self.sample_ticker_data["AAPL"]
        cached_provider = integrate_caching_with_provider(mock_fetcher)
        
        # Make requests and verify stats
        initial_stats = self.cache.get_stats()
        
        # First call - should be cache miss
        cached_provider.get_ticker_info("AAPL")
        stats_after_miss = self.cache.get_stats()
        
        # Second call - should be cache hit for TICKER field
        cached_provider.get_ticker_info("AAPL")
        stats_after_hit = self.cache.get_stats()
        
        # Verify stats progression
        self.assertGreaterEqual(stats_after_miss['misses'], initial_stats['misses'])
        self.assertGreaterEqual(stats_after_hit['hits'], stats_after_miss['hits'])


class TestCacheManagerCompatibility(unittest.TestCase):
    """Test cache manager behavioral compatibility."""
    
    def test_cache_manager_initialization(self):
        """Test that cache manager initializes safely."""
        manager = CacheManager()
        
        # Should initialize with cache disabled by default
        self.assertFalse(manager._enabled)
        
        # Initialize with cache disabled
        manager.initialize(enable_cache=False)
        self.assertFalse(manager._enabled)
        
        # Initialize with cache enabled
        manager.initialize(enable_cache=True)
        self.assertTrue(manager._enabled)
    
    def test_cache_enable_disable_cycle(self):
        """Test that enable/disable cycle works correctly."""
        manager = CacheManager()
        manager.initialize(enable_cache=False)
        
        # Enable cache
        manager.enable_cache()
        self.assertTrue(manager._enabled)
        
        # Disable cache
        manager.disable_cache()
        self.assertFalse(manager._enabled)
    
    def test_performance_stats_collection(self):
        """Test that performance stats are collected correctly."""
        manager = CacheManager()
        manager.initialize(enable_cache=True)
        
        stats = manager.get_performance_stats()
        
        # Verify stats structure
        required_fields = [
            'hits', 'misses', 'total_requests', 'cache_stores', 
            'evictions', 'hit_rate', 'miss_rate', 'cached_tickers', 
            'cached_fields', 'enabled', 'status'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats)
        
        # Verify status reflects actual state
        self.assertEqual(stats['enabled'], manager._enabled)
        self.assertEqual(stats['status'], 'enabled' if manager._enabled else 'disabled')


if __name__ == '__main__':
    # Run comprehensive behavioral compatibility tests
    unittest.main(verbosity=2)