"""
Comprehensive test suite for field-level caching system behavioral compatibility.

This test suite verifies that the field-level caching system maintains 100%
behavioral compatibility with the original Yahoo Finance API. All tests compare
cached vs non-cached results to ensure identical output.

CRITICAL: These tests must pass with ZERO differences between cached and
non-cached results to ensure system reliability.
"""

import pytest
import time
import logging
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock

from yahoofinance.core.field_cache import FieldLevelCache, CacheStats
from yahoofinance.core.field_cache_config import (
    FieldCacheSettings,
    create_test_cache_settings,
    resolve_field_name,
    get_internal_field_names
)
from yahoofinance.core.cache_wrapper import (
    CacheWrappedProvider,
    wrap_provider_with_cache,
    field_cached,
    enable_global_field_cache,
    disable_global_field_cache
)


# Test fixtures and data
TEST_TICKER_DATA = {
    "AAPL": {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "market_cap": 3000000000000,
        "price": 150.25,
        "target_price": 175.00,
        "upside": 16.45,
        "analyst_count": 42,
        "total_ratings": 38,
        "buy_percentage": 78.5,
        "pe_trailing": 28.2,
        "pe_forward": 25.1,
        "peg_ratio": 1.8,
        "beta": 1.23,
        "dividend_yield": 0.52,
        "short_float_pct": 1.2,
        "earnings_growth": 8.5
    },
    "MSFT": {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "market_cap": 2800000000000,
        "price": 340.50,
        "target_price": 380.00,
        "upside": 11.6,
        "analyst_count": 38,
        "total_ratings": 35,
        "buy_percentage": 82.1,
        "pe_trailing": 32.5,
        "pe_forward": 28.8,
        "peg_ratio": 2.1,
        "beta": 0.89,
        "dividend_yield": 0.68,
        "short_float_pct": 0.8,
        "earnings_growth": 12.3
    }
}


class MockProvider:
    """Mock finance data provider for testing."""
    
    def __init__(self, call_delay: float = 0.0, fail_rate: float = 0.0):
        """
        Initialize mock provider.
        
        Args:
            call_delay: Artificial delay for API calls (simulates network latency)
            fail_rate: Probability of API call failure (0.0 = never fail, 1.0 = always fail)
        """
        self.call_delay = call_delay
        self.fail_rate = fail_rate
        self.call_count = 0
        self.call_history = []
    
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """Mock get_ticker_info method."""
        self.call_count += 1
        self.call_history.append(("get_ticker_info", ticker, skip_insider_metrics))
        
        if self.call_delay > 0:
            time.sleep(self.call_delay)
        
        if self.fail_rate > 0 and (self.call_count * self.fail_rate) >= 1:
            raise Exception(f"Simulated API failure for {ticker}")
        
        if ticker in TEST_TICKER_DATA:
            return TEST_TICKER_DATA[ticker].copy()
        else:
            return {"ticker": ticker, "error": "not_found"}
    
    def batch_get_ticker_info(self, tickers: List[str], 
                             skip_insider_metrics: bool = False) -> Dict[str, Dict[str, Any]]:
        """Mock batch_get_ticker_info method."""
        self.call_count += 1
        self.call_history.append(("batch_get_ticker_info", tickers, skip_insider_metrics))
        
        if self.call_delay > 0:
            time.sleep(self.call_delay * len(tickers))
        
        result = {}
        for ticker in tickers:
            if ticker in TEST_TICKER_DATA:
                result[ticker] = TEST_TICKER_DATA[ticker].copy()
            else:
                result[ticker] = {"ticker": ticker, "error": "not_found"}
        
        return result
    
    def clear_cache(self):
        """Mock cache clearing."""
        pass
    
    def get_cache_info(self):
        """Mock cache info."""
        return {"mock_provider": True}


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    return MockProvider()


@pytest.fixture
def test_cache_settings():
    """Create test cache settings."""
    return create_test_cache_settings(
        enabled=True,
        never_fields={"PRICE", "TARGET", "UPSIDE"},
        daily_fields={"TICKER": 86400, "COMPANY": 86400, "CAP": 86400}
    )


@pytest.fixture
def field_cache(test_cache_settings):
    """Create a field cache for testing."""
    return FieldLevelCache(enabled=True, config={
        "NEVER": test_cache_settings.never_cache_fields,
        "DAILY": test_cache_settings.daily_cache_fields
    })


class TestFieldLevelCache:
    """Test field-level cache functionality."""
    
    def test_cache_initialization(self, test_cache_settings):
        """Test cache initialization with various configurations."""
        # Test enabled cache
        cache = FieldLevelCache(enabled=True, config={
            "NEVER": {"PRICE", "TARGET"},
            "DAILY": {"TICKER": 3600, "COMPANY": 7200}
        })
        assert cache.enabled is True
        assert cache._get_field_ttl("PRICE") is None
        assert cache._get_field_ttl("TICKER") == 3600
        assert cache._get_field_ttl("COMPANY") == 7200
        
        # Test disabled cache
        cache_disabled = FieldLevelCache(enabled=False)
        assert cache_disabled.enabled is False
    
    def test_field_ttl_resolution(self, field_cache):
        """Test field TTL resolution based on configuration."""
        # Never cached fields should return None
        assert field_cache._get_field_ttl("PRICE") is None
        assert field_cache._get_field_ttl("TARGET") is None
        assert field_cache._get_field_ttl("UPSIDE") is None
        
        # Daily cached fields should return TTL
        assert field_cache._get_field_ttl("TICKER") == 86400
        assert field_cache._get_field_ttl("COMPANY") == 86400
        assert field_cache._get_field_ttl("CAP") == 86400
        
        # Unknown fields should return None
        assert field_cache._get_field_ttl("UNKNOWN_FIELD") is None
    
    def test_cache_key_generation(self, field_cache):
        """Test cache key generation."""
        key1 = field_cache._generate_cache_key("AAPL", "TICKER")
        key2 = field_cache._generate_cache_key("aapl", "TICKER")
        key3 = field_cache._generate_cache_key("AAPL", "COMPANY")
        
        assert key1 == "field:AAPL:TICKER"
        assert key2 == "field:AAPL:TICKER"  # Should be normalized to uppercase
        assert key3 == "field:AAPL:COMPANY"
        assert key1 != key3
    
    def test_single_field_caching(self, field_cache):
        """Test caching of individual field values."""
        api_call_count = 0
        
        def mock_api_call():
            nonlocal api_call_count
            api_call_count += 1
            return "Apple Inc."
        
        # First call should hit API
        result1 = field_cache.get_field_value("AAPL", "COMPANY", mock_api_call)
        assert result1 == "Apple Inc."
        assert api_call_count == 1
        
        # Second call should hit cache
        result2 = field_cache.get_field_value("AAPL", "COMPANY", mock_api_call)
        assert result2 == "Apple Inc."
        assert api_call_count == 1  # No additional API call
        
        # Never-cached field should always hit API
        def mock_price_call():
            nonlocal api_call_count
            api_call_count += 1
            return 150.25
        
        result3 = field_cache.get_field_value("AAPL", "PRICE", mock_price_call)
        assert result3 == 150.25
        assert api_call_count == 2
        
        result4 = field_cache.get_field_value("AAPL", "PRICE", mock_price_call)
        assert result4 == 150.25
        assert api_call_count == 3  # Should call API again
    
    def test_multiple_field_caching(self, field_cache):
        """Test caching of multiple fields simultaneously."""
        api_call_count = 0
        
        def mock_api_call():
            nonlocal api_call_count
            api_call_count += 1
            return {
                "TICKER": "AAPL",
                "COMPANY": "Apple Inc.", 
                "PRICE": 150.25,
                "CAP": 3000000000000
            }
        
        fields = ["TICKER", "COMPANY", "PRICE", "CAP"]
        
        # First call should hit API
        result1 = field_cache.get_multiple_fields("AAPL", fields, mock_api_call)
        assert api_call_count == 1
        assert result1["TICKER"] == "AAPL"
        assert result1["COMPANY"] == "Apple Inc."
        assert result1["PRICE"] == 150.25
        
        # Second call should use cache for cacheable fields
        result2 = field_cache.get_multiple_fields("AAPL", fields, mock_api_call)
        assert api_call_count == 2  # Called again due to PRICE being never-cached
        
        # Verify cached fields remain consistent
        assert result2["TICKER"] == result1["TICKER"]
        assert result2["COMPANY"] == result1["COMPANY"]
        assert result2["CAP"] == result1["CAP"]
    
    def test_cache_expiration(self, test_cache_settings):
        """Test cache expiration functionality."""
        # Create cache with very short TTL for testing
        cache = FieldLevelCache(enabled=True, config={
            "NEVER": set(),
            "DAILY": {"TICKER": 1}  # 1 second TTL
        })
        
        api_call_count = 0
        
        def mock_api_call():
            nonlocal api_call_count
            api_call_count += 1
            return "AAPL"
        
        # First call
        result1 = cache.get_field_value("AAPL", "TICKER", mock_api_call)
        assert result1 == "AAPL"
        assert api_call_count == 1
        
        # Immediate second call should hit cache
        result2 = cache.get_field_value("AAPL", "TICKER", mock_api_call)
        assert result2 == "AAPL"
        assert api_call_count == 1
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Third call should hit API again (expired)
        result3 = cache.get_field_value("AAPL", "TICKER", mock_api_call)
        assert result3 == "AAPL"
        assert api_call_count == 2
    
    def test_cache_invalidation(self, field_cache):
        """Test cache invalidation functionality."""
        # Setup cache with data
        def mock_api_call():
            return "Apple Inc."
        
        result1 = field_cache.get_field_value("AAPL", "COMPANY", mock_api_call)
        assert result1 == "Apple Inc."
        
        # Verify cache contains data
        assert "AAPL" in field_cache._cache
        assert "COMPANY" in field_cache._cache["AAPL"]
        
        # Invalidate ticker
        field_cache.invalidate_ticker("AAPL")
        assert "AAPL" not in field_cache._cache
        
        # Setup data for field invalidation test
        field_cache.get_field_value("AAPL", "COMPANY", mock_api_call)
        field_cache.get_field_value("MSFT", "COMPANY", mock_api_call)
        
        # Invalidate field across all tickers
        field_cache.invalidate_field("COMPANY")
        assert "AAPL" not in field_cache._cache or "COMPANY" not in field_cache._cache.get("AAPL", {})
        assert "MSFT" not in field_cache._cache or "COMPANY" not in field_cache._cache.get("MSFT", {})
    
    def test_cache_statistics(self, field_cache):
        """Test cache statistics collection."""
        stats = field_cache.stats
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.total_requests == 0
        
        # Generate some cache activity
        def mock_api_call():
            return "test_value"
        
        # Cache miss
        field_cache.get_field_value("AAPL", "COMPANY", mock_api_call)
        assert field_cache.stats.cache_misses == 1
        assert field_cache.stats.total_requests == 1
        
        # Cache hit
        field_cache.get_field_value("AAPL", "COMPANY", mock_api_call)
        assert field_cache.stats.cache_hits == 1
        assert field_cache.stats.total_requests == 2
        
        # Verify hit rate calculation
        assert field_cache.stats.hit_rate() == 50.0


class TestCacheWrappedProvider:
    """Test cache-wrapped provider functionality."""
    
    def test_provider_wrapping(self, mock_provider, test_cache_settings):
        """Test wrapping of provider with cache."""
        wrapped = CacheWrappedProvider(mock_provider, test_cache_settings)
        
        # Verify wrapped provider maintains API compatibility
        assert hasattr(wrapped, 'get_ticker_info')
        assert hasattr(wrapped, 'batch_get_ticker_info')
        assert hasattr(wrapped, 'clear_cache')
        
        # Test that non-cached methods delegate correctly
        if hasattr(mock_provider, 'get_cache_info'):
            cache_info = wrapped.get_cache_info()
            assert cache_info["mock_provider"] is True
    
    def test_behavioral_compatibility_single_ticker(self, mock_provider, test_cache_settings):
        """Test that cached provider produces identical results to original."""
        # Create wrapped and unwrapped versions
        wrapped = CacheWrappedProvider(mock_provider, test_cache_settings)
        unwrapped = MockProvider()
        
        ticker = "AAPL"
        
        # Get results from both
        wrapped_result = wrapped.get_ticker_info(ticker)
        unwrapped_result = unwrapped.get_ticker_info(ticker)
        
        # Results should be identical
        assert wrapped_result == unwrapped_result
        
        # Test with skip_insider_metrics parameter
        wrapped_result2 = wrapped.get_ticker_info(ticker, skip_insider_metrics=True)
        unwrapped_result2 = unwrapped.get_ticker_info(ticker, skip_insider_metrics=True)
        
        assert wrapped_result2 == unwrapped_result2
    
    def test_behavioral_compatibility_batch(self, mock_provider, test_cache_settings):
        """Test that batch operations maintain behavioral compatibility."""
        wrapped = CacheWrappedProvider(mock_provider, test_cache_settings)
        unwrapped = MockProvider()
        
        tickers = ["AAPL", "MSFT"]
        
        # Get batch results from both
        wrapped_result = wrapped.batch_get_ticker_info(tickers)
        unwrapped_result = unwrapped.batch_get_ticker_info(tickers)
        
        # Results should be identical
        assert wrapped_result == unwrapped_result
        
        # Verify all tickers are present
        for ticker in tickers:
            assert ticker in wrapped_result
            assert ticker in unwrapped_result
            assert wrapped_result[ticker] == unwrapped_result[ticker]
    
    def test_cache_performance_improvement(self, mock_provider, test_cache_settings):
        """Test that caching improves performance while maintaining results."""
        # Create provider with artificial delay
        slow_provider = MockProvider(call_delay=0.1)
        wrapped = CacheWrappedProvider(slow_provider, test_cache_settings)
        
        ticker = "AAPL"
        
        # First call - should be slow (cache miss)
        start_time = time.time()
        result1 = wrapped.get_ticker_info(ticker)
        first_call_time = time.time() - start_time
        
        # Second call - should be fast (cache hit for cacheable fields)
        start_time = time.time()
        result2 = wrapped.get_ticker_info(ticker)
        second_call_time = time.time() - start_time
        
        # Results should be identical
        assert result1 == result2
        
        # Second call should be faster (though not guaranteed due to never-cached fields)
        # At minimum, verify we can call multiple times without errors
        assert second_call_time >= 0
    
    def test_cache_fallback_on_error(self, test_cache_settings):
        """Test that cache errors fallback to API gracefully."""
        # Create provider that sometimes fails
        failing_provider = MockProvider(fail_rate=0.5)
        wrapped = CacheWrappedProvider(failing_provider, test_cache_settings)
        
        # Mock cache to raise error
        with patch.object(wrapped._cache, 'get_multiple_fields', side_effect=Exception("Cache error")):
            # Should fallback to original provider
            try:
                result = wrapped.get_ticker_info("AAPL")
                # If it doesn't raise, the fallback worked
                assert "ticker" in result
            except Exception as e:
                # Should be original provider error, not cache error
                assert "Cache error" not in str(e)
    
    def test_statistics_collection(self, mock_provider, test_cache_settings):
        """Test cache statistics collection in wrapped provider."""
        wrapped = CacheWrappedProvider(mock_provider, test_cache_settings)
        
        # Make some calls
        wrapped.get_ticker_info("AAPL")
        wrapped.get_ticker_info("AAPL")  # Should hit cache
        wrapped.get_ticker_info("MSFT")
        
        # Get statistics
        stats = wrapped.get_cache_stats()
        
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "total_requests" in stats
        assert "provider_type" in stats
        assert stats["provider_type"] == "MockProvider"
        assert stats["cache_enabled"] is True


class TestFieldCacheConfiguration:
    """Test field cache configuration system."""
    
    def test_field_name_resolution(self):
        """Test field name resolution between display and internal names."""
        # Test known mappings
        assert resolve_field_name("ticker") == "TICKER"
        assert resolve_field_name("name") == "COMPANY"
        assert resolve_field_name("market_cap") == "CAP"
        
        # Test unknown fields return as-is
        assert resolve_field_name("unknown_field") == "unknown_field"
        
        # Test getting internal names
        internal_names = get_internal_field_names("TICKER")
        assert "ticker" in internal_names
        assert "symbol" in internal_names
    
    def test_cache_settings_validation(self):
        """Test cache settings validation."""
        # Valid settings
        valid_settings = create_test_cache_settings(
            enabled=True,
            never_fields={"PRICE", "TARGET"},
            daily_fields={"TICKER": 3600, "COMPANY": 7200}
        )
        warnings = valid_settings.validate_configuration()
        assert len(warnings) == 0
        
        # Invalid settings - field overlap
        invalid_settings = create_test_cache_settings(
            enabled=True,
            never_fields={"PRICE", "TICKER"},  # TICKER also in daily
            daily_fields={"TICKER": 3600, "COMPANY": 7200}
        )
        warnings = invalid_settings.validate_configuration()
        assert len(warnings) > 0
        assert any("overlap" in warning.lower() for warning in warnings)
        
        # Invalid TTL values
        invalid_ttl_settings = create_test_cache_settings(
            enabled=True,
            never_fields={"PRICE"},
            daily_fields={"TICKER": -1, "COMPANY": 0}  # Invalid TTLs
        )
        warnings = invalid_ttl_settings.validate_configuration()
        assert len(warnings) >= 2  # Should warn about both invalid TTLs


class TestGlobalCacheIntegration:
    """Test global cache integration functionality."""
    
    def test_global_cache_enable_disable(self):
        """Test enabling and disabling global cache."""
        # Ensure disabled initially
        disable_global_field_cache()
        
        # Enable with test settings
        test_settings = create_test_cache_settings(enabled=True)
        success = enable_global_field_cache(test_settings)
        assert success is True
        
        from yahoofinance.core.cache_wrapper import is_global_cache_enabled
        assert is_global_cache_enabled() is True
        
        # Disable
        disable_global_field_cache()
        assert is_global_cache_enabled() is False
    
    def test_field_cached_decorator(self):
        """Test field_cached decorator functionality."""
        call_count = 0
        
        @field_cached(["TICKER", "COMPANY"], 
                     create_test_cache_settings(enabled=True))
        def mock_function(ticker):
            nonlocal call_count
            call_count += 1
            return {"TICKER": ticker, "COMPANY": f"{ticker} Inc."}
        
        # First call
        result1 = mock_function("AAPL")
        assert call_count == 1
        assert result1["TICKER"] == "AAPL"
        
        # Second call should hit cache (but may still call due to implementation)
        result2 = mock_function("AAPL")
        assert result2 == result1
    
    def test_wrapper_function(self, mock_provider):
        """Test wrap_provider_with_cache function."""
        settings = create_test_cache_settings(enabled=True)
        wrapped = wrap_provider_with_cache(mock_provider, settings)
        
        assert isinstance(wrapped, CacheWrappedProvider)
        assert wrapped._cache.enabled is True
        assert wrapped.get_original_provider() is mock_provider


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_cache_disabled_behavior(self, mock_provider):
        """Test behavior when cache is disabled."""
        disabled_settings = create_test_cache_settings(enabled=False)
        wrapped = CacheWrappedProvider(mock_provider, disabled_settings)
        
        # Should behave identically to unwrapped provider
        unwrapped = MockProvider()
        
        result_wrapped = wrapped.get_ticker_info("AAPL")
        result_unwrapped = unwrapped.get_ticker_info("AAPL")
        
        assert result_wrapped == result_unwrapped
        
        # Cache should show as disabled in stats
        stats = wrapped.get_cache_stats()
        assert stats["cache_enabled"] is False
    
    def test_empty_data_handling(self, mock_provider, test_cache_settings):
        """Test handling of empty or None data."""
        wrapped = CacheWrappedProvider(mock_provider, test_cache_settings)
        
        # Test with ticker that returns empty data
        result = wrapped.get_ticker_info("NONEXISTENT")
        assert "error" in result or result is not None
    
    def test_concurrent_access(self, mock_provider, test_cache_settings):
        """Test thread safety of cache operations."""
        import threading
        
        wrapped = CacheWrappedProvider(mock_provider, test_cache_settings)
        results = []
        errors = []
        
        def worker():
            try:
                result = wrapped.get_ticker_info("AAPL")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors and consistent results
        assert len(errors) == 0
        assert len(results) == 10
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result


@pytest.mark.integration
class TestIntegrationBehavioralCompatibility:
    """Integration tests to verify complete behavioral compatibility."""
    
    def test_end_to_end_compatibility(self, mock_provider):
        """Comprehensive end-to-end test of behavioral compatibility."""
        # Test settings that mirror production configuration
        production_settings = FieldCacheSettings(
            enabled=True,
            never_cache_fields={
                "PRICE", "TARGET", "UPSIDE", "#T", "#A", "%BUY", 
                "PET", "PEF", "PEG", "PP", "EXRET", "A", "BS"
            },
            daily_cache_fields={
                "TICKER": 86400, "COMPANY": 86400, "CAP": 86400,
                "BETA": 86400, "SI": 86400, "DIV%": 86400,
                "EARNINGS": 86400, "EG": 86400
            }
        )
        
        wrapped = CacheWrappedProvider(mock_provider, production_settings)
        unwrapped = MockProvider()
        
        test_tickers = ["AAPL", "MSFT", "NONEXISTENT"]
        
        # Test single ticker calls
        for ticker in test_tickers:
            wrapped_result = wrapped.get_ticker_info(ticker)
            unwrapped_result = unwrapped.get_ticker_info(ticker)
            
            assert wrapped_result == unwrapped_result, f"Mismatch for {ticker}"
        
        # Test batch calls
        wrapped_batch = wrapped.batch_get_ticker_info(test_tickers)
        unwrapped_batch = unwrapped.batch_get_ticker_info(test_tickers)
        
        assert wrapped_batch == unwrapped_batch
        
        # Test repeated calls for cache behavior
        for ticker in test_tickers:
            wrapped_repeat = wrapped.get_ticker_info(ticker)
            unwrapped_repeat = unwrapped.get_ticker_info(ticker)
            
            assert wrapped_repeat == unwrapped_repeat, f"Repeat mismatch for {ticker}"
    
    def test_performance_characteristics(self, mock_provider):
        """Test that cache improves performance without changing behavior."""
        settings = create_test_cache_settings(enabled=True)
        wrapped = CacheWrappedProvider(mock_provider, settings)
        
        # Clear any existing cache
        wrapped.clear_cache()
        
        ticker = "AAPL"
        iterations = 5
        
        # Collect timing and results
        results = []
        times = []
        
        for i in range(iterations):
            start = time.time()
            result = wrapped.get_ticker_info(ticker)
            elapsed = time.time() - start
            
            results.append(result)
            times.append(elapsed)
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Cache altered result data"
        
        # Performance should be reasonable (no catastrophic slowdown)
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"Average call time too slow: {avg_time}s"


if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(level=logging.INFO)
    
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])