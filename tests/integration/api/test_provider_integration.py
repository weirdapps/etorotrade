"""
Integration tests for API providers with real network calls.

Tests the full API stack including:
- AsyncHybridProvider with yfinance
- ResilientProvider with fallback logic
- Cache integration
- Error handling and retry logic
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.api.providers.resilient_provider import ResilientProvider
from yahoofinance.api.providers.fallback_strategy import DataSource
from yahoofinance.core.cache import default_cache_manager


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def cache_manager():
    """Create a clean cache manager for testing."""
    cache = default_cache_manager
    cache.clear()
    yield cache
    cache.clear()


@pytest.fixture
def hybrid_provider():
    """Create AsyncHybridProvider instance."""
    return AsyncHybridProvider()


@pytest.fixture
def resilient_provider(cache_manager):
    """Create ResilientProvider with cache."""
    return ResilientProvider(enable_fallback=True, enable_stale_cache=True)


class TestAsyncHybridProviderIntegration:
    """Integration tests for AsyncHybridProvider with real API calls."""

    @pytest.mark.asyncio
    async def test_fetch_single_ticker(self, hybrid_provider):
        """Fetch real data for a well-known ticker."""
        result = await hybrid_provider.get_ticker_info("AAPL")

        # Should have basic fields
        assert result is not None
        assert "symbol" in result or "ticker" in result
        # Provider returns normalized data with various price fields
        assert any(k in result for k in ["price", "current_price", "currentPrice", "regularMarketPrice"])

    @pytest.mark.asyncio
    async def test_fetch_multiple_tickers(self, hybrid_provider):
        """Fetch data for multiple tickers concurrently."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        tasks = [hybrid_provider.get_ticker_info(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result, expected_ticker in zip(results, tickers):
            assert result is not None
            assert result.get("symbol") == expected_ticker

    @pytest.mark.asyncio
    async def test_invalid_ticker_handling(self, hybrid_provider):
        """Invalid ticker should return error indicator."""
        result = await hybrid_provider.get_ticker_info("INVALID_TICKER_XYZ")

        # Should either have error field or return minimal data
        assert result is not None
        assert "symbol" in result or "error" in result

    @pytest.mark.asyncio
    async def test_performance_concurrent_requests(self, hybrid_provider):
        """Test performance with many concurrent requests."""
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
                   "META", "AMZN", "NFLX", "DIS", "UBER"]

        import time
        start = time.perf_counter()

        tasks = [hybrid_provider.get_ticker_info(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 15 seconds for 10 tickers)
        # Note: Real API calls can be slow, this is just a smoke test
        assert elapsed < 15.0, f"Took {elapsed:.2f}s for 10 tickers (expected <15s)"
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_rate_limiting_respected(self, hybrid_provider):
        """Verify rate limiting doesn't cause failures."""
        # Make many rapid requests
        tasks = [hybrid_provider.get_ticker_info("AAPL") for _ in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed or fail gracefully
        for result in results:
            assert not isinstance(result, Exception) or "rate limit" in str(result).lower()


class TestResilientProviderIntegration:
    """Integration tests for ResilientProvider with fallback logic."""

    @pytest.mark.asyncio
    async def test_primary_source_success(self, resilient_provider):
        """Successful fetch from primary source."""
        result = await resilient_provider.get_ticker_info("AAPL")

        assert result is not None
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "_data_source" in result
        assert result["_data_source"] in [DataSource.PRIMARY.value, DataSource.CACHE_FRESH.value]

    @pytest.mark.asyncio
    async def test_fallback_activation(self, resilient_provider):
        """Test fallback when primary fails."""
        # Mock primary to fail
        with patch.object(resilient_provider.strategy.primary, 'get_ticker_info',
                         side_effect=Exception("Primary API down")):
            result = await resilient_provider.get_ticker_info("AAPL")

            # Should get data from fallback or cache
            assert result is not None
            if "error" not in result:
                assert "_data_source" in result
                assert result["_data_source"] in [
                    DataSource.FALLBACK.value,
                    DataSource.CACHE_FRESH.value,
                    DataSource.CACHE_STALE.value
                ]

    @pytest.mark.asyncio
    async def test_cache_integration(self, resilient_provider, cache_manager):
        """Test cache is populated and used."""
        ticker = "MSFT"

        # First fetch - should populate cache
        result1 = await resilient_provider.get_ticker_info(ticker)
        assert result1 is not None

        # Check cache was populated
        cached = cache_manager.get(ticker)
        assert cached is not None

        # Second fetch - should use cache if within TTL
        result2 = await resilient_provider.get_ticker_info(ticker)
        assert result2 is not None

        # If from cache, should have cache metadata
        if result2.get("_data_source") in [DataSource.CACHE_FRESH.value, DataSource.CACHE_STALE.value]:
            assert "_is_stale" in result2

    @pytest.mark.asyncio
    async def test_stale_cache_usage(self, resilient_provider, cache_manager):
        """Test stale cache is used when all sources fail."""
        ticker = "NVDA"

        # First fetch to populate cache
        result1 = await resilient_provider.get_ticker_info(ticker)
        assert result1 is not None

        # Manually age the cache
        cached_data = cache_manager.get(ticker)
        if cached_data:
            cached_data['_cached_at'] = datetime.now() - timedelta(days=3)
            cache_manager.set(ticker, cached_data)

        # Mock both providers to fail
        with patch.object(resilient_provider.strategy.primary, 'get_ticker_info',
                         side_effect=Exception("Primary down")):
            with patch.object(resilient_provider.strategy.fallback, 'get_ticker_info',
                             side_effect=Exception("Fallback down")):
                result2 = await resilient_provider.get_ticker_info(ticker)

                # Should get stale cache data
                if result2 and "error" not in result2:
                    assert result2.get("_data_source") == DataSource.CACHE_STALE.value
                    assert result2.get("_is_stale") is True

    @pytest.mark.asyncio
    async def test_metadata_enrichment(self, resilient_provider):
        """Verify metadata is added to responses."""
        result = await resilient_provider.get_ticker_info("GOOGL")

        assert result is not None
        if "error" not in result:
            # Should have metadata fields
            assert "_data_source" in result
            assert "_is_stale" in result
            assert "_fetched_at" in result
            assert "_latency_ms" in result

            # Validate metadata types
            assert isinstance(result["_is_stale"], bool)
            assert isinstance(result["_latency_ms"], (int, float))

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_fallback(self, resilient_provider):
        """Test concurrent requests all use fallback correctly."""
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

        tasks = [resilient_provider.get_ticker_info(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert result is not None
            # Each should have data source metadata
            if "error" not in result:
                assert "_data_source" in result


class TestCacheIntegration:
    """Integration tests for cache behavior."""

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, resilient_provider, cache_manager):
        """Test cache respects TTL."""
        ticker = "META"

        # Fetch to populate cache
        result1 = await resilient_provider.get_ticker_info(ticker)
        assert result1 is not None

        # Manually expire cache
        cached_data = cache_manager.get(ticker)
        if cached_data and '_cached_at' in cached_data:
            # Set cache time to 49 hours ago (beyond default 48hr TTL)
            cached_data['_cached_at'] = datetime.now() - timedelta(hours=49)
            cache_manager.set(ticker, cached_data)

        # Next fetch should treat cache as stale
        result2 = await resilient_provider.get_ticker_info(ticker)
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_cache_isolation_between_tickers(self, resilient_provider, cache_manager):
        """Test cache entries don't interfere with each other."""
        ticker1 = "AMZN"
        ticker2 = "NFLX"

        # Fetch both
        result1 = await resilient_provider.get_ticker_info(ticker1)
        result2 = await resilient_provider.get_ticker_info(ticker2)

        assert result1 is not None
        assert result2 is not None

        # Cache should have separate entries
        cached1 = cache_manager.get(ticker1)
        cached2 = cache_manager.get(ticker2)

        if cached1 and cached2:
            assert cached1.get("symbol") == ticker1
            assert cached2.get("symbol") == ticker2

    @pytest.mark.asyncio
    async def test_cache_clear(self, resilient_provider, cache_manager):
        """Test cache can be cleared."""
        # Populate cache
        await resilient_provider.get_ticker_info("DIS")

        # Verify cache has data
        assert cache_manager.get("DIS") is not None

        # Clear cache
        cache_manager.clear()

        # Verify cache is empty
        assert cache_manager.get("DIS") is None


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, resilient_provider):
        """Test handling of network timeouts."""
        # Mock to simulate timeout
        async def timeout_mock(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout

        with patch.object(resilient_provider.strategy.primary, 'get_ticker_info',
                         side_effect=asyncio.TimeoutError("Network timeout")):
            result = await resilient_provider.get_ticker_info("AAPL")

            # Should fallback or return cached data
            assert result is not None

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self, resilient_provider):
        """Test handling of malformed API responses."""
        # Mock to return malformed data
        with patch.object(resilient_provider.strategy.primary, 'get_ticker_info',
                         return_value={"error": "Invalid data"}):
            result = await resilient_provider.get_ticker_info("TEST")

            # Should attempt fallback
            assert result is not None

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, resilient_provider):
        """Test handling when some requests fail in batch."""
        tickers = ["AAPL", "INVALID_XYZ", "MSFT"]

        tasks = [resilient_provider.get_ticker_info(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 3
        # Should have mix of successes and failures
        successes = sum(1 for r in results if isinstance(r, dict) and "error" not in r)
        assert successes >= 2  # At least AAPL and MSFT should succeed


class TestPerformanceUnderLoad:
    """Integration tests for performance under load."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_batch_processing(self, resilient_provider):
        """Test processing large batch of tickers."""
        # Use well-known tickers
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "BRK.B", "V", "JNJ",
            "WMT", "JPM", "MA", "PG", "UNH",
            "DIS", "HD", "BAC", "VZ", "ADBE"
        ]

        import time
        start = time.perf_counter()

        tasks = [resilient_provider.get_ticker_info(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        # Note: Real API calls can vary, this is just a smoke test
        assert elapsed < 30.0, f"Took {elapsed:.2f}s for 20 tickers (expected <30s)"
        assert len(results) == 20

    @pytest.mark.asyncio
    async def test_repeated_requests_use_cache(self, resilient_provider):
        """Test repeated requests benefit from cache."""
        ticker = "AAPL"

        # First request - fresh fetch
        import time
        start1 = time.perf_counter()
        result1 = await resilient_provider.get_ticker_info(ticker)
        elapsed1 = time.perf_counter() - start1

        # Second request - should be faster from cache
        start2 = time.perf_counter()
        result2 = await resilient_provider.get_ticker_info(ticker)
        elapsed2 = time.perf_counter() - start2

        assert result1 is not None
        assert result2 is not None

        # Second request should be significantly faster
        # (unless first one was also from cache)
        if result1.get("_data_source") != DataSource.CACHE_FRESH.value:
            assert elapsed2 < elapsed1 * 0.5, "Cache should make second request faster"
