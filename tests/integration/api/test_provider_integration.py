"""
Integration tests for API providers with real network calls.

Tests the full API stack including:
- AsyncHybridProvider with yfinance
- Error handling and retry logic
"""

import pytest
import asyncio
from unittest.mock import patch

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def hybrid_provider():
    """Create AsyncHybridProvider instance."""
    return AsyncHybridProvider()


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
    @pytest.mark.xfail(reason="Network-dependent test - may fail due to API rate limits or timeouts")
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


