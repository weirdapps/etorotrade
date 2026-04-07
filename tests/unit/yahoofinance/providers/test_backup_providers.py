"""
Unit tests for backup data providers (Alpha Vantage and Polygon).

Tests verify that providers:
- Return None/empty gracefully when no API key is set
- Implement proper rate limiting
- Follow the AsyncFinanceDataProvider interface
- Handle errors appropriately
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yahoofinance.api.providers.alpha_vantage_provider import AlphaVantageProvider
from yahoofinance.api.providers.polygon_provider import PolygonProvider
from yahoofinance.api.providers.provider_registry import (
    ProviderRegistry,
    get_provider_registry,
    get_stock_data,
)
from yahoofinance.core.errors import APIError, RateLimitError


class TestAlphaVantageProvider:
    """Tests for Alpha Vantage provider."""

    def test_init_without_api_key(self):
        """Test provider initializes gracefully without API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = AlphaVantageProvider()
            assert provider.api_key is None
            assert provider.daily_request_count == 0
            assert provider.minute_request_count == 0

    def test_init_with_api_key(self):
        """Test provider initializes with API key from environment."""
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}):
            provider = AlphaVantageProvider()
            assert provider.api_key == "test_key"

    @pytest.mark.asyncio
    async def test_get_ticker_info_without_key(self):
        """Test get_ticker_info returns empty dict when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = AlphaVantageProvider()
            result = await provider.get_ticker_info("AAPL")
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_price_data_without_key(self):
        """Test get_price_data returns empty dict when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = AlphaVantageProvider()
            result = await provider.get_price_data("AAPL")
            assert result == {}

    @pytest.mark.asyncio
    async def test_rate_limiting_per_minute(self):
        """Test that rate limiting enforces 5 requests per minute."""
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}):
            provider = AlphaVantageProvider()

            # Simulate 5 requests (should pass)
            for i in range(5):
                await provider._rate_limit()
                assert provider.minute_request_count == i + 1

            # 6th request should require waiting
            start_time = time.time()
            # Mock sleep to avoid actual waiting in tests
            with patch("asyncio.sleep") as mock_sleep:
                await provider._rate_limit()
                # Should have called sleep because we hit the limit
                assert mock_sleep.called

    @pytest.mark.asyncio
    async def test_rate_limiting_per_day(self):
        """Test that rate limiting enforces 25 requests per day."""
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}):
            provider = AlphaVantageProvider()

            # Set daily count to max
            provider.daily_request_count = 25

            # Next request should raise RateLimitError
            with pytest.raises(RateLimitError) as exc_info:
                await provider._rate_limit()

            assert "daily limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_batch_get_ticker_info_without_key(self):
        """Test batch operation returns empty dicts when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = AlphaVantageProvider()
            tickers = ["AAPL", "MSFT", "GOOGL"]
            result = await provider.batch_get_ticker_info(tickers)

            assert len(result) == 3
            for ticker in tickers:
                assert result[ticker] == {}

    def test_cache_info(self):
        """Test get_cache_info returns rate limit statistics."""
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}):
            provider = AlphaVantageProvider()
            provider.daily_request_count = 10
            provider.minute_request_count = 2

            info = provider.get_cache_info()

            assert info["provider"] == "AlphaVantage"
            assert info["api_key_configured"] is True
            assert info["daily_requests_used"] == 10
            assert info["daily_requests_remaining"] == 15
            assert info["minute_requests_used"] == 2

    @pytest.mark.asyncio
    async def test_parse_number(self):
        """Test _parse_number handles various formats."""
        provider = AlphaVantageProvider()

        assert provider._parse_number("123.45") == pytest.approx(123.45)
        assert provider._parse_number("None") is None
        assert provider._parse_number(None) is None
        assert provider._parse_number("") is None
        assert provider._parse_number("invalid") is None

    @pytest.mark.asyncio
    async def test_historical_data_returns_empty(self):
        """Test historical_data returns empty DataFrame (not implemented)."""
        provider = AlphaVantageProvider()
        result = await provider.get_historical_data("AAPL")
        assert result.empty

    @pytest.mark.asyncio
    async def test_earnings_dates_returns_none(self):
        """Test earnings_dates returns None tuple (not implemented)."""
        provider = AlphaVantageProvider()
        result = await provider.get_earnings_dates("AAPL")
        assert result == (None, None)


class TestPolygonProvider:
    """Tests for Polygon.io provider."""

    def test_init_without_api_key(self):
        """Test provider initializes gracefully without API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = PolygonProvider()
            assert provider.api_key is None
            assert provider.minute_request_count == 0

    def test_init_with_api_key(self):
        """Test provider initializes with API key from environment."""
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            provider = PolygonProvider()
            assert provider.api_key == "test_key"

    @pytest.mark.asyncio
    async def test_get_ticker_info_without_key(self):
        """Test get_ticker_info returns empty dict when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = PolygonProvider()
            result = await provider.get_ticker_info("AAPL")
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_price_data_without_key(self):
        """Test get_price_data returns empty dict when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = PolygonProvider()
            result = await provider.get_price_data("AAPL")
            assert result == {}

    @pytest.mark.asyncio
    async def test_rate_limiting_per_minute(self):
        """Test that rate limiting enforces 5 requests per minute."""
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            provider = PolygonProvider()

            # Simulate 5 requests (should pass)
            for i in range(5):
                await provider._rate_limit()
                assert provider.minute_request_count == i + 1

            # 6th request should require waiting
            with patch("asyncio.sleep") as mock_sleep:
                await provider._rate_limit()
                # Should have called sleep because we hit the limit
                assert mock_sleep.called

    @pytest.mark.asyncio
    async def test_batch_get_ticker_info_without_key(self):
        """Test batch operation returns empty dicts when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = PolygonProvider()
            tickers = ["AAPL", "MSFT", "GOOGL"]
            result = await provider.batch_get_ticker_info(tickers)

            assert len(result) == 3
            for ticker in tickers:
                assert result[ticker] == {}

    def test_cache_info(self):
        """Test get_cache_info returns rate limit statistics."""
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            provider = PolygonProvider()
            provider.minute_request_count = 3

            info = provider.get_cache_info()

            assert info["provider"] == "Polygon.io"
            assert info["api_key_configured"] is True
            assert info["minute_requests_used"] == 3
            assert info["minute_requests_remaining"] == 2

    @pytest.mark.asyncio
    async def test_historical_data_returns_empty(self):
        """Test historical_data returns empty DataFrame (not implemented)."""
        provider = PolygonProvider()
        result = await provider.get_historical_data("AAPL")
        assert result.empty

    @pytest.mark.asyncio
    async def test_earnings_dates_returns_none(self):
        """Test earnings_dates returns None tuple (not implemented)."""
        provider = PolygonProvider()
        result = await provider.get_earnings_dates("AAPL")
        assert result == (None, None)


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_registry_initialization(self):
        """Test registry initializes with all providers."""
        registry = ProviderRegistry()

        assert len(registry.providers) == 4
        provider_names = [name for name, _ in registry.providers]
        assert "yfinance" in provider_names
        assert "yahooquery" in provider_names
        assert "alpha_vantage" in provider_names
        assert "polygon" in provider_names

        # Check stats initialized
        for name in provider_names:
            assert name in registry.stats
            assert registry.stats[name]["success"] == 0
            assert registry.stats[name]["failure"] == 0

    @pytest.mark.asyncio
    async def test_registry_tries_providers_in_order(self):
        """Test registry tries providers in correct order."""
        registry = ProviderRegistry()

        # Mock all providers to fail except third one
        for i, (name, provider) in enumerate(registry.providers):
            mock = AsyncMock()
            if i == 2:  # Third provider (alpha_vantage)
                # Return valid data (needs more than just symbol/name)
                mock.return_value = {
                    "symbol": "AAPL",
                    "name": "Apple Inc.",
                    "current_price": 150.0
                }
            else:
                mock.return_value = {}
            provider.get_ticker_info = mock

        result = await registry.get_stock_data("AAPL")

        # Should have gotten data from third provider
        assert result is not None
        assert result["symbol"] == "AAPL"

        # Check that first 3 providers were tried
        for i in range(3):
            _, provider = registry.providers[i]
            provider.get_ticker_info.assert_called_once_with("AAPL", False)

        # Fourth provider should not have been tried
        _, fourth_provider = registry.providers[3]
        fourth_provider.get_ticker_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_registry_returns_none_when_all_fail(self):
        """Test registry returns None when all providers fail."""
        registry = ProviderRegistry()

        # Mock all providers to return empty
        for name, provider in registry.providers:
            provider.get_ticker_info = AsyncMock(return_value={})

        result = await registry.get_stock_data("INVALID")

        assert result is None

        # All providers should have been tried
        for _, provider in registry.providers:
            provider.get_ticker_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_registry_tracks_statistics(self):
        """Test registry tracks success/failure statistics."""
        registry = ProviderRegistry()

        # Mock first provider to succeed (needs valid data)
        first_provider = registry.providers[0][1]
        first_provider.get_ticker_info = AsyncMock(
            return_value={
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "current_price": 150.0
            }
        )

        # Mock others to fail
        for _, provider in registry.providers[1:]:
            provider.get_ticker_info = AsyncMock(return_value={})

        await registry.get_stock_data("AAPL")

        # Check stats
        stats = registry.get_stats()
        assert stats["yfinance"]["success"] == 1
        assert stats["yfinance"]["failure"] == 0

    def test_is_valid_data(self):
        """Test _is_valid_data correctly validates data."""
        registry = ProviderRegistry()

        # Valid data
        assert registry._is_valid_data({"symbol": "AAPL", "name": "Apple", "current_price": 150.0})

        # Invalid: empty
        assert not registry._is_valid_data({})

        # Invalid: only symbol
        assert not registry._is_valid_data({"symbol": "AAPL"})

        # Invalid: only name
        assert not registry._is_valid_data({"name": "Apple"})

        # Valid: symbol + one other field
        assert registry._is_valid_data({"symbol": "AAPL", "current_price": 150.0})

    def test_get_stats_summary(self):
        """Test get_stats_summary returns formatted string."""
        registry = ProviderRegistry()

        # Add some stats
        registry.stats["yfinance"]["success"] = 10
        registry.stats["yfinance"]["failure"] = 2

        summary = registry.get_stats_summary()

        assert "Provider Usage Statistics" in summary
        assert "yfinance" in summary
        assert "10" in summary
        assert "2" in summary

    def test_reset_stats(self):
        """Test reset_stats clears all statistics."""
        registry = ProviderRegistry()

        # Add some stats
        registry.stats["yfinance"]["success"] = 10
        registry.stats["alpha_vantage"]["failure"] = 5

        # Reset
        registry.reset_stats()

        # Verify all reset
        for name in registry.stats:
            assert registry.stats[name]["success"] == 0
            assert registry.stats[name]["failure"] == 0

    def test_clear_caches(self):
        """Test clear_caches calls clear_cache on all providers."""
        registry = ProviderRegistry()

        # Mock clear_cache for all providers
        for _, provider in registry.providers:
            provider.clear_cache = MagicMock()

        registry.clear_caches()

        # Verify all were called
        for _, provider in registry.providers:
            provider.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_convenience_function_get_stock_data(self):
        """Test convenience function uses global registry."""
        with patch("yahoofinance.api.providers.provider_registry.get_provider_registry") as mock_get:
            mock_registry = MagicMock()
            mock_registry.get_stock_data = AsyncMock(return_value={"symbol": "AAPL"})
            mock_get.return_value = mock_registry

            result = await get_stock_data("AAPL")

            assert result["symbol"] == "AAPL"
            mock_registry.get_stock_data.assert_called_once_with("AAPL", False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
