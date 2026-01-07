#!/usr/bin/env python3
"""
Unit tests for AsyncHybridProvider with comprehensive mocking.

Tests cover:
- Mock yfinance responses
- Fallback to yahooquery
- Error handling paths
- Batch processing
- Data supplementation logic
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from yahoofinance.core.errors import APIError, NetworkError, ValidationError, YFinanceError


class TestAsyncHybridProviderBasic:
    """Basic tests for AsyncHybridProvider initialization and configuration."""

    @pytest.fixture
    def mock_yf_provider(self):
        """Create a mock yfinance provider."""
        mock = AsyncMock()
        mock.max_concurrency = 15
        return mock

    @pytest.fixture
    def mock_yq_provider(self):
        """Create a mock yahooquery provider."""
        return AsyncMock()

    @pytest.fixture
    def mock_config(self):
        """Mock provider configuration."""
        return {
            "ENABLE_YAHOOQUERY": False,
        }

    def test_provider_initialization(self, mock_config):
        """Test that provider initializes with correct configuration."""
        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config):

            mock_yf.return_value.max_concurrency = 15

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            assert provider.yf_provider is not None
            assert provider.yq_provider is not None
            assert provider.max_concurrency == 15

    def test_provider_yahooquery_disabled(self, mock_config):
        """Test that yahooquery is disabled by default configuration."""
        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config):

            mock_yf.return_value.max_concurrency = 15

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            assert provider.enable_yahooquery is False

    def test_provider_yahooquery_enabled(self):
        """Test that yahooquery can be enabled via configuration."""
        enabled_config = {"ENABLE_YAHOOQUERY": True}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', enabled_config):

            mock_yf.return_value.max_concurrency = 15

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            assert provider.enable_yahooquery is True


class TestAsyncHybridProviderTickerInfo:
    """Tests for get_ticker_info method."""

    @pytest.fixture
    def sample_yf_response(self):
        """Sample yfinance response data."""
        return {
            "symbol": "AAPL",
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "company": "APPLE INC.",
            "price": 150.0,
            "market_cap": 2500000000000,
            "pe_forward": 22.3,
            "pe_trailing": 25.5,
            "peg_ratio": 1.8,
            "beta": 1.2,
            "upside": 15.0,
            "buy_percentage": 80.0,
            "analyst_count": 35,
            "total_ratings": 35,
            "A": "A",
        }

    @pytest.fixture
    def sample_yq_response(self):
        """Sample yahooquery response data for supplementation."""
        return {
            "symbol": "AAPL",
            "peg_ratio": 1.9,
            "pe_forward": 23.0,
            "beta": 1.25,
        }

    @pytest.mark.asyncio
    async def test_get_ticker_info_yfinance_only(self, sample_yf_response):
        """Test get_ticker_info with yfinance data only (yahooquery disabled)."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config:

            # Setup mocks
            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_instance.get_ticker_info.return_value = sample_yf_response
            mock_yf_class.return_value = mock_yf_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.return_value = "AAPL"
            mock_get_config.return_value = mock_config_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            result = await provider.get_ticker_info("AAPL")

            assert result["ticker"] == "AAPL"
            assert result["price"] == pytest.approx(150.0)
            assert result["data_source"] == "yfinance"
            mock_yf_instance.get_ticker_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ticker_info_with_supplementation(self, sample_yf_response, sample_yq_response):
        """Test get_ticker_info with yahooquery supplementation for missing fields."""
        mock_config = {"ENABLE_YAHOOQUERY": True}

        # Remove peg_ratio to trigger supplementation
        yf_missing_peg = sample_yf_response.copy()
        yf_missing_peg["peg_ratio"] = None

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config:

            # Setup yfinance mock with missing peg_ratio
            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_instance.get_ticker_info.return_value = yf_missing_peg
            mock_yf_class.return_value = mock_yf_instance

            # Setup yahooquery mock
            mock_yq_instance = AsyncMock()
            mock_yq_instance.get_ticker_info.return_value = sample_yq_response
            mock_yq_class.return_value = mock_yq_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.return_value = "AAPL"
            mock_get_config.return_value = mock_config_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            result = await provider.get_ticker_info("AAPL")

            assert result["ticker"] == "AAPL"
            # peg_ratio should be supplemented from yahooquery
            assert result.get("peg_ratio") == pytest.approx(1.9)
            assert result["data_source"] == "hybrid (yf+yq)"

    @pytest.mark.asyncio
    async def test_get_ticker_info_yfinance_error_handling(self):
        """Test error handling when yfinance returns an error."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config:

            # Setup yfinance mock to return error
            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_instance.get_ticker_info.return_value = {"error": "API rate limit exceeded"}
            mock_yf_class.return_value = mock_yf_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.return_value = "INVALID"
            mock_get_config.return_value = mock_config_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            result = await provider.get_ticker_info("INVALID")

            # Should still return a result with error information
            assert "error" in result
            assert result["data_source"] == "none"

    @pytest.mark.asyncio
    async def test_get_ticker_info_yfinance_exception(self):
        """Test handling when yfinance raises an exception."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config:

            # Setup yfinance mock to raise exception
            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_instance.get_ticker_info.side_effect = YFinanceError("Network error")
            mock_yf_class.return_value = mock_yf_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.return_value = "AAPL"
            mock_get_config.return_value = mock_config_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            result = await provider.get_ticker_info("AAPL")

            # Should handle exception gracefully
            assert result["ticker"] == "AAPL"
            assert result["data_source"] == "none"


class TestAsyncHybridProviderBatchProcessing:
    """Tests for batch_get_ticker_info method."""

    @pytest.fixture
    def sample_batch_responses(self):
        """Sample batch response data."""
        return {
            "AAPL": {
                "symbol": "AAPL",
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "price": 150.0,
                "upside": 15.0,
                "buy_percentage": 80.0,
            },
            "MSFT": {
                "symbol": "MSFT",
                "ticker": "MSFT",
                "name": "Microsoft Corporation",
                "price": 300.0,
                "upside": 10.0,
                "buy_percentage": 75.0,
            },
            "GOOGL": {
                "symbol": "GOOGL",
                "ticker": "GOOGL",
                "name": "Alphabet Inc.",
                "price": 120.0,
                "upside": -5.0,
                "buy_percentage": 60.0,
            },
        }

    @pytest.mark.asyncio
    async def test_batch_get_ticker_info_success(self, sample_batch_responses):
        """Test successful batch processing of multiple tickers."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config, \
             patch('yahoofinance.api.providers.async_hybrid_provider.gather_with_concurrency') as mock_gather:

            # Setup mocks
            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_class.return_value = mock_yf_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.side_effect = lambda x: x
            mock_get_config.return_value = mock_config_instance

            # Mock gather to return batch results
            mock_gather.return_value = list(sample_batch_responses.values())

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            tickers = ["AAPL", "MSFT", "GOOGL"]
            result = await provider.batch_get_ticker_info(tickers)

            assert len(result) == 3
            mock_gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_get_ticker_info_empty_list(self):
        """Test batch processing with empty ticker list."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config):

            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_class.return_value = mock_yf_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            result = await provider.batch_get_ticker_info([])

            assert result == {}


class TestAsyncHybridProviderTickerMapping:
    """Tests for ticker mapping functionality."""

    def test_ticker_mapping_crypto(self):
        """Test that crypto tickers are mapped correctly."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config):

            mock_yf_instance = MagicMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_class.return_value = mock_yf_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            assert provider._ticker_mappings.get("BTC") == "BTC-USD"
            assert provider._ticker_mappings.get("ETH") == "ETH-USD"
            assert provider._ticker_mappings.get("SOL") == "SOL-USD"

    def test_ticker_mapping_commodities(self):
        """Test that commodity tickers are mapped correctly."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config):

            mock_yf_instance = MagicMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_class.return_value = mock_yf_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            assert provider._ticker_mappings.get("GOLD") == "GC=F"
            assert provider._ticker_mappings.get("OIL") == "CL=F"
            assert provider._ticker_mappings.get("SILVER") == "SI=F"


class TestAsyncHybridProviderErrorHandling:
    """Tests for error handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test handling of APIError exceptions."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config:

            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_instance.get_ticker_info.side_effect = APIError("Rate limit exceeded")
            mock_yf_class.return_value = mock_yf_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.return_value = "AAPL"
            mock_get_config.return_value = mock_config_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            # Should not raise - should handle gracefully
            result = await provider.get_ticker_info("AAPL")
            assert result is not None
            assert result["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of NetworkError exceptions."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config:

            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_instance.get_ticker_info.side_effect = NetworkError("Connection timeout")
            mock_yf_class.return_value = mock_yf_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.return_value = "AAPL"
            mock_get_config.return_value = mock_config_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            # Should not raise - should handle gracefully
            result = await provider.get_ticker_info("AAPL")
            assert result is not None

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test handling of ValidationError exceptions."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config:

            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_instance.get_ticker_info.side_effect = ValidationError("Invalid ticker format")
            mock_yf_class.return_value = mock_yf_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.return_value = "INVALID$"
            mock_get_config.return_value = mock_config_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            # Should not raise - should handle gracefully
            result = await provider.get_ticker_info("INVALID$")
            assert result is not None


class TestAsyncHybridProviderDataQuality:
    """Tests for data quality and integrity."""

    def test_exret_calculation_logic(self):
        """Test that EXRET calculation logic is correct."""
        # This tests the EXRET calculation formula directly
        # In the provider: EXRET = upside * buy_percentage / 100

        upside = 20.0
        buy_percentage = 80.0
        expected_exret = upside * buy_percentage / 100

        assert expected_exret == pytest.approx(16.0)

        # Test edge cases
        assert 0.0 * 50.0 / 100 == pytest.approx(0.0)  # Zero upside
        assert 10.0 * 0.0 / 100 == pytest.approx(0.0)  # Zero buy_percentage
        assert -5.0 * 60.0 / 100 == pytest.approx(-3.0)  # Negative upside

    @pytest.mark.asyncio
    async def test_essential_fields_preserved(self):
        """Test that essential fields are always present in results."""
        mock_config = {"ENABLE_YAHOOQUERY": False}

        with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooFinanceProvider') as mock_yf_class, \
             patch('yahoofinance.api.providers.async_hybrid_provider.AsyncYahooQueryProvider') as mock_yq_class, \
             patch('yahoofinance.core.config.PROVIDER_CONFIG', mock_config), \
             patch('trade_modules.config_manager.get_config') as mock_get_config:

            # Return minimal data
            mock_yf_instance = AsyncMock()
            mock_yf_instance.max_concurrency = 15
            mock_yf_instance.get_ticker_info.return_value = {"price": 100.0}
            mock_yf_class.return_value = mock_yf_instance

            mock_config_instance = MagicMock()
            mock_config_instance.get_data_fetch_ticker.return_value = "AAPL"
            mock_get_config.return_value = mock_config_instance

            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            provider = AsyncHybridProvider()

            result = await provider.get_ticker_info("AAPL")

            # Essential fields should always exist
            assert "symbol" in result
            assert "ticker" in result
            assert result["symbol"] == "AAPL"
            assert result["ticker"] == "AAPL"
