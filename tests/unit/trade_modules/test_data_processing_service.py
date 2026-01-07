"""
Tests for trade_modules/data_processing_service.py

This module tests the DataProcessingService class for batch ticker processing.
"""

import pytest
import pandas as pd
import logging
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from trade_modules.data_processing_service import DataProcessingService


@pytest.fixture
def logger():
    """Create a mock logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_provider():
    """Create a mock provider with async get_ticker_info."""
    provider = MagicMock()
    provider.get_ticker_info = AsyncMock()
    return provider


@pytest.fixture
def data_service(mock_provider, logger):
    """Create a DataProcessingService instance."""
    return DataProcessingService(mock_provider, logger)


class TestDataProcessingServiceInit:
    """Tests for DataProcessingService initialization."""

    def test_init_with_provider_and_logger(self, mock_provider, logger):
        """Test DataProcessingService initializes correctly."""
        service = DataProcessingService(mock_provider, logger)
        assert service.provider is mock_provider
        assert service.logger is logger


class TestProcessSingleTicker:
    """Tests for _process_single_ticker method."""

    @pytest.mark.asyncio
    async def test_process_single_ticker_success(self, data_service, mock_provider):
        """Test successful single ticker processing."""
        mock_provider.get_ticker_info.return_value = {
            "price": 175.0,
            "market_cap": 3000000000000,
            "volume": 50000000,
            "pe_trailing": 28.5,
            "pe_forward": 25.0,
            "dividend_yield": 0.5,
            "beta": 1.2,
            "target_price": 200.0,
            "upside": 14.3,
            "buy_percentage": 80.0,
            "analyst_count": 15,
            "total_ratings": 20,
        }

        result = await data_service._process_single_ticker("AAPL")

        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["price"] == pytest.approx(175.0)
        assert result["market_cap"] == 3000000000000

    @pytest.mark.asyncio
    async def test_process_single_ticker_no_data(self, data_service, mock_provider):
        """Test single ticker processing when no data returned."""
        mock_provider.get_ticker_info.return_value = None

        result = await data_service._process_single_ticker("INVALID")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_ticker_empty_dict(self, data_service, mock_provider):
        """Test single ticker processing with empty dict returns None."""
        mock_provider.get_ticker_info.return_value = {}

        result = await data_service._process_single_ticker("AAPL")

        # Empty dict is falsy, so returns None (same as no data)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_ticker_normalizes_ticker(self, data_service, mock_provider):
        """Test that ticker is normalized."""
        mock_provider.get_ticker_info.return_value = {
            "price": 175.0,
        }

        result = await data_service._process_single_ticker("  aapl  ")

        assert result is not None
        assert result["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_process_single_ticker_handles_key_error(self, data_service, mock_provider):
        """Test handling of KeyError during processing."""
        mock_provider.get_ticker_info.side_effect = KeyError("missing_key")

        result = await data_service._process_single_ticker("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_ticker_handles_value_error(self, data_service, mock_provider):
        """Test handling of ValueError during processing."""
        mock_provider.get_ticker_info.side_effect = ValueError("invalid value")

        result = await data_service._process_single_ticker("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_ticker_handles_type_error(self, data_service, mock_provider):
        """Test handling of TypeError during processing."""
        mock_provider.get_ticker_info.side_effect = TypeError("type error")

        result = await data_service._process_single_ticker("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_ticker_handles_connection_error(self, data_service, mock_provider):
        """Test handling of ConnectionError during processing."""
        mock_provider.get_ticker_info.side_effect = ConnectionError("connection failed")

        result = await data_service._process_single_ticker("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_ticker_handles_timeout_error(self, data_service, mock_provider):
        """Test handling of TimeoutError during processing."""
        mock_provider.get_ticker_info.side_effect = TimeoutError("timeout")

        result = await data_service._process_single_ticker("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_ticker_handles_generic_exception(self, data_service, mock_provider):
        """Test handling of generic exception during processing."""
        mock_provider.get_ticker_info.side_effect = RuntimeError("unexpected error")

        result = await data_service._process_single_ticker("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_single_ticker_uses_current_price_fallback(self, data_service, mock_provider):
        """Test that current_price is used as fallback for price."""
        mock_provider.get_ticker_info.return_value = {
            "current_price": 175.0,  # No 'price' key, has 'current_price'
        }

        result = await data_service._process_single_ticker("AAPL")

        assert result is not None
        assert result["price"] == pytest.approx(175.0)


class TestProcessTickerBatch:
    """Tests for process_ticker_batch method."""

    @pytest.mark.asyncio
    async def test_process_ticker_batch_success(self, data_service, mock_provider):
        """Test successful batch processing."""
        mock_provider.get_ticker_info.side_effect = [
            {"price": 175.0, "market_cap": 3e12},
            {"price": 380.0, "market_cap": 2.5e12},
        ]

        async def mock_process_batch(items, processor, **kwargs):
            results = {}
            for item in items:
                results[item] = await processor(item)
            return results

        with patch('yahoofinance.utils.async_utils.enhanced.process_batch_async', side_effect=mock_process_batch):
            result = await data_service.process_ticker_batch(["AAPL", "MSFT"])

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "AAPL" in result.index
            assert "MSFT" in result.index

    @pytest.mark.asyncio
    async def test_process_ticker_batch_empty_list(self, data_service):
        """Test batch processing with empty list."""
        async def mock_process_batch(items, processor, **kwargs):
            return {}

        with patch('yahoofinance.utils.async_utils.enhanced.process_batch_async', side_effect=mock_process_batch):
            result = await data_service.process_ticker_batch([])

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_process_ticker_batch_filters_none_results(self, data_service, mock_provider):
        """Test that None results are filtered out."""
        # Set up mock to return data for some tickers, None for others
        async def mock_get_info(ticker):
            if ticker.upper() == "INVALID":
                return None
            return {"price": 175.0, "market_cap": 3e12}

        mock_provider.get_ticker_info = mock_get_info

        async def mock_process_batch(items, processor, **kwargs):
            results = {}
            for item in items:
                results[item] = await processor(item)
            return results

        with patch('yahoofinance.utils.async_utils.enhanced.process_batch_async', side_effect=mock_process_batch):
            result = await data_service.process_ticker_batch(["AAPL", "INVALID", "MSFT"])

            # INVALID should be filtered out (returns None)
            assert len(result) == 2
            assert "INVALID" not in result.index

    @pytest.mark.asyncio
    async def test_process_ticker_batch_custom_batch_size(self, data_service, mock_provider):
        """Test batch processing with custom batch size."""
        mock_provider.get_ticker_info.return_value = {"price": 175.0}

        captured_kwargs = {}

        async def mock_process_batch(items, processor, **kwargs):
            captured_kwargs.update(kwargs)
            results = {}
            for item in items:
                results[item] = await processor(item)
            return results

        with patch('yahoofinance.utils.async_utils.enhanced.process_batch_async', side_effect=mock_process_batch):
            result = await data_service.process_ticker_batch(["AAPL"], batch_size=10)

            assert captured_kwargs["batch_size"] == 10


class TestBackwardCompatibility:
    """Tests for backward compatibility method."""

    @pytest.mark.asyncio
    async def test_process_batch_alias(self, data_service):
        """Test that _process_batch is an alias for process_ticker_batch."""
        with patch.object(data_service, 'process_ticker_batch') as mock_method:
            mock_method.return_value = pd.DataFrame()

            await data_service._process_batch(["AAPL", "MSFT"])

            mock_method.assert_called_once_with(["AAPL", "MSFT"])

    @pytest.mark.asyncio
    async def test_process_batch_ignores_extra_args(self, data_service):
        """Test that _process_batch ignores extra arguments."""
        with patch.object(data_service, 'process_ticker_batch') as mock_method:
            mock_method.return_value = pd.DataFrame()

            # Should not raise error with extra args
            await data_service._process_batch(
                ["AAPL"], "extra_arg", extra_kwarg="value"
            )

            mock_method.assert_called_once()


class TestDataProcessingServiceIntegration:
    """Integration tests for DataProcessingService."""

    @pytest.mark.asyncio
    async def test_process_multiple_tickers_integration(self, data_service, mock_provider):
        """Test processing multiple tickers end-to-end."""
        # Set up provider to return different data for each ticker
        async def mock_get_info(ticker):
            # Handle normalized tickers (GOOGL -> GOOG)
            data = {
                "AAPL": {"price": 175.0, "market_cap": 3e12, "upside": 14.3},
                "MSFT": {"price": 380.0, "market_cap": 2.5e12, "upside": 10.5},
                "GOOG": {"price": 140.0, "market_cap": 1.8e12, "upside": 12.0},
            }
            return data.get(ticker.upper(), None)

        mock_provider.get_ticker_info = mock_get_info

        async def mock_process_batch(items, processor, **kwargs):
            results = {}
            for item in items:
                result = await processor(item)
                results[item] = result
            return results

        with patch('yahoofinance.utils.async_utils.enhanced.process_batch_async', side_effect=mock_process_batch):
            result = await data_service.process_ticker_batch(["AAPL", "MSFT", "GOOGL"])

            assert len(result) == 3
            assert result.loc["AAPL", "price"] == pytest.approx(175.0)
            assert result.loc["MSFT", "price"] == pytest.approx(380.0)
            # GOOGL normalizes to GOOG
            assert "GOOG" in result.index
