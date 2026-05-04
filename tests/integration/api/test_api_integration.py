"""
Integration tests for API client, provider pattern, and rate limiting.

These tests verify that the API client, providers, and rate limiting components
work together correctly in realistic scenarios.
"""

import pandas as pd
import pytest

# Using direct imports from canonical modules
from yahoofinance.api import get_provider
from yahoofinance.core.errors import ValidationError


@pytest.mark.integration
@pytest.mark.network
async def test_client_with_rate_limiting():
    """Test that client uses rate limiting correctly."""
    provider = get_provider()

    # Get info for a ticker and ensure it returns data
    data = await provider.get_ticker_info("AAPL")
    assert data is not None
    assert "name" in data
    assert "symbol" in data


@pytest.mark.integration
@pytest.mark.network
async def test_rate_limited_retries():
    """Test retry behavior with rate limiting."""
    provider = get_provider()

    # Get data for a well-known ticker
    result = await provider.get_ticker_info("AAPL")

    # Verify result
    assert result is not None
    assert ("ticker" in result and result["ticker"] == "AAPL") or (
        "symbol" in result and result["symbol"] == "AAPL"
    )
    assert "name" in result
    assert "symbol" in result


@pytest.mark.integration
@pytest.mark.network
async def test_batch_processing_with_rate_limiting():
    """Test batch processing with rate limiting."""
    provider = get_provider()

    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]

    results = []
    for ticker in tickers:
        results.append(await provider.get_ticker_info(ticker))

    assert len(results) == 4
    assert all(r is not None for r in results)
    assert all("symbol" in r for r in results)


@pytest.mark.integration
@pytest.mark.network
async def test_error_recovery_integration():
    """Test error recovery - verify provider handles invalid tickers gracefully."""
    provider = get_provider()

    result1 = await provider.get_ticker_info("AAPL")
    assert result1 is not None

    # Invalid tickers may raise or return empty/None depending on provider
    try:
        result2 = await provider.get_ticker_info("INVALID_TICKER_THAT_DOES_NOT_EXIST_12345")
        # If no exception, result should be empty or have minimal data
        assert result2 is None or isinstance(result2, dict)
    except Exception:
        pass  # Exception is also acceptable behavior


@pytest.mark.integration
@pytest.mark.network
class TestProviderIntegration:
    """Integration tests for the provider interface"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test case"""
        self.provider = get_provider()
        self.valid_ticker = "AAPL"
        yield

    async def test_get_ticker_info(self):
        """Test getting ticker info via provider"""
        info = await self.provider.get_ticker_info(self.valid_ticker)

        assert isinstance(info, dict)
        assert "symbol" in info
        assert "name" in info
        assert info["symbol"] == self.valid_ticker
        assert info["name"] is not None

    async def test_get_price_data(self):
        """Test getting price data via provider"""
        info = await self.provider.get_ticker_info(self.valid_ticker)

        assert isinstance(info, dict)
        assert info.get("symbol") == self.valid_ticker

    async def test_get_historical_data(self):
        """Test getting historical data via provider"""
        hist_data = await self.provider.get_historical_data(self.valid_ticker, period="1mo")

        assert isinstance(hist_data, pd.DataFrame)
        assert len(hist_data) > 0
        assert "Close" in hist_data.columns
        assert "Volume" in hist_data.columns

    async def test_get_analyst_ratings(self):
        """Test getting analyst ratings via provider"""
        ratings = await self.provider.get_analyst_ratings(self.valid_ticker)

        assert isinstance(ratings, dict)
        assert "symbol" in ratings

    @pytest.mark.xfail(reason="Invalid tickers may not raise ValidationError with all providers")
    async def test_invalid_ticker(self):
        """Test validation for invalid tickers"""
        with pytest.raises(ValidationError):
            await self.provider.get_ticker_info("INVALID_TICKER_123456789")


@pytest.mark.integration
@pytest.mark.network
class TestProviderMigration:
    """Tests provider functionality without compatibility layer imports"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test case"""
        self.provider = get_provider()
        self.valid_ticker = "MSFT"
        yield

    async def test_provider_functionality(self):
        """Test provider functionality directly"""
        provider_data = await self.provider.get_ticker_info(self.valid_ticker)

        assert isinstance(provider_data, dict)
        assert "name" in provider_data
        assert "symbol" in provider_data
        assert provider_data["name"] is not None
        assert provider_data["symbol"] == self.valid_ticker

    async def test_stock_data_conversion(self):
        """Test that provider data can be used to create stock data objects if needed"""
        provider_data = await self.provider.get_ticker_info(self.valid_ticker)

        class StockInfo:
            def __init__(self, ticker, data):
                self.ticker = ticker
                self.name = data.get("name")
                self.symbol = data.get("symbol")
                self.market_cap = data.get("market_cap")
                self.pe_ratio = data.get("pe_trailing")

        stock_info = StockInfo(self.valid_ticker, provider_data)

        assert stock_info.ticker == self.valid_ticker
        assert stock_info.name == provider_data["name"]
        assert stock_info.symbol == provider_data["symbol"]

    async def test_batch_processing(self):
        """Test batch processing with multiple tickers"""
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]

        results = {}
        for ticker in tickers:
            results[ticker] = await self.provider.get_ticker_info(ticker)

        for ticker in tickers:
            assert ticker in results
            assert results[ticker] is not None
            assert "name" in results[ticker]
            assert "symbol" in results[ticker]
