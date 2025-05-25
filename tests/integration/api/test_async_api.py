"""Integration tests for async provider pattern"""

import pandas as pd
import pytest
import pytest_asyncio

from yahoofinance.api import get_provider


# Skip these tests by default as they require a running event loop
pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.network]


class TestAsyncProviderIntegration:
    """Integration tests for the async provider interface"""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Set up test case"""
        self.provider = get_provider(async_mode=True)
        self.valid_ticker = "AAPL"
        yield

    async def test_get_ticker_info(self):
        """Test getting ticker info via async provider"""
        info = await self.provider.get_ticker_info(self.valid_ticker)

        # Verify data structure
        assert isinstance(info, dict)
        assert "symbol" in info
        assert "name" in info
        assert "sector" in info

        # Verify actual data
        assert info["name"] is not None

    async def test_get_price_data(self):
        """Test getting price data via async provider"""
        price_data = await self.provider.get_price_data(self.valid_ticker)

        # Verify data structure
        assert isinstance(price_data, dict)
        assert "current_price" in price_data

        # Verify actual data
        assert price_data.get("current_price") is not None

    async def test_get_historical_data(self):
        """Test getting historical data via async provider"""
        hist_data = await self.provider.get_historical_data(self.valid_ticker, period="1mo")

        # Verify data structure
        assert isinstance(hist_data, pd.DataFrame)
        assert len(hist_data) > 0

        # Verify columns
        assert "Close" in hist_data.columns
        assert "Volume" in hist_data.columns

    async def test_get_analyst_ratings(self):
        """Test getting analyst ratings via async provider"""
        ratings = await self.provider.get_analyst_ratings(self.valid_ticker)

        # Verify data structure
        assert isinstance(ratings, dict)
        assert "symbol" in ratings

    async def test_batch_processing(self):
        """Test batch ticker processing with async provider"""
        tickers = ["AAPL", "MSFT", "GOOG"]
        batch_results = await self.provider.batch_get_ticker_info(tickers)

        # Verify batch results
        assert isinstance(batch_results, dict)
        assert len(batch_results) == len(tickers)

        # Check each ticker result
        for ticker in tickers:
            assert ticker in batch_results
            ticker_data = batch_results[ticker]
            if ticker_data and not isinstance(ticker_data, dict):
                # Skip if ticker_data is None or has error
                continue
            if ticker_data and not ticker_data.get("error"):  # Skip entries with errors
                assert isinstance(ticker_data, dict)
                assert "symbol" in ticker_data  # Changed from 'name' to 'symbol'
