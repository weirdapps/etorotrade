"""
Unit tests for API providers.

This module tests the implementation of API providers to ensure they correctly 
implement the FinanceDataProvider interface and return data in the expected format.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from yahoofinance.api.providers.base import FinanceDataProvider
from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider
from yahoofinance.core.errors import YFinanceError
from yahoofinance.api import get_provider


class TestFinanceProviders:
    """Tests for the provider base classes and factory function."""

    def test_get_provider_returns_yahoo_finance_provider(self):
        """Test that get_provider returns a YahooFinanceProvider by default."""
        provider = get_provider()
        assert isinstance(provider, YahooFinanceProvider)
        
    def test_get_provider_with_async_true(self):
        """Test that get_provider with async_mode=True returns an AsyncYahooFinanceProvider."""
        provider = get_provider(async_mode=True)
        # This is checking the class name since we don't want to import AsyncYahooFinanceProvider
        # directly to avoid potential circular imports
        assert provider.__class__.__name__ == "AsyncYahooFinanceProvider"


@pytest.fixture
def mock_client():
    """Create a mock YFinanceClient."""
    client = Mock()
    
    # Mock ticker info response
    mock_stock_data = Mock()
    mock_stock_data.current_price = 150.0
    mock_stock_data.target_price = 180.0
    mock_stock_data.company_name = "Apple Inc."
    mock_stock_data.price_change_percentage = 2.5
    mock_stock_data.market_cap = 2500000000000
    
    client.get_ticker_info.return_value = mock_stock_data
    
    # Mock price data response
    client.get_price_data.return_value = {
        "price": 150.0,
        "previous_close": 147.5,
        "change": 2.5,
        "change_percent": 1.69,
        "volume": 75000000,
        "avg_volume": 80000000
    }
    
    # Mock historical data response
    hist_data = pd.DataFrame({
        "Date": [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
        "Open": [145.0, 147.0, 148.0],
        "High": [147.0, 149.0, 152.0],
        "Low": [144.0, 146.0, 147.0],
        "Close": [146.0, 148.0, 150.0],
        "Volume": [70000000, 75000000, 80000000]
    })
    client.get_historical_data.return_value = hist_data
    
    # Mock analyst ratings response
    client.get_analyst_ratings.return_value = {
        "buy": 25,
        "hold": 7,
        "sell": 3,
        "target_price": 180.0,
        "total_analysts": 35
    }
    
    # Mock earnings data response
    client.get_earnings_data.return_value = {
        "earnings_date": "2023-04-15",
        "eps_estimate": 1.45,
        "eps_actual": 1.52,
        "surprise_percent": 4.8
    }
    
    # Mock search tickers response
    client.search_tickers.return_value = [
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
        {"symbol": "AAPLE", "name": "Apple Electric", "exchange": "NYSE"}
    ]
    
    return client


class TestYahooFinanceProvider:
    """Tests for the YahooFinanceProvider implementation."""
    
    def test_instantiation(self, mock_client):
        """Test creating a YahooFinanceProvider instance."""
        with patch('yahoofinance.api.providers.yahoo_finance.YFinanceClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            provider = YahooFinanceProvider()
            assert provider.client is not None
    
    def test_get_ticker_info(self, mock_client):
        """Test get_ticker_info method returns correctly formatted data."""
        with patch('yahoofinance.api.providers.yahoo_finance.YFinanceClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Prepare mock stock data with proper attributes
            mock_stock = Mock()
            mock_stock.ticker_object = Mock()
            mock_stock.ticker_object.ticker = "AAPL"
            mock_stock.name = "Apple Inc."
            mock_stock.sector = "Technology"
            mock_stock.market_cap = 2500000000000
            mock_stock.beta = 1.2
            mock_stock.pe_trailing = 25.0
            mock_stock.pe_forward = 20.0
            mock_stock.dividend_yield = 0.6
            mock_stock.current_price = 150.0
            mock_stock.analyst_count = 35
            mock_stock.peg_ratio = 1.5
            mock_stock.short_float_pct = 0.5
            mock_stock.last_earnings = "2023-01-15"
            mock_stock.previous_earnings = "2022-10-15"
            
            mock_client.get_ticker_info.return_value = mock_stock
            
            provider = YahooFinanceProvider()
            result = provider.get_ticker_info("AAPL")
            
            assert isinstance(result, dict)
            assert "ticker" in result
            assert "price" in result
            assert "name" in result
            assert result["price"] == 150.0
            assert result["name"] == "Apple Inc."
    
    def test_get_price_data(self, mock_client):
        """Test get_price_data method returns correctly formatted data."""
        with patch('yahoofinance.api.providers.yahoo_finance.YFinanceClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Mock the PricingAnalyzer
            metrics = {
                'current_price': 150.0,
                'target_price': 180.0,
                'upside_potential': 20.0,
                'price_change': 3.5,
                'price_change_percentage': 2.3
            }
            
            with patch('yahoofinance.pricing.PricingAnalyzer') as mock_pricing:
                mock_pricing_instance = Mock()
                mock_pricing_instance.calculate_price_metrics.return_value = metrics
                mock_pricing.return_value = mock_pricing_instance
                
                provider = YahooFinanceProvider()
                result = provider.get_price_data("AAPL")
                
                assert isinstance(result, dict)
                assert "current_price" in result
                assert "upside_potential" in result
                assert result["current_price"] == 150.0
    
    def test_get_historical_data(self, mock_client):
        """Test get_historical_data method returns correctly formatted data."""
        with patch('yahoofinance.api.providers.yahoo_finance.YFinanceClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            provider = YahooFinanceProvider()
            result = provider.get_historical_data("AAPL", "1y", "1d")
            
            assert isinstance(result, pd.DataFrame)
            assert "Date" in result.columns
            assert "Close" in result.columns
            assert len(result) == 3
    
    def test_get_analyst_ratings(self, mock_client):
        """Test get_analyst_ratings method returns correctly formatted data."""
        with patch('yahoofinance.api.providers.yahoo_finance.YFinanceClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Mock analyst data
            ratings = {
                'positive_percentage': 85.0,
                'total_ratings': 35,
                'ratings_type': 'buy/hold/sell',
                'recommendations': {'buy': 30, 'hold': 3, 'sell': 2}
            }
            
            with patch('yahoofinance.analyst.AnalystData') as mock_analyst:
                mock_analyst_instance = Mock()
                mock_analyst_instance.get_ratings_summary.return_value = ratings
                mock_analyst.return_value = mock_analyst_instance
                
                provider = YahooFinanceProvider()
                result = provider.get_analyst_ratings("AAPL")
                
                assert isinstance(result, dict)
                assert "positive_percentage" in result
                assert "total_ratings" in result
                assert result["positive_percentage"] == 85.0
    
    def test_get_earnings_data(self, mock_client):
        """Test get_earnings_data method returns correctly formatted data."""
        with patch('yahoofinance.api.providers.yahoo_finance.YFinanceClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Mock stock data for basic earnings info
            mock_stock = Mock()
            mock_stock.last_earnings = "2023-04-15"
            mock_stock.previous_earnings = "2023-01-15"
            mock_client.get_ticker_info.return_value = mock_stock
            
            # Skip the EarningsAnalyzer mocking since it doesn't exist
            
            provider = YahooFinanceProvider()
            result = provider.get_earnings_data("AAPL")
            
            assert isinstance(result, dict)
            assert "last_earnings" in result
            assert result["last_earnings"] == "2023-04-15"
    
    def test_search_tickers(self, mock_client):
        """Test search_tickers method returns correctly formatted data."""
        with patch('yahoofinance.api.providers.yahoo_finance.YFinanceClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            provider = YahooFinanceProvider()
            result = provider.search_tickers("Apple")
            
            assert isinstance(result, list)
            assert len(result) == 2
            assert "symbol" in result[0]
            assert "name" in result[0]
            assert result[0]["symbol"] == "AAPL"

    def test_error_handling(self, mock_client):
        """Test that provider properly handles and propagates errors."""
        with patch('yahoofinance.api.providers.yahoo_finance.YFinanceClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            # Make the client raise an exception
            mock_client.get_ticker_info.side_effect = ValueError("API Error")
            
            provider = YahooFinanceProvider()
            
            # The provider should transform the error to YFinanceError
            with pytest.raises(YFinanceError):
                provider.get_ticker_info("AAPL")