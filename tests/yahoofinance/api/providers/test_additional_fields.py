#!/usr/bin/env python3
"""
Test the handling of additional fields like short interest and PEG ratio.
"""
import pytest
from unittest.mock import patch, MagicMock

from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider
from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider


def test_yahoo_finance_short_interest():
    """Test short interest handling in YahooFinanceProvider."""
    # Create the provider
    provider = YahooFinanceProvider()
    
    # Mock the _fetch_ticker_info method to return test data
    ticker_info = {
        "symbol": "AAPL",
        "shortPercentOfFloat": 0.0075, # 0.75%
    }
    
    with patch.object(provider, '_fetch_ticker_info', return_value=ticker_info):
        # Call get_ticker_info
        result = provider.get_ticker_info("AAPL")
        
        # Verify short interest is included and correctly formatted
        assert result["short_percent"] == 0.75
        assert isinstance(result["short_percent"], float)


def test_yahoo_finance_peg_ratio():
    """Test PEG ratio handling in YahooFinanceProvider."""
    # Create the provider
    provider = YahooFinanceProvider()
    
    # Mock the _fetch_ticker_info method to return test data
    ticker_info = {
        "symbol": "AAPL",
        "pegRatio": 1.82,
    }
    
    with patch.object(provider, '_fetch_ticker_info', return_value=ticker_info):
        # Call get_ticker_info
        result = provider.get_ticker_info("AAPL")
        
        # Verify PEG ratio is included
        assert result["peg_ratio"] == 1.82
        assert isinstance(result["peg_ratio"], float)


@pytest.mark.asyncio
async def test_async_hybrid_provider_supplementing_missing_fields():
    """Test AsyncHybridProvider supplementing missing fields."""
    # Enable yahooquery supplementation
    with patch('yahoofinance.api.providers.async_hybrid_provider.PROVIDER_CONFIG', 
              {"ENABLE_YAHOOQUERY": True}):
        
        # Create mock providers
        yf_provider = MagicMock()
        yq_provider = MagicMock()
        
        # Set up return values for YF - missing PEG ratio
        yf_provider.get_ticker_info.return_value = {
            "symbol": "AAPL",
            "ticker": "AAPL",
            "short_percent": 0.75,
            # Missing peg_ratio
        }
        
        # Set up return values for YQ - has PEG ratio
        yq_provider.get_ticker_info.return_value = {
            "symbol": "AAPL",
            "ticker": "AAPL",
            "peg_ratio": 1.82,
        }
        
        # Create the hybrid provider with mocked providers
        provider = AsyncHybridProvider()
        provider.yf_provider = yf_provider
        provider.yq_provider = yq_provider
        provider.enable_yahooquery = True
        
        # Test with ticker missing PEG ratio
        result = await provider.get_ticker_info("AAPL")
        
        # Verify fields are properly combined
        assert result["short_percent"] == 0.75  # From YF
        assert result["peg_ratio"] == 1.82      # From YQ
        
        # Test batch processing
        yf_provider.batch_get_ticker_info.return_value = {
            "AAPL": {"symbol": "AAPL", "ticker": "AAPL", "short_percent": 0.75},
            "MSFT": {"symbol": "MSFT", "ticker": "MSFT", "short_percent": 0.65},
        }
        
        # Disable yahooquery for simplicity in batch test
        provider.enable_yahooquery = False
        
        # Test batch processing
        results = await provider.batch_get_ticker_info(["AAPL", "MSFT"])
        
        # Verify short interest data is present in batch results
        assert results["AAPL"]["short_percent"] == 0.75
        assert results["MSFT"]["short_percent"] == 0.65


def test_missing_fields_handling():
    """Test handling when fields are not available."""
    # Create the provider
    provider = YahooFinanceProvider()
    
    # Mock the _fetch_ticker_info method to return test data with missing fields
    ticker_info = {
        "symbol": "UNKNOWN",
        "shortName": "Unknown Stock",
        "regularMarketPrice": 10.0,
        # Missing short_percent and peg_ratio
    }
    
    with patch.object(provider, '_fetch_ticker_info', return_value=ticker_info):
        # Call get_ticker_info
        result = provider.get_ticker_info("UNKNOWN")
        
        # Verify default values for missing fields
        assert "short_percent" not in result
        assert "peg_ratio" not in result