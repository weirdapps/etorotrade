"""
Unit tests for the EnhancedAsyncYahooFinanceProvider.

This module contains tests for the enhanced async provider implementation,
focusing on circuit breaker integration, error handling, and resilience patterns.
"""

import pytest
import json
import asyncio
import aiohttp
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd

from yahoofinance_v2.api.providers.enhanced_async_yahoo_finance import EnhancedAsyncYahooFinanceProvider
from yahoofinance_v2.utils.network.circuit_breaker import CircuitOpenError, CircuitState
from yahoofinance_v2.core.errors import APIError, ValidationError, RateLimitError, NetworkError, YFinanceError


@pytest.fixture
async def enhanced_provider():
    """Create test provider with disabled circuit breaker for most tests"""
    provider = EnhancedAsyncYahooFinanceProvider(
        max_retries=1,
        retry_delay=0.01,
        max_concurrency=2,
        enable_circuit_breaker=False  # Disable by default for most tests
    )
    yield provider
    # Clean up
    await provider.close()


@pytest.fixture
async def enhanced_provider_with_circuit_breaker():
    """Create test provider with enabled circuit breaker"""
    provider = EnhancedAsyncYahooFinanceProvider(
        max_retries=1,
        retry_delay=0.01,
        max_concurrency=2,
        enable_circuit_breaker=True
    )
    yield provider
    # Clean up
    await provider.close()


@pytest.mark.asyncio
async def test_ensure_session():
    """Test that _ensure_session creates a session when needed"""
    provider = EnhancedAsyncYahooFinanceProvider()
    assert provider._session is None
    
    # First call should create a session
    session = await provider._ensure_session()
    assert provider._session is not None
    assert session is provider._session
    assert not session.closed
    
    # Second call should return the same session
    session2 = await provider._ensure_session()
    assert session2 is session
    
    # Clean up
    await provider.close()
    assert provider._session is None


@pytest.mark.asyncio
async def test_fetch_json_success(enhanced_provider):
    """Test successful JSON fetch with mocked response"""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"test": "data"})
    
    # Mock the session's get method
    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    with patch.object(enhanced_provider, '_ensure_session', return_value=mock_session):
        result = await enhanced_provider._fetch_json("https://example.com")
        
        assert result == {"test": "data"}
        mock_session.get.assert_called_once_with("https://example.com", params=None)


@pytest.mark.asyncio
async def test_fetch_json_rate_limit_error(enhanced_provider):
    """Test handling of rate limit errors"""
    mock_response = AsyncMock()
    mock_response.status = 429
    mock_response.headers = {"Retry-After": "30"}
    
    # Mock the session's get method
    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    with patch.object(enhanced_provider, '_ensure_session', return_value=mock_session):
        with pytest.raises(RateLimitError) as excinfo:
            await enhanced_provider._fetch_json("https://example.com")
        
        assert "rate limit exceeded" in str(excinfo.value).lower()
        assert excinfo.value.retry_after == 30


@pytest.mark.asyncio
async def test_fetch_json_api_error(enhanced_provider):
    """Test handling of API errors"""
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text = AsyncMock(return_value="Internal Server Error")
    
    # Mock the session's get method
    mock_session = AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    with patch.object(enhanced_provider, '_ensure_session', return_value=mock_session):
        with pytest.raises(APIError) as excinfo:
            await enhanced_provider._fetch_json("https://example.com")
        
        assert "API error: 500" in str(excinfo.value)
        assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_fetch_json_network_error(enhanced_provider):
    """Test handling of network errors"""
    # Mock the session's get method to raise network error
    mock_session = AsyncMock()
    mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
    
    with patch.object(enhanced_provider, '_ensure_session', return_value=mock_session):
        with pytest.raises(NetworkError) as excinfo:
            await enhanced_provider._fetch_json("https://example.com")
        
        assert "Network error" in str(excinfo.value)
        assert "Connection failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_fetch_json_with_circuit_breaker(enhanced_provider_with_circuit_breaker):
    """Test circuit breaker integration with fetch_json"""
    # Mock for successful response
    mock_success_response = AsyncMock()
    mock_success_response.status = 200
    mock_success_response.json = AsyncMock(return_value={"test": "data"})
    
    # Mock for error response
    mock_error_response = AsyncMock()
    mock_error_response.status = 500
    mock_error_response.text = AsyncMock(return_value="Server Error")
    
    # Mock session with alternating responses
    mock_session = AsyncMock()
    mock_get_context_managers = [
        mock_success_response,  # First call succeeds
        mock_error_response,    # Next calls fail with 500
        mock_error_response,
        mock_error_response,
        mock_error_response,
    ]
    mock_session.get.return_value.__aenter__.side_effect = mock_get_context_managers
    
    provider = enhanced_provider_with_circuit_breaker
    
    with patch.object(provider, '_ensure_session', return_value=mock_session):
        # First call should succeed
        result = await provider._fetch_json("https://example.com")
        assert result == {"test": "data"}
        
        # Subsequent calls should fail
        for _ in range(3):  # We need 3 failures to trip the circuit
            with pytest.raises(APIError):
                await provider._fetch_json("https://example.com")
        
        # Now the circuit should be open
        with pytest.raises(APIError) as excinfo:
            await provider._fetch_json("https://example.com")
        
        # Verify it's a translated circuit open error
        assert "currently unavailable" in str(excinfo.value)
        assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_get_ticker_info(enhanced_provider):
    """Test get_ticker_info with mocked fetch_json"""
    # Simplified mock response based on actual Yahoo Finance API response
    mock_response = {
        "quoteSummary": {
            "result": [{
                "price": {
                    "regularMarketPrice": {"raw": 150.25},
                    "longName": "Test Company",
                    "shortName": "TEST",
                    "marketCap": {"raw": 2000000000000},
                    "currency": "USD",
                    "exchange": "NMS",
                    "quoteType": "EQUITY"
                },
                "summaryProfile": {
                    "sector": "Technology",
                    "industry": "Software",
                    "country": "United States",
                    "website": "https://example.com"
                },
                "summaryDetail": {
                    "trailingPE": {"raw": 25.5},
                    "dividendYield": {"raw": 0.0165},
                    "beta": {"raw": 1.25},
                    "fiftyTwoWeekHigh": {"raw": 180.0},
                    "fiftyTwoWeekLow": {"raw": 120.0}
                },
                "defaultKeyStatistics": {
                    "forwardPE": {"raw": 22.5},
                    "pegRatio": {"raw": 1.8},
                    "shortPercentOfFloat": {"raw": 0.015}
                },
                "financialData": {
                    "targetMeanPrice": {"raw": 175.0},
                    "recommendationMean": {"raw": 2.1}
                },
                "recommendationTrend": {
                    "trend": [{
                        "period": "0m",
                        "strongBuy": 10,
                        "buy": 15,
                        "hold": 5,
                        "sell": 2,
                        "strongSell": 0
                    }]
                }
            }]
        }
    }
    
    # Mock the fetch_json method
    with patch.object(enhanced_provider, '_fetch_json', AsyncMock(return_value=mock_response)), \
         patch.object(enhanced_provider, 'get_insider_transactions', AsyncMock(return_value=[])):
        
        result = await enhanced_provider.get_ticker_info("AAPL")
        
        # Verify key fields were extracted properly
        assert result["symbol"] == "AAPL"
        assert result["name"] == "Test Company"
        assert result["current_price"] == 150.25
        assert result["target_price"] == 175.0
        assert result["upside"] == pytest.approx(16.47, 0.01)  # (175/150.25 - 1) * 100
        assert result["market_cap"] == 2000000000000
        assert result["market_cap_fmt"] == "2.00T"
        assert result["pe_trailing"] == 25.5
        assert result["pe_forward"] == 22.5
        assert result["peg_ratio"] == 1.8
        assert result["dividend_yield"] == 1.65  # Converted to percentage
        assert result["short_percent"] == 1.5    # Converted to percentage
        assert result["analyst_count"] == 32     # Sum of all ratings


@pytest.mark.asyncio
async def test_batch_get_ticker_info(enhanced_provider):
    """Test batch_get_ticker_info with mocked get_ticker_info"""
    # Mock the get_ticker_info method
    async def mock_get_ticker_info(ticker, skip_insider_metrics=False):
        # Return different data for different tickers
        return {
            "symbol": ticker,
            "name": f"Test {ticker}",
            "current_price": 100.0 + (ord(ticker[-1]) - ord('A')) * 10
        }
    
    with patch.object(enhanced_provider, 'get_ticker_info', side_effect=mock_get_ticker_info):
        result = await enhanced_provider.batch_get_ticker_info(["AAPL", "MSFT", "GOOG"])
        
        # Verify each ticker got processed
        assert len(result) == 3
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOG" in result
        
        # Verify data for each ticker
        assert result["AAPL"]["name"] == "Test AAPL"
        assert result["AAPL"]["current_price"] == 100.0  # 100 + (L-A)*10 = 100 + (76-65)*10 = 100 + 110 = 210
        
        assert result["MSFT"]["name"] == "Test MSFT"
        assert result["MSFT"]["current_price"] == 130.0  # 100 + (T-A)*10 = 100 + (84-65)*10 = 100 + 190 = 290
        
        assert result["GOOG"]["name"] == "Test GOOG"
        assert result["GOOG"]["current_price"] == 110.0  # 100 + (G-A)*10 = 100 + (71-65)*10 = 100 + 60 = 160


@pytest.mark.asyncio
async def test_circuit_breaker_integration_retry_after():
    """Test that CircuitOpenError is translated with proper retry_after"""
    provider = EnhancedAsyncYahooFinanceProvider(enable_circuit_breaker=True)
    
    # Mock CircuitOpenError with metrics containing retry_after
    circuit_error = CircuitOpenError(
        "Circuit is OPEN",
        "yahoofinance_api",
        "OPEN",
        {"time_until_reset": 120}  # 2 minutes until reset
    )
    
    # Mock fetch_json to raise CircuitOpenError
    with patch.object(provider, '_fetch_json', side_effect=circuit_error):
        with pytest.raises(APIError) as excinfo:
            await provider.get_ticker_info("AAPL")
        
        # Check that error was translated properly
        assert "currently unavailable" in str(excinfo.value)
        assert excinfo.value.status_code == 503
        assert excinfo.value.retry_after == 120  # Should preserve retry_after
    
    await provider.close()


@pytest.mark.asyncio
async def test_error_handling_in_batch_operations(enhanced_provider):
    """Test error handling in batch operations"""
    # Mock get_ticker_info to succeed for some tickers and fail for others
    async def mock_get_ticker_info(ticker, skip_insider_metrics=False):
        if ticker == "AAPL":
            return {"symbol": ticker, "name": "Apple Inc.", "current_price": 150.0}
        elif ticker == "MSFT":
            return {"symbol": ticker, "name": "Microsoft Corp.", "current_price": 300.0}
        elif ticker == "ERROR":
            raise APIError("Test API error")
        elif ticker == "NETWORK":
            raise NetworkError("Test network error")
        else:
            raise ValidationError(f"Invalid ticker: {ticker}")
    
    with patch.object(enhanced_provider, 'get_ticker_info', side_effect=mock_get_ticker_info):
        result = await enhanced_provider.batch_get_ticker_info(["AAPL", "MSFT", "ERROR", "NETWORK", "INVALID"])
        
        # Verify successful requests
        assert "AAPL" in result
        assert result["AAPL"]["name"] == "Apple Inc."
        assert result["AAPL"]["current_price"] == 150.0
        
        assert "MSFT" in result
        assert result["MSFT"]["name"] == "Microsoft Corp."
        assert result["MSFT"]["current_price"] == 300.0
        
        # Verify error handling for failing requests
        assert "ERROR" in result
        assert "error" in result["ERROR"]
        assert "Test API error" in result["ERROR"]["error"]
        
        assert "NETWORK" in result
        assert "error" in result["NETWORK"]
        assert "Test network error" in result["NETWORK"]["error"]
        
        assert "INVALID" in result
        assert "error" in result["INVALID"]
        assert "Invalid ticker" in result["INVALID"]["error"]