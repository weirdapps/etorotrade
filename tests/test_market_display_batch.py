import pytest
from unittest.mock import Mock, patch
import pandas as pd
from yahoofinance.display import MarketDisplay, RateLimitTracker
from yahoofinance.client import YFinanceError

@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def mock_stock_info():
    info = Mock()
    info.current_price = 100.0
    info.target_price = 120.0
    info.analyst_count = 10
    info.pe_trailing = 20.5
    info.pe_forward = 18.2
    info.peg_ratio = 1.5
    info.dividend_yield = 2.5
    info.beta = 1.1
    info.short_float_pct = 2.0
    info.last_earnings = "2024-01-15"
    info.insider_buy_pct = 60.0
    info.insider_transactions = 5
    return info

@pytest.fixture
def mock_pricing():
    pricing = Mock()
    pricing.calculate_price_metrics.return_value = {
        'current_price': 100.0,
        'target_price': 120.0,
        'upside_potential': 20.0
    }
    return pricing

@pytest.fixture
def mock_analyst():
    analyst = Mock()
    analyst.get_ratings_summary.return_value = {
        'positive_percentage': 75.0,
        'total_ratings': 10
    }
    return analyst

@pytest.fixture
def display(mock_client, mock_pricing, mock_analyst):
    with patch('yahoofinance.display.PricingAnalyzer') as mock_pricing_class, \
         patch('yahoofinance.display.AnalystData') as mock_analyst_class:
        mock_pricing_class.return_value = mock_pricing
        mock_analyst_class.return_value = mock_analyst
        display = MarketDisplay(client=mock_client)
        return display

def mock_tqdm_wrapper(iterable, **kwargs):
    """Wrapper for tqdm that ignores extra arguments"""
    return iter(iterable)

def test_process_tickers_batch_size(display, mock_client, mock_stock_info):
    """Test processing tickers with different batch sizes"""
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    # Test with batch size of 2
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    with patch('yahoofinance.display.tqdm', side_effect=mock_tqdm_wrapper):
        reports = display._process_tickers(tickers, batch_size=2)
        
        assert len(reports) == 4
        assert mock_client.get_ticker_info.call_count == 4

def test_process_tickers_partial_failure(display, mock_client, mock_stock_info, mock_analyst):
    """Test batch processing with some failed tickers"""
    def mock_get_info(ticker):
        if ticker in ["AAPL", "MSFT"]:
            return mock_stock_info
        raise YFinanceError(f"API error for {ticker}")
    
    mock_client.get_ticker_info.side_effect = mock_get_info
    mock_analyst.get_ratings_summary.return_value = None  # Simulate failed ratings
    
    tickers = ["AAPL", "INVALID", "MSFT", "ERROR"]
    with patch('yahoofinance.display.tqdm', side_effect=mock_tqdm_wrapper):
        reports = display._process_tickers(tickers, batch_size=2)
        
        assert len(reports) == 4  # All tickers reported, some as not found
        assert sum(1 for r in reports if not r['_not_found']) == 2  # Two successful
        assert sum(1 for r in reports if r['_not_found']) == 2  # Two failed

def test_process_tickers_rate_limit_handling(display, mock_client, mock_stock_info):
    """Test rate limit handling during batch processing"""
    # Setup rate limit error sequence
    call_count = 0
    def mock_get_info(ticker):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:  # First two calls hit rate limit
            raise YFinanceError("Too many requests")
        return mock_stock_info
    
    mock_client.get_ticker_info.side_effect = mock_get_info
    
    with patch('time.sleep') as mock_sleep:  # Mock sleep to speed up tests
        with patch('yahoofinance.display.tqdm', side_effect=mock_tqdm_wrapper):
            reports = display._process_tickers(["AAPL"], batch_size=1)
            
            assert len(reports) == 1
            assert mock_sleep.call_count >= 2  # Verify backoff was applied

def test_process_tickers_batch_delay(display, mock_client, mock_stock_info):
    """Test batch delay adjustments based on success rate"""
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    tickers = ["AAPL", "GOOGL", "MSFT"]
    with patch('time.sleep') as mock_sleep:
        with patch('yahoofinance.display.tqdm', side_effect=mock_tqdm_wrapper):
            reports = display._process_tickers(tickers, batch_size=1)
            
            assert len(reports) == 3
            assert mock_sleep.call_count >= 2  # Verify inter-batch delays

def test_process_tickers_empty_batch(display):
    """Test processing empty ticker list"""
    reports = display._process_tickers([], batch_size=5)
    assert len(reports) == 0

def test_process_tickers_duplicate_handling(display, mock_client, mock_stock_info):
    """Test handling of duplicate tickers"""
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    tickers = ["AAPL", "AAPL", "AAPL"]
    with patch('yahoofinance.display.tqdm', side_effect=mock_tqdm_wrapper):
        reports = display._process_tickers(tickers, batch_size=2)
        
        assert len(reports) == 1  # Should deduplicate
        assert mock_client.get_ticker_info.call_count == 1  # Should only call once

def test_process_tickers_error_threshold(display, mock_client):
    """Test handling of tickers that exceed error threshold"""
    # Setup ticker to always error
    mock_client.get_ticker_info.side_effect = YFinanceError("API Error")
    
    with patch('yahoofinance.display.tqdm', side_effect=mock_tqdm_wrapper):
        reports = display._process_tickers(["AAPL"], batch_size=1)
        
        assert len(reports) == 1
        assert reports[0]['_not_found'] is True  # Should mark as not found

def test_process_tickers_batch_success_rate(display, mock_client, mock_stock_info):
    """Test batch delay adjustment based on success rate"""
    # Setup mixed success/failure sequence
    call_count = 0
    def mock_get_info(ticker):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:  # Every other call fails
            raise YFinanceError("API Error")
        return mock_stock_info
    
    mock_client.get_ticker_info.side_effect = mock_get_info
    
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    with patch('time.sleep') as mock_sleep:
        with patch('yahoofinance.display.tqdm', side_effect=mock_tqdm_wrapper):
            reports = display._process_tickers(tickers, batch_size=2)
            
            # Verify some reports are marked as not found due to errors
            not_found_count = sum(1 for r in reports if r.get('_not_found', False))
            assert not_found_count > 0

def test_process_tickers_rate_limit_recovery(display, mock_client, mock_stock_info):
    """Test recovery from rate limit errors"""
    # Setup recovery sequence
    call_count = 0
    def mock_get_info(ticker):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:  # First two calls hit rate limit
            raise YFinanceError("Too many requests")
        return mock_stock_info
    
    mock_client.get_ticker_info.side_effect = mock_get_info
    
    with patch('time.sleep'):  # Mock sleep to speed up tests
        with patch('yahoofinance.display.tqdm', side_effect=mock_tqdm_wrapper):
            reports = display._process_tickers(["AAPL"], batch_size=1)
            
            assert len(reports) == 1
            # Verify the report is eventually successful after retries
            assert not reports[0].get('_not_found', False)