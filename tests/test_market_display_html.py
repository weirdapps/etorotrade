import pytest
from unittest.mock import Mock, patch, mock_open
import pandas as pd
from yahoofinance.display import MarketDisplay
from yahoofinance.client import YFinanceError

@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def mock_stock_info():
    info = Mock()
    info.current_price = 100.0
    info.price_change_percentage = 5.0
    info.mtd_change = 3.0
    info.ytd_change = 10.0
    info.two_year_change = 20.0
    info.beta = 1.2
    info.alpha = 0.5
    info.sharpe_ratio = 1.8
    info.sortino_ratio = 2.1
    info.cash_percentage = 15.0
    return info

@pytest.fixture
def display(mock_client):
    return MarketDisplay(client=mock_client)

def test_generate_market_metrics_success(display, mock_client, mock_stock_info):
    """Test successful market metrics generation"""
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    metrics = display._generate_market_metrics(['AAPL'])
    
    assert 'AAPL' in metrics
    assert metrics['AAPL']['value'] == pytest.approx(5.0)
    assert metrics['AAPL']['label'] == 'AAPL'
    assert metrics['AAPL']['is_percentage'] is True

def test_generate_market_metrics_partial_failure(display, mock_client, mock_stock_info):
    """Test market metrics generation with some failed tickers"""
    def mock_get_info(ticker):
        if ticker == 'AAPL':
            return mock_stock_info
        raise YFinanceError(f"API error for {ticker}")
    
    mock_client.get_ticker_info.side_effect = mock_get_info
    
    metrics = display._generate_market_metrics(['AAPL', 'INVALID'])
    
    assert 'AAPL' in metrics
    assert 'INVALID' not in metrics
    assert len(metrics) == 1

def test_generate_market_metrics_no_price(display, mock_client):
    """Test handling of tickers with no price data"""
    mock_info = Mock()
    mock_info.current_price = None
    mock_info.price_change_percentage = None
    mock_client.get_ticker_info.return_value = mock_info
    
    metrics = display._generate_market_metrics(['AAPL'])
    
    assert len(metrics) == 0

@patch('builtins.open', new_callable=mock_open)
def test_write_html_file_success(mock_file, display):
    """Test successful HTML file writing"""
    display._write_html_file('<html>test</html>', 'test.html')
    
    mock_file.assert_called_once()
    mock_file().write.assert_called_once_with('<html>test</html>')

@patch('builtins.open')
def test_write_html_file_error(mock_file, display):
    """Test HTML file writing error handling"""
    mock_file.side_effect = IOError("Write error")
    
    display._write_html_file('<html>test</html>', 'test.html')
    # Should log error but not raise exception

def test_generate_market_html_success(display, mock_client, mock_stock_info):
    """Test successful market HTML generation"""
    # Mock file operations
    with patch('builtins.open', new_callable=mock_open):
        # Mock ticker loading
        with patch.object(display, '_load_tickers_from_file') as mock_load:
            mock_load.return_value = ['AAPL']
            
            # Mock client response
            mock_client.get_ticker_info.return_value = mock_stock_info
            
            # Should complete without errors
            display.generate_market_html()

def test_generate_market_html_no_tickers(display):
    """Test market HTML generation with no tickers"""
    with patch.object(display, '_load_tickers_from_file') as mock_load:
        mock_load.return_value = []
        
        display.generate_market_html()
        # Should log error but not raise exception

def test_generate_portfolio_html_success(display, mock_client, mock_stock_info):
    """Test successful portfolio HTML generation"""
    # Mock file operations
    with patch('builtins.open', new_callable=mock_open):
        # Mock ticker loading
        with patch.object(display, '_load_tickers_from_file') as mock_load:
            mock_load.return_value = ['AAPL']
            
            # Mock client response
            mock_client.get_ticker_info.return_value = mock_stock_info
            
            # Should complete without errors
            display.generate_portfolio_html()

def test_generate_portfolio_html_partial_data(display, mock_client):
    """Test portfolio HTML generation with partial data"""
    # Create mock with missing data
    partial_info = Mock()
    partial_info.current_price = 100.0
    partial_info.price_change_percentage = 5.0
    partial_info.mtd_change = None  # Missing MTD
    partial_info.ytd_change = 10.0
    partial_info.two_year_change = None  # Missing 2YR
    partial_info.beta = 1.2
    partial_info.alpha = None  # Missing Alpha
    partial_info.sharpe_ratio = 1.8
    partial_info.sortino_ratio = None  # Missing Sortino
    partial_info.cash_percentage = 15.0
    
    # Mock file operations
    with patch('builtins.open', new_callable=mock_open):
        # Mock ticker loading
        with patch.object(display, '_load_tickers_from_file') as mock_load:
            mock_load.return_value = ['AAPL']
            
            # Mock client response
            mock_client.get_ticker_info.return_value = partial_info
            
            # Should complete without errors
            display.generate_portfolio_html()

def test_generate_portfolio_html_api_error(display, mock_client):
    """Test portfolio HTML generation with API errors"""
    # Mock file operations
    with patch('builtins.open', new_callable=mock_open):
        # Mock ticker loading
        with patch.object(display, '_load_tickers_from_file') as mock_load:
            mock_load.return_value = ['AAPL']
            
            # Mock API error
            mock_client.get_ticker_info.side_effect = YFinanceError("API Error")
            
            # Should handle error gracefully
            display.generate_portfolio_html()

def test_generate_portfolio_html_no_tickers(display):
    """Test portfolio HTML generation with no tickers"""
    with patch.object(display, '_load_tickers_from_file') as mock_load:
        mock_load.return_value = []
        
        display.generate_portfolio_html()
        # Should log error but not raise exception

def test_generate_portfolio_html_file_error(display, mock_client, mock_stock_info):
    """Test portfolio HTML generation with file write error"""
    # Mock ticker loading
    with patch.object(display, '_load_tickers_from_file') as mock_load:
        mock_load.return_value = ['AAPL']
        
        # Mock client response
        mock_client.get_ticker_info.return_value = mock_stock_info
        
        # Mock file write error
        with patch('builtins.open') as mock_file:
            mock_file.side_effect = IOError("Write error")
            
            # Should handle error gracefully
            display.generate_portfolio_html()