import pytest
from unittest.mock import patch, Mock
import sys
from io import StringIO
from yahoofinance._metrics import show_available_metrics, main

@pytest.fixture
def mock_stock_info():
    return {
        # Valuation metrics
        'trailingPE': 25.5,
        'forwardPE': 20.1,
        'priceToBook': 10.2,
        'enterpriseValue': 2000000000,
        
        # Growth & Margins
        'revenueGrowth': 0.15,
        'profitMargins': 0.25,
        'grossMargins': 0.45,
        
        # Financial Health
        'currentRatio': 1.5,
        'debtToEquity': 0.8,
        'returnOnEquity': 0.2,
        
        # Market Data
        'beta': 1.2,
        'marketCap': 1500000000,
        'shortRatio': 2.5,
        
        # Dividends
        'dividendYield': 0.03,
        'payoutRatio': 0.4,
        
        # Earnings
        'trailingEps': 5.5,
        'forwardEps': 6.2
    }

@pytest.fixture
def mock_yf_ticker(mock_stock_info):
    mock_ticker = Mock()
    mock_ticker.info = mock_stock_info
    return mock_ticker

def test_show_available_metrics(mock_yf_ticker, capsys):
    with patch('yfinance.Ticker', return_value=mock_yf_ticker):
        show_available_metrics('AAPL')
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check header
        assert 'Available metrics for AAPL' in output
        assert '=' * 80 in output
        
        # Check categories
        assert 'Valuation:' in output
        assert 'Growth & Margins:' in output
        assert 'Financial Health:' in output
        assert 'Market Data:' in output
        assert 'Dividends:' in output
        assert 'Earnings:' in output
        
        # Check some specific metrics
        assert 'trailingPE' in output
        assert '25.5' in output
        assert 'beta' in output
        assert '1.2' in output
        assert 'dividendYield' in output
        assert '0.03' in output

def test_show_available_metrics_missing_values(mock_yf_ticker, capsys):
    # Modify mock to have some None values
    mock_yf_ticker.info = {
        'trailingPE': None,
        'beta': 1.2,
        'dividendYield': None
    }
    
    with patch('yfinance.Ticker', return_value=mock_yf_ticker):
        show_available_metrics('AAPL')
        
        captured = capsys.readouterr()
        output = captured.out
        
        # None values should not be printed
        assert 'trailingPE' not in output
        assert 'beta                      = 1.2' in output
        assert 'dividendYield' not in output

def test_show_available_metrics_empty_info(capsys):
    mock_ticker = Mock()
    mock_ticker.info = {}
    
    with patch('yfinance.Ticker', return_value=mock_ticker):
        show_available_metrics('AAPL')
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should still show categories but no metrics
        assert 'Available metrics for AAPL' in output
        assert 'Valuation:' in output
        assert 'Growth & Margins:' in output
        assert 'trailingPE' not in output
        assert 'beta' not in output

def test_main_with_valid_args():
    test_args = ['script.py', 'AAPL']
    with patch.object(sys, 'argv', test_args), \
         patch('yahoofinance._metrics.show_available_metrics') as mock_show:
        main()
        mock_show.assert_called_once_with('AAPL')

def test_main_with_invalid_args(capsys):
    test_args = ['script.py']
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        
        captured = capsys.readouterr()
        assert "Usage: python -m yahoofinance.metrics_debug TICKER" in captured.out

def test_main_lowercase_ticker():
    test_args = ['script.py', 'aapl']
    with patch.object(sys, 'argv', test_args), \
         patch('yahoofinance._metrics.show_available_metrics') as mock_show:
        main()
        # Should convert ticker to uppercase
        mock_show.assert_called_once_with('AAPL')

def test_metrics_categories_structure():
    """Test that all expected metric categories are present"""
    with patch('yfinance.Ticker') as mock_ticker:
        # Capture the categories by calling the function
        with patch('builtins.print'):  # Suppress output
            show_available_metrics('AAPL')
            
        # Get the source code
        import inspect
        source = inspect.getsource(show_available_metrics)
        
        # Check for expected categories in the source
        expected_categories = [
            "Valuation",
            "Growth & Margins",
            "Financial Health",
            "Market Data",
            "Dividends",
            "Earnings"
        ]
        
        for category in expected_categories:
            assert category in source, f"Missing category: {category}"