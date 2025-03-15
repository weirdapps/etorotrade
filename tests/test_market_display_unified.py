#!/usr/bin/env python3
"""
Unified tests for market display functionality

This test file verifies all aspects of the MarketDisplay class:
- Core functionality: Loading tickers, generating reports, sorting data
- Batch processing: Handling multiple tickers with rate limiting
- HTML generation: Creating dashboards for market and portfolio data
"""

import unittest
import pytest
from unittest.mock import Mock, patch, mock_open, PropertyMock
import pandas as pd
from io import StringIO
from yahoofinance.display import MarketDisplay, RateLimitTracker
from yahoofinance.client import YFinanceClient, YFinanceError
from yahoofinance.formatting import DisplayConfig


class TestMarketDisplayCore(unittest.TestCase):
    """Test core functionality of MarketDisplay."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_client = Mock(spec=YFinanceClient)
        self.test_input_dir = "test_input"
        self.display = MarketDisplay(
            client=self.mock_client,
            input_dir=self.test_input_dir
        )
        # Mock the pricing analyzer
        self.display.pricing = Mock()
        self.display.analyst = Mock()

    def test_load_tickers_from_file(self):
        """Test loading tickers from file using helper method."""
        mock_csv_content = 'ticker\nAAPL\nGOOGL\nMSFT'
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))):
            tickers = self.display._load_tickers_from_file('portfolio.csv', 'ticker')
            self.assertEqual(sorted(tickers), ['AAPL', 'GOOGL', 'MSFT'])

    def test_load_tickers_from_file_not_found(self):
        """Test file not found handling in helper method."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError), \
             self.assertRaises(FileNotFoundError):
            self.display._load_tickers_from_file('nonexistent.csv', 'ticker')

    def test_load_tickers_from_file_invalid_column(self):
        """Test invalid column handling in helper method."""
        mock_csv_content = 'wrong_column\nAAPL\nGOOGL\nMSFT'
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))), \
             self.assertRaises(KeyError):
            self.display._load_tickers_from_file('portfolio.csv', 'ticker')

    def test_load_tickers_from_input_empty(self):
        """Test empty input handling in helper method."""
        with patch('builtins.input', return_value=''):
            tickers = self.display._load_tickers_from_input()
            self.assertEqual(tickers, [])

    def test_load_tickers_from_input_duplicates(self):
        """Test duplicate removal in helper method."""
        with patch('builtins.input', return_value='AAPL, AAPL, MSFT, MSFT'):
            tickers = self.display._load_tickers_from_input()
            self.assertEqual(sorted(tickers), ['AAPL', 'MSFT'])

    def test_sort_market_data(self):
        """Test market data sorting."""
        mock_data = pd.DataFrame([
            {'_not_found': False, '_sort_exret': 10, '_sort_earnings': 5, '_ticker': 'AAPL'},
            {'_not_found': False, '_sort_exret': 5, '_sort_earnings': 10, '_ticker': 'GOOGL'},
            {'_not_found': True, '_ticker': 'INVALID'}
        ])
        sorted_df = self.display._sort_market_data(mock_data)
        self.assertEqual(sorted_df.iloc[0]['_ticker'], 'AAPL')
        self.assertEqual(sorted_df.iloc[1]['_ticker'], 'GOOGL')
        self.assertEqual(sorted_df.iloc[2]['_ticker'], 'INVALID')

    def test_format_dataframe(self):
        """Test DataFrame formatting."""
        mock_data = pd.DataFrame([
            {'_not_found': False, '_sort_exret': 10, '_sort_earnings': 5, '_ticker': 'AAPL', 'price': 150}
        ])
        formatted_df = self.display._format_dataframe(mock_data)
        self.assertIn('#', formatted_df.columns)
        self.assertNotIn('_not_found', formatted_df.columns)
        self.assertNotIn('_sort_exret', formatted_df.columns)
        self.assertEqual(formatted_df.iloc[0]['#'], 1)

    def test_load_tickers_manual_input(self):
        """Test loading tickers from manual input."""
        with patch('builtins.input', return_value='AAPL, GOOGL, MSFT'):
            tickers = self.display.load_tickers('I')
            self.assertEqual(sorted(tickers), ['AAPL', 'GOOGL', 'MSFT'])

    def test_load_tickers_portfolio(self):
        """Test loading tickers from portfolio.csv."""
        mock_csv_content = 'ticker\nAAPL\nGOOGL\nMSFT'
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))):
            tickers = self.display.load_tickers('P')
            self.assertEqual(sorted(tickers), ['AAPL', 'GOOGL', 'MSFT'])

    def test_load_tickers_market(self):
        """Test loading tickers from market.csv."""
        mock_csv_content = 'symbol\nAAPL\nGOOGL\nMSFT'
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))):
            tickers = self.display.load_tickers('M')
            self.assertEqual(sorted(tickers), ['AAPL', 'GOOGL', 'MSFT'])

    def test_load_tickers_invalid_source(self):
        """Test loading tickers with invalid source."""
        tickers = self.display.load_tickers('X')
        self.assertEqual(tickers, [])

    def test_load_tickers_file_not_found(self):
        """Test handling of missing file."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError):
            tickers = self.display.load_tickers('P')
            self.assertEqual(tickers, [])

    def test_load_tickers_empty_input(self):
        """Test handling of empty manual input."""
        with patch('builtins.input', return_value=''):
            tickers = self.display.load_tickers('I')
            self.assertEqual(tickers, [])

    def test_load_tickers_duplicate_removal(self):
        """Test that duplicate tickers are removed."""
        with patch('builtins.input', return_value='AAPL, AAPL, MSFT, MSFT'):
            tickers = self.display.load_tickers('I')
            self.assertEqual(sorted(tickers), ['AAPL', 'MSFT'])

    def test_generate_stock_report(self):
        """Test stock report generation."""
        mock_price_metrics = {
            'current_price': 150.0,
            'target_price': 180.0,
            'upside_potential': 20.0
        }
        mock_ratings = {
            'positive_percentage': 80.0,
            'total_ratings': 10
        }
        mock_stock_info = Mock(
            analyst_count=10,
            pe_trailing=20.5,
            pe_forward=18.2,
            peg_ratio=1.5,
            dividend_yield=2.5,
            beta=1.1,
            short_float_pct=2.0,
            last_earnings='2024-01-01',
            insider_buy_pct=75.0,
            insider_transactions=5
        )
        
        self.display.pricing.calculate_price_metrics.return_value = mock_price_metrics
        self.display.analyst.get_ratings_summary.return_value = mock_ratings
        self.mock_client.get_ticker_info.return_value = mock_stock_info
        
        report = self.display.generate_stock_report('AAPL')
        
        self.assertEqual(report['ticker'], 'AAPL')
        self.assertEqual(report['price'], 150.0)
        self.assertEqual(report['target_price'], 180.0)
        self.assertEqual(report['upside'], 20.0)
        self.assertEqual(report['analyst_count'], 10)
        self.assertEqual(report['buy_percentage'], 80.0)
        self.assertFalse(report['_not_found'])

    def test_display_configuration(self):
        """Test display configuration options."""
        config = DisplayConfig(use_colors=False)
        display = MarketDisplay(
            client=self.mock_client,
            config=config
        )
        self.assertFalse(display.formatter.config.use_colors)


# Pytest-style tests for batch processing
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


# Pytest-style tests for HTML generation
@pytest.fixture
def html_display(mock_client):
    return MarketDisplay(client=mock_client)

@pytest.fixture
def html_stock_info():
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

def test_generate_market_metrics_success(html_display, mock_client, html_stock_info):
    """Test successful market metrics generation"""
    mock_client.get_ticker_info.return_value = html_stock_info
    
    metrics = html_display._generate_market_metrics(['AAPL'])
    
    assert 'AAPL' in metrics
    assert metrics['AAPL']['value'] == pytest.approx(5.0)
    assert metrics['AAPL']['label'] == 'AAPL'
    assert metrics['AAPL']['is_percentage'] is True

def test_generate_market_metrics_partial_failure(html_display, mock_client, html_stock_info):
    """Test market metrics generation with some failed tickers"""
    def mock_get_info(ticker):
        if ticker == 'AAPL':
            return html_stock_info
        raise YFinanceError(f"API error for {ticker}")
    
    mock_client.get_ticker_info.side_effect = mock_get_info
    
    metrics = html_display._generate_market_metrics(['AAPL', 'INVALID'])
    
    assert 'AAPL' in metrics
    assert 'INVALID' not in metrics
    assert len(metrics) == 1

def test_generate_market_metrics_no_price(html_display, mock_client):
    """Test handling of tickers with no price data"""
    mock_info = Mock()
    mock_info.current_price = None
    mock_info.price_change_percentage = None
    mock_client.get_ticker_info.return_value = mock_info
    
    metrics = html_display._generate_market_metrics(['AAPL'])
    
    assert len(metrics) == 0

@patch('builtins.open', new_callable=mock_open)
def test_write_html_file_success(mock_file, html_display):
    """Test successful HTML file writing"""
    html_display._write_html_file('<html>test</html>', 'test.html')
    
    mock_file.assert_called_once()
    mock_file().write.assert_called_once_with('<html>test</html>')

@patch('builtins.open')
def test_write_html_file_error(mock_file, html_display):
    """Test HTML file writing error handling"""
    mock_file.side_effect = IOError("Write error")
    
    html_display._write_html_file('<html>test</html>', 'test.html')
    # Should log error but not raise exception

def test_generate_market_html_success(html_display, mock_client, html_stock_info):
    """Test successful market HTML generation"""
    # Mock file operations
    with patch('builtins.open', new_callable=mock_open):
        # Mock ticker loading
        with patch.object(html_display, '_load_tickers_from_file') as mock_load:
            mock_load.return_value = ['AAPL']
            
            # Mock client response
            mock_client.get_ticker_info.return_value = html_stock_info
            
            # Should complete without errors
            html_display.generate_market_html()

def test_generate_market_html_no_tickers(html_display):
    """Test market HTML generation with no tickers"""
    with patch.object(html_display, '_load_tickers_from_file') as mock_load:
        mock_load.return_value = []
        
        html_display.generate_market_html()
        # Should log error but not raise exception

def test_generate_portfolio_html_success(html_display, mock_client, html_stock_info):
    """Test successful portfolio HTML generation"""
    # Mock file operations
    with patch('builtins.open', new_callable=mock_open):
        # Mock ticker loading
        with patch.object(html_display, '_load_tickers_from_file') as mock_load:
            mock_load.return_value = ['AAPL']
            
            # Mock client response
            mock_client.get_ticker_info.return_value = html_stock_info
            
            # Should complete without errors
            html_display.generate_portfolio_html()

def test_generate_portfolio_html_partial_data(html_display, mock_client):
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
        with patch.object(html_display, '_load_tickers_from_file') as mock_load:
            mock_load.return_value = ['AAPL']
            
            # Mock client response
            mock_client.get_ticker_info.return_value = partial_info
            
            # Should complete without errors
            html_display.generate_portfolio_html()

def test_generate_portfolio_html_api_error(html_display, mock_client):
    """Test portfolio HTML generation with API errors"""
    # Mock file operations
    with patch('builtins.open', new_callable=mock_open):
        # Mock ticker loading
        with patch.object(html_display, '_load_tickers_from_file') as mock_load:
            mock_load.return_value = ['AAPL']
            
            # Mock API error
            mock_client.get_ticker_info.side_effect = YFinanceError("API Error")
            
            # Should handle error gracefully
            html_display.generate_portfolio_html()


if __name__ == "__main__":
    unittest.main()