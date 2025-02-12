import pytest
from unittest.mock import Mock, patch, mock_open
import pandas as pd
from datetime import datetime
from yahoofinance.display import MarketDisplay
from yahoofinance.client import YFinanceError
from yahoofinance.formatting import DisplayConfig

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
    info.analyst_count = 10
    info.pe_trailing = 20.5
    info.pe_forward = 18.2
    info.peg_ratio = 1.5
    info.dividend_yield = 2.5
    info.short_float_pct = 3.0
    info.last_earnings = "2024-01-15"
    info.insider_buy_pct = 60.0
    info.insider_transactions = 5
    return info

@pytest.fixture
def display(mock_client):
    with patch('yahoofinance.display.PricingAnalyzer') as mock_pricing:
        instance = mock_pricing.return_value
        display = MarketDisplay(client=mock_client)
        display.pricing = instance
        return display

def test_init_default():
    display = MarketDisplay()
    assert display.input_dir == "yahoofinance/input"
    assert display.client is not None
    assert display.formatter is not None

def test_init_custom():
    client = Mock()
    config = DisplayConfig()
    display = MarketDisplay(client=client, config=config, input_dir="custom/input")
    assert display.input_dir == "custom/input"
    assert display.client == client
    assert display.formatter.config == config

@patch('pandas.read_csv')
def test_load_tickers_from_file(mock_read_csv, display):
    mock_df = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT']
    })
    mock_read_csv.return_value = mock_df
    
    tickers = display._load_tickers_from_file('portfolio.csv', 'ticker')
    assert tickers == ['AAPL', 'GOOGL', 'MSFT']
    mock_read_csv.assert_called_once()

@patch('builtins.input', return_value='AAPL, GOOGL, MSFT')
def test_load_tickers_from_input(mock_input, display):
    tickers = display._load_tickers_from_input()
    assert set(tickers) == {'AAPL', 'GOOGL', 'MSFT'}
    mock_input.assert_called_once()

def test_create_empty_report(display):
    report = display._create_empty_report('AAPL')
    assert report['ticker'] == 'AAPL'
    assert report['price'] == 0
    assert report['_not_found'] is True
    assert all(key in report for key in [
        'target_price', 'upside', 'analyst_count', 'buy_percentage',
        'total_ratings', 'pe_trailing', 'pe_forward', 'peg_ratio',
        'dividend_yield', 'beta', 'short_float_pct', 'last_earnings',
        'insider_buy_pct', 'insider_transactions'
    ])

def test_generate_stock_report_success(display, mock_client, mock_stock_info):
    mock_client.get_ticker_info.return_value = mock_stock_info
    display.pricing.calculate_price_metrics.return_value = {
        'current_price': 100.0,
        'target_price': 120.0,
        'upside_potential': 20.0
    }

    report = display.generate_stock_report('AAPL')
    
    assert report['ticker'] == 'AAPL'
    assert report['price'] == 100.0
    assert report['target_price'] == 120.0
    assert report['upside'] == 20.0
    assert report['_not_found'] is False

def test_generate_stock_report_no_price_data(display, mock_client):
    display.pricing.calculate_price_metrics.return_value = None
    
    report = display.generate_stock_report('AAPL')
    assert report['_not_found'] is True
    assert report['price'] == 0

def test_generate_stock_report_api_error(display, mock_client):
    mock_client.get_ticker_info.side_effect = YFinanceError("API Error")
    
    report = display.generate_stock_report('AAPL')
    assert report['_not_found'] is True
    assert report['price'] == 0

def test_sort_market_data(display):
    data = [
        {'ticker': 'AAPL', '_not_found': False, '_sort_exret': 10, '_sort_earnings': 5},
        {'ticker': 'GOOGL', '_not_found': False, '_sort_exret': 5, '_sort_earnings': 10},
        {'ticker': 'INVALID', '_not_found': True, '_ticker': 'INVALID'}
    ]
    df = pd.DataFrame(data)
    
    sorted_df = display._sort_market_data(df)
    
    assert len(sorted_df) == 3
    assert sorted_df.iloc[0]['ticker'] == 'AAPL'
    assert sorted_df.iloc[-1]['ticker'] == 'INVALID'

def test_format_dataframe(display):
    data = [
        {'ticker': 'AAPL', '_not_found': False, '_sort_exret': 10, '_sort_earnings': 5, '_ticker': 'AAPL'},
        {'ticker': 'GOOGL', '_not_found': False, '_sort_exret': 5, '_sort_earnings': 10, '_ticker': 'GOOGL'}
    ]
    df = pd.DataFrame(data)
    
    formatted_df = display._format_dataframe(df)
    
    assert '#' in formatted_df.columns
    assert '_not_found' not in formatted_df.columns
    assert '_sort_exret' not in formatted_df.columns
    assert '_sort_earnings' not in formatted_df.columns
    assert '_ticker' not in formatted_df.columns

@patch('yahoofinance.display.tqdm')
def test_process_tickers(mock_tqdm, display, mock_client, mock_stock_info):
    mock_client.get_ticker_info.return_value = mock_stock_info
    display.pricing.calculate_price_metrics.return_value = {
        'current_price': 100.0,
        'target_price': 120.0,
        'upside_potential': 20.0
    }
    mock_tqdm.return_value = ['AAPL', 'GOOGL']
    
    reports = display._process_tickers(['AAPL', 'GOOGL'])
    assert len(reports) == 2
    assert all('_ticker' in report for report in reports)

@patch('builtins.open', new_callable=mock_open)
def test_write_html_file(mock_file, display):
    display._write_html_file('<html>test</html>', 'test.html')
    mock_file.assert_called_once()
    mock_file().write.assert_called_once_with('<html>test</html>')

def test_display_report_no_tickers(display):
    with pytest.raises(ValueError, match="No valid tickers provided"):
        display.display_report([])

@patch('yahoofinance.display.tabulate')
def test_display_report_success(mock_tabulate, display, mock_client, mock_stock_info):
    mock_client.get_ticker_info.return_value = mock_stock_info
    display.pricing.calculate_price_metrics.return_value = {
        'current_price': 100.0,
        'target_price': 120.0,
        'upside_potential': 20.0
    }
    
    with patch('builtins.print') as mock_print:
        display.display_report(['AAPL'])
        assert mock_print.call_count >= 3  # Header + timestamp + table
        assert mock_tabulate.call_count == 1

def test_generate_market_metrics(display, mock_client, mock_stock_info):
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    metrics = display._generate_market_metrics(['AAPL'])
    assert 'AAPL' in metrics
    assert metrics['AAPL']['value'] == 5.0
    assert metrics['AAPL']['is_percentage'] is True

@patch('yahoofinance.utils.FormatUtils')
def test_generate_market_html(mock_format_utils, display):
    with patch.object(display, '_load_tickers_from_file') as mock_load:
        mock_load.return_value = ['AAPL', 'GOOGL']
        with patch.object(display, '_generate_market_metrics') as mock_metrics:
            mock_metrics.return_value = {'AAPL': {'value': 5.0}}
            with patch.object(display, '_write_html_file') as mock_write:
                display.generate_market_html()
                assert mock_load.call_count == 1
                assert mock_metrics.call_count == 1
                assert mock_write.call_count == 1

@patch('yahoofinance.utils.FormatUtils')
def test_generate_portfolio_html(mock_format_utils, display, mock_client, mock_stock_info):
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    with patch.object(display, '_load_tickers_from_file') as mock_load:
        mock_load.return_value = ['AAPL']
        with patch.object(display, '_write_html_file') as mock_write:
            display.generate_portfolio_html()
            assert mock_load.call_count == 1
            assert mock_write.call_count == 1