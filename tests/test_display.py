import pytest
from unittest.mock import Mock, patch, mock_open
import pandas as pd
from datetime import datetime
from yahoofinance.display import MarketDisplay
from yahoofinance.core.errors import YFinanceError
from yahoofinance.formatting import DisplayConfig

@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.get_ticker_info.return_value = {
        'ticker': 'AAPL',
        'name': 'Apple Inc.',
        'current_price': 100.0,
        'target_price': 120.0,
        'upside_potential': 20.0,
        'beta': 1.2,
        'pe_trailing': 20.5,
        'pe_forward': 18.2,
        'peg_ratio': 1.5,
        'dividend_yield': 2.5,
        'short_float_pct': 3.0,
        'analyst_count': 10,
    }
    provider.get_price_data.return_value = {
        'current_price': 100.0,
        'target_price': 120.0,
        'upside_potential': 20.0,
    }
    provider.get_analyst_ratings.return_value = {
        'positive_percentage': 75.0,
        'total_ratings': 10,
        'recommendations': {
            'buy': 7,
            'hold': 2,
            'sell': 1
        }
    }
    return provider

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
def display(mock_provider):
    with patch('yahoofinance.api.get_provider', return_value=mock_provider):
        with patch('yahoofinance.display.PricingAnalyzer') as mock_pricing:
            instance = mock_pricing.return_value
            display = MarketDisplay()
            display.provider = mock_provider
            display.pricing = instance
            return display

@patch('yahoofinance.api.get_provider')
def test_init_default(mock_get_provider):
    mock_provider = Mock()
    mock_get_provider.return_value = mock_provider
    
    display = MarketDisplay()
    assert display.input_dir == "yahoofinance/input"
    assert display.provider is not None
    assert display.formatter is not None

def test_init_custom():
    config = DisplayConfig()
    mock_provider = Mock()
    
    with patch('yahoofinance.api.get_provider', return_value=mock_provider):
        display = MarketDisplay(config=config, input_dir="custom/input")
        assert display.input_dir == "custom/input"
        assert display.provider is not None  # Just check it exists
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

@pytest.mark.skip(reason="Needs to be updated for provider pattern")
def test_generate_stock_report_success(display, mock_provider):
    # Mock provider responses are already set up in the fixture

    report = display.generate_stock_report('AAPL')
    
    assert report['ticker'] == 'AAPL'
    assert report['price'] == pytest.approx(100.0)
    assert report['target_price'] == pytest.approx(120.0)
    assert report['upside'] == pytest.approx(20.0)
    assert report['_not_found'] is False

def test_generate_stock_report_no_price_data(display, mock_client):
    display.pricing.calculate_price_metrics.return_value = None
    
    report = display.generate_stock_report('AAPL')
    assert report['_not_found'] is True
    assert report['price'] == 0

def test_generate_stock_report_api_error(display, mock_provider):
    mock_provider.get_ticker_info.side_effect = YFinanceError("API Error")
    
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
def test_process_tickers(mock_tqdm, display, mock_provider):
    # Set up the mock_tqdm to return an iterable instead of list
    mock_tqdm.return_value.__iter__.return_value = iter(['AAPL', 'GOOGL'])
    
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
def test_display_report_success(mock_tabulate, display, mock_provider):
    # Mock provider responses are already set up in the fixture
    
    with patch('builtins.print') as mock_print:
        display.display_report(['AAPL'])
        assert mock_print.call_count >= 3  # Header + timestamp + table
        assert mock_tabulate.call_count == 1

@pytest.mark.skip(reason="Needs to be updated for provider pattern")
def test_generate_market_metrics(display, mock_provider):
    # Provider already mocked in fixture
    
    metrics = display._generate_market_metrics(['AAPL'])
    assert 'AAPL' in metrics
    assert metrics['AAPL']['value'] == pytest.approx(5.0)
    assert metrics['AAPL']['is_percentage'] is True

@patch('yahoofinance.utils.FormatUtils')
def test_generate_market_html(mock_format_utils, display):
    mock_format_utils.generate_market_html.return_value = '<html>test</html>'
    mock_format_utils.format_market_metrics.return_value = {'formatted': 'metrics'}
    
    with patch.object(display, '_generate_market_metrics') as mock_metrics:
        mock_metrics.return_value = {'AAPL': {'value': 5.0}}
        with patch.object(display, '_write_html_file') as mock_write:
            mock_write.return_value = '/path/to/file.html'
            result = display.generate_market_html(['AAPL', 'GOOGL'])
            assert mock_metrics.call_count == 1
            mock_write.assert_called_once_with('<html>test</html>', 'index.html')
            assert result == '/path/to/file.html'

@patch('yahoofinance.utils.FormatUtils')
def test_generate_portfolio_html(mock_format_utils, display, mock_provider):
    # Provider already mocked in fixture
    mock_format_utils.generate_market_html.return_value = '<html>test</html>'
    mock_format_utils.format_market_metrics.return_value = {'formatted': 'metrics'}
    
    with patch.object(display, '_process_tickers') as mock_process:
        mock_process.return_value = [{'raw': {'ticker': 'AAPL'}}]
        with patch.object(display, '_write_html_file') as mock_write:
            mock_write.return_value = '/path/to/file.html'
            result = display.generate_portfolio_html(['AAPL'])
            assert mock_process.call_count == 1
            mock_write.assert_called_once_with('<html>test</html>', 'portfolio.html')
            assert result == '/path/to/file.html'

@pytest.mark.skip(reason="Needs to be updated for provider pattern")
@patch('pandas.DataFrame.to_csv')
def test_save_to_csv(mock_to_csv, display):
    df = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL'],
        'price': [100.0, 200.0]
    })
    
    # Test market source
    display._save_to_csv(df, 'M')
    
    # Test was called at least once
    assert mock_to_csv.call_count >= 1

@patch('yahoofinance.display.tabulate')
@patch('pandas.DataFrame.to_csv')
def test_display_report_with_csv_saving(mock_to_csv, mock_tabulate, display, mock_provider):
    # Provider already mocked in fixture
    
    with patch('builtins.print'):
        # Test market source - should call to_csv
        display.display_report(['AAPL'], 'M')
        assert mock_to_csv.call_count > 0
        
        # Reset mock
        mock_to_csv.reset_mock()
        
        # Test portfolio source - should call to_csv 
        display.display_report(['AAPL'], 'P')
        assert mock_to_csv.call_count > 0
        
        # Reset mock
        mock_to_csv.reset_mock()
        
        # Test manual input (no CSV saving)
        display.display_report(['AAPL'], 'I')
        mock_to_csv.assert_not_called()