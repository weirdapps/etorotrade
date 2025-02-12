import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
from yahoofinance.index import (
    get_previous_trading_day_close,
    calculate_weekly_dates,
    get_previous_month_ends,
    fetch_changes,
    update_html,
    INDICES
)

@pytest.fixture
def mock_yf_data():
    mock_data = pd.DataFrame({
        'Close': [100.0, 105.0, 110.0]
    }, index=[
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3)
    ])
    return mock_data

@pytest.fixture
def mock_html():
    return """
    <html>
        <body>
            <span id="DJI30">old_value</span>
            <span id="SP500">old_value</span>
        </body>
    </html>
    """

def test_get_previous_trading_day_close(mock_yf_data):
    with patch('yfinance.download', return_value=mock_yf_data):
        price, date = get_previous_trading_day_close('^DJI', datetime(2024, 1, 3))
        assert float(price.iloc[-1]) == 110.0
        assert date == datetime(2024, 1, 3).date()

def test_calculate_weekly_dates():
    with patch('yahoofinance.index.datetime') as mock_datetime:
        # Mock today as a Wednesday (weekday 2)
        mock_datetime.today.return_value = datetime(2024, 1, 10)  # A Wednesday
        previous_friday, last_friday = calculate_weekly_dates()
        
        # Last Friday should be Jan 5, Previous Friday should be Dec 29
        assert last_friday.date() == datetime(2024, 1, 5).date()
        assert previous_friday.date() == datetime(2023, 12, 29).date()

def test_get_previous_month_ends():
    with patch('yahoofinance.index.datetime') as mock_datetime:
        # Mock current date as Feb 15, 2024
        mock_now = datetime(2024, 2, 15)
        mock_datetime.today.return_value = mock_now
        
        prev_prev_month_end, prev_month_end = get_previous_month_ends()
        
        # Previous month end should be Jan 31, 2024
        # Previous previous month end should be Dec 31, 2023
        assert prev_month_end == datetime(2024, 1, 31).date()
        assert prev_prev_month_end == datetime(2023, 12, 31).date()

def test_fetch_changes(mock_yf_data):
    with patch('yfinance.download', return_value=mock_yf_data):
        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 3).date()
        
        changes = fetch_changes(start_date, end_date)
        
        # Check if we have results for all indices
        assert len(changes) == len(INDICES)
        
        # Check structure of first result
        first_change = changes[0]
        assert 'Index' in first_change
        assert 'Change Percent' in first_change
        assert any('Previous' in key for key in first_change.keys())
        assert any('Current' in key for key in first_change.keys())

def test_update_html(tmp_path):
    html_file = tmp_path / "test.html"
    
    test_data = [
        {
            'Index': 'DJI30',
            'Change Percent': '+1.23%',
            'Current (2024-01-03)': '15000.00'
        },
        {
            'Index': 'SP500',
            'Change Percent': '-0.45%',
            'Current (2024-01-03)': '4000.00'
        },
        {
            'Index': 'NQ100',
            'Change Percent': '+0.75%',
            'Current (2024-01-03)': '16000.00'
        },
        {
            'Index': 'VIX',
            'Change Percent': '-1.20%',
            'Current (2024-01-03)': '15.50'
        }
    ]

    formatted_metrics = [
        {'id': 'DJI30', 'value': '+1.23%', 'label': 'DJI30 (2024-01-03)'},
        {'id': 'SP500', 'value': '-0.45%', 'label': 'SP500 (2024-01-03)'},
        {'id': 'NQ100', 'value': '+0.75%', 'label': 'NQ100 (2024-01-03)'},
        {'id': 'VIX', 'value': '-1.20%', 'label': 'VIX (2024-01-03)'}
    ]
    
    mock_utils = Mock()
    mock_utils.format_market_metrics.return_value = formatted_metrics
    mock_utils.generate_market_html.return_value = "<html>Mocked HTML</html>"
    
    with patch('yahoofinance.index.FormatUtils', return_value=mock_utils):
        update_html(test_data, str(html_file))
        
        # Verify format_market_metrics was called
        assert mock_utils.format_market_metrics.call_count == 1
        # Verify generate_market_html was called
        assert mock_utils.generate_market_html.call_count == 1
        assert "Market Performance" in mock_utils.generate_market_html.call_args[1]['title']

@patch('builtins.input', side_effect=['W'])
def test_main_weekly(mock_input, mock_yf_data):
    with patch('yfinance.download', return_value=mock_yf_data), \
         patch('yahoofinance.index.datetime') as mock_datetime, \
         patch('yahoofinance.index.display_results') as mock_display, \
         patch('yahoofinance.index.update_html') as mock_update:
        
        mock_datetime.today.return_value = datetime(2024, 1, 10)  # A Wednesday
        from yahoofinance.index import main
        main()
        
        # Verify display_results was called
        assert mock_display.call_count == 1
        # Verify update_html was called
        assert mock_update.call_count == 1

@patch('builtins.input', side_effect=['M'])
def test_main_monthly(mock_input, mock_yf_data):
    with patch('yfinance.download', return_value=mock_yf_data), \
         patch('yahoofinance.index.datetime') as mock_datetime, \
         patch('yahoofinance.index.display_results') as mock_display, \
         patch('yahoofinance.index.update_html') as mock_update:
        
        mock_datetime.today.return_value = datetime(2024, 2, 15)
        from yahoofinance.index import main
        main()
        
        # Verify display_results was called
        assert mock_display.call_count == 1
        # Verify update_html was called
        assert mock_update.call_count == 1

@patch('builtins.input', side_effect=['X', 'W'])
def test_main_invalid_input(mock_input, mock_yf_data):
    with patch('yfinance.download', return_value=mock_yf_data), \
         patch('yahoofinance.index.datetime') as mock_datetime, \
         patch('yahoofinance.index.display_results') as mock_display, \
         patch('yahoofinance.index.update_html') as mock_update:
        
        mock_datetime.today.return_value = datetime(2024, 1, 10)  # A Wednesday
        from yahoofinance.index import main
        main()
        
        # Verify it eventually called display_results after invalid input
        assert mock_display.call_count == 1
        # Verify update_html was called
        assert mock_update.call_count == 1