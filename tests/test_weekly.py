import pytest
from datetime import datetime, timedelta, date
import pandas as pd
from yahoofinance.weekly import (
    calculate_dates,
    get_previous_trading_day_close,
    fetch_weekly_change,
    update_html
)

def test_calculate_dates():
    # Test that the function returns two dates
    last_friday, previous_friday = calculate_dates()
    
    # Check that both dates are datetime objects
    assert isinstance(last_friday, datetime)
    assert isinstance(previous_friday, datetime)
    
    # Check that previous_friday is 7 days before last_friday
    assert (last_friday - previous_friday).days == 7
    
    # Check that both dates are Fridays
    assert last_friday.weekday() == 4  # Friday is 4 in Python's weekday()
    assert previous_friday.weekday() == 4

def test_get_previous_trading_day_close(mocker):
    # Mock yfinance download function
    mock_data = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0],
        'Date': pd.date_range(start='2024-01-01', periods=3)
    }).set_index('Date')
    
    mocker.patch('yfinance.download', return_value=mock_data)
    
    # Test the function
    price, actual_date = get_previous_trading_day_close('^GSPC', datetime.now())
    
    # Check that we get a float price and a date
    assert isinstance(price, float)
    assert isinstance(actual_date, date)
    assert price == pytest.approx(102.0)

def test_fetch_weekly_change(mocker):
    # Mock get_previous_trading_day_close
    def mock_trading_day_close(ticker, date):
        if date == last_friday:
            return 110.0, date.date()
        return 100.0, date.date()
    
    last_friday = datetime.now()
    previous_friday = last_friday - timedelta(days=7)
    
    mocker.patch(
        'yahoofinance.weekly.get_previous_trading_day_close',
        side_effect=mock_trading_day_close
    )
    
    # Test the function
    results = fetch_weekly_change(last_friday, previous_friday)
    
    # Check results
    assert isinstance(results, list)
    assert len(results) > 0
    assert 'Index' in results[0]
    assert 'Change Percent' in results[0]
    
    # Check that the change percentage is correct
    # 110 is 10% higher than 100
    assert '+10.00%' in results[0]['Change Percent']

def test_update_html(tmp_path):
    # Create a temporary HTML file
    html_content = """
    <html>
    <body>
        <div id="DJI30">old value</div>
        <div id="SP500">old value</div>
    </body>
    </html>
    """
    html_file = tmp_path / "test.html"
    html_file.write_text(html_content)
    
    # Test data
    test_data = [
        {
            'Index': 'DJI30',
            'Change Percent': '+5.00%'
        },
        {
            'Index': 'SP500',
            'Change Percent': '+3.00%'
        }
    ]
    
    # Update the HTML
    update_html(test_data, str(html_file))
    
    # Read the updated file
    updated_content = html_file.read_text()
    
    # Check that values were updated
    assert '>+5.00%<' in updated_content
    assert '>+3.00%<' in updated_content