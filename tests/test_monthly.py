import pytest
from unittest.mock import patch, Mock, call
import pandas as pd
from datetime import datetime, timedelta
import pytz
from yahoofinance.monthly import (
    get_last_business_day,
    get_previous_trading_day_close,
    get_previous_month_ends,
    fetch_monthly_change,
    main
)

@pytest.fixture
def mock_yf_download():
    with patch('yahoofinance.monthly.yf.download') as mock:
        yield mock

def test_get_last_business_day():
    # Test end of month
    assert get_last_business_day(2024, 2) == datetime(2024, 2, 29).date()
    # Test end of year
    assert get_last_business_day(2024, 12) == datetime(2024, 12, 31).date()

def test_get_previous_trading_day_close(mock_yf_download):
    # Create mock data with pandas Series for Close
    mock_data = pd.DataFrame({
        'Close': pd.Series([100.0, 101.0, 102.0], 
            index=[datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)])
    })
    mock_yf_download.return_value = mock_data

    # Test successful case
    test_date = datetime(2024, 1, 4).date()
    price, date = get_previous_trading_day_close('AAPL', test_date)
    assert price == 102.0
    assert date == datetime(2024, 1, 3).date()

    # Test empty data case
    mock_yf_download.return_value = pd.DataFrame()
    price, date = get_previous_trading_day_close('AAPL', test_date)
    assert price is None
    assert date is None

def test_get_previous_month_ends():
    with patch('yahoofinance.monthly.datetime') as mock_datetime:
        # Mock current time to a known date
        mock_now = datetime(2024, 3, 15, tzinfo=pytz.timezone('Europe/Athens'))
        mock_datetime.now.return_value = mock_now

        prev_prev, prev = get_previous_month_ends()
        
        # February 29, 2024 and January 31, 2024
        assert prev.strftime('%Y-%m-%d') == '2024-02-29'
        assert prev_prev.strftime('%Y-%m-%d') == '2024-01-31'

def test_fetch_monthly_change(mock_yf_download):
    # Mock data for two different dates
    mock_data1 = pd.DataFrame({
        'Close': pd.Series([100.0], index=[datetime(2024, 1, 31)])
    })
    mock_data2 = pd.DataFrame({
        'Close': pd.Series([110.0], index=[datetime(2024, 2, 29)])
    })
    
    # Need to provide enough mock data for all indices
    mock_yf_download.side_effect = [mock_data1, mock_data2] * len(['DJI30', 'SP500', 'NQ100', 'VIX'])

    start_date = datetime(2024, 1, 31).date()
    end_date = datetime(2024, 2, 29).date()
    
    results, _, _ = fetch_monthly_change(start_date, end_date)
    
    assert len(results) > 0
    first_result = results[0]
    assert 'Index' in first_result
    assert 'Change Percent' in first_result
    assert '+10.00%' in first_result['Change Percent']

def test_main():
    with patch('yahoofinance.monthly.get_previous_month_ends') as mock_ends, \
         patch('yahoofinance.monthly.fetch_monthly_change') as mock_fetch, \
         patch('yahoofinance.monthly.pd.DataFrame') as mock_df, \
         patch('yahoofinance.monthly.tabulate') as mock_tabulate, \
         patch('yahoofinance.monthly.datetime') as mock_datetime:
        
        # Mock the return values
        mock_ends.return_value = (
            datetime(2024, 1, 31).date(),
            datetime(2024, 2, 29).date()
        )
        
        test_data = [{'Index': 'DJI30', 'Previous Month': '100.00', 'This Month': '110.00', 'Change Percent': '+10.00%'}]
        mock_fetch.return_value = (test_data, None, None)
        
        df = pd.DataFrame(test_data)
        mock_df.return_value = df
        
        mock_datetime.now.return_value = datetime(2024, 3, 15, tzinfo=pytz.timezone('Europe/Athens'))
        
        # Run main function
        main()
        
        # Verify the function calls
        mock_ends.assert_called_once()
        mock_fetch.assert_called_once()
        
        # Check that DataFrame was called correctly
        mock_df.assert_has_calls([
            call(test_data, columns=['Index', 'Previous Month', 'This Month', 'Change Percent'])
        ], any_order=True)
        mock_tabulate.assert_called_once()