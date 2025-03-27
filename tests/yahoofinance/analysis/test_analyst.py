import pytest
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import Mock, patch
from yahoofinance.compat.analyst import AnalystData
from yahoofinance.core.config import POSITIVE_GRADES
from yahoofinance.core.errors import ValidationError, YFinanceError

@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def analyst_data(mock_client):
    return AnalystData(mock_client)

@pytest.fixture
def sample_ratings_df():
    return pd.DataFrame({
        'GradeDate': pd.date_range(start='2024-01-01', periods=3),
        'Firm': ['Firm A', 'Firm B', 'Firm C'],
        'FromGrade': ['Hold', 'Sell', 'Buy'],
        'ToGrade': ['Buy', 'Hold', 'Strong Buy'],
        'Action': ['up', 'up', 'up']
    }).reset_index()  # Add index column to match actual data structure

def test_validate_date_valid(analyst_data):
    # Should not raise any exception
    analyst_data._validate_date('2024-01-01')
    analyst_data._validate_date(None)

def test_validate_date_invalid(analyst_data):
    with pytest.raises(ValidationError):
        analyst_data._validate_date('invalid-date')

def test_safe_float_conversion(analyst_data):
    assert analyst_data._safe_float_conversion('123.45') == pytest.approx(123.45)
    assert analyst_data._safe_float_conversion('1,234.56') == pytest.approx(1234.56)
    assert analyst_data._safe_float_conversion(None) is None
    assert analyst_data._safe_float_conversion('invalid') is None

def test_fetch_ratings_data_success(analyst_data, mock_client, sample_ratings_df):
    # Setup mock
    mock_stock = Mock()
    mock_stock._stock = Mock()
    mock_stock._stock.upgrades_downgrades = sample_ratings_df
    mock_client.get_ticker_info.return_value = mock_stock

    # Test with specific start date
    result = analyst_data.fetch_ratings_data('AAPL', '2024-01-01')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert all(col in result.columns for col in ['GradeDate', 'Firm', 'FromGrade', 'ToGrade', 'Action'])

def test_fetch_ratings_data_no_data(analyst_data, mock_client):
    # Setup mock for empty data
    mock_stock = Mock()
    mock_stock._stock = Mock()
    mock_stock._stock.upgrades_downgrades = None
    mock_client.get_ticker_info.return_value = mock_stock

    result = analyst_data.fetch_ratings_data('AAPL')
    assert result is None

def test_fetch_ratings_data_error(analyst_data, mock_client):
    mock_client.get_ticker_info.side_effect = Exception("API Error")
    
    with pytest.raises(YFinanceError):
        analyst_data.fetch_ratings_data('AAPL')

def test_get_ratings_summary_success(analyst_data, mock_client, sample_ratings_df):
    # Setup mocks
    mock_stock = Mock()
    mock_stock._stock = Mock()
    mock_stock._stock.upgrades_downgrades = sample_ratings_df
    mock_stock.last_earnings = '2024-01-01'
    mock_client.get_ticker_info.return_value = mock_stock

    # Test with earnings date
    result = analyst_data.get_ratings_summary('AAPL', use_earnings_date=True)
    assert isinstance(result, dict)
    assert 'positive_percentage' in result
    assert 'total_ratings' in result
    assert 'ratings_type' in result
    assert result['total_ratings'] == 3
    assert result['ratings_type'] == 'E'

    # Test with specific start date
    result = analyst_data.get_ratings_summary('AAPL', '2024-01-01', use_earnings_date=False)
    assert isinstance(result, dict)
    assert result['total_ratings'] == 3
    assert result['ratings_type'] == 'E'

def test_get_ratings_summary_fallback(analyst_data, mock_client, sample_ratings_df):
    # Setup mocks for no post-earnings data
    mock_stock = Mock()
    mock_stock._stock = Mock()
    # First call returns empty DataFrame (no post-earnings data)
    # Second call returns sample data (all-time data)
    mock_stock._stock.upgrades_downgrades = pd.DataFrame()
    mock_stock.last_earnings = '2024-01-01'
    mock_client.get_ticker_info.return_value = mock_stock

    # Test fallback to all-time data
    with patch.object(analyst_data, 'fetch_ratings_data') as mock_fetch:
        mock_fetch.side_effect = [None, sample_ratings_df]  # First None for earnings, then data for all-time
        result = analyst_data.get_ratings_summary('AAPL', use_earnings_date=True)
        
        assert isinstance(result, dict)
        assert result['total_ratings'] == 3
        assert result['ratings_type'] == 'A'  # Should indicate all-time ratings

def test_get_ratings_summary_no_data(analyst_data, mock_client):
    # Setup mock for no data
    mock_stock = Mock()
    mock_stock._stock = Mock()
    mock_stock._stock.upgrades_downgrades = None
    mock_stock.last_earnings = None
    mock_client.get_ticker_info.return_value = mock_stock

    result = analyst_data.get_ratings_summary('AAPL')
    assert result['positive_percentage'] is None
    assert result['total_ratings'] is None
    assert result['ratings_type'] is None

def test_get_recent_changes_success(analyst_data, mock_client, sample_ratings_df):
    # Setup mock with data within the last 30 days
    today = datetime.now()
    recent_dates = pd.date_range(end=today, periods=3)
    
    df = pd.DataFrame({
        'GradeDate': recent_dates,
        'Firm': ['Firm A', 'Firm B', 'Firm C'],
        'FromGrade': ['Hold', 'Sell', 'Buy'],
        'ToGrade': ['Buy', 'Hold', 'Strong Buy'],
        'Action': ['up', 'up', 'up']
    })
    
    mock_stock = Mock()
    mock_stock._stock = Mock()
    mock_stock._stock.upgrades_downgrades = df
    mock_client.get_ticker_info.return_value = mock_stock

    result = analyst_data.get_recent_changes('AAPL', days=30)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(change, dict) for change in result)
    assert all(key in result[0] for key in ['date', 'firm', 'from_grade', 'to_grade', 'action'])

def test_get_recent_changes_invalid_days(analyst_data):
    with pytest.raises(ValidationError):
        analyst_data.get_recent_changes('AAPL', days=0)
    
    with pytest.raises(ValidationError):
        analyst_data.get_recent_changes('AAPL', days=-1)

def test_get_recent_changes_no_data(analyst_data, mock_client):
    # Setup mock for no data
    mock_stock = Mock()
    mock_stock._stock = Mock()
    mock_stock._stock.upgrades_downgrades = None
    mock_client.get_ticker_info.return_value = mock_stock

    result = analyst_data.get_recent_changes('AAPL')
    assert isinstance(result, list)
    assert len(result) == 0

def test_positive_grades_constant():
    assert isinstance(POSITIVE_GRADES, set)
    assert len(POSITIVE_GRADES) > 0
    assert all(isinstance(grade, str) for grade in POSITIVE_GRADES)
    assert "Buy" in POSITIVE_GRADES
    assert "Strong Buy" in POSITIVE_GRADES