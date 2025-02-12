import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
from yahoofinance.insiders import InsiderAnalyzer, START_DATE_COL

@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def analyzer(mock_client):
    return InsiderAnalyzer(mock_client)

@pytest.fixture
def sample_insider_df():
    return pd.DataFrame({
        START_DATE_COL: [
            '2024-01-15',
            '2024-01-16',
            '2024-01-17',
            '2024-01-18'
        ],
        'Text': [
            'Purchase at price $100',
            'Sale at price $105',
            'Purchase at price $102',
            'Sale at price $107'
        ]
    })

@pytest.fixture
def mock_stock_info():
    info = Mock()
    info.previous_earnings = '2024-01-01'
    info._stock = Mock()
    return info

def test_init(mock_client):
    analyzer = InsiderAnalyzer(mock_client)
    assert analyzer.client == mock_client

def test_get_insider_metrics_no_earnings(analyzer, mock_client):
    # Setup mock with no earnings data
    mock_stock_info = Mock()
    mock_stock_info.previous_earnings = None
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] is None
    assert result['transaction_count'] is None
    mock_client.get_ticker_info.assert_called_once_with('AAPL', skip_insider_metrics=True)

def test_get_insider_metrics_no_transactions(analyzer, mock_client):
    # Setup mock with earnings but no transactions
    mock_stock_info = Mock()
    mock_stock_info.previous_earnings = '2024-01-01'
    mock_stock_info._stock = Mock()
    mock_stock_info._stock.insider_transactions = None
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] is None
    assert result['transaction_count'] is None

def test_get_insider_metrics_empty_transactions(analyzer, mock_client):
    # Setup mock with empty transactions DataFrame
    mock_stock_info = Mock()
    mock_stock_info.previous_earnings = '2024-01-01'
    mock_stock_info._stock = Mock()
    mock_stock_info._stock.insider_transactions = pd.DataFrame()
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] is None
    assert result['transaction_count'] is None

def test_get_insider_metrics_no_recent_transactions(analyzer, mock_client, sample_insider_df):
    # Setup mock with old transactions
    mock_stock_info = Mock()
    mock_stock_info.previous_earnings = '2024-02-01'  # After all sample transactions
    mock_stock_info._stock = Mock()
    mock_stock_info._stock.insider_transactions = sample_insider_df
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] is None
    assert result['transaction_count'] is None

def test_get_insider_metrics_success(analyzer, mock_client, sample_insider_df):
    # Setup mock with valid transactions
    mock_stock_info = Mock()
    mock_stock_info.previous_earnings = '2024-01-01'  # Before sample transactions
    mock_stock_info._stock = Mock()
    mock_stock_info._stock.insider_transactions = sample_insider_df
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] == pytest.approx(50.0)  # 2 purchases out of 4 transactions
    assert result['transaction_count'] == 4

def test_get_insider_metrics_only_purchases(analyzer, mock_client):
    # Create DataFrame with only purchases
    df = pd.DataFrame({
        START_DATE_COL: ['2024-01-15', '2024-01-16'],
        'Text': [
            'Purchase at price $100',
            'Purchase at price $102'
        ]
    })
    
    mock_stock_info = Mock()
    mock_stock_info.previous_earnings = '2024-01-01'
    mock_stock_info._stock = Mock()
    mock_stock_info._stock.insider_transactions = df
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] == pytest.approx(100.0)
    assert result['transaction_count'] == 2

def test_get_insider_metrics_only_sales(analyzer, mock_client):
    # Create DataFrame with only sales
    df = pd.DataFrame({
        START_DATE_COL: ['2024-01-15', '2024-01-16'],
        'Text': [
            'Sale at price $100',
            'Sale at price $102'
        ]
    })
    
    mock_stock_info = Mock()
    mock_stock_info.previous_earnings = '2024-01-01'
    mock_stock_info._stock = Mock()
    mock_stock_info._stock.insider_transactions = df
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] == pytest.approx(0.0)
    assert result['transaction_count'] == 2

def test_get_insider_metrics_no_valid_transactions(analyzer, mock_client):
    # Create DataFrame with no valid purchase/sale transactions
    df = pd.DataFrame({
        START_DATE_COL: ['2024-01-15', '2024-01-16'],
        'Text': [
            'Other transaction',
            'Another transaction'
        ]
    })
    
    mock_stock_info = Mock()
    mock_stock_info.previous_earnings = '2024-01-01'
    mock_stock_info._stock = Mock()
    mock_stock_info._stock.insider_transactions = df
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] is None
    assert result['transaction_count'] is None

def test_get_insider_metrics_error(analyzer, mock_client):
    # Setup mock to raise an exception
    mock_client.get_ticker_info.side_effect = Exception("Test error")
    
    result = analyzer.get_insider_metrics('AAPL')
    
    assert result['insider_buy_pct'] is None
    assert result['transaction_count'] is None