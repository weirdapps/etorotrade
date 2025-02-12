import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd
from yahoofinance.pricing import (
    PricingAnalyzer,
    PriceTarget,
    PriceData,
    YFinanceError,
    ValidationError
)

@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def analyzer(mock_client):
    return PricingAnalyzer(mock_client)

@pytest.fixture
def mock_stock_info():
    info = Mock()
    info.current_price = 100.0
    info.target_price = 120.0
    info.analyst_count = 10
    info._stock = Mock()
    return info

def test_init(mock_client):
    analyzer = PricingAnalyzer(mock_client)
    assert analyzer.client == mock_client

def test_safe_float_conversion(analyzer):
    assert analyzer._safe_float_conversion("123.45") == pytest.approx(123.45)
    assert analyzer._safe_float_conversion("1,234.56") == pytest.approx(1234.56)
    assert analyzer._safe_float_conversion(None) is None
    assert analyzer._safe_float_conversion("invalid") is None
    assert analyzer._safe_float_conversion(100) == pytest.approx(100.0)

def test_get_current_price_success(analyzer, mock_client, mock_stock_info):
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    price = analyzer.get_current_price("AAPL")
    assert price == pytest.approx(100.0)
    mock_client.get_ticker_info.assert_called_once_with("AAPL")

def test_get_current_price_none(analyzer, mock_client):
    mock_stock_info = Mock()
    mock_stock_info.current_price = None
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    price = analyzer.get_current_price("AAPL")
    assert price is None

def test_get_current_price_error(analyzer, mock_client):
    mock_client.get_ticker_info.side_effect = Exception("API Error")
    
    with pytest.raises(YFinanceError):
        analyzer.get_current_price("AAPL")

def test_get_historical_prices_success(analyzer, mock_client):
    # Create sample historical data
    hist_data = pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 98.0],
        "Close": [101.0, 102.0],
        "Volume": [1000000, 1100000]
    }, index=[datetime(2024, 1, 1), datetime(2024, 1, 2)])
    
    mock_stock = Mock()
    mock_stock.history.return_value = hist_data
    mock_stock_info = Mock()
    mock_stock_info._stock = mock_stock
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    prices = analyzer.get_historical_prices("AAPL", "1mo")
    
    assert len(prices) == 2
    assert isinstance(prices[0], PriceData)
    assert prices[0].date == datetime(2024, 1, 1)
    assert prices[0].open == pytest.approx(100.0)
    assert prices[0].close == pytest.approx(101.0)

def test_get_historical_prices_empty(analyzer, mock_client):
    mock_stock = Mock()
    mock_stock.history.return_value = pd.DataFrame()
    mock_stock_info = Mock()
    mock_stock_info._stock = mock_stock
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    prices = analyzer.get_historical_prices("AAPL")
    assert len(prices) == 0

def test_get_historical_prices_invalid_period(analyzer):
    with pytest.raises(ValidationError):
        analyzer.get_historical_prices("AAPL", "invalid")

def test_get_historical_prices_error(analyzer, mock_client):
    mock_client.get_ticker_info.side_effect = Exception("API Error")
    
    with pytest.raises(YFinanceError):
        analyzer.get_historical_prices("AAPL")

def test_get_price_targets_success(analyzer, mock_client, mock_stock_info):
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    targets = analyzer.get_price_targets("AAPL")
    
    assert isinstance(targets, PriceTarget)
    assert targets.mean == pytest.approx(120.0)
    assert targets.num_analysts == 10
    assert targets.high is None  # Not implemented yet
    assert targets.low is None   # Not implemented yet

def test_get_price_targets_no_data(analyzer, mock_client):
    mock_stock_info = Mock()
    mock_stock_info.target_price = None
    mock_stock_info.analyst_count = None
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    targets = analyzer.get_price_targets("AAPL")
    
    assert isinstance(targets, PriceTarget)
    assert targets.mean is None
    assert targets.num_analysts == 0

def test_get_price_targets_error(analyzer, mock_client):
    mock_client.get_ticker_info.side_effect = Exception("API Error")
    
    with pytest.raises(YFinanceError):
        analyzer.get_price_targets("AAPL")

def test_calculate_price_metrics_success(analyzer, mock_client, mock_stock_info):
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    metrics = analyzer.calculate_price_metrics("AAPL")
    
    assert metrics["current_price"] == pytest.approx(100.0)
    assert metrics["target_price"] == pytest.approx(120.0)
    assert metrics["upside_potential"] == pytest.approx(20.0, rel=1e-9)  # Use approx for floating point comparison

def test_calculate_price_metrics_no_current_price(analyzer, mock_client):
    mock_stock_info = Mock()
    mock_stock_info.current_price = None
    mock_stock_info.target_price = 120.0
    mock_stock_info.analyst_count = 10
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    metrics = analyzer.calculate_price_metrics("AAPL")
    
    assert metrics["current_price"] is None
    assert metrics["target_price"] == pytest.approx(120.0)
    assert metrics["upside_potential"] is None

def test_calculate_price_metrics_no_target(analyzer, mock_client):
    mock_stock_info = Mock()
    mock_stock_info.current_price = 100.0
    mock_stock_info.target_price = None
    mock_stock_info.analyst_count = 0
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    metrics = analyzer.calculate_price_metrics("AAPL")
    
    assert metrics["current_price"] == pytest.approx(100.0)
    assert metrics["target_price"] is None
    assert metrics["upside_potential"] is None

def test_calculate_price_metrics_zero_price(analyzer, mock_client):
    mock_stock_info = Mock()
    mock_stock_info.current_price = 0.0
    mock_stock_info.target_price = 120.0
    mock_stock_info.analyst_count = 10
    mock_client.get_ticker_info.return_value = mock_stock_info
    
    metrics = analyzer.calculate_price_metrics("AAPL")
    
    assert metrics["current_price"] == pytest.approx(0.0)
    assert metrics["target_price"] == 120.0
    assert metrics["upside_potential"] is None  # Division by zero handled

def test_calculate_price_metrics_error(analyzer, mock_client):
    mock_client.get_ticker_info.side_effect = Exception("API Error")
    
    with pytest.raises(YFinanceError):
        analyzer.calculate_price_metrics("AAPL")

def test_price_target_namedtuple():
    target = PriceTarget(mean=100.0, high=120.0, low=80.0, num_analysts=10)
    assert target.mean == pytest.approx(100.0)
    assert target.high == pytest.approx(120.0)
    assert target.low == pytest.approx(80.0)
    assert target.num_analysts == 10

def test_price_data_namedtuple():
    data = PriceData(
        date=datetime(2024, 1, 1),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000000,
        adjusted_close=102.0
    )
    assert data.date == datetime(2024, 1, 1)
    assert data.open == pytest.approx(100.0)
    assert data.high == pytest.approx(105.0)
    assert data.low == pytest.approx(95.0)
    assert data.close == pytest.approx(102.0)
    assert data.volume == 1000000  # Integer, no need for approx
    assert data.adjusted_close == pytest.approx(102.0)