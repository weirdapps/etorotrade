import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from yahoofinance.analysis.metrics import PriceData, PriceTarget, PricingAnalyzer
from yahoofinance.core.errors import ValidationError, YFinanceError


@pytest.fixture
def mock_provider():
    provider = Mock()
    return provider


@pytest.fixture
def analyzer(mock_provider):
    return PricingAnalyzer(provider=mock_provider)


@pytest.fixture
def sample_ticker_info():
    """Create sample ticker info for testing."""
    return {
        "price": 100.0,
        "target_price": 120.0,
        "highest_target_price": 140.0,
        "lowest_target_price": 90.0,
        "median_target_price": 115.0,
        "upside": 20.0,
        "analyst_count": 10,
        "symbol": "AAPL",
        "change": 2.5,
        "change_percent": 2.5,
        "volume": 1000000,
        "average_volume": 1200000,
        "high_52week": 150.0,
        "low_52week": 80.0,
        "from_high": -33.3,
        "from_low": 25.0,
    }


def test_init(mock_provider):
    """Test that the analyzer initializes correctly."""
    analyzer = PricingAnalyzer(provider=mock_provider)
    assert analyzer.provider == mock_provider


def test_get_price_data(analyzer, mock_provider, sample_ticker_info):
    """Test getting price data."""
    # Set mock provider to return sample data
    mock_provider.get_ticker_info.return_value = sample_ticker_info

    # Call method
    price_data = analyzer.get_price_data("AAPL")

    # Verify result
    assert price_data.price == pytest.approx(100.0, abs=1e-9)
    assert price_data.change == pytest.approx(2.5, abs=1e-9)
    assert price_data.change_percent == pytest.approx(2.5, abs=1e-9)
    assert price_data.volume == 1000000
    assert price_data.average_volume == 1200000
    assert price_data.volume_ratio == pytest.approx(1000000 / 1200000, abs=1e-9)
    assert price_data.high_52week == pytest.approx(150.0, abs=1e-9)
    assert price_data.low_52week == pytest.approx(80.0, abs=1e-9)
    assert price_data.from_high == pytest.approx(-33.3, abs=1e-9)
    assert price_data.from_low == pytest.approx(25.0, abs=1e-9)

    # Verify provider was called correctly
    mock_provider.get_ticker_info.assert_called_once_with("AAPL")


def test_get_price_data_error(analyzer, mock_provider):
    """Test handling errors when getting price data."""
    # Set mock provider to raise a YFinanceError
    mock_provider.get_ticker_info.side_effect = YFinanceError("API Error")

    # Call method - should not raise but return empty object
    price_data = analyzer.get_price_data("AAPL")

    # Verify result is an empty PriceData object
    assert isinstance(price_data, PriceData)
    assert price_data.price is None
    assert price_data.change is None

    # Verify provider was called correctly
    mock_provider.get_ticker_info.assert_called_once_with("AAPL")


@pytest.mark.asyncio
async def test_get_price_data_async(analyzer, mock_provider, sample_ticker_info):
    """Test getting price data asynchronously."""
    # Set analyzer to use async provider
    analyzer.is_async = True

    # Set mock provider to return sample data
    mock_provider.get_ticker_info = Mock()
    mock_provider.get_ticker_info.return_value = asyncio.Future()
    mock_provider.get_ticker_info.return_value.set_result(sample_ticker_info)

    # Call method
    price_data = await analyzer.get_price_data_async("AAPL")

    # Verify result
    assert price_data.price == pytest.approx(100.0, abs=1e-9)
    assert price_data.change == pytest.approx(2.5, abs=1e-9)
    assert price_data.volume == 1000000

    # Verify provider was called correctly
    mock_provider.get_ticker_info.assert_called_once_with("AAPL")


def test_get_price_target(analyzer, mock_provider, sample_ticker_info):
    """Test getting price target data."""
    # Set mock provider to return sample data
    mock_provider.get_ticker_info.return_value = sample_ticker_info

    # Call method
    price_target = analyzer.get_price_target("AAPL")

    # Verify result
    assert price_target.average == pytest.approx(120.0, abs=1e-9)
    assert price_target.median == pytest.approx(115.0, abs=1e-9)
    assert price_target.high == pytest.approx(140.0, abs=1e-9)
    assert price_target.low == pytest.approx(90.0, abs=1e-9)
    assert price_target.upside == pytest.approx(20.0, abs=1e-9)
    assert price_target.analyst_count == 10

    # Verify provider was called correctly
    mock_provider.get_ticker_info.assert_called_once_with("AAPL")


def test_get_price_target_missing_data(analyzer, mock_provider):
    """Test getting price target with missing data."""
    # Set mock provider to return data with missing values
    ticker_info = {"price": 100.0, "analyst_count": 0}
    mock_provider.get_ticker_info.return_value = ticker_info

    # Call method
    price_target = analyzer.get_price_target("AAPL")

    # Verify result
    assert price_target.average is None
    assert price_target.median is None
    assert price_target.high is None
    assert price_target.low is None
    assert price_target.upside is None
    assert price_target.analyst_count == 0


def test_get_price_target_error(analyzer, mock_provider):
    """Test handling errors when getting price target."""
    # Set mock provider to raise a YFinanceError
    mock_provider.get_ticker_info.side_effect = YFinanceError("API Error")

    # Call method - should not raise but return empty object
    price_target = analyzer.get_price_target("AAPL")

    # Verify result is an empty PriceTarget object
    assert isinstance(price_target, PriceTarget)
    assert price_target.average is None
    assert price_target.median is None


@pytest.mark.asyncio
async def test_get_price_target_async(analyzer, mock_provider, sample_ticker_info):
    """Test getting price target asynchronously."""
    # Set analyzer to use async provider
    analyzer.is_async = True

    # Set mock provider to return sample data
    mock_provider.get_ticker_info = Mock()
    mock_provider.get_ticker_info.return_value = asyncio.Future()
    mock_provider.get_ticker_info.return_value.set_result(sample_ticker_info)

    # Call method
    price_target = await analyzer.get_price_target_async("AAPL")

    # Verify result
    assert price_target.average == pytest.approx(120.0, abs=1e-9)
    assert price_target.median == pytest.approx(115.0, abs=1e-9)
    assert price_target.analyst_count == 10

    # Verify provider was called correctly
    mock_provider.get_ticker_info.assert_called_once_with("AAPL")


def test_get_all_metrics(analyzer, mock_provider, sample_ticker_info):
    """Test getting all metrics."""
    # Set mock provider to return sample data
    mock_provider.get_ticker_info.return_value = sample_ticker_info

    # Call method
    metrics = analyzer.get_all_metrics("AAPL")

    # Verify result
    assert metrics["price"] == pytest.approx(100.0, abs=1e-9)
    assert metrics["target_price"] == pytest.approx(120.0, abs=1e-9)
    assert metrics["analyst_count"] == 10
    assert metrics["high_52week"] == pytest.approx(150.0, abs=1e-9)

    # Verify provider was called correctly
    mock_provider.get_ticker_info.assert_called_once_with("AAPL")


def test_get_all_metrics_error(analyzer, mock_provider):
    """Test handling errors when getting all metrics."""
    # Set mock provider to raise a YFinanceError
    mock_provider.get_ticker_info.side_effect = YFinanceError("API Error")

    # Call method - should not raise but return empty dict
    metrics = analyzer.get_all_metrics("AAPL")

    # Verify result is an empty dict
    assert isinstance(metrics, dict)
    assert len(metrics) == 0

    # Verify provider was called correctly
    mock_provider.get_ticker_info.assert_called_once_with("AAPL")


@pytest.mark.asyncio
async def test_get_all_metrics_async(analyzer, mock_provider, sample_ticker_info):
    """Test getting all metrics asynchronously."""
    # Set analyzer to use async provider
    analyzer.is_async = True

    # Set mock provider to return sample data
    mock_provider.get_ticker_info = Mock()
    mock_provider.get_ticker_info.return_value = asyncio.Future()
    mock_provider.get_ticker_info.return_value.set_result(sample_ticker_info)

    # Call method
    metrics = await analyzer.get_all_metrics_async("AAPL")

    # Verify result
    assert metrics["price"] == pytest.approx(100.0, abs=1e-9)
    assert metrics["target_price"] == pytest.approx(120.0, abs=1e-9)

    # Verify provider was called correctly
    mock_provider.get_ticker_info.assert_called_once_with("AAPL")


def test_price_data_namedtuple():
    """Test that PriceData class works correctly."""
    data = PriceData(
        price=100.0,
        change=2.5,
        change_percent=2.5,
        volume=1000000,
        average_volume=1200000,
        volume_ratio=0.83,
        high_52week=150.0,
        low_52week=80.0,
        from_high=-33.3,
        from_low=25.0,
    )
    assert data.price == pytest.approx(100.0, abs=1e-9)
    assert data.change == pytest.approx(2.5, abs=1e-9)
    assert data.change_percent == pytest.approx(2.5, abs=1e-9)
    assert data.volume == 1000000
    assert data.average_volume == 1200000
    assert data.volume_ratio == pytest.approx(0.83, abs=1e-9)
    assert data.high_52week == pytest.approx(150.0, abs=1e-9)
    assert data.low_52week == pytest.approx(80.0, abs=1e-9)
    assert data.from_high == pytest.approx(-33.3, abs=1e-9)
    assert data.from_low == pytest.approx(25.0, abs=1e-9)


def test_price_target_namedtuple():
    """Test that PriceTarget class works correctly."""
    target = PriceTarget(
        average=120.0, median=115.0, high=140.0, low=90.0, upside=20.0, analyst_count=10
    )
    assert target.average == pytest.approx(120.0, abs=1e-9)
    assert target.median == pytest.approx(115.0, abs=1e-9)
    assert target.high == pytest.approx(140.0, abs=1e-9)
    assert target.low == pytest.approx(90.0, abs=1e-9)
    assert target.upside == pytest.approx(20.0, abs=1e-9)
    assert target.analyst_count == 10
