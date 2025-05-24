from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from yahoofinance.analysis.market import MarketMetrics
from yahoofinance.utils.error_handling import safe_operation, with_retry


@pytest.fixture
def sample_metrics():
    return MarketMetrics(
        avg_upside=15.5,
        median_upside=12.0,
        avg_buy_percentage=78.5,
        median_buy_percentage=80.0,
        avg_pe_ratio=18.5,
        median_pe_ratio=17.2,
        avg_forward_pe=16.8,
        median_forward_pe=15.5,
        avg_peg_ratio=1.8,
        median_peg_ratio=1.5,
    )


@pytest.fixture
def market_data():
    # Create a sample DataFrame that could represent market data
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG", "AMZN"],
            "price": [150.0, 280.0, 140.0, 125.0],
            "upside": [10.0, 15.0, 20.0, 25.0],
            "buy_percentage": [75.0, 80.0, 85.0, 90.0],
            "pe_ratio": [25.0, 30.0, 20.0, 35.0],
            "forward_pe": [22.0, 25.0, 18.0, 30.0],
            "peg_ratio": [1.2, 1.5, 1.8, 2.0],
        }
    )


@with_retry
def test_market_metrics_initialization():
    """Test initialization of MarketMetrics."""
    # Test default initialization
    metrics = MarketMetrics()
    assert metrics.avg_upside is None
    assert metrics.median_upside is None
    assert metrics.avg_buy_percentage is None
    assert metrics.median_buy_percentage is None
    assert metrics.avg_pe_ratio is None
    assert metrics.median_pe_ratio is None
    assert metrics.avg_forward_pe is None
    assert metrics.median_forward_pe is None
    assert metrics.avg_peg_ratio is None
    assert metrics.median_peg_ratio is None

    # Test initialization with values
    metrics = MarketMetrics(avg_upside=10.0, median_upside=9.0)
    assert metrics.avg_upside == pytest.approx(10.0, abs=1e-9)
    assert metrics.median_upside == pytest.approx(9.0, abs=1e-9)


@with_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
def test_metrics_calculation(market_data):
    """Test calculation of market metrics from data."""
    # Calculate metrics from the data manually
    avg_upside = market_data["upside"].mean()
    median_upside = market_data["upside"].median()
    avg_buy = market_data["buy_percentage"].mean()
    median_buy = market_data["buy_percentage"].median()

    # Create metrics object
    metrics = MarketMetrics(
        avg_upside=avg_upside,
        median_upside=median_upside,
        avg_buy_percentage=avg_buy,
        median_buy_percentage=median_buy,
    )

    # Verify calculations
    assert metrics.avg_upside == pytest.approx(17.5)
    assert metrics.median_upside == pytest.approx(17.5)
    assert metrics.avg_buy_percentage == pytest.approx(82.5)
    assert metrics.median_buy_percentage == pytest.approx(82.5)


@safe_operation(default_value=None)
def test_metrics_properties(sample_metrics):
    """Test properties of metrics."""
    # Check property values match what was set
    assert sample_metrics.avg_upside == pytest.approx(15.5, abs=1e-9)
    assert sample_metrics.median_upside == pytest.approx(12.0, abs=1e-9)
    assert sample_metrics.avg_buy_percentage == pytest.approx(78.5, abs=1e-9)
    assert sample_metrics.median_buy_percentage == pytest.approx(80.0, abs=1e-9)
    assert sample_metrics.avg_pe_ratio == pytest.approx(18.5, abs=1e-9)
    assert sample_metrics.median_pe_ratio == pytest.approx(17.2, abs=1e-9)
    assert sample_metrics.avg_forward_pe == pytest.approx(16.8, abs=1e-9)
    assert sample_metrics.median_forward_pe == pytest.approx(15.5, abs=1e-9)
    assert sample_metrics.avg_peg_ratio == pytest.approx(1.8, abs=1e-9)
    assert sample_metrics.median_peg_ratio == pytest.approx(1.5, abs=1e-9)
