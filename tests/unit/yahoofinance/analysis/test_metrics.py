"""
Tests for yahoofinance/analysis/metrics.py

This module tests the PriceData, PriceTarget dataclasses and PricingAnalyzer class.
"""

import pytest
from unittest.mock import MagicMock

from yahoofinance.analysis.metrics import (
    PriceData,
    PriceTarget,
    PricingAnalyzer,
)


class TestPriceData:
    """Tests for PriceData dataclass."""

    def test_default_values(self):
        """Test default values are None."""
        data = PriceData()
        assert data.price is None
        assert data.change is None
        assert data.change_percent is None
        assert data.volume is None
        assert data.average_volume is None
        assert data.volume_ratio is None
        assert data.high_52week is None
        assert data.low_52week is None
        assert data.from_high is None
        assert data.from_low is None

    def test_with_values(self):
        """Test initialization with values."""
        data = PriceData(
            price=175.50,
            change=2.50,
            change_percent=1.45,
            volume=50000000,
            average_volume=40000000,
            volume_ratio=1.25,
            high_52week=200.0,
            low_52week=150.0,
            from_high=-12.25,
            from_low=17.0,
        )

        assert data.price == 175.50
        assert data.change == 2.50
        assert data.change_percent == 1.45
        assert data.volume == 50000000
        assert data.average_volume == 40000000
        assert data.volume_ratio == 1.25
        assert data.high_52week == 200.0
        assert data.low_52week == 150.0
        assert data.from_high == -12.25
        assert data.from_low == 17.0

    def test_partial_initialization(self):
        """Test partial initialization."""
        data = PriceData(price=175.50, volume=50000000)

        assert data.price == 175.50
        assert data.volume == 50000000
        assert data.change is None


class TestPriceTarget:
    """Tests for PriceTarget dataclass."""

    def test_default_values(self):
        """Test default values are None."""
        target = PriceTarget()
        assert target.average is None
        assert target.median is None
        assert target.high is None
        assert target.low is None
        assert target.upside is None
        assert target.analyst_count is None

    def test_with_values(self):
        """Test initialization with values."""
        target = PriceTarget(
            average=200.0,
            median=195.0,
            high=250.0,
            low=180.0,
            upside=14.0,
            analyst_count=25,
        )

        assert target.average == 200.0
        assert target.median == 195.0
        assert target.high == 250.0
        assert target.low == 180.0
        assert target.upside == 14.0
        assert target.analyst_count == 25

    def test_partial_initialization(self):
        """Test partial initialization."""
        target = PriceTarget(average=200.0, upside=14.0)

        assert target.average == 200.0
        assert target.upside == 14.0
        assert target.median is None


class MockSyncProvider:
    """Mock synchronous provider."""

    def get_ticker_info(self, ticker: str):
        """Synchronous get_ticker_info."""
        return {
            "price": 175.0,
            "change": 2.50,
            "change_percent": 1.45,
            "volume": 50000000,
            "average_volume": 40000000,
            "high_52week": 200.0,
            "low_52week": 150.0,
            "target_price": 200.0,
            "target_median": 195.0,
            "target_high": 250.0,
            "target_low": 180.0,
            "upside": 14.0,
            "analyst_count": 25,
        }


class MockAsyncProvider:
    """Mock asynchronous provider that satisfies __await__ check."""

    class AwaitableMethod:
        """Method that has __await__ attribute and is callable."""

        def __await__(self):
            """Make this awaitable."""
            async def inner():
                return {"price": 175.0, "change": 2.50, "target_price": 200.0}
            return inner().__await__()

        def __call__(self, ticker: str):
            """Make this callable, returns coroutine."""
            async def inner():
                return {"price": 175.0, "change": 2.50, "target_price": 200.0}
            return inner()

    def __init__(self):
        """Initialize with awaitable method."""
        self.get_ticker_info = self.AwaitableMethod()


class TestPricingAnalyzerInit:
    """Tests for PricingAnalyzer initialization."""

    def test_init_with_sync_provider(self):
        """Test initialization with synchronous provider."""
        provider = MockSyncProvider()
        analyzer = PricingAnalyzer(provider=provider)

        assert analyzer.provider is provider
        assert analyzer.is_async is False

    def test_init_with_async_provider(self):
        """Test initialization with asynchronous provider."""
        provider = MockAsyncProvider()
        analyzer = PricingAnalyzer(provider=provider)

        assert analyzer.provider is provider
        assert analyzer.is_async is True

    def test_init_with_none_creates_default_provider(self):
        """Test initialization with None creates default provider."""
        analyzer = PricingAnalyzer(provider=None)

        assert analyzer.provider is not None


class TestPricingAnalyzerGetPriceData:
    """Tests for PricingAnalyzer.get_price_data method."""

    def test_get_price_data_returns_price_data(self):
        """Test get_price_data returns PriceData object."""
        provider = MockSyncProvider()
        analyzer = PricingAnalyzer(provider=provider)

        result = analyzer.get_price_data("AAPL")

        assert isinstance(result, PriceData)
        assert result.price == 175.0


class TestDataclassEquality:
    """Tests for dataclass equality."""

    def test_price_data_equality(self):
        """Test PriceData equality."""
        data1 = PriceData(price=175.0, volume=50000000)
        data2 = PriceData(price=175.0, volume=50000000)

        assert data1 == data2

    def test_price_data_inequality(self):
        """Test PriceData inequality."""
        data1 = PriceData(price=175.0)
        data2 = PriceData(price=180.0)

        assert data1 != data2

    def test_price_target_equality(self):
        """Test PriceTarget equality."""
        target1 = PriceTarget(average=200.0, upside=14.0)
        target2 = PriceTarget(average=200.0, upside=14.0)

        assert target1 == target2
