#!/usr/bin/env python3
"""
ITERATION 10: Core Types Tests
Target: Test StockData dataclass and type conversions
"""

import pytest
from yahoofinance.core.types import StockData


class TestStockDataInitialization:
    """Test StockData initialization."""

    def test_create_empty_stock_data(self):
        """Create StockData with default values."""
        stock = StockData()
        assert stock.name == "N/A"
        assert stock.sector == "N/A"
        assert stock.recommendation_key == "N/A"
        assert stock.market_cap is None
        assert stock.current_price is None

    def test_create_stock_data_with_basic_info(self):
        """Create StockData with basic information."""
        stock = StockData(
            name="Apple Inc.",
            sector="Technology",
            market_cap=3000000000000,
            current_price=150.25,
        )
        assert stock.name == "Apple Inc."
        assert stock.sector == "Technology"
        assert stock.market_cap == 3000000000000
        assert stock.current_price == 150.25

    def test_create_stock_data_with_all_fields(self):
        """Create StockData with all fields populated."""
        stock = StockData(
            name="Test Company",
            sector="Technology",
            market_cap=1000000000,
            current_price=100.0,
            target_price=120.0,
            price_change_percentage=5.0,
            mtd_change=3.0,
            ytd_change=15.0,
            two_year_change=50.0,
            recommendation_mean=2.5,
            recommendation_key="buy",
            analyst_count=15,
            pe_trailing=20.0,
            pe_forward=18.0,
            peg_ratio=1.5,
            quick_ratio=1.2,
            current_ratio=1.5,
            debt_to_equity=0.5,
            short_float_pct=2.0,
            short_ratio=3.0,
            beta=1.1,
            alpha=0.05,
            sharpe_ratio=1.8,
            sortino_ratio=2.0,
            cash_percentage=10.0,
            ma50=105.0,
            ma200=95.0,
            dividend_yield=2.5,
            last_earnings="2024-01-15",
            previous_earnings="2023-10-15",
            insider_buy_pct=5.0,
            insider_transactions=10,
        )
        assert stock.name == "Test Company"
        assert stock.current_price == 100.0
        assert stock.target_price == 120.0
        assert stock.analyst_count == 15
        assert stock.pe_trailing == 20.0
        assert stock.beta == 1.1


class TestStockDataToDict:
    """Test StockData.to_dict() method."""

    def test_to_dict_basic(self):
        """Convert StockData to dictionary."""
        stock = StockData(name="Apple", sector="Tech")
        result = stock.to_dict()
        assert isinstance(result, dict)
        assert result["name"] == "Apple"
        assert result["sector"] == "Tech"

    def test_to_dict_excludes_ticker_object(self):
        """to_dict() excludes ticker_object."""
        stock = StockData(name="Test", ticker_object="some_object")
        result = stock.to_dict()
        assert "ticker_object" not in result
        assert "name" in result

    def test_to_dict_with_none_values(self):
        """to_dict() includes None values."""
        stock = StockData(name="Test", market_cap=None)
        result = stock.to_dict()
        assert "market_cap" in result
        assert result["market_cap"] is None

    def test_to_dict_with_all_fields(self):
        """to_dict() includes all non-ticker fields."""
        stock = StockData(
            name="Test",
            sector="Tech",
            market_cap=1000000,
            current_price=100.0,
            target_price=110.0,
            analyst_count=10,
        )
        result = stock.to_dict()
        assert "name" in result
        assert "sector" in result
        assert "market_cap" in result
        assert "current_price" in result
        assert "target_price" in result
        assert "analyst_count" in result


class TestStockDataFromDict:
    """Test StockData.from_dict() class method."""

    def test_from_dict_basic(self):
        """Create StockData from dictionary."""
        data = {"name": "Apple", "sector": "Technology"}
        stock = StockData.from_dict(data)
        assert stock.name == "Apple"
        assert stock.sector == "Technology"

    def test_from_dict_with_all_fields(self):
        """Create StockData from complete dictionary."""
        data = {
            "name": "Test Company",
            "sector": "Tech",
            "market_cap": 1000000000,
            "current_price": 100.0,
            "target_price": 110.0,
            "analyst_count": 15,
            "pe_trailing": 20.0,
            "beta": 1.1,
        }
        stock = StockData.from_dict(data)
        assert stock.name == "Test Company"
        assert stock.market_cap == 1000000000
        assert stock.current_price == 100.0
        assert stock.analyst_count == 15

    def test_from_dict_filters_unknown_keys(self):
        """from_dict() filters out unknown keys."""
        data = {
            "name": "Test",
            "sector": "Tech",
            "unknown_field": "should be ignored",
            "another_unknown": 123,
        }
        # Should not raise error
        stock = StockData.from_dict(data)
        assert stock.name == "Test"
        assert stock.sector == "Tech"
        # Unknown fields are filtered out

    def test_from_dict_empty_dict(self):
        """Create StockData from empty dictionary."""
        stock = StockData.from_dict({})
        assert stock.name == "N/A"
        assert stock.sector == "N/A"
        assert stock.market_cap is None


class TestStockDataRoundTrip:
    """Test round-trip conversion (to_dict -> from_dict)."""

    def test_round_trip_conversion(self):
        """Convert to dict and back maintains data."""
        original = StockData(
            name="Apple",
            sector="Tech",
            market_cap=3000000000000,
            current_price=150.0,
            analyst_count=20,
        )
        dict_data = original.to_dict()
        restored = StockData.from_dict(dict_data)

        assert restored.name == original.name
        assert restored.sector == original.sector
        assert restored.market_cap == original.market_cap
        assert restored.current_price == original.current_price
        assert restored.analyst_count == original.analyst_count

    def test_round_trip_with_none_values(self):
        """Round-trip preserves None values."""
        original = StockData(name="Test", market_cap=None, current_price=None)
        dict_data = original.to_dict()
        restored = StockData.from_dict(dict_data)

        assert restored.name == original.name
        assert restored.market_cap is None
        assert restored.current_price is None


class TestStockDataDefaults:
    """Test default values of StockData fields."""

    def test_string_fields_default_to_na(self):
        """String fields default to 'N/A'."""
        stock = StockData()
        assert stock.name == "N/A"
        assert stock.sector == "N/A"
        assert stock.recommendation_key == "N/A"

    def test_numeric_fields_default_to_none(self):
        """Numeric fields default to None."""
        stock = StockData()
        assert stock.market_cap is None
        assert stock.current_price is None
        assert stock.target_price is None
        assert stock.pe_trailing is None
        assert stock.beta is None

    def test_ticker_object_defaults_to_none(self):
        """ticker_object defaults to None."""
        stock = StockData()
        assert stock.ticker_object is None
