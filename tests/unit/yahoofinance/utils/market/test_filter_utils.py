"""
Tests for yahoofinance/utils/market/filter_utils.py

This module tests market filtering utilities.
"""

import pytest

from yahoofinance.utils.market.filter_utils import (
    filter_by_market_cap,
    filter_by_sector,
)


class TestFilterByMarketCap:
    """Tests for filter_by_market_cap function."""

    def test_no_filters_returns_all(self):
        """Test that no filters returns all stocks."""
        stocks = [
            {"ticker": "AAPL", "market_cap": 3e12},
            {"ticker": "MSFT", "market_cap": 2.5e12},
        ]
        result = filter_by_market_cap(stocks)
        assert len(result) == 2

    def test_min_cap_filter(self):
        """Test minimum market cap filter."""
        stocks = [
            {"ticker": "AAPL", "market_cap": 3e12},
            {"ticker": "SMALL", "market_cap": 1e9},
            {"ticker": "MSFT", "market_cap": 2.5e12},
        ]
        result = filter_by_market_cap(stocks, min_cap=100e9)
        assert len(result) == 2
        tickers = [s["ticker"] for s in result]
        assert "SMALL" not in tickers

    def test_max_cap_filter(self):
        """Test maximum market cap filter."""
        stocks = [
            {"ticker": "AAPL", "market_cap": 3e12},
            {"ticker": "SMALL", "market_cap": 1e9},
            {"ticker": "MSFT", "market_cap": 2.5e12},
        ]
        result = filter_by_market_cap(stocks, max_cap=100e9)
        assert len(result) == 1
        assert result[0]["ticker"] == "SMALL"

    def test_min_and_max_cap_filter(self):
        """Test both min and max market cap filters."""
        stocks = [
            {"ticker": "MEGA", "market_cap": 3e12},
            {"ticker": "MID", "market_cap": 50e9},
            {"ticker": "SMALL", "market_cap": 1e9},
        ]
        result = filter_by_market_cap(stocks, min_cap=10e9, max_cap=100e9)
        assert len(result) == 1
        assert result[0]["ticker"] == "MID"

    def test_stocks_without_market_cap_skipped(self):
        """Test that stocks without market cap are skipped."""
        stocks = [
            {"ticker": "AAPL", "market_cap": 3e12},
            {"ticker": "UNKNOWN", "price": 100},  # No market_cap
        ]
        result = filter_by_market_cap(stocks, min_cap=1e9)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_empty_list(self):
        """Test empty list returns empty."""
        result = filter_by_market_cap([], min_cap=1e9)
        assert result == []


class TestFilterBySector:
    """Tests for filter_by_sector function."""

    def test_no_filters_returns_all(self):
        """Test that no filters returns all stocks."""
        stocks = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "JPM", "sector": "Financial Services"},
        ]
        result = filter_by_sector(stocks)
        assert len(result) == 2

    def test_include_single_sector(self):
        """Test including a single sector."""
        stocks = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "JPM", "sector": "Financial Services"},
            {"ticker": "MSFT", "sector": "Technology"},
        ]
        result = filter_by_sector(stocks, sectors="Technology")
        assert len(result) == 2
        assert all(s["sector"] == "Technology" for s in result)

    def test_include_multiple_sectors(self):
        """Test including multiple sectors."""
        stocks = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "JPM", "sector": "Financial Services"},
            {"ticker": "JNJ", "sector": "Healthcare"},
        ]
        result = filter_by_sector(stocks, sectors=["Technology", "Healthcare"])
        assert len(result) == 2
        tickers = [s["ticker"] for s in result]
        assert "JPM" not in tickers

    def test_exclude_single_sector(self):
        """Test excluding a single sector."""
        stocks = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "XOM", "sector": "Energy"},
            {"ticker": "MSFT", "sector": "Technology"},
        ]
        result = filter_by_sector(stocks, exclude_sectors="Energy")
        assert len(result) == 2
        tickers = [s["ticker"] for s in result]
        assert "XOM" not in tickers

    def test_exclude_multiple_sectors(self):
        """Test excluding multiple sectors."""
        stocks = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "XOM", "sector": "Energy"},
            {"ticker": "DUK", "sector": "Utilities"},
        ]
        result = filter_by_sector(stocks, exclude_sectors=["Energy", "Utilities"])
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_stocks_without_sector_skipped(self):
        """Test that stocks without sector are skipped."""
        stocks = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "UNKNOWN", "price": 100},  # No sector
        ]
        result = filter_by_sector(stocks, sectors="Technology")
        assert len(result) == 1

    def test_include_and_exclude_together(self):
        """Test that include and exclude work together."""
        stocks = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "AMZN", "sector": "Technology"},  # Would be excluded
            {"ticker": "JPM", "sector": "Financial Services"},
        ]
        # Include Technology but somehow exclude wouldn't affect since we're including
        result = filter_by_sector(stocks, sectors=["Technology"], exclude_sectors=["Consumer Cyclical"])
        # Since exclude doesn't have "Technology", both tech stocks should be included
        assert len(result) == 2

    def test_empty_list(self):
        """Test empty list returns empty."""
        result = filter_by_sector([], sectors="Technology")
        assert result == []


class TestFilterIntegration:
    """Integration tests for filter functions."""

    def test_chain_market_cap_and_sector(self):
        """Test chaining market cap and sector filters."""
        stocks = [
            {"ticker": "AAPL", "market_cap": 3e12, "sector": "Technology"},
            {"ticker": "SMALL_TECH", "market_cap": 1e9, "sector": "Technology"},
            {"ticker": "JPM", "market_cap": 500e9, "sector": "Financial Services"},
        ]
        # First filter by sector
        tech_stocks = filter_by_sector(stocks, sectors="Technology")
        # Then filter by market cap
        result = filter_by_market_cap(tech_stocks, min_cap=100e9)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_preserve_original_data(self):
        """Test that original stock data is preserved."""
        stocks = [
            {"ticker": "AAPL", "market_cap": 3e12, "price": 175, "custom_field": "value"},
        ]
        result = filter_by_market_cap(stocks, min_cap=1e9)
        assert result[0]["custom_field"] == "value"
        assert result[0]["price"] == 175
