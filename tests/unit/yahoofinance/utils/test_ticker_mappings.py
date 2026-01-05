#!/usr/bin/env python3
"""
ITERATION 11: Ticker Mappings Tests
Target: Test ticker mapping compatibility functions
"""

import pytest
from yahoofinance.utils import ticker_mappings


class TestTickerMappingsConstants:
    """Test that mapping constants are accessible."""

    def test_dual_listed_mappings_exists(self):
        """DUAL_LISTED_MAPPINGS constant exists."""
        assert hasattr(ticker_mappings, 'DUAL_LISTED_MAPPINGS')
        assert isinstance(ticker_mappings.DUAL_LISTED_MAPPINGS, dict)

    def test_reverse_mappings_exists(self):
        """REVERSE_MAPPINGS constant exists."""
        assert hasattr(ticker_mappings, 'REVERSE_MAPPINGS')
        assert isinstance(ticker_mappings.REVERSE_MAPPINGS, dict)

    def test_dual_listed_tickers_exists(self):
        """DUAL_LISTED_TICKERS constant exists."""
        assert hasattr(ticker_mappings, 'DUAL_LISTED_TICKERS')
        assert isinstance(ticker_mappings.DUAL_LISTED_TICKERS, set)

    def test_ticker_geography_exists(self):
        """TICKER_GEOGRAPHY constant exists."""
        assert hasattr(ticker_mappings, 'TICKER_GEOGRAPHY')
        assert isinstance(ticker_mappings.TICKER_GEOGRAPHY, dict)


class TestGetNormalizedTicker:
    """Test get_normalized_ticker function."""

    def test_get_normalized_ticker_function_exists(self):
        """get_normalized_ticker function is callable."""
        assert callable(ticker_mappings.get_normalized_ticker)

    def test_get_normalized_ticker_returns_string(self):
        """get_normalized_ticker returns a string."""
        result = ticker_mappings.get_normalized_ticker("AAPL")
        assert isinstance(result, str)


class TestGetUsTicker:
    """Test get_us_ticker function."""

    def test_get_us_ticker_function_exists(self):
        """get_us_ticker function is callable."""
        assert callable(ticker_mappings.get_us_ticker)

    def test_get_us_ticker_returns_string(self):
        """get_us_ticker returns a string."""
        result = ticker_mappings.get_us_ticker("AAPL")
        assert isinstance(result, str)


class TestIsDualListed:
    """Test is_dual_listed function."""

    def test_is_dual_listed_function_exists(self):
        """is_dual_listed function is callable."""
        assert callable(ticker_mappings.is_dual_listed)

    def test_is_dual_listed_returns_bool(self):
        """is_dual_listed returns a boolean."""
        result = ticker_mappings.is_dual_listed("AAPL")
        assert isinstance(result, bool)


class TestGetDisplayTicker:
    """Test get_display_ticker function."""

    def test_get_display_ticker_function_exists(self):
        """get_display_ticker function is callable."""
        assert callable(ticker_mappings.get_display_ticker)

    def test_get_display_ticker_returns_string(self):
        """get_display_ticker returns a string."""
        result = ticker_mappings.get_display_ticker("AAPL")
        assert isinstance(result, str)


class TestGetDataFetchTicker:
    """Test get_data_fetch_ticker function."""

    def test_get_data_fetch_ticker_function_exists(self):
        """get_data_fetch_ticker function is callable."""
        assert callable(ticker_mappings.get_data_fetch_ticker)

    def test_get_data_fetch_ticker_returns_string(self):
        """get_data_fetch_ticker returns a string."""
        result = ticker_mappings.get_data_fetch_ticker("AAPL")
        assert isinstance(result, str)


class TestGetTickerGeography:
    """Test get_ticker_geography function."""

    def test_get_ticker_geography_function_exists(self):
        """get_ticker_geography function is callable."""
        assert callable(ticker_mappings.get_ticker_geography)

    def test_get_ticker_geography_returns_string(self):
        """get_ticker_geography returns a string."""
        result = ticker_mappings.get_ticker_geography("AAPL")
        assert isinstance(result, str)


class TestAreEquivalentTickers:
    """Test are_equivalent_tickers function."""

    def test_are_equivalent_tickers_function_exists(self):
        """are_equivalent_tickers function is callable."""
        assert callable(ticker_mappings.are_equivalent_tickers)

    def test_are_equivalent_tickers_returns_bool(self):
        """are_equivalent_tickers returns a boolean."""
        result = ticker_mappings.are_equivalent_tickers("AAPL", "AAPL")
        assert isinstance(result, bool)

    def test_are_equivalent_tickers_same_ticker(self):
        """Same ticker is equivalent to itself."""
        result = ticker_mappings.are_equivalent_tickers("AAPL", "AAPL")
        assert result is True


class TestGetAllEquivalentTickers:
    """Test get_all_equivalent_tickers function."""

    def test_get_all_equivalent_tickers_function_exists(self):
        """get_all_equivalent_tickers function is callable."""
        assert callable(ticker_mappings.get_all_equivalent_tickers)

    def test_get_all_equivalent_tickers_returns_collection(self):
        """get_all_equivalent_tickers returns a collection."""
        result = ticker_mappings.get_all_equivalent_tickers("AAPL")
        # Should return a set or list
        assert hasattr(result, '__iter__')
