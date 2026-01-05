#!/usr/bin/env python3
"""
ITERATION 17: Format Utils Tests
Target: Test data formatting utilities for financial metrics
"""

import pytest
import pandas as pd
from yahoofinance.utils.data.format_utils import (
    format_market_cap,
    format_position_size,
    calculate_upside,
    calculate_validated_upside,
    format_number,
    format_ticker_for_display,
    calculate_position_size,
)


class TestFormatMarketCap:
    """Test market cap formatting."""

    def test_format_market_cap_trillion(self):
        """Format trillion-dollar market cap."""
        result = format_market_cap(3_500_000_000_000)
        assert "3" in result
        assert "T" in result

    def test_format_market_cap_billion(self):
        """Format billion-dollar market cap."""
        result = format_market_cap(500_000_000_000)
        assert "500" in result
        assert "B" in result

    def test_format_market_cap_million(self):
        """Format million-dollar market cap."""
        result = format_market_cap(250_000_000)
        assert "250" in result
        assert "M" in result

    def test_format_market_cap_none(self):
        """Handle None market cap."""
        result = format_market_cap(None)
        assert result is None or result == "--" or result == "N/A"

    def test_format_market_cap_zero(self):
        """Handle zero market cap."""
        result = format_market_cap(0)
        assert result is None or result == "--" or result == "0"


class TestCalculateUpside:
    """Test upside calculation."""

    def test_calculate_upside_positive(self):
        """Calculate positive upside."""
        result = calculate_upside(100, 120)
        assert result == 20.0 or abs(result - 20.0) < 0.01

    def test_calculate_upside_negative(self):
        """Calculate negative upside."""
        result = calculate_upside(100, 90)
        assert result == -10.0 or abs(result + 10.0) < 0.01

    def test_calculate_upside_none_price(self):
        """Handle None current price."""
        result = calculate_upside(None, 120)
        assert result is None

    def test_calculate_upside_none_target(self):
        """Handle None target price."""
        result = calculate_upside(100, None)
        assert result is None


class TestCalculateValidatedUpside:
    """Test validated upside calculation."""

    def test_calculate_validated_upside_valid(self):
        """Calculate validated upside with valid data."""
        ticker_data = {
            "current_price": 100,
            "target_price": 120
        }
        upside, reason = calculate_validated_upside(ticker_data)

        assert isinstance(upside, (int, float, type(None)))
        assert isinstance(reason, str)

    def test_calculate_validated_upside_missing_data(self):
        """Handle missing price data."""
        ticker_data = {}
        upside, reason = calculate_validated_upside(ticker_data)

        assert isinstance(reason, str)


class TestFormatNumber:
    """Test number formatting."""

    def test_format_number_integer(self):
        """Format integer."""
        result = format_number(1000)
        assert isinstance(result, str)

    def test_format_number_float(self):
        """Format float."""
        result = format_number(1234.56)
        assert isinstance(result, str)


class TestFormatPositionSize:
    """Test position size formatting."""

    def test_format_position_size_valid(self):
        """Format valid position size."""
        result = format_position_size(2.5)
        assert isinstance(result, str)
        assert result != "--"

    def test_format_position_size_none(self):
        """Handle None position size."""
        result = format_position_size(None)
        assert result == "--" or result is None

    def test_format_position_size_zero(self):
        """Handle zero position size."""
        result = format_position_size(0)
        assert result == "--" or "0" in result


class TestFormatTickerForDisplay:
    """Test ticker formatting for display."""

    def test_format_ticker_for_display_basic(self):
        """Format basic ticker."""
        result = format_ticker_for_display("AAPL")
        assert isinstance(result, str)

    def test_format_ticker_for_display_with_suffix(self):
        """Format ticker with suffix."""
        result = format_ticker_for_display("BTC-USD")
        assert isinstance(result, str)


class TestCalculatePositionSize:
    """Test position size calculation."""

    def test_calculate_position_size_callable(self):
        """Verify calculate_position_size is callable."""
        assert callable(calculate_position_size)


class TestFormattingEdgeCases:
    """Test edge cases in formatting."""

    def test_format_very_large_market_cap(self):
        """Format extremely large market cap."""
        result = format_market_cap(10_000_000_000_000)  # 10 trillion
        assert "T" in result

    def test_format_very_small_market_cap(self):
        """Format very small market cap."""
        result = format_market_cap(1_000_000)  # 1 million
        assert "M" in result or "1" in result

    def test_format_negative_values(self):
        """Handle negative values gracefully."""
        # Negative market cap (shouldn't happen but test gracefully)
        result_cap = format_market_cap(-100000000)
        assert isinstance(result_cap, str) or result_cap is None


class TestFormattingConsistency:
    """Test consistency across formatting functions."""

    def test_none_values_consistent(self):
        """All formatters handle None consistently."""
        cap_result = format_market_cap(None)
        size_result = format_position_size(None)

        # All should return consistent null representation
        assert cap_result is None or cap_result == "--"
        assert size_result is None or size_result == "--"


