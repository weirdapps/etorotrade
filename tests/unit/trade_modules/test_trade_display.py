"""
Tests for trade_modules/trade_display.py

This module tests the DisplayFormatter class for formatting trading data.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from trade_modules.trade_display import (
    DisplayFormatter,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_CYAN,
    COLOR_RESET,
    COLOR_BOLD,
)


@pytest.fixture
def formatter():
    """Create a DisplayFormatter instance with colors enabled."""
    return DisplayFormatter(use_colors=True)


@pytest.fixture
def formatter_no_colors():
    """Create a DisplayFormatter instance without colors."""
    return DisplayFormatter(use_colors=False)


@pytest.fixture
def sample_opportunities():
    """Create sample opportunities DataFrames."""
    buy_df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "price": [175.0, 380.0],
        "market_cap": [3e12, 2.5e12],
        "pe_ratio": [28.5, 32.0],
        "price_target": [200.0, 420.0],
        "expected_return": [14.3, 10.5],
        "confidence_score": [0.85, 0.75],
    })
    buy_df = buy_df.set_index("ticker")

    sell_df = pd.DataFrame({
        "ticker": ["NFLX", "INTC"],
        "price": [400.0, 30.0],
        "market_cap": [180e9, 130e9],
        "expected_return": [-5.0, -8.0],
        "exret": [-5.0, -8.0],
        "confidence_score": [0.8, 0.7],
        "action": ["S", "S"],
    })
    sell_df = sell_df.set_index("ticker")

    hold_df = pd.DataFrame({
        "ticker": ["GOOGL", "AMD"],
        "price": [140.0, 120.0],
        "dividend_yield": [0.5, 0.0],
        "pe_ratio": [22.0, 45.0],
        "expected_return": [2.0, 3.0],
        "confidence_score": [0.6, 0.55],
    })
    hold_df = hold_df.set_index("ticker")

    return {
        "buy_opportunities": buy_df,
        "sell_opportunities": sell_df,
        "hold_opportunities": hold_df,
    }


class TestDisplayFormatterInit:
    """Tests for DisplayFormatter initialization."""

    def test_init_with_colors(self):
        """Test initialization with colors enabled."""
        formatter = DisplayFormatter(use_colors=True)
        assert formatter.use_colors is True

    def test_init_without_colors(self):
        """Test initialization with colors disabled."""
        formatter = DisplayFormatter(use_colors=False)
        assert formatter.use_colors is False

    def test_default_colors(self):
        """Test default color setting."""
        formatter = DisplayFormatter()
        assert formatter.use_colors is True


class TestFormatSectionTitle:
    """Tests for _format_section_title method."""

    def test_buy_section_title_with_colors(self, formatter):
        """Test buy section title gets green color."""
        title = formatter._format_section_title("Buy Opportunities")
        assert COLOR_GREEN in title
        assert "Buy Opportunities" in title
        assert COLOR_RESET in title

    def test_sell_section_title_with_colors(self, formatter):
        """Test sell section title gets red color."""
        title = formatter._format_section_title("Sell Opportunities")
        assert COLOR_RED in title
        assert "Sell Opportunities" in title

    def test_hold_section_title_with_colors(self, formatter):
        """Test hold section title gets yellow color."""
        title = formatter._format_section_title("Hold Opportunities")
        assert COLOR_YELLOW in title
        assert "Hold Opportunities" in title

    def test_generic_section_title_with_colors(self, formatter):
        """Test generic section title gets cyan color."""
        title = formatter._format_section_title("Other Section")
        assert COLOR_CYAN in title
        assert "Other Section" in title

    def test_section_title_without_colors(self, formatter_no_colors):
        """Test section title without colors."""
        title = formatter_no_colors._format_section_title("Buy Opportunities")
        assert COLOR_GREEN not in title
        assert "=== Buy Opportunities ===" == title


class TestGetDisplayColumns:
    """Tests for _get_display_columns method."""

    def test_buy_columns(self, formatter):
        """Test columns for buy opportunities."""
        columns = formatter._get_display_columns("buy_opportunities")
        assert "price" in columns
        assert "market_cap" in columns
        assert "price_target" in columns
        assert "expected_return" in columns

    def test_sell_columns(self, formatter):
        """Test columns for sell opportunities."""
        columns = formatter._get_display_columns("sell_opportunities")
        assert "price" in columns
        assert "action" in columns
        assert "exret" in columns

    def test_hold_columns(self, formatter):
        """Test columns for hold opportunities."""
        columns = formatter._get_display_columns("hold_opportunities")
        assert "price" in columns
        assert "dividend_yield" in columns
        assert "pe_ratio" in columns

    def test_unknown_action_type(self, formatter):
        """Test default columns for unknown action type."""
        columns = formatter._get_display_columns("unknown_type")
        assert "ticker" in columns
        assert "price" in columns


class TestFormatDisplayColumns:
    """Tests for _format_display_columns method."""

    def test_format_market_cap(self, formatter):
        """Test market cap formatting."""
        df = pd.DataFrame({
            "market_cap": [3e12, 500e9, 100e6],
        })
        result = formatter._format_display_columns(df)
        assert "T" in result.iloc[0]["market_cap"]  # Trillion
        assert "B" in result.iloc[1]["market_cap"]  # Billion

    def test_format_price_large(self, formatter):
        """Test price formatting for large values."""
        df = pd.DataFrame({
            "price": [1500.0, 50.0, 5.0],
        })
        result = formatter._format_display_columns(df)
        # Large price: no decimals
        assert result.iloc[0]["price"] == "$1,500"
        # Medium price: 1 decimal
        assert result.iloc[1]["price"] == "$50.0"
        # Small price: 2 decimals
        assert result.iloc[2]["price"] == "$5.00"

    def test_format_percentage_columns(self, formatter):
        """Test percentage column formatting."""
        df = pd.DataFrame({
            "expected_return": [14.3, 10.5],
            "confidence_score": [0.85, 0.75],
        })
        result = formatter._format_display_columns(df)
        # Should have % in formatted values
        assert "%" in result.iloc[0]["expected_return"]

    def test_format_ratio_columns(self, formatter):
        """Test ratio column formatting."""
        df = pd.DataFrame({
            "pe_ratio": [28.5, 32.123],
            "beta": [1.234, 0.9],
        })
        result = formatter._format_display_columns(df)
        assert result.iloc[0]["pe_ratio"] == "28.50"
        assert result.iloc[0]["beta"] == "1.23"

    def test_format_empty_dataframe(self, formatter):
        """Test formatting empty DataFrame."""
        df = pd.DataFrame()
        result = formatter._format_display_columns(df)
        assert result.empty

    def test_format_roe_column(self, formatter):
        """Test ROE column formatting."""
        df = pd.DataFrame({
            "ROE": [0.1983, 0.25, None],
        })
        result = formatter._format_display_columns(df)
        # Should convert 0.1983 to percentage representation
        assert "19.8" in result.iloc[0]["ROE"]
        assert result.iloc[2]["ROE"] == "N/A"

    def test_format_de_column(self, formatter):
        """Test DE column formatting."""
        df = pd.DataFrame({
            "DE": [1.5, 0.75, None],
        })
        result = formatter._format_display_columns(df)
        assert result.iloc[0]["DE"] == "1.5"
        assert result.iloc[2]["DE"] == "N/A"


class TestApplyColorCoding:
    """Tests for _apply_color_coding method."""

    def test_color_coding_disabled(self, formatter_no_colors):
        """Test that color coding is skipped when disabled."""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [175.0],
        })
        result = formatter_no_colors._apply_color_coding(df, "buy_opportunities")
        # Should return unchanged when colors disabled
        assert result.equals(df)

    def test_color_coding_empty_df(self, formatter):
        """Test color coding with empty DataFrame."""
        df = pd.DataFrame()
        result = formatter._apply_color_coding(df, "buy_opportunities")
        assert result.empty


class TestPrepareDisplayDataframe:
    """Tests for _prepare_display_dataframe method."""

    def test_prepare_empty_dataframe(self, formatter):
        """Test preparing empty DataFrame."""
        df = pd.DataFrame()
        result = formatter._prepare_display_dataframe(df, "buy_opportunities")
        assert result.empty

    def test_prepare_selects_appropriate_columns(self, formatter):
        """Test that appropriate columns are selected."""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [175.0],
            "market_cap": [3e12],
            "extra_column": ["extra"],
        })
        result = formatter._prepare_display_dataframe(df, "buy_opportunities")
        # Extra column should be filtered out
        assert "extra_column" not in result.columns


class TestFormatTradingOpportunities:
    """Tests for format_trading_opportunities method."""

    def test_format_table_output(self, formatter, sample_opportunities):
        """Test formatting as table."""
        result = formatter.format_trading_opportunities(sample_opportunities, "table")
        assert isinstance(result, str)
        # Should contain section titles
        assert "Buy" in result or "Sell" in result or "Hold" in result

    def test_format_json_output(self, formatter, sample_opportunities):
        """Test formatting as JSON."""
        result = formatter.format_trading_opportunities(sample_opportunities, "json")
        assert isinstance(result, str)
        # JSON output should contain price data
        assert "175" in result or "380" in result

    def test_format_empty_opportunities(self, formatter):
        """Test formatting empty opportunities."""
        empty_opportunities = {
            "buy_opportunities": pd.DataFrame(),
            "sell_opportunities": pd.DataFrame(),
            "hold_opportunities": pd.DataFrame(),
        }
        result = formatter.format_trading_opportunities(empty_opportunities, "table")
        assert isinstance(result, str)

    def test_format_handles_error(self, formatter):
        """Test error handling during formatting."""
        # Pass invalid data
        invalid_opportunities = {"buy_opportunities": "not_a_dataframe"}
        result = formatter.format_trading_opportunities(invalid_opportunities, "table")
        assert "Error" in result


class TestDisplayFormatterIntegration:
    """Integration tests for DisplayFormatter."""

    def test_full_format_workflow(self, formatter, sample_opportunities):
        """Test complete formatting workflow."""
        result = formatter.format_trading_opportunities(sample_opportunities, "table")

        # Should have formatted output
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_preserves_data(self, formatter_no_colors):
        """Test that formatting preserves essential data."""
        opportunities = {
            "buy_opportunities": pd.DataFrame({
                "ticker": ["AAPL"],
                "price": [175.0],
                "market_cap": [3e12],
            }).set_index("ticker"),
            "sell_opportunities": pd.DataFrame(),
            "hold_opportunities": pd.DataFrame(),
        }

        result = formatter_no_colors.format_trading_opportunities(opportunities, "table")

        # Original price data should be visible in output
        assert "175" in result or "$175" in result

    def test_different_output_formats(self, formatter, sample_opportunities):
        """Test that different formats produce different outputs."""
        table_result = formatter.format_trading_opportunities(sample_opportunities, "table")
        json_result = formatter.format_trading_opportunities(sample_opportunities, "json")

        # JSON should contain brackets and braces
        assert "{" in json_result or "[" in json_result
        # Results should be different
        assert table_result != json_result
