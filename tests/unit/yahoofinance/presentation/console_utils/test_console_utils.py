"""
Tests for yahoofinance/presentation/console_utils/

This module tests the console utility classes for colors, formatters, and tables.
"""

import pytest
import pandas as pd

from yahoofinance.presentation.console_utils.colors import ConsoleColors
from yahoofinance.presentation.console_utils.formatters import ConsoleFormatter
from yahoofinance.presentation.console_utils.tables import TableRenderer


class TestConsoleColors:
    """Tests for ConsoleColors class."""

    def test_color_codes_defined(self):
        """Test that color codes are defined."""
        assert ConsoleColors.GREEN == "\033[92m"
        assert ConsoleColors.RED == "\033[91m"
        assert ConsoleColors.YELLOW == "\033[93m"
        assert ConsoleColors.RESET == "\033[0m"

    def test_colorize_positive_value(self):
        """Test colorize with positive value returns green."""
        result = ConsoleColors.colorize("test", 5.0)
        assert ConsoleColors.GREEN in result
        assert "test" in result
        assert ConsoleColors.RESET in result

    def test_colorize_negative_value(self):
        """Test colorize with negative value returns red."""
        result = ConsoleColors.colorize("test", -3.0)
        assert ConsoleColors.RED in result
        assert "test" in result
        assert ConsoleColors.RESET in result

    def test_colorize_zero_value(self):
        """Test colorize with zero value returns yellow."""
        result = ConsoleColors.colorize("test", 0.0)
        assert ConsoleColors.YELLOW in result
        assert "test" in result
        assert ConsoleColors.RESET in result

    def test_colorize_small_positive(self):
        """Test colorize with small positive value."""
        result = ConsoleColors.colorize("price", 0.001)
        assert ConsoleColors.GREEN in result

    def test_colorize_small_negative(self):
        """Test colorize with small negative value."""
        result = ConsoleColors.colorize("price", -0.001)
        assert ConsoleColors.RED in result


class TestConsoleFormatter:
    """Tests for ConsoleFormatter class."""

    def test_format_number_none(self):
        """Test format_number with None value."""
        result = ConsoleFormatter.format_number(None)
        assert result == "--"

    def test_format_number_dash(self):
        """Test format_number with dash value."""
        result = ConsoleFormatter.format_number("--")
        assert result == "--"

    def test_format_number_int(self):
        """Test format_number with integer."""
        result = ConsoleFormatter.format_number(1234)
        assert result == "1,234.00"

    def test_format_number_float(self):
        """Test format_number with float."""
        result = ConsoleFormatter.format_number(1234.567)
        assert result == "1,234.57"

    def test_format_number_large(self):
        """Test format_number with large number."""
        result = ConsoleFormatter.format_number(1234567.89)
        assert result == "1,234,567.89"

    def test_format_number_string(self):
        """Test format_number with string value."""
        result = ConsoleFormatter.format_number("test")
        assert result == "test"

    def test_format_number_invalid(self):
        """Test format_number with invalid value."""
        result = ConsoleFormatter.format_number({"invalid": "object"})
        # Should convert to string
        assert "{" in result or result == "--"

    def test_format_percentage_none(self):
        """Test format_percentage with None value."""
        result = ConsoleFormatter.format_percentage(None)
        assert result == "--"

    def test_format_percentage_dash(self):
        """Test format_percentage with dash value."""
        result = ConsoleFormatter.format_percentage("--")
        assert result == "--"

    def test_format_percentage_float(self):
        """Test format_percentage with float."""
        result = ConsoleFormatter.format_percentage(15.5)
        assert result == "15.5%"

    def test_format_percentage_int(self):
        """Test format_percentage with integer."""
        result = ConsoleFormatter.format_percentage(20)
        assert result == "20.0%"

    def test_format_percentage_string_number(self):
        """Test format_percentage with string number."""
        result = ConsoleFormatter.format_percentage("25.5")
        assert result == "25.5%"

    def test_format_percentage_negative(self):
        """Test format_percentage with negative value."""
        result = ConsoleFormatter.format_percentage(-5.5)
        assert result == "-5.5%"

    def test_format_percentage_invalid(self):
        """Test format_percentage with invalid value."""
        result = ConsoleFormatter.format_percentage("not_a_number")
        assert result == "--"


class TestTableRenderer:
    """Tests for TableRenderer class."""

    def test_render_dataframe_basic(self):
        """Test render_dataframe with basic DataFrame."""
        df = pd.DataFrame({
            "name": ["Apple", "Microsoft"],
            "price": [175.0, 380.0],
        })

        result = TableRenderer.render_dataframe(df)

        assert isinstance(result, str)
        assert "Apple" in result
        assert "Microsoft" in result
        assert "175" in result
        assert "380" in result

    def test_render_dataframe_with_index(self):
        """Test render_dataframe preserves index."""
        df = pd.DataFrame({
            "price": [175.0, 380.0],
        }, index=["AAPL", "MSFT"])

        result = TableRenderer.render_dataframe(df)

        assert "AAPL" in result
        assert "MSFT" in result

    def test_render_dataframe_empty(self):
        """Test render_dataframe with empty DataFrame."""
        df = pd.DataFrame()

        result = TableRenderer.render_dataframe(df)

        assert isinstance(result, str)

    def test_render_dataframe_custom_headers(self):
        """Test render_dataframe with custom headers."""
        df = pd.DataFrame({
            "name": ["Apple"],
            "price": [175.0],
        })

        result = TableRenderer.render_dataframe(df, headers=["Name", "Price"])

        assert isinstance(result, str)

    def test_render_dataframe_numeric_columns(self):
        """Test render_dataframe with various numeric types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        })

        result = TableRenderer.render_dataframe(df)

        assert "1.5" in result
        assert "2.5" in result
        assert "3.5" in result


class TestConsoleUtilsIntegration:
    """Integration tests for console utilities."""

    def test_format_and_colorize(self):
        """Test formatting and colorizing together."""
        value = 15.5
        formatted = ConsoleFormatter.format_percentage(value)
        colorized = ConsoleColors.colorize(formatted, value)

        assert "15.5%" in colorized
        assert ConsoleColors.GREEN in colorized

    def test_render_formatted_data(self):
        """Test rendering pre-formatted data."""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [ConsoleFormatter.format_number(175.0)],
            "change": [ConsoleFormatter.format_percentage(5.5)],
        })

        result = TableRenderer.render_dataframe(df)

        # Table may format numbers differently, just check core content
        assert "175" in result
        assert "5.5%" in result
