#!/usr/bin/env python3
"""
ITERATION 28: Display Manager Tests
Target: Test unified display formatting for console, CSV, and HTML
File: trade_modules/display_manager.py (197 statements, 0% coverage)
"""

import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch


class TestDisplayManagerInit:
    """Test DisplayManager initialization."""

    def test_init_with_default_config(self):
        """Initialize with default config."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()

        assert manager.config is not None
        assert manager.color_reset == "\033[0m"

    def test_init_with_custom_config(self):
        """Initialize with custom config."""
        from trade_modules.display_manager import DisplayManager
        from trade_modules.trade_config import TradeConfig

        custom_config = TradeConfig()
        manager = DisplayManager(config=custom_config)

        assert manager.config is custom_config


class TestPrepareDataFrame:
    """Test DataFrame preparation."""

    def test_prepare_empty_dataframe(self):
        """Prepare empty DataFrame returns empty copy."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame()

        result = manager.prepare_dataframe(df, "p", output_type="console")

        assert result.empty
        assert len(result) == 0

    def test_prepare_filters_missing_columns(self):
        """Prepare filters out columns not in DataFrame."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})

        # Config might request columns that don't exist
        result = manager.prepare_dataframe(df, "p", output_type="console")

        # Should only include available columns
        assert not result.empty


class TestFormatMethods:
    """Test format convenience methods."""

    def test_format_console(self):
        """Format for console output."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})

        result = manager.format_console(df, "p")

        assert isinstance(result, pd.DataFrame)

    def test_format_csv(self):
        """Format for CSV output."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})

        result = manager.format_csv(df, "p")

        assert isinstance(result, pd.DataFrame)

    def test_format_html(self):
        """Format for HTML output."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})

        result = manager.format_html(df, "p")

        assert isinstance(result, pd.DataFrame)


class TestValueFormatting:
    """Test value formatting."""

    def test_format_value_na_returns_dash(self):
        """Format NA value returns '--'."""
        from trade_modules.display_manager import DisplayManager
        import numpy as np

        manager = DisplayManager()

        result = manager._format_value(np.nan, {}, "console")

        assert result == "--"

    def test_format_value_none_returns_dash(self):
        """Format None value returns '--'."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()

        result = manager._format_value(None, {}, "console")

        assert result == "--"

    def test_format_value_text_type(self):
        """Format text type value."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()

        result = manager._format_value("AAPL", {"type": "text"}, "console")

        assert result == "AAPL"


class TestCurrencyFormatting:
    """Test currency formatting."""

    def test_format_currency_basic(self):
        """Format basic currency value."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "currency", "decimals": 2, "symbol": "$"}

        result = manager._format_currency(150.50, format_rule, "console")

        assert result == "$150.50"

    def test_format_currency_high_value(self):
        """Format high currency value uses fewer decimals."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {
            "type": "currency",
            "decimals": 4,
            "symbol": "$",
            "threshold_high_decimals": 100
        }

        result = manager._format_currency(1500.1234, format_rule, "console")

        # Should use max 2 decimals for values > threshold
        assert result.startswith("$1,500.")
        assert len(result.split(".")[1]) <= 2

    def test_format_currency_with_commas(self):
        """Format currency with thousand separators."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "currency", "decimals": 2, "symbol": "$"}

        result = manager._format_currency(1234567.89, format_rule, "console")

        assert result == "$1,234,567.89"

    def test_format_currency_invalid_value(self):
        """Format invalid currency value returns string."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "currency", "decimals": 2, "symbol": "$"}

        result = manager._format_currency("invalid", format_rule, "console")

        assert result == "invalid"


class TestPercentageFormatting:
    """Test percentage formatting."""

    def test_format_percentage_basic(self):
        """Format basic percentage value."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "percentage", "decimals": 1, "suffix": "%"}

        result = manager._format_percentage(12.5, format_rule, "console")

        assert "12.5%" in result

    def test_format_percentage_positive_color_console(self):
        """Format positive percentage with green color for console."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {
            "type": "percentage",
            "decimals": 1,
            "suffix": "%",
            "color_positive": "green"
        }

        result = manager._format_percentage(10.5, format_rule, "console")

        assert "\033[92m" in result  # Green color code
        assert "10.5%" in result
        assert manager.color_reset in result

    def test_format_percentage_negative_color_console(self):
        """Format negative percentage with red color for console."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {
            "type": "percentage",
            "decimals": 1,
            "suffix": "%",
            "color_negative": "red"
        }

        result = manager._format_percentage(-5.2, format_rule, "console")

        assert "\033[91m" in result  # Red color code
        assert "-5.2%" in result
        assert manager.color_reset in result

    def test_format_percentage_no_color_for_csv(self):
        """Format percentage without color for CSV."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {
            "type": "percentage",
            "decimals": 1,
            "suffix": "%",
            "color_positive": "green"
        }

        result = manager._format_percentage(10.5, format_rule, "csv")

        assert result == "10.5%"
        assert "\033[" not in result  # No color codes


class TestMarketCapFormatting:
    """Test market cap formatting."""

    def test_format_market_cap_trillion(self):
        """Format trillion market cap."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "market_cap", "decimals": 1, "units": ["M", "B", "T"]}

        result = manager._format_market_cap(2_500_000_000_000, format_rule, "console")

        assert result == "2.5T"

    def test_format_market_cap_billion(self):
        """Format billion market cap."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "market_cap", "decimals": 1, "units": ["M", "B", "T"]}

        result = manager._format_market_cap(150_000_000_000, format_rule, "console")

        assert result == "150.0B"

    def test_format_market_cap_million(self):
        """Format million market cap."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "market_cap", "decimals": 1, "units": ["M", "B", "T"]}

        result = manager._format_market_cap(500_000_000, format_rule, "console")

        assert result == "500.0M"

    def test_format_market_cap_small(self):
        """Format small market cap below million."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "market_cap", "decimals": 1, "units": ["M", "B", "T"]}

        result = manager._format_market_cap(500_000, format_rule, "console")

        assert result == "500,000"


class TestDecimalFormatting:
    """Test decimal formatting."""

    def test_format_decimal_basic(self):
        """Format basic decimal value."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "decimal", "decimals": 2}

        result = manager._format_decimal(3.14159, format_rule, "console")

        assert result == "3.14"

    def test_format_decimal_custom_decimals(self):
        """Format decimal with custom decimal places."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "decimal", "decimals": 4}

        result = manager._format_decimal(2.718281828, format_rule, "console")

        assert result == "2.7183"


class TestActionFormatting:
    """Test action formatting."""

    def test_format_action_buy_console(self):
        """Format BUY action for console."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {
            "type": "action",
            "colors": {
                "BUY": {"name": "BUY", "console": "\033[92m"}
            }
        }

        result = manager._format_action("buy", format_rule, "console")

        assert "\033[92m" in result
        assert "BUY" in result

    def test_format_action_sell_console(self):
        """Format SELL action for console."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {
            "type": "action",
            "colors": {
                "SELL": {"name": "SELL", "console": "\033[91m"}
            }
        }

        result = manager._format_action("sell", format_rule, "console")

        assert "\033[91m" in result
        assert "SELL" in result

    def test_format_action_csv_no_color(self):
        """Format action for CSV without colors."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {
            "type": "action",
            "colors": {
                "BUY": {"name": "BUY", "console": "\033[92m"}
            }
        }

        result = manager._format_action("buy", format_rule, "csv")

        assert result == "BUY"
        assert "\033[" not in result

    def test_format_action_unknown(self):
        """Format unknown action returns uppercase."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        format_rule = {"type": "action", "colors": {}}

        result = manager._format_action("unknown", format_rule, "console")

        assert result == "UNKNOWN"


class TestSaveCSV:
    """Test CSV saving."""

    def test_save_csv_creates_file(self, tmp_path):
        """Save CSV creates file."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})
        output_file = str(tmp_path / "output.csv")

        manager.save_csv(df, output_file, "p")

        assert os.path.exists(output_file)

    def test_save_csv_creates_directory(self, tmp_path):
        """Save CSV creates directory if not exists."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})
        output_file = str(tmp_path / "nested" / "dir" / "output.csv")

        manager.save_csv(df, output_file, "p")

        assert os.path.exists(output_file)

    def test_save_csv_content(self, tmp_path):
        """Save CSV with correct content."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL", "MSFT"], "PRICE": [150.0, 300.0]})
        output_file = str(tmp_path / "output.csv")

        manager.save_csv(df, output_file, "p")

        # Read back and verify
        saved_df = pd.read_csv(output_file)
        assert len(saved_df) == 2


class TestSaveHTML:
    """Test HTML saving."""

    def test_save_html_creates_file(self, tmp_path):
        """Save HTML creates file."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})
        output_file = str(tmp_path / "output.html")

        manager.save_html(df, output_file, "p")

        assert os.path.exists(output_file)

    def test_save_html_with_title(self, tmp_path):
        """Save HTML with custom title."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})
        output_file = str(tmp_path / "output.html")

        manager.save_html(df, output_file, "p", title="Custom Title")

        # Read content and verify title
        with open(output_file, 'r') as f:
            content = f.read()
        assert "Custom Title" in content

    def test_save_html_empty_dataframe(self, tmp_path):
        """Save empty DataFrame to HTML."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame()
        output_file = str(tmp_path / "output.html")

        manager.save_html(df, output_file, "p")

        assert os.path.exists(output_file)


class TestGenerateHTMLTable:
    """Test HTML table generation."""

    def test_generate_html_table_basic(self):
        """Generate basic HTML table."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"TICKER": ["AAPL"], "PRICE": [150.0]})

        html = manager._generate_html_table(df, "Test Title")

        assert "<!DOCTYPE html>" in html
        assert "<table>" in html
        assert "Test Title" in html
        assert "TICKER" in html
        assert "AAPL" in html

    def test_generate_html_table_with_styling(self):
        """Generate HTML table includes CSS styling."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()
        df = pd.DataFrame({"ACTION": ["BUY"]})

        html = manager._generate_html_table(df, "Test")

        assert "<style>" in html
        assert "action-buy" in html
        assert "action-sell" in html


class TestGetHTMLCSSClass:
    """Test HTML CSS class determination."""

    def test_get_css_class_action_buy(self):
        """Get CSS class for BUY action."""
        from trade_modules.display_manager import DisplayManager
        from unittest.mock import Mock

        manager = DisplayManager()
        # Mock config to return action format rule
        manager.config.get_format_rule = Mock(return_value={"type": "action"})

        css_class = manager._get_html_css_class("ACTION", "BUY")

        assert "buy" in css_class.lower()

    def test_get_css_class_action_sell(self):
        """Get CSS class for SELL action."""
        from trade_modules.display_manager import DisplayManager
        from unittest.mock import Mock

        manager = DisplayManager()
        # Mock config to return action format rule
        manager.config.get_format_rule = Mock(return_value={"type": "action"})

        css_class = manager._get_html_css_class("ACTION", "SELL")

        assert "sell" in css_class.lower()

    def test_get_css_class_positive_value(self):
        """Get CSS class for positive value."""
        from trade_modules.display_manager import DisplayManager
        from unittest.mock import Mock

        manager = DisplayManager()
        # Mock config to return non-action format rule
        manager.config.get_format_rule = Mock(return_value={"type": "percentage"})

        css_class = manager._get_html_css_class("UPSIDE", "+12.5%")

        assert css_class == "positive"

    def test_get_css_class_negative_value(self):
        """Get CSS class for negative value."""
        from trade_modules.display_manager import DisplayManager
        from unittest.mock import Mock

        manager = DisplayManager()
        # Mock config to return non-action format rule
        manager.config.get_format_rule = Mock(return_value={"type": "percentage"})

        css_class = manager._get_html_css_class("UPSIDE", "-5.2%")

        assert css_class == "negative"


class TestGetOptionTitle:
    """Test option title generation."""

    def test_get_option_title_portfolio(self):
        """Get title for portfolio option."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()

        title = manager.get_option_title("p")

        assert title == "Portfolio Analysis"

    def test_get_option_title_market(self):
        """Get title for market option."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()

        title = manager.get_option_title("m")

        assert title == "Market Analysis"

    def test_get_option_title_trade_buy(self):
        """Get title for trade buy sub-option."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()

        title = manager.get_option_title("t", "b")

        assert title == "Buy Opportunities"

    def test_get_option_title_trade_sell(self):
        """Get title for trade sell sub-option."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()

        title = manager.get_option_title("t", "s")

        assert title == "Sell Opportunities"

    def test_get_option_title_unknown(self):
        """Get title for unknown option."""
        from trade_modules.display_manager import DisplayManager

        manager = DisplayManager()

        title = manager.get_option_title("x")

        assert "Analysis" in title


