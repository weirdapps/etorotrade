#!/usr/bin/env python3
"""
ITERATION 6: Display Formatter Tests
Target: Test DisplayFormatter utility functions for financial data formatting
"""

import pytest
from yahoofinance.presentation.formatter import DisplayFormatter, Color, DisplayConfig


class TestDisplayFormatter:
    """Test DisplayFormatter formatting utilities."""

    @pytest.fixture
    def formatter(self):
        """Create DisplayFormatter instance."""
        return DisplayFormatter()

    def test_format_price_high_value(self, formatter):
        """Format price >= $1000 with 0 decimals."""
        assert formatter.format_price(1500.99) == "$1,501"
        assert formatter.format_price(25000.45) == "$25,000"
        assert formatter.format_price(1000.0) == "$1,000"

    def test_format_price_medium_value(self, formatter):
        """Format price >= $10 with 1 decimal."""
        assert formatter.format_price(15.99) == "$16.0"
        assert formatter.format_price(99.95) == "$100.0"
        assert formatter.format_price(10.0) == "$10.0"

    def test_format_price_low_value(self, formatter):
        """Format price < $10 with 2 decimals."""
        assert formatter.format_price(5.99) == "$5.99"
        assert formatter.format_price(0.50) == "$0.50"
        assert formatter.format_price(9.99) == "$9.99"

    def test_format_price_zero_none(self, formatter):
        """Handle zero and None values."""
        assert formatter.format_price(0) == "--"
        assert formatter.format_price(None) == "--"

    def test_format_percentage_normal(self, formatter):
        """Format normal percentage values."""
        assert formatter.format_percentage(10.5) == "10.5%"
        assert formatter.format_percentage(50.0) == "50.0%"
        assert formatter.format_percentage(-5.5) == "-5.5%"

    def test_format_percentage_extreme_high(self, formatter):
        """Format extreme high percentages as 'H'."""
        assert formatter.format_percentage(150.0) == "H"
        assert formatter.format_percentage(999.9) == "H"
        assert formatter.format_percentage(99.5) == "H"

    def test_format_percentage_extreme_low(self, formatter):
        """Format extreme low percentages as 'L'."""
        assert formatter.format_percentage(-150.0) == "L"
        assert formatter.format_percentage(-999.9) == "L"
        assert formatter.format_percentage(-99.5) == "L"

    def test_format_percentage_zero_none(self, formatter):
        """Handle zero and None values."""
        assert formatter.format_percentage(0) == "--"
        assert formatter.format_percentage(None) == "--"

    def test_format_ratio(self, formatter):
        """Format ratio values."""
        assert formatter.format_ratio(1.5) == "1.50"
        assert formatter.format_ratio(25.0) == "25.00"
        assert formatter.format_ratio(0.05) == "0.05"

    def test_format_ratio_zero_none(self, formatter):
        """Handle zero and None values."""
        assert formatter.format_ratio(0) == "--"
        assert formatter.format_ratio(None) == "--"


class TestColorEnum:
    """Test Color enum constants."""

    def test_color_constants_exist(self):
        """Verify all color constants are defined."""
        assert Color.GREEN.value == "\033[92m"
        assert Color.RED.value == "\033[91m"
        assert Color.YELLOW.value == "\033[93m"
        assert Color.BLUE.value == "\033[94m"
        assert Color.PURPLE.value == "\033[95m"
        assert Color.CYAN.value == "\033[96m"
        assert Color.WHITE.value == "\033[97m"
        assert Color.RESET.value == "\033[0m"
        assert Color.BOLD.value == "\033[1m"
        assert Color.UNDERLINE.value == "\033[4m"


class TestDisplayConfig:
    """Test DisplayConfig dataclass."""

    def test_display_config_defaults(self):
        """Verify default configuration values."""
        config = DisplayConfig()
        assert config.compact_mode is False
        assert config.show_colors is True
        assert config.max_name_length == 14
        assert config.date_format == "%Y-%m-%d"
        assert config.show_headers is True
        assert config.max_columns is None
        assert config.sort_column is None
        assert config.reverse_sort is False

    def test_display_config_custom_values(self):
        """Create DisplayConfig with custom values."""
        config = DisplayConfig(
            compact_mode=True,
            show_colors=False,
            max_name_length=20,
            max_columns=10,
            sort_column="upside",
            reverse_sort=True,
        )
        assert config.compact_mode is True
        assert config.show_colors is False
        assert config.max_name_length == 20
        assert config.max_columns == 10
        assert config.sort_column == "upside"
        assert config.reverse_sort is True

    def test_display_config_reorder_columns(self):
        """Verify reorder_columns is initialized from config."""
        config = DisplayConfig()
        # Should be initialized from STANDARD_DISPLAY_COLUMNS in __post_init__
        assert config.reorder_columns is not None
        assert isinstance(config.reorder_columns, list)
