#!/usr/bin/env python3
"""
ITERATION 24: Display Helpers Tests
Target: Test display and reporting utility functions
File: yahoofinance/utils/display_helpers.py (245 statements, 26% coverage)
"""

import pytest
import pandas as pd
from yahoofinance.utils.display_helpers import (
    handle_manual_input_tickers,
    get_output_file_and_title,
    format_market_cap,
    TICKER_COL,
    COMPANY_COL,
    ACTION_COL,
    PRICE_COL,
    TARGET_COL,
    UPSIDE_COL,
)


class TestHandleManualInputTickers:
    """Test manual ticker input processing."""

    def test_comma_separated_tickers(self):
        """Process comma-separated tickers."""
        result = handle_manual_input_tickers(["AAPL,MSFT,GOOGL"])
        assert "AAPL" in result
        assert "MSFT" in result
        # normalize_ticker converts GOOGL to GOOG
        assert "GOOG" in result
        assert len(result) == 3

    def test_space_separated_tickers(self):
        """Process space-separated tickers."""
        result = handle_manual_input_tickers(["AAPL MSFT GOOGL"])
        assert len(result) == 3
        assert "AAPL" in result

    def test_mixed_separators(self):
        """Process tickers with mixed separators."""
        result = handle_manual_input_tickers(["AAPL, MSFT; GOOGL"])
        assert len(result) == 3

    def test_single_ticker(self):
        """Process single ticker."""
        result = handle_manual_input_tickers(["AAPL"])
        assert result == ["AAPL"]

    def test_empty_list(self):
        """Handle empty list input."""
        result = handle_manual_input_tickers([])
        assert result == []

    def test_whitespace_trimming(self):
        """Trim whitespace from tickers."""
        result = handle_manual_input_tickers(["  AAPL  ,  MSFT  "])
        assert "AAPL" in result
        assert "MSFT" in result

    def test_uppercase_conversion(self):
        """Convert tickers to uppercase."""
        result = handle_manual_input_tickers(["aapl,msft"])
        assert "AAPL" in result
        assert "MSFT" in result

    def test_duplicate_removal(self):
        """Handle duplicate tickers."""
        # Function doesn't explicitly remove duplicates, but normalizes them
        result = handle_manual_input_tickers(["AAPL,AAPL,MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result


class TestGetOutputFileAndTitle:
    """Test output file and title determination."""

    def test_market_source(self):
        """Get output file for market source."""
        file, title = get_output_file_and_title("M", "/output")
        assert file == "/output/market.csv"
        assert title == "Market Analysis"

    def test_portfolio_source(self):
        """Get output file for portfolio source."""
        file, title = get_output_file_and_title("P", "/output")
        assert file == "/output/portfolio.csv"
        assert title == "Portfolio Analysis"

    def test_manual_source(self):
        """Get output file for manual input source."""
        file, title = get_output_file_and_title("I", "/output")
        assert file == "/output/manual.csv"
        assert title == "Manual Ticker Analysis"

    def test_custom_output_dir(self):
        """Use custom output directory."""
        file, title = get_output_file_and_title("M", "/custom/path")
        assert file.startswith("/custom/path/")

    def test_unknown_source_defaults_to_manual(self):
        """Unknown source defaults to manual."""
        file, title = get_output_file_and_title("X", "/output")
        assert file == "/output/manual.csv"
        assert title == "Manual Ticker Analysis"


class TestFormatMarketCap:
    """Test market cap formatting."""

    def test_format_market_cap_with_value(self):
        """Format market cap with valid value."""
        row = pd.Series({"CAP": "3.5T"})
        result = format_market_cap(row)
        assert isinstance(result, str)

    def test_format_market_cap_missing(self):
        """Handle missing market cap."""
        row = pd.Series({})
        result = format_market_cap(row)
        # Should handle gracefully
        assert isinstance(result, str) or result is None


class TestColumnConstants:
    """Test column name constants."""

    def test_ticker_column_defined(self):
        """TICKER column constant is defined."""
        assert TICKER_COL == "TICKER"

    def test_company_column_defined(self):
        """COMPANY column constant is defined."""
        assert COMPANY_COL == "COMPANY"

    def test_action_column_defined(self):
        """ACTION column constant is defined."""
        assert ACTION_COL == "ACTION"

    def test_price_column_defined(self):
        """PRICE column constant is defined."""
        assert PRICE_COL == "PRICE"

    def test_target_column_defined(self):
        """TARGET column constant is defined."""
        assert TARGET_COL == "TARGET"

    def test_upside_column_defined(self):
        """UPSIDE column constant is defined."""
        assert UPSIDE_COL == "UPSIDE"


class TestTickerInputEdgeCases:
    """Test edge cases in ticker input handling."""

    def test_semicolon_separator(self):
        """Process semicolon-separated tickers."""
        result = handle_manual_input_tickers(["AAPL;MSFT;GOOGL"])
        assert len(result) == 3

    def test_multiple_spaces(self):
        """Handle multiple spaces between tickers."""
        result = handle_manual_input_tickers(["AAPL    MSFT    GOOGL"])
        assert len(result) == 3

    def test_trailing_commas(self):
        """Handle trailing commas."""
        result = handle_manual_input_tickers(["AAPL,MSFT,"])
        assert len(result) == 2

    def test_leading_commas(self):
        """Handle leading commas."""
        result = handle_manual_input_tickers([",AAPL,MSFT"])
        assert len(result) == 2

    def test_special_characters_in_ticker(self):
        """Handle tickers with special characters (e.g., BRK.B)."""
        result = handle_manual_input_tickers(["BRK.B,AAPL"])
        assert len(result) == 2


class TestOutputFileGeneration:
    """Test output file path generation."""

    def test_output_file_has_csv_extension(self):
        """All output files have .csv extension."""
        sources = ["M", "P", "I"]
        for source in sources:
            file, _ = get_output_file_and_title(source, "/output")
            assert file.endswith(".csv")

    def test_output_file_paths_unique(self):
        """Different sources generate different file paths."""
        market_file, _ = get_output_file_and_title("M", "/output")
        portfolio_file, _ = get_output_file_and_title("P", "/output")
        manual_file, _ = get_output_file_and_title("I", "/output")

        assert market_file != portfolio_file
        assert market_file != manual_file
        assert portfolio_file != manual_file

    def test_titles_are_descriptive(self):
        """All titles are descriptive strings."""
        sources = ["M", "P", "I"]
        for source in sources:
            _, title = get_output_file_and_title(source, "/output")
            assert isinstance(title, str)
            assert len(title) > 0
            assert "Analysis" in title


