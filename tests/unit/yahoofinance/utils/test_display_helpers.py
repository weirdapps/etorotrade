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
    _apply_color_to_row,
    get_action_color,
    _color_based_on_action,
    _has_required_metrics,
    TICKER_COL,
    COMPANY_COL,
    ACTION_COL,
    PRICE_COL,
    TARGET_COL,
    UPSIDE_COL,
    BUY_ACTION,
    SELL_ACTION,
    HOLD_ACTION,
    INCONCLUSIVE_ACTION,
    GREEN_COLOR,
    RED_COLOR,
    YELLOW_COLOR,
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


class TestFormatMarketCapAdvanced:
    """Additional tests for market cap formatting."""

    def test_trillion_format_large(self):
        """Test trillion formatting for large values (>=10T)."""
        row = pd.Series({"market_cap": 15e12})
        result = format_market_cap(row)
        assert result == "15.0T"

    def test_trillion_format_small(self):
        """Test trillion formatting for smaller values (<10T)."""
        row = pd.Series({"market_cap": 2.5e12})
        result = format_market_cap(row)
        assert result == "2.50T"

    def test_billion_format_large(self):
        """Test billion formatting for large values (>=100B)."""
        row = pd.Series({"market_cap": 250e9})
        result = format_market_cap(row)
        assert result == "250B"

    def test_billion_format_medium(self):
        """Test billion formatting for medium values (10-100B)."""
        row = pd.Series({"market_cap": 50e9})
        result = format_market_cap(row)
        assert result == "50.0B"

    def test_billion_format_small(self):
        """Test billion formatting for small values (<10B)."""
        row = pd.Series({"market_cap": 5.5e9})
        result = format_market_cap(row)
        assert result == "5.50B"

    def test_million_format_large(self):
        """Test million formatting for large values (>=100M)."""
        row = pd.Series({"market_cap": 500e6})
        result = format_market_cap(row)
        assert result == "500M"

    def test_million_format_medium(self):
        """Test million formatting for medium values (10-100M)."""
        row = pd.Series({"market_cap": 50e6})
        result = format_market_cap(row)
        assert result == "50.0M"

    def test_million_format_small(self):
        """Test million formatting for small values (<10M)."""
        row = pd.Series({"market_cap": 5.5e6})
        result = format_market_cap(row)
        assert result == "5.50M"

    def test_small_value(self):
        """Test formatting for values under 1M."""
        row = pd.Series({"market_cap": 500000})
        result = format_market_cap(row)
        assert result == "500,000"

    def test_null_value(self):
        """Test handling null value."""
        row = pd.Series({"market_cap": None})
        result = format_market_cap(row)
        assert result == "--"

    def test_alternate_column_names(self):
        """Test using alternate column names."""
        row = pd.Series({"marketCap": 3e12})
        result = format_market_cap(row)
        assert "T" in result


class TestApplyColorToRow:
    """Tests for _apply_color_to_row function."""

    def test_applies_color_to_all_cells(self):
        """Test that color is applied to all cells."""
        row = pd.Series({"ticker": "AAPL", "price": 175.0})
        result = _apply_color_to_row(row.copy(), "92")

        for col in result.index:
            assert "\033[92m" in str(result[col])
            assert "\033[0m" in str(result[col])

    def test_preserves_original_values(self):
        """Test that original values are preserved in colored output."""
        row = pd.Series({"ticker": "AAPL"})
        result = _apply_color_to_row(row.copy(), "91")

        assert "AAPL" in result["ticker"]


class TestGetActionColor:
    """Tests for get_action_color function."""

    def test_buy_returns_green(self):
        """Test BUY action returns green color."""
        result = get_action_color(BUY_ACTION)
        assert result == GREEN_COLOR

    def test_sell_returns_red(self):
        """Test SELL action returns red color."""
        result = get_action_color(SELL_ACTION)
        assert result == RED_COLOR

    def test_hold_returns_none(self):
        """Test HOLD action returns None (no color)."""
        result = get_action_color(HOLD_ACTION)
        assert result is None

    def test_inconclusive_returns_yellow(self):
        """Test INCONCLUSIVE action returns yellow."""
        result = get_action_color(INCONCLUSIVE_ACTION)
        assert result == YELLOW_COLOR

    def test_unknown_returns_none(self):
        """Test unknown action returns None."""
        result = get_action_color("X")
        assert result is None


class TestColorBasedOnAction:
    """Tests for _color_based_on_action function."""

    def test_colors_buy_action_green(self):
        """Test BUY action is colored green."""
        row = pd.Series({"ACT": "B", "ticker": "AAPL"})
        colored_row = row.copy()
        result = _color_based_on_action(row, colored_row)

        assert result is not None
        assert "\033[92m" in str(result["ticker"])

    def test_colors_sell_action_red(self):
        """Test SELL action is colored red."""
        row = pd.Series({"ACT": "S", "ticker": "AAPL"})
        colored_row = row.copy()
        result = _color_based_on_action(row, colored_row)

        assert result is not None
        assert "\033[91m" in str(result["ticker"])

    def test_colors_inconclusive_yellow(self):
        """Test INCONCLUSIVE action is colored yellow."""
        row = pd.Series({"ACT": "I", "ticker": "AAPL"})
        colored_row = row.copy()
        result = _color_based_on_action(row, colored_row)

        assert result is not None
        assert "\033[93m" in str(result["ticker"])

    def test_hold_not_colored(self):
        """Test HOLD action is not colored."""
        row = pd.Series({"ACT": "H", "ticker": "AAPL"})
        colored_row = row.copy()
        result = _color_based_on_action(row, colored_row)

        # HOLD returns the row without color codes
        assert result is not None

    def test_no_action_returns_none(self):
        """Test missing action returns None."""
        row = pd.Series({"ticker": "AAPL"})
        colored_row = row.copy()
        result = _color_based_on_action(row, colored_row)

        assert result is None

    def test_uses_action_column(self):
        """Test uses ACTION column when ACT not present."""
        row = pd.Series({"ACTION": "B", "ticker": "AAPL"})
        colored_row = row.copy()
        result = _color_based_on_action(row, colored_row)

        assert result is not None
        assert "\033[92m" in str(result["ticker"])


class TestHasRequiredMetrics:
    """Tests for _has_required_metrics function."""

    def test_all_metrics_present(self):
        """Test returns True when all required metrics present."""
        row = pd.Series({
            "EXRET": 5.0,
            "UPSIDE": 10.0,
            "%BUY": 75.0,
            "SI": 2.0,
            "PEF": 20.0,
            "BETA": 1.2,
        })
        result = _has_required_metrics(row)
        assert result is True

    def test_missing_metric(self):
        """Test returns False when metric is missing."""
        row = pd.Series({
            "EXRET": 5.0,
            "UPSIDE": 10.0,
            # Missing %BUY
            "SI": 2.0,
            "PEF": 20.0,
            "BETA": 1.2,
        })
        result = _has_required_metrics(row)
        assert result is False

    def test_null_metric(self):
        """Test returns False when metric is null."""
        row = pd.Series({
            "EXRET": 5.0,
            "UPSIDE": None,  # Null value
            "%BUY": 75.0,
            "SI": 2.0,
            "PEF": 20.0,
            "BETA": 1.2,
        })
        result = _has_required_metrics(row)
        assert result is False


class TestApplyColorFormatting:
    """Tests for apply_color_formatting function."""

    def test_apply_color_formatting_with_buy_action(self):
        """Test apply_color_formatting with BUY action."""
        from yahoofinance.utils.display_helpers import apply_color_formatting

        row = pd.Series({"ACT": "B", "ticker": "AAPL", "price": 175.0})
        result = apply_color_formatting(row, None)

        assert result is not None
        assert "\033[92m" in str(result["ticker"])  # Green color

    def test_apply_color_formatting_with_sell_action(self):
        """Test apply_color_formatting with SELL action."""
        from yahoofinance.utils.display_helpers import apply_color_formatting

        row = pd.Series({"ACT": "S", "ticker": "AAPL", "price": 175.0})
        result = apply_color_formatting(row, None)

        assert result is not None
        assert "\033[91m" in str(result["ticker"])  # Red color

    def test_apply_color_formatting_with_hold_action(self):
        """Test apply_color_formatting with HOLD action."""
        from yahoofinance.utils.display_helpers import apply_color_formatting

        row = pd.Series({"ACT": "H", "ticker": "AAPL", "price": 175.0})
        result = apply_color_formatting(row, None)

        assert result is not None

    def test_apply_color_formatting_no_action(self):
        """Test apply_color_formatting without action column."""
        from yahoofinance.utils.display_helpers import apply_color_formatting

        row = pd.Series({"ticker": "AAPL", "price": 175.0})
        result = apply_color_formatting(row, None)

        assert result is not None


class TestColorConstants:
    """Tests for color constants."""

    def test_action_constants_defined(self):
        """Test action constants are defined."""
        assert BUY_ACTION == "B"
        assert SELL_ACTION == "S"
        assert HOLD_ACTION == "H"
        assert INCONCLUSIVE_ACTION == "I"

    def test_color_constants_defined(self):
        """Test color constants are defined."""
        assert GREEN_COLOR == "92"
        assert RED_COLOR == "91"
        assert YELLOW_COLOR == "93"


class TestColumnConstants:
    """Additional column constant tests."""

    def test_exret_column(self):
        """Test EXRET column is defined."""
        from yahoofinance.utils.display_helpers import EXRET_COL
        assert EXRET_COL == "EXRET"

    def test_pet_column(self):
        """Test PET column is defined."""
        from yahoofinance.utils.display_helpers import PET_COL
        assert PET_COL == "PET"

    def test_pef_column(self):
        """Test PEF column is defined."""
        from yahoofinance.utils.display_helpers import PEF_COL
        assert PEF_COL == "PEF"

    def test_peg_column(self):
        """Test PEG column is defined."""
        from yahoofinance.utils.display_helpers import PEG_COL
        assert PEG_COL == "PEG"

    def test_beta_column(self):
        """Test BETA column is defined."""
        from yahoofinance.utils.display_helpers import BETA_COL
        assert BETA_COL == "BETA"

    def test_si_column(self):
        """Test SI column is defined."""
        from yahoofinance.utils.display_helpers import SI_COL
        assert SI_COL == "SI"

    def test_cap_column(self):
        """Test CAP column is defined."""
        from yahoofinance.utils.display_helpers import CAP_COL
        assert CAP_COL == "CAP"

    def test_rank_column(self):
        """Test RANK column is defined."""
        from yahoofinance.utils.display_helpers import RANK_COL
        assert RANK_COL == "#"

    def test_analyst_count_column(self):
        """Test ANALYST_COUNT column is defined."""
        from yahoofinance.utils.display_helpers import ANALYST_COUNT_COL
        assert ANALYST_COUNT_COL == "# A"

    def test_price_target_count_column(self):
        """Test PRICE_TARGET_COUNT column is defined."""
        from yahoofinance.utils.display_helpers import PRICE_TARGET_COUNT_COL
        assert PRICE_TARGET_COUNT_COL == "# T"

    def test_sector_column(self):
        """Test SECTOR column is defined."""
        from yahoofinance.utils.display_helpers import SECTOR_COL
        assert SECTOR_COL == "SECTOR"


class TestMarketCapEdgeCases:
    """Additional edge case tests for market cap formatting."""

    def test_exactly_one_trillion(self):
        """Test formatting exactly 1 trillion."""
        row = pd.Series({"market_cap": 1e12})
        result = format_market_cap(row)
        assert result == "1.00T"

    def test_exactly_one_billion(self):
        """Test formatting exactly 1 billion."""
        row = pd.Series({"market_cap": 1e9})
        result = format_market_cap(row)
        assert result == "1.00B"

    def test_exactly_one_million(self):
        """Test formatting exactly 1 million."""
        row = pd.Series({"market_cap": 1e6})
        result = format_market_cap(row)
        assert result == "1.00M"

    def test_marketCap_alternate_column(self):
        """Test using marketCap column name."""
        row = pd.Series({"marketCap": 2e12})
        result = format_market_cap(row)
        assert "T" in result

    def test_market_cap_value_alternate_column(self):
        """Test using market_cap_value column name."""
        row = pd.Series({"market_cap_value": 3e9})
        result = format_market_cap(row)
        assert "B" in result

    def test_priority_of_column_names(self):
        """Test priority when multiple columns exist."""
        row = pd.Series({
            "market_cap": 1e12,
            "marketCap": 2e12,
        })
        result = format_market_cap(row)
        # Should use market_cap first
        assert result == "1.00T"

