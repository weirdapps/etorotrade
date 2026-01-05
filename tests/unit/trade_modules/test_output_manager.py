#!/usr/bin/env python3
"""
ITERATION 14: Output Manager Tests
Target: Test output management, file exports, and display formatting
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from trade_modules.output_manager import (
    ensure_output_directory,
    _setup_output_files,
    _prepare_csv_dataframe,
    _clean_si_value,
    _add_ranking_column,
    get_column_alignments,
    _get_color_by_title,
    _apply_color_to_dataframe,
    create_empty_results_file,
    _sort_display_dataframe,
    format_display_dataframe,
    _format_price_value,
    _format_percentage_value,
    _format_numeric_value,
    _format_date_value,
    _format_size_value,
    _format_beta_value,
    _format_pe_value,
    _format_peg_value,
    _format_eg_pp_value,
    _format_basic_percentage_value,
    _format_buy_percentage_value,
    prepare_display_dataframe,
    export_results_to_files,
    display_and_save_results,
    OutputManager,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_RESET,
)


class TestEnsureOutputDirectory:
    """Test output directory creation."""

    def test_ensure_output_directory_creates_new_dir(self):
        """Create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "test_output")
            assert not os.path.exists(test_dir)

            ensure_output_directory(test_dir)

            assert os.path.exists(test_dir)

    def test_ensure_output_directory_existing_dir(self):
        """Handle existing directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Directory already exists
            ensure_output_directory(tmpdir)
            assert os.path.exists(tmpdir)


class TestSetupOutputFiles:
    """Test output file path setup."""

    def test_setup_output_files_market_source(self):
        """Market source returns market-specific file paths."""
        buy_file, sell_file, hold_file = _setup_output_files("market")

        assert "market_buy.csv" in buy_file
        assert "market_sell.csv" in sell_file
        assert "market_hold.csv" in hold_file

    def test_setup_output_files_portfolio_source(self):
        """Portfolio source returns portfolio-specific file paths."""
        buy_file, sell_file, hold_file = _setup_output_files("portfolio")

        assert "portfolio_buy.csv" in buy_file
        assert "portfolio_sell.csv" in sell_file
        assert "portfolio_hold.csv" in hold_file

    def test_setup_output_files_manual_source(self):
        """Manual source returns default file paths."""
        buy_file, sell_file, hold_file = _setup_output_files("manual")

        assert isinstance(buy_file, str)
        assert isinstance(sell_file, str)
        assert isinstance(hold_file, str)


class TestCleanSIValue:
    """Test short interest value cleaning."""

    def test_clean_si_value_valid_number(self):
        """Clean valid short interest number."""
        result = _clean_si_value(2.5)
        assert result == "2.5%"

    def test_clean_si_value_with_percent(self):
        """Clean value already containing %."""
        result = _clean_si_value("3.2%")
        assert result == "3.2%"

    def test_clean_si_value_none(self):
        """Handle None value."""
        result = _clean_si_value(None)
        assert result == "--"

    def test_clean_si_value_empty_string(self):
        """Handle empty string."""
        result = _clean_si_value("")
        assert result == "--"

    def test_clean_si_value_na(self):
        """Handle pd.NA value."""
        result = _clean_si_value(pd.NA)
        assert result == "--"

    def test_clean_si_value_invalid(self):
        """Handle invalid value."""
        result = _clean_si_value("invalid")
        assert result == "--"


class TestAddRankingColumn:
    """Test ranking column addition."""

    def test_add_ranking_column(self):
        """Add ranking column to dataframe."""
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOGL"]})
        result = _add_ranking_column(df)

        assert "#" in result.columns
        assert list(result["#"]) == [1, 2, 3]
        assert result.columns[0] == "#"


class TestGetColumnAlignments:
    """Test column alignment logic."""

    def test_get_column_alignments_left_aligned(self):
        """Left-aligned columns."""
        df = pd.DataFrame({"#": [1], "TICKER": ["AAPL"], "NAME": ["Apple"], "BS": ["B"]})
        alignments = get_column_alignments(df)

        assert alignments == ["left", "left", "left", "left"]

    def test_get_column_alignments_right_aligned(self):
        """Right-aligned numeric columns."""
        df = pd.DataFrame({"PRICE": [150.0], "TARGET": [180.0], "UPSIDE": [20.0]})
        alignments = get_column_alignments(df)

        assert alignments == ["right", "right", "right"]

    def test_get_column_alignments_center_aligned(self):
        """Center-aligned other columns."""
        df = pd.DataFrame({"CAP": ["3T"], "BETA": [1.2]})
        alignments = get_column_alignments(df)

        assert alignments == ["center", "center"]


class TestGetColorByTitle:
    """Test color code selection by title."""

    def test_get_color_by_title_buy(self):
        """Buy-related title returns green."""
        assert _get_color_by_title("Buy Opportunities") == COLOR_GREEN

    def test_get_color_by_title_sell(self):
        """Sell-related title returns red."""
        assert _get_color_by_title("Sell Candidates") == COLOR_RED

    def test_get_color_by_title_hold(self):
        """Hold-related title returns yellow."""
        # "hold" keyword needs to be in title_lower
        assert _get_color_by_title("hold") == COLOR_YELLOW

    def test_get_color_by_title_neutral(self):
        """Neutral title returns reset."""
        assert _get_color_by_title("Market Analysis") == COLOR_RESET


class TestFormatPriceValue:
    """Test price value formatting."""

    def test_format_price_value_large(self):
        """Format large price (>=1000) with 0 decimals."""
        result = _format_price_value(1250.75)
        assert result == "$1,251"

    def test_format_price_value_medium(self):
        """Format medium price (>=10) with 1 decimal."""
        result = _format_price_value(125.75)
        assert result == "$125.8"

    def test_format_price_value_small(self):
        """Format small price (<10) with 2 decimals."""
        result = _format_price_value(5.125)
        assert result == "$5.12"

    def test_format_price_value_none(self):
        """Handle None value."""
        result = _format_price_value(None)
        assert result == "--"

    def test_format_price_value_zero(self):
        """Handle zero value."""
        result = _format_price_value(0)
        assert result == "--"


class TestFormatPercentageValue:
    """Test percentage value formatting."""

    def test_format_percentage_value_normal(self):
        """Format normal percentage."""
        result = _format_percentage_value(15.5)
        assert result == "15.5%"

    def test_format_percentage_value_high(self):
        """Format high percentage (>99) as H."""
        result = _format_percentage_value(150.0)
        assert result == "H"

    def test_format_percentage_value_low(self):
        """Format low percentage (<-99) as L."""
        result = _format_percentage_value(-150.0)
        assert result == "L"

    def test_format_percentage_value_none(self):
        """Handle None value."""
        result = _format_percentage_value(None)
        assert result == "--"


class TestFormatNumericValue:
    """Test numeric value formatting."""

    def test_format_numeric_value_valid(self):
        """Format valid numeric value."""
        result = _format_numeric_value(12.345)
        assert result == "12.3"

    def test_format_numeric_value_none(self):
        """Handle None value."""
        result = _format_numeric_value(None)
        assert result == "--"


class TestFormatDateValue:
    """Test date value formatting."""

    def test_format_date_value_with_dashes(self):
        """Format date with dashes to YYYYMMDD."""
        result = _format_date_value("2024-01-15")
        assert result == "20240115"

    def test_format_date_value_already_formatted(self):
        """Handle already formatted date."""
        result = _format_date_value("20240115")
        assert result == "20240115"

    def test_format_date_value_none(self):
        """Handle None value."""
        result = _format_date_value(None)
        assert result == "--"


class TestFormatSizeValue:
    """Test position size value formatting."""

    def test_format_size_value_valid(self):
        """Format valid position size."""
        result = _format_size_value(2.5)
        # format_position_size returns string representation
        assert isinstance(result, str)
        assert result != "--"

    def test_format_size_value_none(self):
        """Handle None value."""
        result = _format_size_value(None)
        assert result == "--"


class TestFormatBetaValue:
    """Test beta value formatting."""

    def test_format_beta_value_normal(self):
        """Format normal beta value."""
        result = _format_beta_value(1.25)
        # 1 decimal formatting
        assert result == "1.2"

    def test_format_beta_value_high(self):
        """Format high beta (>9) as H."""
        result = _format_beta_value(12.5)
        assert result == "H"

    def test_format_beta_value_low(self):
        """Format low beta (<-9) as L."""
        result = _format_beta_value(-12.5)
        assert result == "L"


class TestFormatPEValue:
    """Test P/E ratio value formatting."""

    def test_format_pe_value_normal(self):
        """Format normal P/E value."""
        result = _format_pe_value(20.5)
        assert result == "20.5"

    def test_format_pe_value_high(self):
        """Format high P/E (>99) as H."""
        result = _format_pe_value(150.0)
        assert result == "H"


class TestFormatPEGValue:
    """Test PEG ratio value formatting."""

    def test_format_peg_value_normal(self):
        """Format normal PEG value."""
        result = _format_peg_value(1.5)
        assert result == "1.5"

    def test_format_peg_value_high(self):
        """Format high PEG (>9) as H."""
        result = _format_peg_value(12.0)
        assert result == "H"


class TestFormatEGPPValue:
    """Test EG/PP value formatting."""

    def test_format_eg_pp_value_normal(self):
        """Format normal EG/PP value."""
        result = _format_eg_pp_value(25.5)
        assert result == "25.5"

    def test_format_eg_pp_value_high(self):
        """Format high value (>99) as H."""
        result = _format_eg_pp_value(150.0)
        assert result == "H"

    def test_format_eg_pp_value_low(self):
        """Format low value (<-99) as L."""
        result = _format_eg_pp_value(-150.0)
        assert result == "L"


class TestFormatBasicPercentageValue:
    """Test basic percentage value formatting."""

    def test_format_basic_percentage_value_valid(self):
        """Format valid percentage."""
        result = _format_basic_percentage_value(2.5)
        assert result == "2.5%"

    def test_format_basic_percentage_value_zero(self):
        """Handle zero value."""
        result = _format_basic_percentage_value(0)
        assert result == "--"


class TestFormatBuyPercentageValue:
    """Test buy percentage value formatting."""

    def test_format_buy_percentage_value_valid(self):
        """Format valid buy percentage (no decimals)."""
        result = _format_buy_percentage_value(75.8)
        assert result == "76%"

    def test_format_buy_percentage_value_none(self):
        """Handle None value."""
        result = _format_buy_percentage_value(None)
        assert result == "--"


class TestFormatDisplayDataframe:
    """Test comprehensive dataframe formatting."""

    def test_format_display_dataframe_price_columns(self):
        """Format price columns correctly."""
        df = pd.DataFrame({
            "PRICE": [150.5, 25.75, 5.125],
            "TARGET": [180.0, 30.0, 6.0]
        })
        result = format_display_dataframe(df)

        # Check that values are formatted (contain $ or --)
        assert all("$" in str(v) or v == "--" for v in result["PRICE"])

    def test_format_display_dataframe_percentage_columns(self):
        """Format percentage columns correctly."""
        df = pd.DataFrame({
            "UPSIDE": [15.5, 150.0, -150.0],
            "EXRET": [10.0, None, 5.0]
        })
        result = format_display_dataframe(df)

        # Check UPSIDE formatting
        assert result["UPSIDE"].iloc[0] == "15.5%"
        assert result["UPSIDE"].iloc[1] == "H"  # >99
        assert result["UPSIDE"].iloc[2] == "L"  # <-99


class TestPrepareDisplayDataframe:
    """Test display dataframe preparation."""

    def test_prepare_display_dataframe_adds_ranking(self):
        """Prepare dataframe adds ranking column."""
        df = pd.DataFrame({
            "TICKER": ["AAPL", "MSFT"],
            "PRICE": [150.0, 300.0]
        })
        result = prepare_display_dataframe(df)

        assert "#" in result.columns
        assert result.columns[0] == "#"

    def test_prepare_display_dataframe_formats_values(self):
        """Prepare dataframe formats all values."""
        df = pd.DataFrame({
            "TICKER": ["AAPL"],
            "PRICE": [150.5]
        })
        result = prepare_display_dataframe(df)

        # Price should be formatted
        assert "$" in str(result["PRICE"].iloc[0])


class TestCreateEmptyResultsFile:
    """Test empty results file creation."""

    @patch('trade_modules.output_manager.HTMLGenerator')
    def test_create_empty_results_file(self, mock_html_gen):
        """Create empty CSV and HTML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "empty_results.csv")

            create_empty_results_file(output_file)

            # CSV should exist
            assert os.path.exists(output_file)

            # Should have called HTML generator
            mock_html_gen.return_value.generate_stock_table.assert_called_once()


class TestDisplayAndSaveResults:
    """Test display and save results functionality."""

    @patch('trade_modules.output_manager.HTMLGenerator')
    @patch('trade_modules.output_manager.tabulate')
    @patch('builtins.print')
    def test_display_and_save_results_with_data(self, mock_print, mock_tabulate, mock_html_gen):
        """Display and save results with data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.csv")
            df = pd.DataFrame({
                "#": [1],
                "TICKER": ["AAPL"],
                "PRICE": ["$150.00"]
            })

            display_and_save_results(df, "Test Results", output_file)

            # CSV should exist
            assert os.path.exists(output_file)

            # Should have printed
            assert mock_print.called

    @patch('builtins.print')
    def test_display_and_save_results_empty(self, mock_print):
        """Display and save results with empty dataframe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "empty.csv")
            df = pd.DataFrame()

            display_and_save_results(df, "Empty Results", output_file)

            # Should create empty file
            assert os.path.exists(output_file)


class TestExportResultsToFiles:
    """Test results export to files."""

    @patch('trade_modules.output_manager.display_and_save_results')
    def test_export_results_to_files_all_types(self, mock_display_save):
        """Export all result types (buy/sell/hold)."""
        results_dict = {
            "buy_opportunities": pd.DataFrame({"ticker": ["AAPL"]}),
            "sell_candidates": pd.DataFrame({"ticker": ["MSFT"]}),
            "hold_candidates": pd.DataFrame({"ticker": ["GOOGL"]})
        }

        output_files = export_results_to_files(results_dict, "market")

        # Should have called display_and_save_results 3 times
        assert mock_display_save.call_count == 3

        # Should return output file paths
        assert "buy" in output_files
        assert "sell" in output_files
        assert "hold" in output_files


class TestOutputManager:
    """Test OutputManager class."""

    def test_output_manager_initialization(self):
        """Initialize OutputManager."""
        manager = OutputManager()

        assert manager.output_dir is not None
        assert manager.logger is not None

    def test_output_manager_custom_output_dir(self):
        """Initialize with custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(output_dir=tmpdir)
            assert manager.output_dir == tmpdir

    @patch('trade_modules.output_manager.prepare_display_dataframe')
    @patch('trade_modules.output_manager.export_results_to_files')
    def test_save_analysis_results(self, mock_export, mock_prepare):
        """Save analysis results."""
        manager = OutputManager()

        # Mock prepare to return same dataframe
        mock_prepare.side_effect = lambda df: df
        mock_export.return_value = {"buy": "buy.csv"}

        results = {
            "buy_opportunities": pd.DataFrame({"ticker": ["AAPL"]})
        }

        output_files = manager.save_analysis_results(results)

        assert mock_export.called
        assert isinstance(output_files, dict)

    def test_generate_summary_report(self):
        """Generate summary report text."""
        manager = OutputManager()

        results = {
            "buy_opportunities": pd.DataFrame({"ticker": ["AAPL", "MSFT"]}),
            "sell_candidates": pd.DataFrame({"ticker": ["GOOGL"]})
        }

        summary = manager.generate_summary_report(results)

        assert "ANALYSIS SUMMARY" in summary
        assert "Buy Opportunities" in summary
        assert "2 items" in summary
