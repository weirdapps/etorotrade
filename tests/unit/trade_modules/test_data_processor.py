"""
Tests for trade_modules/data_processor.py

This module tests data processing and formatting functions.
"""

import pytest
import pandas as pd
import numpy as np

from trade_modules.data_processor import (
    process_market_data,
    format_company_names,
    _clean_company_name,
    format_numeric_columns,
    _safe_numeric_format,
    format_percentage_columns,
    _safe_percentage_format,
    format_earnings_date,
    _format_date_string,
)


class TestProcessMarketData:
    """Tests for process_market_data function."""

    def test_process_with_ma_columns(self):
        """Test processing market data with moving average columns."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "price": [175, 350, 140],
            "ma50": [170, 340, 150],
            "ma200": [160, 330, 145],
        })
        result = process_market_data(df)
        assert "in_uptrend" in result.columns
        # AAPL: 175 > 170 > 160 = uptrend
        assert result.iloc[0]["in_uptrend"] == True
        # GOOGL: 140 < 150 (ma50) = not uptrend
        assert result.iloc[2]["in_uptrend"] == False

    def test_process_without_ma_columns(self):
        """Test processing without moving average columns."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "price": [175, 350],
        })
        result = process_market_data(df)
        assert "in_uptrend" in result.columns
        assert result["in_uptrend"].all() == False

    def test_process_empty_df(self):
        """Test processing empty DataFrame."""
        df = pd.DataFrame()
        result = process_market_data(df)
        assert "in_uptrend" in result.columns


class TestCleanCompanyName:
    """Tests for _clean_company_name function."""

    def test_remove_inc_suffix(self):
        """Test removing Inc. suffix."""
        assert _clean_company_name("Apple Inc.") == "Apple"
        assert _clean_company_name("Apple, Inc.") == "Apple"
        assert _clean_company_name("Microsoft Inc") == "Microsoft"

    def test_remove_corp_suffix(self):
        """Test removing Corp. suffix."""
        assert _clean_company_name("Microsoft Corp.") == "Microsoft"
        assert _clean_company_name("Microsoft Corporation") == "Microsoft"

    def test_remove_ltd_suffix(self):
        """Test removing Ltd. suffix."""
        assert _clean_company_name("Samsung Ltd.") == "Samsung"
        assert _clean_company_name("Samsung Limited") == "Samsung"

    def test_remove_llc_suffix(self):
        """Test removing LLC suffix."""
        assert _clean_company_name("Private Company, LLC") == "Private Company"
        assert _clean_company_name("Private Company LLC") == "Private Company"

    def test_truncate_long_names(self):
        """Test that long names are truncated."""
        long_name = "A Very Very Very Long Company Name That Exceeds Thirty Characters"
        result = _clean_company_name(long_name)
        assert len(result) <= 30
        assert result.endswith("...")

    def test_handle_none_and_nan(self):
        """Test handling None and NaN values."""
        assert _clean_company_name(None) == "N/A"
        assert _clean_company_name(np.nan) == "N/A"
        assert _clean_company_name("") == "N/A"


class TestFormatCompanyNames:
    """Tests for format_company_names function."""

    def test_format_company_names(self):
        """Test formatting company names in DataFrame."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "company_name": ["Apple Inc.", "Microsoft Corporation"],
        })
        result = format_company_names(df)
        assert result.iloc[0]["company_name"] == "Apple"
        assert result.iloc[1]["company_name"] == "Microsoft"

    def test_format_with_nan_values(self):
        """Test formatting with NaN company names."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "UNKNOWN"],
            "company_name": ["Apple Inc.", np.nan],
        })
        result = format_company_names(df)
        assert result.iloc[0]["company_name"] == "Apple"
        assert result.iloc[1]["company_name"] == "N/A"

    def test_format_without_company_column(self):
        """Test formatting without company_name column."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
        })
        result = format_company_names(df)
        assert "company_name" not in result.columns


class TestSafeNumericFormat:
    """Tests for _safe_numeric_format function."""

    def test_format_valid_number(self):
        """Test formatting valid numbers."""
        assert _safe_numeric_format(123.456, "{:.2f}") == "123.46"
        assert _safe_numeric_format(100, "{:.0f}") == "100"

    def test_format_string_number(self):
        """Test formatting string representation of numbers."""
        assert _safe_numeric_format("123.456", "{:.2f}") == "123.46"

    def test_format_na_values(self):
        """Test formatting NA values."""
        assert _safe_numeric_format(None, "{:.2f}") == "--"
        assert _safe_numeric_format(np.nan, "{:.2f}") == "--"
        assert _safe_numeric_format("", "{:.2f}") == "--"
        assert _safe_numeric_format("--", "{:.2f}") == "--"

    def test_format_invalid_values(self):
        """Test formatting invalid values."""
        assert _safe_numeric_format("not a number", "{:.2f}") == "--"
        assert _safe_numeric_format([], "{:.2f}") == "--"


class TestFormatNumericColumns:
    """Tests for format_numeric_columns function."""

    def test_format_multiple_columns(self):
        """Test formatting multiple numeric columns."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "price": [175.456, 350.789],
            "volume": [1000000, 2000000],
        })
        result = format_numeric_columns(df, ["price"], "{:.2f}")
        assert result.iloc[0]["price"] == "175.46"

    def test_format_missing_column(self):
        """Test formatting with missing column."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
        })
        result = format_numeric_columns(df, ["price"], "{:.2f}")
        assert "price" not in result.columns


class TestSafePercentageFormat:
    """Tests for _safe_percentage_format function."""

    def test_format_valid_percentage(self):
        """Test formatting valid percentage."""
        assert _safe_percentage_format(25.5) == "25.5%"
        assert _safe_percentage_format(100) == "100.0%"

    def test_format_string_percentage(self):
        """Test formatting string percentage."""
        assert _safe_percentage_format("15.5") == "15.5%"

    def test_format_na_percentage(self):
        """Test formatting NA percentage values."""
        assert _safe_percentage_format(None) == "--"
        assert _safe_percentage_format(np.nan) == "--"
        assert _safe_percentage_format("") == "--"
        assert _safe_percentage_format("--") == "--"

    def test_format_invalid_percentage(self):
        """Test formatting invalid percentage values."""
        assert _safe_percentage_format("invalid") == "--"


class TestFormatPercentageColumns:
    """Tests for format_percentage_columns function."""

    def test_format_percentage_columns(self):
        """Test formatting percentage columns."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "upside": [15.5, 20.3],
            "buy_percentage": [85.0, 90.0],
        })
        result = format_percentage_columns(df, ["upside", "buy_percentage"])
        assert result.iloc[0]["upside"] == "15.5%"
        assert result.iloc[0]["buy_percentage"] == "85.0%"

    def test_format_with_nan(self):
        """Test formatting with NaN values."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "UNKNOWN"],
            "upside": [15.5, np.nan],
        })
        result = format_percentage_columns(df, ["upside"])
        assert result.iloc[0]["upside"] == "15.5%"
        assert result.iloc[1]["upside"] == "--"


class TestFormatDateString:
    """Tests for _format_date_string function."""

    def test_format_full_date(self):
        """Test formatting full date string."""
        assert _format_date_string("2024-01-15") == "2024-01-15"
        assert _format_date_string("2024-01-15 00:00:00") == "2024-01-15"

    def test_format_short_date(self):
        """Test formatting short date string."""
        assert _format_date_string("Jan 15") == "Jan 15"

    def test_format_na_date(self):
        """Test formatting NA date values."""
        assert _format_date_string(None) == "--"
        assert _format_date_string(np.nan) == "--"
        assert _format_date_string("") == "--"


class TestFormatEarningsDate:
    """Tests for format_earnings_date function."""

    def test_format_earnings_date(self):
        """Test formatting earnings date column."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "earnings_date": ["2024-01-25 00:00:00", "2024-02-15"],
        })
        result = format_earnings_date(df)
        assert result.iloc[0]["earnings_date"] == "2024-01-25"
        assert result.iloc[1]["earnings_date"] == "2024-02-15"

    def test_format_without_earnings_column(self):
        """Test formatting without earnings_date column."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
        })
        result = format_earnings_date(df)
        assert "earnings_date" not in result.columns


class TestDataProcessorIntegration:
    """Integration tests for data processor functions."""

    def test_full_processing_pipeline(self):
        """Test a complete data processing pipeline."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "company_name": ["Apple Inc.", "Microsoft Corporation", "Alphabet Inc"],
            "price": [175.456, 350.789, 140.123],
            "upside": [15.5, 20.3, 10.1],
            "ma50": [170, 340, 150],
            "ma200": [160, 330, 145],
            "earnings_date": ["2024-01-25 00:00:00", "2024-02-15", np.nan],
        })

        # Process market data
        result = process_market_data(df)

        # Format company names
        result = format_company_names(result)

        # Format percentages
        result = format_percentage_columns(result, ["upside"])

        # Format earnings
        result = format_earnings_date(result)

        assert result.iloc[0]["company_name"] == "Apple"
        assert result.iloc[0]["upside"] == "15.5%"
        assert result.iloc[0]["earnings_date"] == "2024-01-25"
        assert "in_uptrend" in result.columns

    def test_empty_df_processing(self):
        """Test processing empty DataFrame."""
        df = pd.DataFrame()
        result = process_market_data(df)
        result = format_company_names(result)
        result = format_percentage_columns(result, ["upside"])
        result = format_earnings_date(result)
        assert len(result) == 0


class TestAddMarketCapColumn:
    """Tests for add_market_cap_column function."""

    def test_add_market_cap_from_cap_strings(self):
        """Test adding market cap from CAP string column."""
        from trade_modules.data_processor import add_market_cap_column
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "SMALL"],
            "CAP": ["3.0T", "2.8T", "500M"],
        })
        result = add_market_cap_column(df)
        assert "market_cap" in result.columns
        assert result.iloc[0]["market_cap"] == pytest.approx(3.0e12)
        assert result.iloc[1]["market_cap"] == pytest.approx(2.8e12)
        assert result.iloc[2]["market_cap"] == pytest.approx(500e6)

    def test_add_market_cap_billion(self):
        """Test parsing billion market cap."""
        from trade_modules.data_processor import add_market_cap_column
        df = pd.DataFrame({
            "ticker": ["JPM"],
            "CAP": ["500B"],
        })
        result = add_market_cap_column(df)
        assert result.iloc[0]["market_cap"] == pytest.approx(500e9)

    def test_add_market_cap_without_cap_column(self):
        """Test handling DataFrame without CAP column."""
        from trade_modules.data_processor import add_market_cap_column
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [175],
        })
        result = add_market_cap_column(df)
        assert "market_cap" not in result.columns


class TestParseMarketCapString:
    """Tests for _parse_market_cap_string function."""

    def test_parse_trillion(self):
        """Test parsing trillion values."""
        from trade_modules.data_processor import _parse_market_cap_string
        assert _parse_market_cap_string("3.0T") == pytest.approx(3.0e12)
        assert _parse_market_cap_string("1.5t") == pytest.approx(1.5e12)

    def test_parse_billion(self):
        """Test parsing billion values."""
        from trade_modules.data_processor import _parse_market_cap_string
        assert _parse_market_cap_string("500B") == pytest.approx(500e9)
        assert _parse_market_cap_string("50.5b") == pytest.approx(50.5e9)

    def test_parse_million(self):
        """Test parsing million values."""
        from trade_modules.data_processor import _parse_market_cap_string
        assert _parse_market_cap_string("500M") == pytest.approx(500e6)
        assert _parse_market_cap_string("250.5m") == pytest.approx(250.5e6)

    def test_parse_plain_number(self):
        """Test parsing plain numeric string."""
        from trade_modules.data_processor import _parse_market_cap_string
        assert _parse_market_cap_string("1000000") == pytest.approx(1000000.0)

    def test_parse_invalid_values(self):
        """Test parsing invalid values returns None."""
        from trade_modules.data_processor import _parse_market_cap_string
        assert _parse_market_cap_string("--") is None
        assert _parse_market_cap_string("") is None
        assert _parse_market_cap_string(None) is None
        assert _parse_market_cap_string(np.nan) is None
        assert _parse_market_cap_string("invalid") is None


class TestCalculateExpectedReturn:
    """Tests for calculate_expected_return function."""

    def test_calculate_exret(self):
        """Test calculating expected return."""
        from trade_modules.data_processor import calculate_expected_return
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "upside": [20.0, 15.0],
            "buy_percentage": [80.0, 60.0],
        })
        result = calculate_expected_return(df)
        assert "EXRET" in result.columns
        # EXRET = upside * (buy_percentage / 100)
        assert result.iloc[0]["EXRET"] == pytest.approx(20.0 * 0.80)  # 16.0
        assert result.iloc[1]["EXRET"] == pytest.approx(15.0 * 0.60)  # 9.0

    def test_calculate_exret_missing_columns(self):
        """Test calculating EXRET with missing columns."""
        from trade_modules.data_processor import calculate_expected_return
        df = pd.DataFrame({
            "ticker": ["AAPL"],
        })
        result = calculate_expected_return(df)
        assert "EXRET" in result.columns
        assert result.iloc[0]["EXRET"] == pytest.approx(0.0)

    def test_calculate_exret_with_nan(self):
        """Test calculating EXRET with NaN values."""
        from trade_modules.data_processor import calculate_expected_return
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "upside": [20.0, np.nan],
            "buy_percentage": [80.0, 60.0],
        })
        result = calculate_expected_return(df)
        assert result.iloc[0]["EXRET"] == pytest.approx(16.0)
        assert result.iloc[1]["EXRET"] == pytest.approx(0.0)


class TestNormalizeDataframeColumns:
    """Tests for normalize_dataframe_columns function."""

    def test_rename_columns(self):
        """Test renaming columns with mapping."""
        from trade_modules.data_processor import normalize_dataframe_columns
        df = pd.DataFrame({
            "old_name": [1, 2, 3],
            "another_col": ["a", "b", "c"],
        })
        mapping = {"old_name": "new_name"}
        result = normalize_dataframe_columns(df, mapping)
        assert "new_name" in result.columns
        assert "old_name" not in result.columns
        assert "another_col" in result.columns

    def test_rename_multiple_columns(self):
        """Test renaming multiple columns."""
        from trade_modules.data_processor import normalize_dataframe_columns
        df = pd.DataFrame({
            "col1": [1],
            "col2": [2],
            "col3": [3],
        })
        mapping = {"col1": "column_one", "col2": "column_two"}
        result = normalize_dataframe_columns(df, mapping)
        assert "column_one" in result.columns
        assert "column_two" in result.columns
        assert "col3" in result.columns

    def test_rename_nonexistent_column(self):
        """Test renaming with nonexistent column in mapping."""
        from trade_modules.data_processor import normalize_dataframe_columns
        df = pd.DataFrame({
            "existing": [1, 2],
        })
        mapping = {"nonexistent": "new_name"}
        result = normalize_dataframe_columns(df, mapping)
        assert "new_name" not in result.columns
        assert "existing" in result.columns


class TestValidateRequiredColumns:
    """Tests for validate_required_columns function."""

    def test_all_columns_present(self):
        """Test validation when all required columns are present."""
        from trade_modules.data_processor import validate_required_columns
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [175],
            "volume": [1000000],
        })
        is_valid, missing = validate_required_columns(df, ["ticker", "price"])
        assert is_valid is True
        assert missing == []

    def test_missing_columns(self):
        """Test validation with missing columns."""
        from trade_modules.data_processor import validate_required_columns
        df = pd.DataFrame({
            "ticker": ["AAPL"],
        })
        is_valid, missing = validate_required_columns(df, ["ticker", "price", "volume"])
        assert is_valid is False
        assert "price" in missing
        assert "volume" in missing

    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        from trade_modules.data_processor import validate_required_columns
        df = pd.DataFrame()
        is_valid, missing = validate_required_columns(df, ["ticker"])
        assert is_valid is False
        assert "ticker" in missing

    def test_none_dataframe(self):
        """Test validation with None DataFrame."""
        from trade_modules.data_processor import validate_required_columns
        is_valid, missing = validate_required_columns(None, ["ticker"])
        assert is_valid is False


class TestCleanDataframeForOutput:
    """Tests for clean_dataframe_for_output function."""

    def test_replace_nan_in_string_columns(self):
        """Test replacing NaN in string columns with '--'."""
        from trade_modules.data_processor import clean_dataframe_for_output
        df = pd.DataFrame({
            "ticker": ["AAPL", np.nan, "GOOGL"],
        })
        result = clean_dataframe_for_output(df)
        assert result.iloc[1]["ticker"] == "--"

    def test_replace_nan_in_numeric_columns(self):
        """Test replacing NaN in numeric columns with 0."""
        from trade_modules.data_processor import clean_dataframe_for_output
        df = pd.DataFrame({
            "price": [175.0, np.nan, 140.0],
        })
        result = clean_dataframe_for_output(df)
        assert result.iloc[1]["price"] == 0

    def test_replace_infinity(self):
        """Test replacing infinity values."""
        from trade_modules.data_processor import clean_dataframe_for_output
        df = pd.DataFrame({
            "ratio": [1.5, np.inf, -np.inf],
        })
        result = clean_dataframe_for_output(df)
        assert result.iloc[1]["ratio"] == 0
        assert result.iloc[2]["ratio"] == 0


class TestApplyDataFilters:
    """Tests for apply_data_filters function."""

    def test_equality_filter(self):
        """Test filtering by equality."""
        from trade_modules.data_processor import apply_data_filters
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "sector": ["Tech", "Tech", "Tech"],
            "signal": ["BUY", "SELL", "HOLD"],
        })
        filters = {"signal": "BUY"}
        result = apply_data_filters(df, filters)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"

    def test_list_filter(self):
        """Test filtering by list inclusion."""
        from trade_modules.data_processor import apply_data_filters
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "signal": ["BUY", "SELL", "HOLD", "BUY"],
        })
        filters = {"signal": ["BUY", "HOLD"]}
        result = apply_data_filters(df, filters)
        assert len(result) == 3
        assert "MSFT" not in result["ticker"].values

    def test_range_filter_min(self):
        """Test filtering by minimum value."""
        from trade_modules.data_processor import apply_data_filters
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "SMALL"],
            "market_cap": [3e12, 2.8e12, 1e9],
        })
        filters = {"market_cap": {"min": 1e12}}
        result = apply_data_filters(df, filters)
        assert len(result) == 2
        assert "SMALL" not in result["ticker"].values

    def test_range_filter_max(self):
        """Test filtering by maximum value."""
        from trade_modules.data_processor import apply_data_filters
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "SMALL"],
            "market_cap": [3e12, 2.8e12, 1e9],
        })
        filters = {"market_cap": {"max": 1e12}}
        result = apply_data_filters(df, filters)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "SMALL"

    def test_range_filter_min_max(self):
        """Test filtering by min and max range."""
        from trade_modules.data_processor import apply_data_filters
        df = pd.DataFrame({
            "ticker": ["MEGA", "LARGE", "SMALL"],
            "market_cap": [3e12, 500e9, 1e9],
        })
        filters = {"market_cap": {"min": 100e9, "max": 1e12}}
        result = apply_data_filters(df, filters)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "LARGE"

    def test_filter_missing_column(self):
        """Test filtering with nonexistent column."""
        from trade_modules.data_processor import apply_data_filters
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
        })
        filters = {"nonexistent": "value"}
        result = apply_data_filters(df, filters)
        assert len(result) == 2


class TestDataProcessorClass:
    """Tests for DataProcessor class."""

    def test_process_ticker_data(self):
        """Test processing raw ticker data."""
        from trade_modules.data_processor import DataProcessor
        processor = DataProcessor()
        raw_data = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "current_price": 175.50,
            "target_price": 200.00,
            "market_cap": 3e12,
            "pe_forward": 25.5,
            "upside": 14.0,
            "buy_percentage": 85.0,
        }
        result = processor.process_ticker_data(raw_data)
        assert result["ticker"] == "AAPL"
        assert result["company_name"] == "Apple Inc."
        # Note: numeric fields are converted to strings by process_ticker_data
        assert float(result["market_cap"]) == pytest.approx(3e12)
        assert float(result["upside"]) == pytest.approx(14.0)

    def test_process_ticker_data_empty_ticker(self):
        """Test processing data with empty ticker."""
        from trade_modules.data_processor import DataProcessor
        processor = DataProcessor()
        raw_data = {
            "ticker": "",
            "current_price": 100,
        }
        result = processor.process_ticker_data(raw_data)
        assert result["ticker"] == "--"

    def test_safe_numeric_conversion(self):
        """Test safe numeric conversion."""
        from trade_modules.data_processor import DataProcessor
        processor = DataProcessor()
        assert processor._safe_numeric_conversion(123.45) == pytest.approx(123.45)
        assert processor._safe_numeric_conversion("100") == pytest.approx(100.0)
        assert processor._safe_numeric_conversion(None) == pytest.approx(0.0)
        assert processor._safe_numeric_conversion("--") == pytest.approx(0.0)
        assert processor._safe_numeric_conversion("invalid") == pytest.approx(0.0)

    def test_safe_percentage_conversion(self):
        """Test safe percentage conversion."""
        from trade_modules.data_processor import DataProcessor
        processor = DataProcessor()
        # Already a percentage (> 1)
        assert processor._safe_percentage_conversion(85.0) == pytest.approx(85.0)
        # Decimal that should be converted
        assert processor._safe_percentage_conversion(0.85) == pytest.approx(85.0)
        # NA values
        assert processor._safe_percentage_conversion(None) == pytest.approx(0.0)
        assert processor._safe_percentage_conversion("--") == pytest.approx(0.0)
