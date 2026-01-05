#!/usr/bin/env python3
"""
ITERATION 33: Trade Criteria Tests
Target: Test trade criteria evaluation utilities
File: yahoofinance/utils/trade_criteria.py (75 statements, 53% coverage)
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch


class TestConstants:
    """Test module constants."""

    def test_action_constants(self):
        """Verify action constants."""
        from yahoofinance.utils.trade_criteria import (
            BUY_ACTION, SELL_ACTION, HOLD_ACTION, INCONCLUSIVE_ACTION, NO_ACTION
        )

        assert BUY_ACTION == "B"
        assert SELL_ACTION == "S"
        assert HOLD_ACTION == "H"
        assert INCONCLUSIVE_ACTION == "I"
        assert NO_ACTION == ""

    def test_column_constants(self):
        """Verify column name constants."""
        from yahoofinance.utils.trade_criteria import (
            UPSIDE, BUY_PERCENTAGE_COL, PE_FORWARD, PE_TRAILING
        )

        assert UPSIDE == "upside"
        assert BUY_PERCENTAGE_COL == "buy_percentage"
        assert PE_FORWARD == "pe_forward"
        assert PE_TRAILING == "pe_trailing"


class TestCheckConfidenceCriteria:
    """Test check_confidence_criteria function."""

    def test_sufficient_coverage(self):
        """Accept row with sufficient analyst coverage."""
        from yahoofinance.utils.trade_criteria import check_confidence_criteria

        row = {
            "analyst_count": 10,
            "total_ratings": 10
        }

        result = check_confidence_criteria(row, {})

        assert result is True

    def test_insufficient_analyst_count(self):
        """Reject row with insufficient analyst count."""
        from yahoofinance.utils.trade_criteria import check_confidence_criteria

        row = {
            "analyst_count": 3,
            "total_ratings": 10
        }

        result = check_confidence_criteria(row, {})

        assert result is False

    def test_insufficient_total_ratings(self):
        """Reject row with insufficient total ratings."""
        from yahoofinance.utils.trade_criteria import check_confidence_criteria

        row = {
            "analyst_count": 10,
            "total_ratings": 3
        }

        result = check_confidence_criteria(row, {})

        assert result is False

    def test_legacy_column_names(self):
        """Handle legacy column names."""
        from yahoofinance.utils.trade_criteria import check_confidence_criteria

        row = {
            "# A": 10,
            "# T": 10
        }

        result = check_confidence_criteria(row, {})

        assert result is True

    def test_none_values(self):
        """Handle None values gracefully."""
        from yahoofinance.utils.trade_criteria import check_confidence_criteria

        row = {
            "analyst_count": None,
            "total_ratings": None
        }

        result = check_confidence_criteria(row, {})

        assert result is False

    def test_string_numbers(self):
        """Convert string numbers to numeric."""
        from yahoofinance.utils.trade_criteria import check_confidence_criteria

        row = {
            "analyst_count": "10",
            "total_ratings": "10"
        }

        result = check_confidence_criteria(row, {})

        assert result is True


class TestNormalizeRowForCriteria:
    """Test normalize_row_for_criteria function."""

    def test_normalize_upside(self):
        """Normalize UPSIDE field."""
        from yahoofinance.utils.trade_criteria import normalize_row_for_criteria

        row = {"UPSIDE": 15.5}

        result = normalize_row_for_criteria(row)

        assert result["upside"] == pytest.approx(15.5)

    def test_normalize_buy_percentage(self):
        """Normalize % BUY field."""
        from yahoofinance.utils.trade_criteria import normalize_row_for_criteria

        row = {"% BUY": 60.0}

        result = normalize_row_for_criteria(row)

        assert result["buy_percentage"] == pytest.approx(60.0)

    def test_normalize_pe_ratios(self):
        """Normalize P/E ratio fields."""
        from yahoofinance.utils.trade_criteria import normalize_row_for_criteria

        row = {
            "PEF": 20.5,
            "PET": 22.0
        }

        result = normalize_row_for_criteria(row)

        assert result["pe_forward"] == pytest.approx(20.5)
        assert result["pe_trailing"] == pytest.approx(22.0)

    def test_preserve_other_fields(self):
        """Preserve fields not explicitly normalized."""
        from yahoofinance.utils.trade_criteria import normalize_row_for_criteria

        row = {
            "UPSIDE": 15.0,
            "custom_field": "value",
            "another_field": 42
        }

        result = normalize_row_for_criteria(row)

        assert result["custom_field"] == "value"
        assert result["another_field"] == 42

    def test_normalize_all_standard_fields(self):
        """Normalize all standard fields."""
        from yahoofinance.utils.trade_criteria import normalize_row_for_criteria

        row = {
            "UPSIDE": 15.0,
            "% BUY": 60.0,
            "PEF": 20.5,
            "PET": 22.0,
            "PEG": 1.5,
            "BETA": 1.2,
            "# A": 10,
            "# T": 10
        }

        result = normalize_row_for_criteria(row)

        assert result["upside"] == pytest.approx(15.0)
        assert result["buy_percentage"] == pytest.approx(60.0)
        assert result["pe_forward"] == pytest.approx(20.5)
        assert result["pe_trailing"] == pytest.approx(22.0)
        assert result["peg_ratio"] == pytest.approx(1.5)
        assert result["beta"] == pytest.approx(1.2)
        assert result["analyst_count"] == 10
        assert result["total_ratings"] == 10


class TestNormalizeRowColumns:
    """Test normalize_row_columns function."""

    def test_normalize_dict(self):
        """Normalize dictionary row."""
        from yahoofinance.utils.trade_criteria import normalize_row_columns

        row = {"UPSIDE": 15.0, "% BUY": 60.0}

        result = normalize_row_columns(row)

        assert result["upside"] == pytest.approx(15.0)
        assert result["buy_percentage"] == pytest.approx(60.0)

    def test_normalize_series(self):
        """Normalize pandas Series row."""
        from yahoofinance.utils.trade_criteria import normalize_row_columns

        row = pd.Series({"UPSIDE": 15.0, "% BUY": 60.0})

        result = normalize_row_columns(row)

        assert result["upside"] == pytest.approx(15.0)
        assert result["buy_percentage"] == pytest.approx(60.0)

    def test_with_column_mapping(self):
        """Accept column mapping parameter (for compatibility)."""
        from yahoofinance.utils.trade_criteria import normalize_row_columns

        row = {"UPSIDE": 15.0}
        mapping = {"UPSIDE": "upside"}

        result = normalize_row_columns(row, column_mapping=mapping)

        assert result["upside"] == pytest.approx(15.0)


class TestCalculateAction:
    """Test calculate_action function."""

    @patch('yahoofinance.utils.trade_criteria.calculate_action_for_row')
    def test_calculate_action_calls_delegate(self, mock_calc):
        """Calculate action delegates to calculate_action_for_row."""
        from yahoofinance.utils.trade_criteria import calculate_action

        mock_calc.return_value = ("B", "Meets criteria")
        ticker_data = {"UPSIDE": 15.0}

        result = calculate_action(ticker_data)

        assert result == "B"
        mock_calc.assert_called_once()


class TestEvaluateTradeCriteria:
    """Test evaluate_trade_criteria function."""

    @patch('yahoofinance.utils.trade_criteria.calculate_action')
    def test_evaluate_delegates_to_calculate_action(self, mock_calc):
        """Evaluate criteria delegates to calculate_action."""
        from yahoofinance.utils.trade_criteria import evaluate_trade_criteria

        mock_calc.return_value = "S"
        ticker_data = {"UPSIDE": 5.0}

        result = evaluate_trade_criteria(ticker_data)

        assert result == "S"
        mock_calc.assert_called_once_with(ticker_data)


class TestFormatNumericValues:
    """Test format_numeric_values function."""

    def test_format_percentage_strings(self):
        """Format percentage strings to floats."""
        from yahoofinance.utils.trade_criteria import format_numeric_values

        df = pd.DataFrame({
            "value": ["10%", "20%", "30%"]
        })

        result = format_numeric_values(df, ["value"])

        assert result["value"].iloc[0] == pytest.approx(10.0)
        assert result["value"].iloc[1] == pytest.approx(20.0)
        assert result["value"].iloc[2] == pytest.approx(30.0)

    def test_format_numeric_strings(self):
        """Format numeric strings to floats."""
        from yahoofinance.utils.trade_criteria import format_numeric_values

        df = pd.DataFrame({
            "value": ["10", "20", "30"]
        })

        result = format_numeric_values(df, ["value"])

        assert result["value"].iloc[0] == pytest.approx(10.0)
        assert result["value"].iloc[1] == pytest.approx(20.0)
        assert result["value"].iloc[2] == pytest.approx(30.0)

    def test_handle_missing_values(self):
        """Handle missing values as NaN."""
        from yahoofinance.utils.trade_criteria import format_numeric_values

        df = pd.DataFrame({
            "value": [10, None, 30]
        })

        result = format_numeric_values(df, ["value"])

        assert result["value"].iloc[0] == pytest.approx(10.0)
        assert pd.isna(result["value"].iloc[1])
        assert result["value"].iloc[2] == pytest.approx(30.0)

    def test_handle_invalid_values(self):
        """Coerce invalid values to NaN."""
        from yahoofinance.utils.trade_criteria import format_numeric_values

        df = pd.DataFrame({
            "value": ["10", "invalid", "30"]
        })

        result = format_numeric_values(df, ["value"])

        assert result["value"].iloc[0] == pytest.approx(10.0)
        assert pd.isna(result["value"].iloc[1])
        assert result["value"].iloc[2] == pytest.approx(30.0)

    def test_preserve_non_numeric_columns(self):
        """Preserve columns not in numeric_columns list."""
        from yahoofinance.utils.trade_criteria import format_numeric_values

        df = pd.DataFrame({
            "numeric": ["10", "20"],
            "text": ["A", "B"]
        })

        result = format_numeric_values(df, ["numeric"])

        assert result["numeric"].iloc[0] == pytest.approx(10.0)
        assert result["text"].iloc[0] == "A"

    def test_handle_already_numeric(self):
        """Handle already numeric values."""
        from yahoofinance.utils.trade_criteria import format_numeric_values

        df = pd.DataFrame({
            "value": [10.0, 20.0, 30.0]
        })

        result = format_numeric_values(df, ["value"])

        assert result["value"].iloc[0] == pytest.approx(10.0)
        assert result["value"].iloc[1] == pytest.approx(20.0)


class TestCalculateActionForRow:
    """Test calculate_action_for_row function."""

    @patch('yahoofinance.utils.trade_criteria.calculate_action')
    def test_calculate_action_for_dict_row(self, mock_calc_action):
        """Calculate action for dictionary row."""
        from yahoofinance.utils.trade_criteria import calculate_action_for_row

        # Mock the calculate_action function from analysis_engine
        with patch('trade_modules.analysis_engine.calculate_action') as mock_engine:
            result_df = pd.DataFrame([{"BS": "B"}])
            mock_engine.return_value = result_df

            row = {"UPSIDE": 15.0, "% BUY": 60.0}
            action, reason = calculate_action_for_row(row, {})

            assert action == "B"
            assert "centralized TradeConfig" in reason

    @patch('yahoofinance.utils.trade_criteria.calculate_action')
    def test_calculate_action_for_series_row(self, mock_calc_action):
        """Calculate action for pandas Series row."""
        from yahoofinance.utils.trade_criteria import calculate_action_for_row

        with patch('trade_modules.analysis_engine.calculate_action') as mock_engine:
            result_df = pd.DataFrame([{"BS": "S"}])
            mock_engine.return_value = result_df

            row = pd.Series({"UPSIDE": 5.0, "% BUY": 40.0})
            action, reason = calculate_action_for_row(row, {})

            assert action == "S"


class TestModuleStructure:
    """Test module structure."""

    def test_module_has_logger(self):
        """Module has logger."""
        from yahoofinance.utils import trade_criteria

        assert hasattr(trade_criteria, 'logger')

    def test_module_docstring(self):
        """Module has docstring."""
        from yahoofinance.utils import trade_criteria

        assert trade_criteria.__doc__ is not None
        assert "Trade criteria" in trade_criteria.__doc__
