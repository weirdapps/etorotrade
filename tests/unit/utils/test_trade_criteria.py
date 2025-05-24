"""
Unit tests for trade criteria utilities.
"""

import pandas as pd
import pytest

from yahoofinance.utils.trade_criteria import (
    calculate_action_for_row,
    check_confidence_criteria,
    check_pe_condition,
    format_numeric_values,
    meets_buy_criteria,
    meets_sell_criteria,
)


# Test trading criteria config
@pytest.fixture
def trading_criteria():
    return {
        "CONFIDENCE": {
            "MIN_ANALYST_COUNT": 5,
            "MIN_PRICE_TARGETS": 5,
        },
        "SELL": {
            "SELL_MAX_UPSIDE": 5.0,
            "SELL_MIN_BUY_PERCENTAGE": 65.0,
            "SELL_MIN_FORWARD_PE": 45.0,
            "SELL_MIN_PEG": 3.0,
            "SELL_MIN_SHORT_INTEREST": 4.0,
            "SELL_MIN_BETA": 3.0,
            "SELL_MAX_EXRET": 10.0,
        },
        "BUY": {
            "BUY_MIN_UPSIDE": 20.0,
            "BUY_MIN_BUY_PERCENTAGE": 85.0,  # Updated to match config.py
            "BUY_MIN_BETA": 0.2,
            "BUY_MAX_BETA": 3.0,
            "BUY_MIN_FORWARD_PE": 0.5,
            "BUY_MAX_FORWARD_PE": 45.0,
            "BUY_MAX_PEG": 3.0,
            "BUY_MAX_SHORT_INTEREST": 3.0,
            "BUY_MIN_EXRET": 10.0,
        },
    }


# Test data for different scenarios
@pytest.fixture
def test_data():
    return {
        "buy_stock": {
            "ticker": "BUY",
            "upside": 25.0,
            "buy_percentage": 90.0,  # Increased to pass the new 85% threshold
            "pe_trailing": 20.0,
            "pe_forward": 15.0,
            "peg_ratio": 1.2,
            "beta": 1.5,
            "short_percent": 2.0,
            "analyst_count": 10,
            "total_ratings": 8,
            "EXRET": 22.5,  # 25 * 0.9
        },
        "sell_stock_low_upside": {
            "ticker": "SELL1",
            "upside": 3.0,  # Below MAX_UPSIDE threshold
            "buy_percentage": 70.0,
            "pe_trailing": 25.0,
            "pe_forward": 20.0,
            "peg_ratio": 2.0,
            "beta": 1.8,
            "short_percent": 3.5,
            "analyst_count": 8,
            "total_ratings": 7,
            "EXRET": 2.1,  # 3 * 0.7
        },
        "sell_stock_high_beta": {
            "ticker": "SELL2",
            "upside": 15.0,
            "buy_percentage": 75.0,
            "pe_trailing": 22.0,
            "pe_forward": 20.0,
            "peg_ratio": 1.8,
            "beta": 3.5,  # Above MAX_BETA threshold
            "short_percent": 3.0,
            "analyst_count": 7,
            "total_ratings": 6,
            "EXRET": 11.25,  # 15 * 0.75
        },
        "hold_stock": {
            "ticker": "HOLD",
            "upside": 12.0,  # Not enough for BUY
            "buy_percentage": 80.0,  # Not enough for BUY
            "pe_trailing": 18.0,
            "pe_forward": 16.0,
            "peg_ratio": 1.5,
            "beta": 1.2,
            "short_percent": 2.5,
            "analyst_count": 6,
            "total_ratings": 5,
            "EXRET": 9.6,  # 12 * 0.8
        },
        "insufficient_confidence": {
            "ticker": "LOWCONF",
            "upside": 30.0,
            "buy_percentage": 90.0,
            "pe_trailing": 15.0,
            "pe_forward": 12.0,
            "peg_ratio": 1.0,
            "beta": 1.0,
            "short_percent": 1.0,
            "analyst_count": 3,  # Below MIN_ANALYST_COUNT
            "total_ratings": 2,  # Below MIN_PRICE_TARGETS
            "EXRET": 27.0,  # 30 * 0.9
        },
    }


def test_check_confidence_criteria(trading_criteria, test_data):
    """Test confidence criteria checking."""
    # Should meet confidence criteria
    assert check_confidence_criteria(test_data["buy_stock"], trading_criteria) is True
    assert check_confidence_criteria(test_data["sell_stock_low_upside"], trading_criteria) is True
    assert check_confidence_criteria(test_data["hold_stock"], trading_criteria) is True

    # Should not meet confidence criteria
    assert (
        check_confidence_criteria(test_data["insufficient_confidence"], trading_criteria) is False
    )


def test_meets_sell_criteria(trading_criteria, test_data):
    """Test sell criteria checking."""
    # Should meet sell criteria due to low upside
    is_sell, reason = meets_sell_criteria(test_data["sell_stock_low_upside"], trading_criteria)
    assert is_sell is True
    assert "upside" in reason.lower()

    # Should meet sell criteria due to high beta
    is_sell, reason = meets_sell_criteria(test_data["sell_stock_high_beta"], trading_criteria)
    assert is_sell is True
    assert "beta" in reason.lower()

    # Should not meet sell criteria
    is_sell, reason = meets_sell_criteria(test_data["buy_stock"], trading_criteria)
    assert is_sell is False
    assert reason is None


def test_meets_buy_criteria(trading_criteria, test_data):
    """Test buy criteria checking."""
    # Should meet buy criteria
    is_buy, reason = meets_buy_criteria(test_data["buy_stock"], trading_criteria)
    assert is_buy is True
    assert reason is None

    # Should not meet buy criteria due to low upside
    is_buy, reason = meets_buy_criteria(test_data["hold_stock"], trading_criteria)
    assert is_buy is False
    assert "upside" in reason.lower()

    # First check shows insufficient upside, so test for that
    is_buy, reason = meets_buy_criteria(test_data["sell_stock_high_beta"], trading_criteria)
    assert is_buy is False
    assert "upside" in reason.lower()


def test_check_pe_condition(trading_criteria, test_data):
    """Test PE condition checking."""
    # Should pass PE condition (forward PE < trailing PE)
    assert check_pe_condition(test_data["buy_stock"], trading_criteria["BUY"]) is True

    # Create a test case for negative trailing PE (growth case)
    growth_stock = test_data["buy_stock"].copy()
    growth_stock["pe_trailing"] = -5.0
    assert check_pe_condition(growth_stock, trading_criteria["BUY"]) is True

    # Should fail PE condition (forward PE > trailing PE)
    worse_pe = test_data["buy_stock"].copy()
    worse_pe["pe_forward"] = 25.0  # Higher than trailing PE of 20.0
    assert check_pe_condition(worse_pe, trading_criteria["BUY"]) is False


def test_calculate_action_for_row(trading_criteria, test_data):
    """Test action calculation for individual rows."""
    # Should be a BUY
    action, reason = calculate_action_for_row(test_data["buy_stock"], trading_criteria)
    assert action == "B"
    assert "meets all buy criteria" in reason.lower()

    # Should be a SELL due to low upside
    action, reason = calculate_action_for_row(test_data["sell_stock_low_upside"], trading_criteria)
    assert action == "S"
    assert "upside" in reason.lower()

    # Should be a SELL due to high beta
    action, reason = calculate_action_for_row(test_data["sell_stock_high_beta"], trading_criteria)
    assert action == "S"
    assert "beta" in reason.lower()

    # Hold stock has EXRET < 10, so it should be a SELL
    action, reason = calculate_action_for_row(test_data["hold_stock"], trading_criteria)
    assert action == "S"
    assert "expected return" in reason.lower()

    # Should have 'I' action due to insufficient confidence
    action, reason = calculate_action_for_row(
        test_data["insufficient_confidence"], trading_criteria
    )
    assert action == "I"  # Updated to match INCONCLUSIVE_ACTION
    assert "insufficient" in reason.lower()


def test_format_numeric_values():
    """Test formatting of numeric values."""
    # Create test DataFrame with various numeric formats
    df = pd.DataFrame(
        {
            "upside": ["25%", "15.5%", "10", None],
            "beta": ["1.5", 2.0, None, "3.2"],
            "text": ["A", "B", "C", "D"],  # Non-numeric column
        }
    )

    numeric_columns = ["upside", "beta"]
    result_df = format_numeric_values(df, numeric_columns)

    # Check that values are properly converted - using pytest.approx for floating point comparisons
    assert result_df["upside"].iloc[0] == pytest.approx(25.0)
    assert result_df["upside"].iloc[1] == pytest.approx(15.5)
    assert result_df["upside"].iloc[2] == pytest.approx(10.0)
    assert pd.isna(result_df["upside"].iloc[3])

    assert result_df["beta"].iloc[0] == pytest.approx(1.5)
    assert result_df["beta"].iloc[1] == pytest.approx(2.0)
    assert pd.isna(result_df["beta"].iloc[2])
    assert result_df["beta"].iloc[3] == pytest.approx(3.2)

    # Text column should remain unchanged
    assert result_df["text"].equals(df["text"])
