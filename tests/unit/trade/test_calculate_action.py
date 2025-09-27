"""
Tests for the calculate_action function in trade.py
"""

import pandas as pd
import pytest

from trade_modules.analysis_engine import calculate_action


@pytest.fixture
def sample_data():
    """Sample ticker data for testing."""
    return pd.DataFrame(
        [
            {
                "ticker": "BUY",
                "upside": 40.0,  # Increased to meet MICRO tier min_upside (35)
                "buy_percentage": 95.0,  # Increased to meet MICRO tier min_buy_percentage (92)
                "pe_trailing": 20.0,
                "pe_forward": 15.0,
                "peg_ratio": 1.2,
                "beta": 1.5,
                "short_percent": 1.0,  # Below max_short_interest for MICRO tier
                "analyst_count": 10,  # Above min_analysts for MICRO tier (6)
                "total_ratings": 8,
                "EXRET": 38.0,  # 40 * 95 / 100 = 38, above MICRO tier min_exret (30)
                "market_cap": 1_000_000_000,  # $1B - MICRO tier
            },
            {
                "ticker": "SELL",
                "upside": 3.0,  # Below SELL_MAX_UPSIDE threshold
                "buy_percentage": 70.0,
                "pe_trailing": 25.0,
                "pe_forward": 20.0,
                "peg_ratio": 2.0,
                "beta": 1.8,
                "short_percent": 3.5,
                "analyst_count": 8,
                "total_ratings": 7,
                "market_cap": 800_000_000,  # $800M - above $500M requirement
            },
            {
                "ticker": "HOLD",
                "upside": 15.0,  # Not enough for BUY_MIN_UPSIDE
                "buy_percentage": 80.0,  # Not enough for BUY_MIN_BUY_PERCENTAGE
                "pe_trailing": 18.0,
                "pe_forward": 16.0,
                "peg_ratio": 1.5,
                "beta": 1.2,
                "short_percent": 1.5,  # At BUY_MAX_SHORT_INTEREST, below SELL_MIN_SHORT_INTEREST
                "analyst_count": 6,
                "total_ratings": 5,
                "EXRET": 8.0,  # Between SELL_MAX_EXRET (5.0) and BUY_MIN_EXRET (15.0)
                "market_cap": 600_000_000,  # $600M - above $500M requirement
            },
            {
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
                "market_cap": 700_000_000,  # $700M - above $500M requirement
            },
        ]
    )


def test_calculate_action(sample_data):
    """Test that actions are calculated correctly."""
    result = calculate_action(sample_data)

    # Check that BS column was added (not ACTION)
    assert "BS" in result.columns

    # Check specific tickers got correct actions
    assert result.loc[result["ticker"] == "BUY", "BS"].iloc[0] == "B"

    # SELL ticker gets 'S' because upside 3% is below micro tier sell threshold (max_upside: 12.5%)
    assert result.loc[result["ticker"] == "SELL", "BS"].iloc[0] == "S"

    # HOLD ticker gets 'S' because EXRET 8% is below micro tier sell max_exret (9%)
    assert result.loc[result["ticker"] == "HOLD", "BS"].iloc[0] == "S"

    assert (
        result.loc[result["ticker"] == "LOWCONF", "BS"].iloc[0] == "I"
    )  # 'I' for INCONCLUSIVE due to insufficient analyst coverage (3 < 4 minimum)
