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
                "upside": 40.0,  # Strong upside for SMALL tier
                "buy_percentage": 95.0,  # Strong buy% for SMALL tier
                "pe_trailing": 20.0,
                "pe_forward": 15.0,
                "peg_ratio": 1.2,
                "beta": 1.5,
                "short_percent": 1.0,
                "analyst_count": 10,
                "total_ratings": 8,
                "EXRET": 38.0,  # 40 * 95 / 100 = 38
                "market_cap": 5_000_000_000,  # $5B - SMALL tier (above $1B minimum)
            },
            {
                "ticker": "SELL",
                "upside": -8.0,  # Negative upside - strong SELL signal (hard trigger)
                "buy_percentage": 30.0,  # Very low buy% - triggers hard SELL
                "pe_trailing": 25.0,
                "pe_forward": 20.0,
                "peg_ratio": 2.0,
                "beta": 1.8,
                "short_percent": 3.5,
                "analyst_count": 8,
                "total_ratings": 7,
                "EXRET": -2.4,  # Negative EXRET
                "market_cap": 3_000_000_000,  # $3B - SMALL tier (above $1B minimum)
            },
            {
                "ticker": "HOLD",
                "upside": 12.0,  # Not enough for BUY in SMALL tier
                "buy_percentage": 72.0,  # Between thresholds
                "pe_trailing": 18.0,
                "pe_forward": 16.0,
                "peg_ratio": 1.5,
                "beta": 1.2,
                "short_percent": 1.5,
                "analyst_count": 8,
                "total_ratings": 7,
                "EXRET": 8.6,  # Moderate EXRET
                "market_cap": 3_000_000_000,  # $3B - SMALL tier (above $1B minimum)
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
                "analyst_count": 3,  # Below MIN_ANALYST_COUNT (6)
                "total_ratings": 2,  # Below MIN_PRICE_TARGETS (6)
                "market_cap": 3_000_000_000,  # $3B - SMALL tier (above $1B minimum)
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

    # SELL ticker gets 'S' because of hard trigger: negative upside + very low buy%
    assert result.loc[result["ticker"] == "SELL", "BS"].iloc[0] == "S"

    # HOLD ticker gets 'H' because it's between BUY and SELL thresholds
    assert result.loc[result["ticker"] == "HOLD", "BS"].iloc[0] == "H"

    assert (
        result.loc[result["ticker"] == "LOWCONF", "BS"].iloc[0] == "I"
    )  # 'I' for INCONCLUSIVE due to insufficient analyst coverage (3 < 4 minimum)
