"""
Trend Continuation Override Tests.

Verifies the trend_continuation override in signals.py correctly rescues
stocks whose price outran analyst targets but remain in confirmed uptrends.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from trade_modules.analysis_engine import calculate_action

_CLEAR_EARNINGS = {
    "earnings_date": None,
    "days_until": None,
    "status": "clear",
    "should_hold": False,
    "conviction_boost": False,
    "conviction_adjustment": 0,
}


@pytest.fixture(autouse=True)
def _mock_earnings_proximity():
    with patch(
        "trade_modules.earnings_proximity.check_earnings_proximity",
        return_value=_CLEAR_EARNINGS,
    ):
        yield


def _make_stock(
    ticker="TEST",
    upside=-10.0,
    buy_pct=85.0,
    pct_52w=80,
    above_200dma=True,
    analyst_momentum=2,
    market_cap=600_000_000_000,
    pe_forward=25.0,
    pe_trailing=28.0,
    roe=20.0,
    de=50.0,
    fcf_yield=3.0,
):
    """Build a DataFrame row with defaults suited for trend-continuation testing."""
    data = {
        "ticker": ticker,
        "upside": upside,
        "buy_percentage": buy_pct,
        "pct_from_52w_high": pct_52w,
        "above_200dma": above_200dma,
        "analyst_momentum": analyst_momentum,
        "market_cap": market_cap,
        "analyst_count": 20,
        "total_ratings": 20,
        "pe_forward": pe_forward,
        "pe_trailing": pe_trailing,
        "EXRET": upside * buy_pct / 100,
        "return_on_equity": roe,
        "debt_to_equity": de,
        "fcf_yield": fcf_yield,
        "peg_ratio": 1.5,
        "short_percent": 2.0,
        "beta": 1.1,
        "revenue_growth": 15.0,
        "pe_vs_sector": 1.2,
        "target_dispersion": 30.0,
        "price": 150.0,
        "two_hundred_day_avg": 130.0,
    }
    return pd.DataFrame([data]).set_index("ticker")


class TestTrendContinuationOverride:
    """CIO v45.0: Verify trend_continuation override rescues negative-upside winners."""

    def test_override_fires_negative_upside_within_floor(self):
        """Stock with upside=-15% should get BUY via trend continuation."""
        df = _make_stock(upside=-15.0, buy_pct=85.0, pct_52w=85, above_200dma=True)
        result = calculate_action(df)
        assert result.loc["TEST", "BS"] == "B", (
            f"Expected BUY via trend continuation for upside=-15%, got {result.loc['TEST', 'BS']}"
        )

    def test_override_fires_small_negative_upside(self):
        """Stock with upside=-5% (like SNDK) should get BUY."""
        df = _make_stock(upside=-5.0, buy_pct=82.0, pct_52w=99, above_200dma=True)
        result = calculate_action(df)
        assert result.loc["TEST", "BS"] == "B", (
            f"Expected BUY for upside=-5% with 82% buy consensus, got {result.loc['TEST', 'BS']}"
        )

    def test_override_blocked_below_floor(self):
        """Stock with upside=-35% (below -30 floor) should NOT get BUY via TC."""
        # Also set pct_52w=60 to avoid momentum track catching this stock
        df = _make_stock(upside=-35.0, buy_pct=90.0, pct_52w=60, above_200dma=True)
        result = calculate_action(df)
        assert result.loc["TEST", "BS"] != "B", (
            f"Expected non-BUY for upside=-35% (below floor), got {result.loc['TEST', 'BS']}"
        )

    def test_override_blocked_low_buy_pct(self):
        """Stock with buy%=70 (below 80 threshold) should NOT get BUY via override."""
        df = _make_stock(upside=-10.0, buy_pct=70.0, pct_52w=85, above_200dma=True)
        result = calculate_action(df)
        assert result.loc["TEST", "BS"] != "B", (
            f"Expected non-BUY for buy%=70 (below tc threshold), got {result.loc['TEST', 'BS']}"
        )

    def test_override_blocked_below_200dma(self):
        """Stock below 200DMA should NOT get BUY via override."""
        df = _make_stock(
            upside=-10.0,
            buy_pct=85.0,
            pct_52w=85,
            above_200dma=False,
        )
        df.at["TEST", "price"] = 120.0
        df.at["TEST", "two_hundred_day_avg"] = 130.0
        result = calculate_action(df)
        assert result.loc["TEST", "BS"] != "B", (
            f"Expected non-BUY for stock below 200DMA, got {result.loc['TEST', 'BS']}"
        )

    def test_override_blocked_low_52w(self):
        """Stock at 60% of 52w high (below 70 tc threshold) should not get BUY."""
        df = _make_stock(upside=-10.0, buy_pct=85.0, pct_52w=60, above_200dma=True)
        result = calculate_action(df)
        assert result.loc["TEST", "BS"] != "B", (
            f"Expected non-BUY for 52w=60% (below tc threshold), got {result.loc['TEST', 'BS']}"
        )


class TestStockSplitPEAwareness:
    """CIO v45.0: Verify stock-split distorted PET does not block BUY."""

    def test_split_distorted_pet_skipped(self):
        """PET/PEF > 10 should bypass trailing PE gate."""
        df = _make_stock(
            upside=20.0,
            buy_pct=85.0,
            pe_trailing=685.0,
            pe_forward=25.0,
            pct_52w=99,
        )
        result = calculate_action(df)
        assert result.loc["TEST", "BS"] == "B", (
            f"Expected BUY despite PET=685 (split distortion), got {result.loc['TEST', 'BS']}"
        )

    def test_normal_pet_still_gates(self):
        """Normal PET/PEF ratio should still apply trailing PE gate."""
        # Set pct_52w=60 to avoid momentum track bypass
        df = _make_stock(
            upside=20.0,
            buy_pct=85.0,
            pe_trailing=120.0,
            pe_forward=25.0,
            pct_52w=60,
        )
        result = calculate_action(df)
        assert result.loc["TEST", "BS"] != "B", (
            f"Expected non-BUY for PET=120 (normal, above max), got {result.loc['TEST', 'BS']}"
        )
