"""Tests for factor attribution with PriceService integration."""

import json
from unittest.mock import patch

import pandas as pd

from trade_modules.factor_attribution import (
    BASE_FACTORS,
    _is_hit,
    compute_factor_attribution,
)


class TestIsHit:
    def test_buy_positive_alpha_is_hit(self):
        assert _is_hit("BUY", 5.0, 2.0) is True  # alpha = 3

    def test_buy_negative_alpha_is_miss(self):
        assert _is_hit("BUY", 1.0, 3.0) is False  # alpha = -2

    def test_sell_negative_alpha_is_hit(self):
        assert _is_hit("SELL", -3.0, 2.0) is True  # alpha = -5

    def test_sell_positive_alpha_is_miss(self):
        assert _is_hit("SELL", 5.0, 2.0) is False  # alpha = 3

    def test_hold_near_benchmark_is_hit(self):
        assert _is_hit("HOLD", 3.0, 2.5) is True  # alpha = 0.5

    def test_hold_far_from_benchmark_is_miss(self):
        assert _is_hit("HOLD", 10.0, 2.0) is False  # alpha = 8


class TestBaseFactors:
    """Verify base factor definitions fire correctly."""

    def test_high_buy_pct_fires(self):
        entry = {"buy_pct": 85}
        assert BASE_FACTORS["high_buy_pct"]["test"](entry) is True

    def test_high_buy_pct_no_fire(self):
        entry = {"buy_pct": 60}
        assert BASE_FACTORS["high_buy_pct"]["test"](entry) is False

    def test_strong_exret_fires(self):
        entry = {"exret": 35}
        assert BASE_FACTORS["strong_exret"]["test"](entry) is True


class TestComputeFactorAttribution:
    """Integration test with mocked action log and prices."""

    def test_basic_attribution(self, tmp_path):
        """Should compute attribution from action log entries."""
        log_path = tmp_path / "action_log.jsonl"
        entries = []
        for i in range(10):
            entries.append(
                {
                    "ticker": f"TICK{i}",
                    "committee_date": "2026-01-02",
                    "action": "BUY",
                    "conviction": 70 + i,
                    "price_at_recommendation": 100.0,
                    "buy_pct": 80 + i,
                    "exret": 20 + i,
                }
            )

        with open(log_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        # Mock price fetching to avoid yfinance calls
        dates = pd.date_range("2026-01-02", periods=50, freq="B")
        mock_prices = {}
        for i in range(10):
            tkr = f"TICK{i}"
            mock_prices[tkr] = {
                d.strftime("%Y-%m-%d"): 100 + i + (j * 0.5) for j, d in enumerate(dates)
            }
        mock_prices["SPY"] = {d.strftime("%Y-%m-%d"): 500 + (j * 0.2) for j, d in enumerate(dates)}

        with patch(
            "trade_modules.factor_attribution._fetch_prices_as_df",
            return_value=(pd.DataFrame(), mock_prices),
        ):
            result = compute_factor_attribution(
                action_log_path=log_path,
                horizons=[7, 14],
                output_path=tmp_path / "attribution.json",
            )

        assert "factors" in result
        assert result["entries_evaluated"] > 0
        assert "high_buy_pct" in result["factors"]
