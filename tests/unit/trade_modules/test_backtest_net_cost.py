"""Tests for net-of-cost alpha in signal-level backtest.

Validates that eToro tier-based transaction costs are correctly imported
from committee_backtester and applied to produce net_alpha columns in
the backtest engine.
"""

import numpy as np
import pandas as pd
import pytest


class TestRoundTripCostImport:
    """Verify _round_trip_cost_pct is importable and returns sane values."""

    def test_import_and_mega_tier(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        cost = _round_trip_cost_pct(tier="MEGA", holding_days=7)
        # MEGA spread = 9 bps one-way = 0.18% round-trip, no financing
        assert cost == 0.18

    def test_large_tier(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        cost = _round_trip_cost_pct(tier="LARGE", holding_days=7)
        assert cost == 0.24  # 12 bps * 2

    def test_mid_tier(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        cost = _round_trip_cost_pct(tier="MID", holding_days=7)
        assert cost == 0.36  # 18 bps * 2

    def test_small_tier(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        cost = _round_trip_cost_pct(tier="SMALL", holding_days=7)
        assert cost == 0.6  # 30 bps * 2

    def test_micro_tier(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        cost = _round_trip_cost_pct(tier="MICRO", holding_days=7)
        assert cost == 1.2  # 60 bps * 2

    def test_crypto(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        cost = _round_trip_cost_pct(tier="MID", holding_days=30, is_crypto=True)
        assert cost == 2.0  # Crypto 100 bps * 2

    def test_unknown_tier_defaults_to_mid(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        cost = _round_trip_cost_pct(tier="UNKNOWN", holding_days=7)
        assert cost == 0.36  # Falls back to 18 bps default

    def test_financing_adds_cost(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        cost_no_fin = _round_trip_cost_pct(tier="MID", holding_days=30)
        cost_with_fin = _round_trip_cost_pct(tier="MID", holding_days=30, financing_apr=0.064)
        assert cost_with_fin > cost_no_fin


class TestNetAlphaComputation:
    """Verify net_alpha = alpha - round_trip_cost conceptually."""

    def test_net_alpha_lower_than_gross(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        gross_alpha = 0.61  # % (typical BUY T+7)
        cost = _round_trip_cost_pct(tier="MID", holding_days=7)
        net_alpha = gross_alpha - cost
        assert net_alpha < gross_alpha
        assert net_alpha == pytest.approx(0.25, abs=0.01)

    def test_mega_cap_preserves_more_alpha(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        gross_alpha = 0.61
        mega_cost = _round_trip_cost_pct(tier="MEGA", holding_days=7)
        mid_cost = _round_trip_cost_pct(tier="MID", holding_days=7)
        assert gross_alpha - mega_cost > gross_alpha - mid_cost

    def test_micro_cap_can_consume_all_alpha(self):
        from trade_modules.committee_backtester import _round_trip_cost_pct

        gross_alpha = 0.61
        micro_cost = _round_trip_cost_pct(tier="MICRO", holding_days=7)
        net_alpha = gross_alpha - micro_cost
        assert net_alpha < 0  # 1.2% cost > 0.61% alpha


class TestBacktestEngineNetAlpha:
    """Integration tests for net_alpha in BacktestEngine.calculate_returns."""

    def _make_engine(self):
        from trade_modules.backtest_engine import BacktestEngine

        engine = BacktestEngine.__new__(BacktestEngine)
        engine.horizons = [7]
        return engine

    def _make_price_data(self, ticker, prices, start_date="2026-01-14"):
        """Build a price DataFrame with trading-day index."""
        dates = pd.bdate_range(start=start_date, periods=len(prices))
        return pd.DataFrame({ticker: prices}, index=dates)

    def _make_spy_data(self, prices, start_date="2026-01-14"):
        dates = pd.bdate_range(start=start_date, periods=len(prices))
        return pd.Series(prices, index=dates, name="SPY")

    def test_calculate_returns_includes_net_alpha(self):
        engine = self._make_engine()
        # Stock goes from 100 to 105 (+5%), SPY 100 to 102 (+2%), alpha = 3%
        prices = [100.0] + [100.0] * 6 + [105.0] + [105.0] * 2
        spy_prices = [100.0] + [100.0] * 6 + [102.0] + [102.0] * 2

        signals_df = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "signal": "B",
                    "date": "2026-01-14",
                    "price_at_signal": 100.0,
                    "tier": "MID",
                    "region": "US",
                }
            ]
        )

        price_data = self._make_price_data("TEST", prices)
        spy_data = self._make_spy_data(spy_prices)

        results = engine.calculate_returns(signals_df, price_data, spy_data, horizon=7)

        assert len(results) == 1
        row = results.iloc[0]
        assert "net_alpha" in results.columns
        assert "round_trip_cost" in results.columns
        assert row["alpha"] == pytest.approx(3.0, abs=0.1)
        assert row["round_trip_cost"] == 0.36  # MID tier
        assert row["net_alpha"] == pytest.approx(3.0 - 0.36, abs=0.1)
        assert row["net_alpha"] < row["alpha"]

    def test_net_alpha_nan_when_alpha_nan(self):
        """When SPY data is missing, alpha is NaN, so net_alpha should also be NaN."""
        engine = self._make_engine()
        prices = [100.0] * 10
        signals_df = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "signal": "B",
                    "date": "2026-01-14",
                    "price_at_signal": 100.0,
                    "tier": "MID",
                    "region": "US",
                }
            ]
        )
        price_data = self._make_price_data("TEST", prices)
        spy_data = pd.Series(dtype=float)  # Empty SPY

        results = engine.calculate_returns(signals_df, price_data, spy_data, horizon=7)
        assert len(results) == 1
        assert np.isnan(results.iloc[0]["net_alpha"])

    def test_missing_tier_defaults_to_mid(self):
        """When tier is absent, cost should use MID default."""
        engine = self._make_engine()
        prices = [100.0] + [100.0] * 6 + [105.0] + [105.0] * 2
        spy_prices = [100.0] + [100.0] * 6 + [102.0] + [102.0] * 2

        signals_df = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "signal": "B",
                    "date": "2026-01-14",
                    "price_at_signal": 100.0,
                    # No "tier" key
                    "region": "US",
                }
            ]
        )

        price_data = self._make_price_data("TEST", prices)
        spy_data = self._make_spy_data(spy_prices)

        results = engine.calculate_returns(signals_df, price_data, spy_data, horizon=7)
        assert results.iloc[0]["round_trip_cost"] == 0.36  # MID default

    def test_crypto_ticker_uses_crypto_spread(self):
        engine = self._make_engine()
        prices = [100.0] + [100.0] * 6 + [110.0] + [110.0] * 2
        spy_prices = [100.0] + [100.0] * 6 + [102.0] + [102.0] * 2

        signals_df = pd.DataFrame(
            [
                {
                    "ticker": "BTC-USD",
                    "signal": "B",
                    "date": "2026-01-14",
                    "price_at_signal": 100.0,
                    "tier": "MEGA",
                    "region": "US",
                }
            ]
        )

        price_data = self._make_price_data("BTC-USD", prices)
        spy_data = self._make_spy_data(spy_prices)

        results = engine.calculate_returns(signals_df, price_data, spy_data, horizon=7)
        assert results.iloc[0]["round_trip_cost"] == 2.0  # Crypto, not MEGA


class TestComputeGroupStatsNetAlpha:
    """Test that _compute_group_stats returns net alpha statistics."""

    def test_net_alpha_stats_present(self):
        from trade_modules.backtest_engine import BacktestEngine

        gdf = pd.DataFrame(
            {
                "stock_return": [5.0, 3.0, -1.0],
                "alpha": [3.0, 1.0, -2.0],
                "net_alpha": [2.64, 0.64, -2.36],
                "beta": [1.0, 1.0, 1.0],
                "alpha_eur": [3.0, 1.0, -2.0],
                "beta_adj_alpha_eur": [2.5, 0.5, -2.5],
            }
        )

        stats = BacktestEngine._compute_group_stats(gdf, signal_type="B")

        assert "net_avg_alpha" in stats
        assert "net_alpha_hit_rate" in stats
        assert stats["net_avg_alpha"] is not None
        assert stats["net_alpha_hit_rate"] is not None
        # 2 of 3 net_alphas > 0
        assert stats["net_alpha_hit_rate"] == pytest.approx(66.7, abs=1.0)
        # avg of [2.64, 0.64, -2.36]
        assert stats["net_avg_alpha"] == pytest.approx(0.31, abs=0.01)

    def test_net_alpha_stats_without_column(self):
        """Gracefully handle DataFrames without net_alpha column."""
        from trade_modules.backtest_engine import BacktestEngine

        gdf = pd.DataFrame(
            {
                "stock_return": [5.0],
                "alpha": [3.0],
                # No net_alpha column
                "beta": [1.0],
                "alpha_eur": [3.0],
                "beta_adj_alpha_eur": [2.5],
            }
        )

        stats = BacktestEngine._compute_group_stats(gdf, signal_type="B")
        assert stats["net_avg_alpha"] is None
        assert stats["net_alpha_hit_rate"] is None


class TestBuildReportNetAlpha:
    """Test that build_report includes net alpha fields in headline."""

    def test_headline_contains_net_alpha_fields(self):
        import os
        import sys

        sys.path.insert(
            0,
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"),
        )
        from run_weekly_backtest import build_report

        signal_summary = {
            "t7_B": {
                "count": 100,
                "hit_rate": 55.0,
                "alpha_hit_rate": 52.0,
                "avg_alpha": 0.61,
                "net_avg_alpha": 0.25,
                "net_alpha_hit_rate": 48.0,
            },
            "t7_S": {
                "count": 50,
                "hit_rate": 60.0,
                "alpha_hit_rate": 58.0,
                "avg_alpha": -1.5,
                "net_avg_alpha": -1.86,
                "net_alpha_hit_rate": 62.0,
            },
        }

        report = build_report(
            signal_summary=signal_summary,
            committee_result={"status": "complete"},
            scorecard={"buy_recommendations": {}, "conviction_calibration": {}},
            calibration={"sufficient_data": False, "modifiers": {}},
        )

        headline = report["headline"]
        assert headline["buy_net_avg_alpha_t7"] == 0.25
        assert headline["buy_net_alpha_hit_rate_t7"] == 48.0
        assert headline["sell_net_avg_alpha_t7"] == -1.86
        assert headline["sell_net_alpha_hit_rate_t7"] == 62.0
