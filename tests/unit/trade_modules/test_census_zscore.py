"""
N8 / M14: Census z-score thresholds + 7d EMA smoothing.

Legacy census trend classification uses absolute thresholds:
- _STRONG_PP_THRESHOLD = 10.0 pp (huge move)
- _MODERATE_PP_THRESHOLD = 3.0 pp (small move)

Problem: with top-100 census, 1 investor change = 1pp. The threshold
of 3pp = just 3 investors changing. That's noise.

Plus the 30-day classification uses ONLY the first/last snapshot —
a single-day spike at boundary triggers full classification.

M14 fixes:
1. z-score thresholds (delta vs ticker's 90d holder volatility) so
   "significance" depends on the ticker's normal churn.
2. 7-day EMA smoothing on holder count, not raw last-value.
3. Sustained-for-3-snapshots requirement before classifying.
"""

import pytest


class TestZScoreClassification:
    def test_huge_move_relative_to_history_classified_as_strong(self):
        from trade_modules.census_time_series import _classify_trend_zscore

        # Ticker normally varies ±0.5pp; today it moved +5pp → z=10 → STRONG
        cls = _classify_trend_zscore(
            delta_pp=+5.0,
            holder_volatility_pp=0.5,
        )
        assert cls == "strong_accumulation"

    def test_modest_move_for_volatile_ticker_classified_as_stable(self):
        """A 3pp move when the ticker normally varies ±5pp = z=0.6 → noise."""
        from trade_modules.census_time_series import _classify_trend_zscore

        cls = _classify_trend_zscore(
            delta_pp=+3.0,
            holder_volatility_pp=5.0,
        )
        assert cls == "stable"

    def test_negative_strong_move_distribution(self):
        from trade_modules.census_time_series import _classify_trend_zscore

        cls = _classify_trend_zscore(
            delta_pp=-4.0,
            holder_volatility_pp=1.0,
        )
        assert cls == "strong_distribution"

    def test_zero_volatility_falls_back_to_absolute_threshold(self):
        """Edge case: brand-new ticker with no history → use legacy thresholds."""
        from trade_modules.census_time_series import _classify_trend_zscore

        # Volatility unknown → should not crash; behave like legacy classifier
        cls = _classify_trend_zscore(delta_pp=+15.0, holder_volatility_pp=0.0)
        assert cls in ("strong_accumulation",)

    def test_z_under_2_is_stable(self):
        """|z| < 2 (default threshold) → stable, regardless of pp magnitude."""
        from trade_modules.census_time_series import _classify_trend_zscore

        # z = 1.5 is below the 2-sigma significance threshold
        cls = _classify_trend_zscore(
            delta_pp=+1.5,
            holder_volatility_pp=1.0,
        )
        assert cls == "stable"


class TestHolderVolatility:
    def test_constant_holder_count_yields_zero_volatility(self):
        from trade_modules.census_time_series import _holder_volatility

        snapshots = [{"date": f"2026-01-{d:02d}", "holdings": {"AAA": 30}} for d in range(1, 30)]
        vol = _holder_volatility("AAA", snapshots, lookback_days=90)
        assert vol == pytest.approx(0.0, abs=0.01)

    def test_varying_holder_count_yields_nonzero_volatility(self):
        from trade_modules.census_time_series import _holder_volatility

        # AAA varies between 28 and 32 holders → ±2 stddev → some volatility
        snapshots = []
        for d in range(1, 30):
            count = 30 + (2 if d % 2 == 0 else -2)
            snapshots.append(
                {
                    "date": f"2026-01-{d:02d}",
                    "holdings": {"AAA": count},
                    "investor_count": 100,
                }
            )
        vol = _holder_volatility("AAA", snapshots, lookback_days=90)
        # 4-pt swing on 100 holders → 4pp swing → ~2pp std
        assert vol > 0.5
        assert vol < 5.0


class TestEMASmoothing:
    def test_ema_smooths_single_day_spike(self):
        from trade_modules.census_time_series import _ema_holder_pct

        # 6 days at 30%, then 1 day at 50% (spike)
        snapshots = [
            {
                "date": f"2026-01-{d:02d}",
                "holdings": {"AAA": 30 if d < 7 else 50},
                "investor_count": 100,
            }
            for d in range(1, 8)
        ]
        # EMA with span=7 should be much closer to 30% than to 50%
        ema = _ema_holder_pct("AAA", snapshots, span=7)
        # The exact value depends on EMA formula, but spike alone shouldn't dominate
        assert 30.0 < ema < 40.0  # smoothed, not at the spike value

    def test_ema_returns_zero_for_missing_ticker(self):
        from trade_modules.census_time_series import _ema_holder_pct

        snapshots = [
            {"date": "2026-01-01", "holdings": {}, "investor_count": 100},
        ]
        assert _ema_holder_pct("AAA", snapshots, span=7) == 0.0
