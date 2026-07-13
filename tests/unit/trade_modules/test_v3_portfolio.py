"""TDD tests for scripts.v3_portfolio pure helpers.

Covers:
  - trend_regime: uptrend → risk_on/1.0; downtrend → risk_off/0.65;
    choppy → neutral/0.90; short series (<200 obs) → neutral/0.90.
  - build_target_rows: weights>0 kept and sorted desc by target_pct;
    zero/absent-weight rows excluded; all required keys present incl SL/TP.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.v3_portfolio import build_target_rows, trend_regime
from trade_modules.v3.conditioning import DEPLOYMENT_BY_REGIME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(days: int, values: list[float] | None = None) -> pd.Series:
    idx = pd.date_range("2022-01-03", periods=days, freq="B")
    if values is not None:
        return pd.Series(values, index=idx)
    # Default: flat series at 100
    return pd.Series([100.0] * days, index=idx)


def _uptrend(days: int = 300) -> pd.Series:
    """Monotonically rising series — last > sma200 AND sma50 > sma200."""
    idx = pd.date_range("2022-01-03", periods=days, freq="B")
    return pd.Series(np.linspace(80.0, 150.0, days), index=idx)


def _downtrend(days: int = 300) -> pd.Series:
    """Monotonically falling series — last < sma200."""
    idx = pd.date_range("2022-01-03", periods=days, freq="B")
    return pd.Series(np.linspace(150.0, 80.0, days), index=idx)


def _neutral_series(days: int = 300) -> pd.Series:
    """Guaranteed neutral: last > sma200 AND sma50 < sma200.

    Construction (verified analytically):
      days 0–249  : stable at 100.0  (sets the sma200 baseline)
      days 250–298: dip to 95.0      (pulls sma50 below sma200)
      day  299    : bounce to 102.0  (last > sma200 ≈ 98.79 but sma50 ≈ 95.14 < 98.79)
    """
    idx = pd.date_range("2022-01-03", periods=days, freq="B")
    values = [100.0] * 250 + [95.0] * 49 + [102.0]
    assert len(values) == days
    return pd.Series(values, index=idx)


def _scored_frame(tickers, convictions=None, weights_nonzero=True):
    """Minimal scored frame for build_target_rows tests."""
    n = len(tickers)
    if convictions is None:
        convictions = list(np.linspace(2.0, 0.5, n))
    df = pd.DataFrame(
        {
            "conviction": convictions,
            "rank": list(range(1, n + 1)),
            "price": [100.0 + i * 10 for i in range(n)],
            "sector": ["Tech", "Health", "Energy"][:n] + ["Tech"] * max(0, n - 3),
            "name": [f"Company {t}" for t in tickers],
            "stop_loss": [90.0 + i for i in range(n)],
            "take_profit": [120.0 + i for i in range(n)],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return df


def _weights(tickers, values):
    return pd.Series(dict(zip(tickers, values)))


# ---------------------------------------------------------------------------
# trend_regime tests
# ---------------------------------------------------------------------------


class TestTrendRegime:
    def test_uptrend_returns_risk_on(self):
        regime, mult = trend_regime(_uptrend())
        assert regime == "risk_on"
        assert mult == pytest.approx(1.0)

    def test_downtrend_returns_risk_off(self):
        regime, mult = trend_regime(_downtrend())
        assert regime == "risk_off"
        assert mult == pytest.approx(0.65)

    def test_choppy_returns_neutral(self):
        regime, mult = trend_regime(_neutral_series())
        assert regime == "neutral"
        assert mult == pytest.approx(0.90)

    def test_short_series_returns_neutral(self):
        """Fewer than 200 observations → forced neutral."""
        for length in (0, 1, 50, 199):
            regime, mult = trend_regime(_series(length))
            assert regime == "neutral", f"length={length} should be neutral"
            assert mult == pytest.approx(0.90), f"length={length} multiplier wrong"

    def test_exactly_200_obs_does_not_crash(self):
        """200 obs: rolling(200).mean() has exactly one non-NaN value."""
        s = _series(200, [100.0] * 200)
        regime, mult = trend_regime(s)
        # All equal → sma200==last==sma50 → not strictly below → not risk_off;
        # sma50 not > sma200 → not risk_on; lands in neutral.
        assert regime == "neutral"
        assert mult == pytest.approx(0.90)

    def test_returns_tuple_of_str_and_float(self):
        regime, mult = trend_regime(_uptrend())
        assert isinstance(regime, str)
        assert isinstance(mult, float)


# ---------------------------------------------------------------------------
# DEPLOYMENT_BY_REGIME tests (Change 2 — regime deployment 85-95%)
# ---------------------------------------------------------------------------


class TestDeploymentByRegime:
    def test_mapping_values(self):
        assert DEPLOYMENT_BY_REGIME["risk_off"] == pytest.approx(0.85)
        assert DEPLOYMENT_BY_REGIME["neutral"] == pytest.approx(0.90)
        assert DEPLOYMENT_BY_REGIME["risk_on"] == pytest.approx(0.95)

    def test_averages_about_ninety(self):
        vals = [DEPLOYMENT_BY_REGIME[r] for r in ("risk_off", "neutral", "risk_on")]
        assert sum(vals) / len(vals) == pytest.approx(0.90)

    def test_ordered_by_risk_appetite(self):
        assert (
            DEPLOYMENT_BY_REGIME["risk_off"]
            < DEPLOYMENT_BY_REGIME["neutral"]
            < DEPLOYMENT_BY_REGIME["risk_on"]
        )

    def test_covers_every_trend_regime_label(self):
        # Every regime trend_regime can emit must have a deployment target.
        for series in (_uptrend(), _downtrend(), _neutral_series()):
            regime, _ = trend_regime(series)
            assert regime in DEPLOYMENT_BY_REGIME


# ---------------------------------------------------------------------------
# build_target_rows tests
# ---------------------------------------------------------------------------


class TestBuildTargetRows:
    _TICKERS = ["AAPL", "MSFT", "GOOG"]

    def _result(self, weights_list):
        """Fake build_portfolio result dict."""
        return {"weights": _weights(self._TICKERS, weights_list)}

    def test_returns_list(self):
        scored = _scored_frame(self._TICKERS)
        result = self._result([0.10, 0.08, 0.05])
        rows = build_target_rows(scored, result)
        assert isinstance(rows, list)

    def test_zero_weight_excluded(self):
        scored = _scored_frame(self._TICKERS)
        result = self._result([0.10, 0.0, 0.05])
        rows = build_target_rows(scored, result)
        tickers_out = [r["ticker"] for r in rows]
        assert "MSFT" not in tickers_out
        assert "AAPL" in tickers_out
        assert "GOOG" in tickers_out

    def test_absent_weight_excluded(self):
        """Tickers in scored but not in result['weights'] are omitted."""
        scored = _scored_frame(self._TICKERS)
        # Provide weights only for 2 of the 3 tickers
        result = {"weights": _weights(["AAPL", "MSFT"], [0.10, 0.08])}
        rows = build_target_rows(scored, result)
        tickers_out = [r["ticker"] for r in rows]
        assert "GOOG" not in tickers_out

    def test_sorted_descending_by_target_pct(self):
        scored = _scored_frame(self._TICKERS)
        # Deliberately unordered weights
        result = self._result([0.05, 0.15, 0.10])
        rows = build_target_rows(scored, result)
        pcts = [r["target_pct"] for r in rows]
        assert pcts == sorted(pcts, reverse=True)

    def test_all_required_keys_present(self):
        scored = _scored_frame(self._TICKERS)
        result = self._result([0.10, 0.08, 0.05])
        rows = build_target_rows(scored, result)
        required = {
            "ticker",
            "name",
            "sector",
            "conviction",
            "rank",
            "target_pct",
            "price",
            "stop_loss",
            "take_profit",
        }
        for row in rows:
            assert required <= set(row.keys()), f"Missing keys in row: {set(row.keys())}"

    def test_target_pct_matches_weight(self):
        scored = _scored_frame(self._TICKERS)
        weights = [0.10, 0.08, 0.05]
        result = self._result(weights)
        rows = build_target_rows(scored, result)
        by_ticker = {r["ticker"]: r for r in rows}
        assert by_ticker["AAPL"]["target_pct"] == pytest.approx(0.10)
        assert by_ticker["GOOG"]["target_pct"] == pytest.approx(0.05)

    def test_stop_loss_and_take_profit_present(self):
        scored = _scored_frame(self._TICKERS)
        result = self._result([0.10, 0.08, 0.05])
        rows = build_target_rows(scored, result)
        for row in rows:
            # Values may be NaN (missing data) but the key must exist
            assert "stop_loss" in row
            assert "take_profit" in row

    def test_empty_weights_returns_empty_list(self):
        scored = _scored_frame(self._TICKERS)
        result = {"weights": pd.Series(dtype=float)}
        rows = build_target_rows(scored, result)
        assert rows == []

    def test_all_zero_weights_returns_empty_list(self):
        scored = _scored_frame(self._TICKERS)
        result = self._result([0.0, 0.0, 0.0])
        rows = build_target_rows(scored, result)
        assert rows == []
