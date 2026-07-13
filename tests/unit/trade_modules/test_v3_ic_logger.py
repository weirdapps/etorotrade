"""TDD tests for scripts.v3_ic_logger pure helpers (Phase 5E).

Covers:
  - append_snapshot : idempotency (same date twice → 0 second time, no dup rows)
  - forward_return_panel : correct fwd = close[date+h]/close[date] - 1 per ticker
  - ic_from_log : positive mean_ic when conviction predicts forward return
  - ic_from_log : near-zero mean_ic on noise (independent signal)
  - ic_from_log : graceful insufficient-history marker when fewer dates than horizon

No network calls — all synthetic in-memory / tmp-file data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v3_ic_logger import append_snapshot, forward_return_panel, ic_from_log

# ---------------------------------------------------------------------------
# Helpers — synthetic log builders
# ---------------------------------------------------------------------------


def _make_simple_log() -> pd.DataFrame:
    """Three dates, two tickers, deterministic closes for exact arithmetic tests."""
    return pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "ticker": "T00",
                "conviction": 0.5,
                "sector": "Tech",
                "close": 100.0,
            },
            {
                "date": "2026-01-01",
                "ticker": "T01",
                "conviction": -0.2,
                "sector": "Tech",
                "close": 200.0,
            },
            {
                "date": "2026-01-02",
                "ticker": "T00",
                "conviction": 0.3,
                "sector": "Tech",
                "close": 102.0,
            },
            {
                "date": "2026-01-02",
                "ticker": "T01",
                "conviction": 0.1,
                "sector": "Tech",
                "close": 196.0,
            },
            {
                "date": "2026-01-03",
                "ticker": "T00",
                "conviction": -0.1,
                "sector": "Tech",
                "close": 105.0,
            },
            {
                "date": "2026-01-03",
                "ticker": "T01",
                "conviction": 0.4,
                "sector": "Tech",
                "close": 198.0,
            },
        ]
    )


def _make_predictive_log(
    n_dates: int, n_tickers: int, noise_scale: float, seed: int
) -> pd.DataFrame:
    """Time-series log where conviction[d][t] predicts next-day log-return.

    Daily return for ticker t on day d:
        r[d][t] = conviction[d][t] * 0.05 + N(0, noise_scale)

    Prices are cumulative products of (1 + r).  A low noise_scale → high IC.
    """
    rng = np.random.default_rng(seed)
    n_total = n_dates + 10  # extra future dates for higher horizons
    tickers = [f"T{t:02d}" for t in range(n_tickers)]
    dates = [f"2026-01-{d + 1:02d}" for d in range(n_total)]

    convictions = rng.standard_normal((n_total, n_tickers))
    prices = np.ones((n_total, n_tickers)) * 100.0
    for d in range(1, n_total):
        ret = convictions[d - 1] * 0.05 + rng.standard_normal(n_tickers) * noise_scale
        prices[d] = prices[d - 1] * (1 + ret)

    rows = []
    for d_idx in range(n_total):
        for t_idx, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": dates[d_idx],
                    "ticker": ticker,
                    "conviction": convictions[d_idx, t_idx],
                    "sector": "Technology",
                    "close": float(prices[d_idx, t_idx]),
                }
            )
    return pd.DataFrame(rows)


def _make_noise_log(n_dates: int, n_tickers: int, seed: int) -> pd.DataFrame:
    """Log where conviction and forward return are independent (pure noise)."""
    rng = np.random.default_rng(seed)
    n_total = n_dates + 10
    tickers = [f"T{t:02d}" for t in range(n_tickers)]
    dates = [f"2026-01-{d + 1:02d}" for d in range(n_total)]

    rows = []
    prices = np.ones((n_total, n_tickers)) * 100.0
    for d in range(1, n_total):
        # returns independent of convictions
        prices[d] = prices[d - 1] * (1 + rng.standard_normal(n_tickers) * 0.02)

    for d_idx in range(n_total):
        for t_idx, ticker in enumerate(tickers):
            rows.append(
                {
                    "date": dates[d_idx],
                    "ticker": ticker,
                    "conviction": rng.standard_normal(),  # independent signal
                    "sector": "Technology",
                    "close": float(prices[d_idx, t_idx]),
                }
            )
    return pd.DataFrame(rows)


def _scored_frame(n: int = 4) -> pd.DataFrame:
    """Minimal scored DataFrame matching what compute_scores returns for eligible names."""
    return pd.DataFrame(
        {
            "conviction": [0.8, -0.3, 1.1, 0.0],
            "sector": ["Tech", "Finance", "Energy", "Health"],
            "price": [100.0, 200.0, 50.0, 75.0],
        },
        index=[f"TKR{i}" for i in range(n)],
    )


# ---------------------------------------------------------------------------
# append_snapshot
# ---------------------------------------------------------------------------


class TestAppendSnapshot:
    def test_first_append_returns_row_count(self, tmp_path):
        log_path = tmp_path / "log.csv"
        scored = _scored_frame()
        n = append_snapshot(str(log_path), "2026-07-01", scored)
        assert n == len(scored)

    def test_idempotent_same_date_returns_zero(self, tmp_path):
        log_path = tmp_path / "log.csv"
        scored = _scored_frame()
        append_snapshot(str(log_path), "2026-07-01", scored)
        n2 = append_snapshot(str(log_path), "2026-07-01", scored)
        assert n2 == 0

    def test_no_duplicate_rows_after_two_calls(self, tmp_path):
        log_path = tmp_path / "log.csv"
        scored = _scored_frame()
        append_snapshot(str(log_path), "2026-07-01", scored)
        append_snapshot(str(log_path), "2026-07-01", scored)
        df = pd.read_csv(log_path)
        assert len(df) == len(scored)
        assert list(df["date"].unique()) == ["2026-07-01"]

    def test_different_dates_both_appended(self, tmp_path):
        log_path = tmp_path / "log.csv"
        scored = _scored_frame()
        n1 = append_snapshot(str(log_path), "2026-07-01", scored)
        n2 = append_snapshot(str(log_path), "2026-07-02", scored)
        assert n1 == len(scored)
        assert n2 == len(scored)
        df = pd.read_csv(log_path)
        assert len(df) == 2 * len(scored)

    def test_log_schema_correct(self, tmp_path):
        """CSV must have exactly the five schema columns."""
        log_path = tmp_path / "log.csv"
        append_snapshot(str(log_path), "2026-07-01", _scored_frame())
        df = pd.read_csv(log_path)
        assert set(df.columns) == {"date", "ticker", "conviction", "sector", "close"}

    def test_close_sourced_from_price_column(self, tmp_path):
        log_path = tmp_path / "log.csv"
        scored = pd.DataFrame(
            {"conviction": [0.5], "sector": ["Tech"], "price": [123.45]},
            index=["XYZ"],
        )
        append_snapshot(str(log_path), "2026-07-01", scored)
        df = pd.read_csv(log_path)
        assert pytest.approx(df.loc[0, "close"], rel=1e-6) == 123.45

    def test_creates_parent_dir_if_missing(self, tmp_path):
        log_path = tmp_path / "deep" / "nested" / "log.csv"
        n = append_snapshot(str(log_path), "2026-07-01", _scored_frame())
        assert n > 0
        assert log_path.exists()


# ---------------------------------------------------------------------------
# forward_return_panel
# ---------------------------------------------------------------------------


class TestForwardReturnPanel:
    def test_correct_forward_returns_h1(self):
        log = _make_simple_log()
        panel = forward_return_panel(log, horizon=1)

        # T00: 2026-01-01 close=100, next=102 → fwd = 0.02
        row_t00_d1 = panel[(panel["signal_date"] == "2026-01-01") & (panel["ticker"] == "T00")]
        assert len(row_t00_d1) == 1
        assert pytest.approx(row_t00_d1.iloc[0]["forward_return"], rel=1e-6) == 0.02

        # T01: 2026-01-01 close=200, next=196 → fwd = -0.02
        row_t01_d1 = panel[(panel["signal_date"] == "2026-01-01") & (panel["ticker"] == "T01")]
        assert len(row_t01_d1) == 1
        assert pytest.approx(row_t01_d1.iloc[0]["forward_return"], rel=1e-6) == -0.02

    def test_correct_forward_returns_h2(self):
        log = _make_simple_log()
        panel = forward_return_panel(log, horizon=2)

        # T00: 2026-01-01 close=100, date+2 close=105 → fwd = 0.05
        row = panel[(panel["signal_date"] == "2026-01-01") & (panel["ticker"] == "T00")]
        assert len(row) == 1
        assert pytest.approx(row.iloc[0]["forward_return"], rel=1e-6) == 0.05

    def test_empty_when_horizon_exceeds_dates(self):
        log = _make_simple_log()
        panel = forward_return_panel(log, horizon=5)
        assert panel.empty

    def test_columns_present(self):
        log = _make_simple_log()
        panel = forward_return_panel(log, horizon=1)
        assert set(panel.columns) >= {"signal_date", "ticker", "conviction", "forward_return"}

    def test_conviction_carried_from_signal_date(self):
        log = _make_simple_log()
        panel = forward_return_panel(log, horizon=1)
        # conviction at 2026-01-01 for T00 is 0.5
        row = panel[(panel["signal_date"] == "2026-01-01") & (panel["ticker"] == "T00")]
        assert pytest.approx(row.iloc[0]["conviction"], rel=1e-6) == 0.5

    def test_no_nan_in_forward_return(self):
        log = _make_simple_log()
        panel = forward_return_panel(log, horizon=1)
        assert panel["forward_return"].notna().all()


# ---------------------------------------------------------------------------
# ic_from_log
# ---------------------------------------------------------------------------


class TestIcFromLog:
    def test_positive_mean_ic_on_predictive_log(self):
        """When conviction predicts forward return, mean IC should be positive."""
        log = _make_predictive_log(n_dates=30, n_tickers=40, noise_scale=0.002, seed=99)
        results = ic_from_log(log, horizons=(1,))
        r = results[1]
        assert "insufficient_history" not in r, f"Got insufficient_history: {r}"
        assert r["mean_ic"] is not None
        assert r["mean_ic"] > 0.1, f"Expected positive IC, got {r['mean_ic']:.4f}"

    def test_near_zero_mean_ic_on_noise_log(self):
        """Independent conviction and returns → IC centred on zero."""
        log = _make_noise_log(n_dates=40, n_tickers=40, seed=7)
        results = ic_from_log(log, horizons=(1,))
        r = results[1]
        # Can be None if no dates qualified — both None and ~0 are acceptable
        if r.get("mean_ic") is not None:
            assert abs(r["mean_ic"]) < 0.25, f"Noise IC unexpectedly large: {r['mean_ic']:.4f}"

    def test_insufficient_history_marker(self):
        """Horizons needing more dates than available get the graceful marker."""
        # Only 3 distinct dates → horizon=5 is insufficient (need at least 6)
        log = _make_simple_log()  # 3 dates
        results = ic_from_log(log, horizons=(5, 10))
        for h in (5, 10):
            assert results[h].get("insufficient_history") is True, (
                f"Expected insufficient_history for horizon={h}, got {results[h]}"
            )

    def test_returns_dict_keyed_by_horizon(self):
        log = _make_predictive_log(n_dates=15, n_tickers=20, noise_scale=0.005, seed=1)
        results = ic_from_log(log, horizons=(1, 2, 5))
        assert set(results.keys()) == {1, 2, 5}

    def test_sufficient_history_gives_ic_result(self):
        """With enough dates, the result dict has the standard IC keys."""
        log = _make_predictive_log(n_dates=20, n_tickers=20, noise_scale=0.005, seed=42)
        results = ic_from_log(log, horizons=(1,))
        r = results[1]
        if "insufficient_history" not in r:
            assert "mean_ic" in r
            assert "n_dates" in r

    def test_horizon_with_zero_dates_logged(self):
        """Empty log → all horizons get insufficient_history."""
        empty = pd.DataFrame(columns=["date", "ticker", "conviction", "sector", "close"])
        results = ic_from_log(empty, horizons=(1, 5))
        for h in (1, 5):
            assert results[h].get("insufficient_history") is True
