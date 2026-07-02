"""TDD tests for scripts.signals_v2_validate — the S2 signal-validation core.

These cover ONLY the pure functions; ``main()`` is network-bound (yfinance) and
excluded from coverage.

The paramount property is NO LOOK-AHEAD: the price-factor score at rebalance
position ``t`` must depend ONLY on prices up to and including ``t``. A leak would
make the whole out-of-sample verdict a lie, so it is tested first and hardest.

Run in isolation:
    cd ~/SourceCode/etorotrade
    python3 -m pytest tests/unit/trade_modules/test_signals_v2_validate.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.signals_v2_validate import (
    build_signal_rows,
    exit_bucket_alpha,
    price_factor_score,
    rebalance_dates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panel(n_days: int, tickers, builder) -> pd.DataFrame:
    """Build a (dates x tickers) price panel.

    ``builder(ticker, idx_array)`` returns a 1D price array of length n_days.
    """
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    arr = np.arange(n_days)
    data = {t: builder(t, arr) for t in tickers}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# NO-LOOK-AHEAD — the critical correctness test
# ---------------------------------------------------------------------------


def test_no_lookahead_score_invariant_to_future_bars():
    """price_factor_score at position k MUST be identical whether or not the
    bars AFTER k exist / are mutated.

    Construct a panel whose final bars are a huge spike. Score at k < last must
    not move when the future spike is present vs. absent — proving the score
    reads no price after k.
    """
    n = 400
    tickers = ["A", "B", "C", "D"]

    # Distinct gentle upward ramps so the cross-section has real spread at k.
    def build(t, arr):
        slope = {"A": 0.4, "B": 0.7, "C": 1.0, "D": 1.3}[t]
        return 100.0 + slope * arr

    full = _panel(n, tickers, build)

    k = 300  # a position well before the end

    # Score using the full panel (future bars present).
    score_full = price_factor_score(full, k)

    # Now inject a massive future spike AFTER k on the full panel, then rescore.
    spiked = full.copy()
    spiked.iloc[k + 1 :] = spiked.iloc[k + 1 :] * 1000.0  # obscene future move
    score_spiked = price_factor_score(spiked, k)

    # And a version truncated at k (no future bars exist at all).
    truncated = full.iloc[: k + 1].copy()
    score_trunc = price_factor_score(truncated, k)

    # All three must be IDENTICAL (bitwise-close): the score at k ignores >k.
    pd.testing.assert_series_equal(
        score_full.sort_index(), score_spiked.sort_index(), check_exact=False, rtol=0, atol=0
    )
    pd.testing.assert_series_equal(
        score_full.sort_index(), score_trunc.sort_index(), check_exact=False, rtol=0, atol=0
    )


def test_no_lookahead_mutating_before_k_does_change_score():
    """Sanity counter-test: mutating a bar AT/BEFORE k SHOULD change the score
    (otherwise the invariance test above could pass trivially)."""
    n = 400
    tickers = ["A", "B", "C", "D"]

    def build(t, arr):
        slope = {"A": 0.4, "B": 0.7, "C": 1.0, "D": 1.3}[t]
        return 100.0 + slope * arr

    full = _panel(n, tickers, build)
    k = 300
    base = price_factor_score(full, k)

    perturbed = full.copy()
    # Change a mid-window bar for ticker A (well within the momentum window).
    perturbed.iloc[100, perturbed.columns.get_loc("A")] *= 1.5
    after = price_factor_score(perturbed, k)

    assert not base.equals(after), "score at k must react to price changes at/<k"


def test_price_factor_score_returns_series_over_tickers():
    n = 300
    tickers = ["A", "B", "C"]
    panel = _panel(n, tickers, lambda t, arr: 100.0 + arr * (ord(t[0]) - 64))
    s = price_factor_score(panel, 260)
    assert isinstance(s, pd.Series)
    assert set(s.index) <= set(tickers)
    # At least the non-degenerate names get finite scores.
    assert s.notna().any()


# ---------------------------------------------------------------------------
# rebalance_dates — spacing + warmup
# ---------------------------------------------------------------------------


def test_rebalance_dates_spacing_and_warmup():
    idx = pd.date_range("2020-01-01", periods=1000, freq="B")
    positions = rebalance_dates(idx, freq_days=30, warmup=253)
    # Positions are integer offsets into the index.
    assert all(isinstance(p, (int, np.integer)) for p in positions)
    # First rebalance is at/after warmup.
    assert positions[0] >= 253
    # Spacing is ~freq_days.
    diffs = np.diff(positions)
    assert all(d == 30 for d in diffs)
    # All within range.
    assert max(positions) < len(idx)


def test_rebalance_dates_short_index_returns_empty():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    assert rebalance_dates(idx, freq_days=30, warmup=253) == []


# ---------------------------------------------------------------------------
# build_signal_rows — forward return window + shape
# ---------------------------------------------------------------------------


def test_build_signal_rows_forward_return_uses_t_to_t_plus_h():
    """On a strictly rising ramp with a known daily slope, the forward return for
    a rebalance at t over horizon h must equal close[t+h]/close[t]-1."""
    n = 700
    tickers = ["A", "B", "C", "D", "E"]

    # Different positive slopes -> a real cross-section; all strictly rising.
    slopes = {"A": 0.2, "B": 0.5, "C": 0.9, "D": 1.4, "E": 2.0}

    def build(t, arr):
        return 100.0 + slopes[t] * arr

    panel = _panel(n, tickers, build)
    # SPY flat-ish so alpha ~ the stock's own return.
    spy = pd.Series(100.0 + 0.0 * np.arange(n), index=panel.index, name="SPY")

    horizon = 30
    rows = build_signal_rows(
        panel, spy, freq_days=60, horizon_days=horizon, buy_pct=0.2, sell_pct=0.2
    )
    assert rows, "expected non-empty rows"

    # Pick any B row and verify its forward return matches the ramp exactly.
    for r in rows:
        t_pos = None
        # Recover t from signal_date.
        sig_date = pd.Timestamp(r["signal_date"])
        t_pos = panel.index.get_loc(sig_date)
        tkr = r["ticker"]
        expected_stock = panel[tkr].iloc[t_pos + horizon] / panel[tkr].iloc[t_pos] - 1.0
        expected_spy = spy.iloc[t_pos + horizon] / spy.iloc[t_pos] - 1.0
        expected_alpha = expected_stock - expected_spy
        assert r["alpha"] == pytest.approx(expected_alpha, rel=1e-9, abs=1e-12)
        assert r["horizon"] == horizon


def test_build_signal_rows_rising_set_gives_positive_buy_alpha():
    """A universe of strictly-rising names vs a flat benchmark → the B-signal
    rows must carry positive alpha."""
    n = 700
    tickers = [f"T{i}" for i in range(10)]

    def build(t, arr):
        slope = 0.2 + 0.2 * int(t[1:])  # monotone, all positive
        return 100.0 + slope * arr

    panel = _panel(n, tickers, build)
    spy = pd.Series(100.0, index=panel.index, name="SPY")  # flat

    rows = build_signal_rows(panel, spy, freq_days=90, horizon_days=30)
    buys = [r for r in rows if r["signal"] == "B"]
    assert buys, "expected some B rows"
    assert np.mean([r["alpha"] for r in buys]) > 0


def test_build_signal_rows_only_where_forward_exists():
    """No row may be emitted for a rebalance whose t+horizon is out of range."""
    n = 320
    tickers = ["A", "B", "C"]
    panel = _panel(n, tickers, lambda t, arr: 100.0 + arr * (ord(t[0]) - 64))
    spy = pd.Series(100.0 + np.arange(n) * 0.1, index=panel.index, name="SPY")

    horizon = 30
    rows = build_signal_rows(panel, spy, freq_days=30, horizon_days=horizon)
    max_pos = len(panel) - 1
    for r in rows:
        t_pos = panel.index.get_loc(pd.Timestamp(r["signal_date"]))
        assert t_pos + horizon <= max_pos


def test_build_signal_rows_net_alpha_subtracts_costs():
    """net_alpha must equal alpha minus a flat 20bps round-trip cost."""
    n = 700
    tickers = ["A", "B", "C", "D"]
    panel = _panel(n, tickers, lambda t, arr: 100.0 + arr * (ord(t[0]) - 63))
    spy = pd.Series(100.0, index=panel.index, name="SPY")
    rows = build_signal_rows(panel, spy, freq_days=90, horizon_days=30)
    assert rows
    for r in rows:
        assert r["net_alpha"] == pytest.approx(r["alpha"] - 0.0020, abs=1e-12)
        assert r["tier"] == "NA"


# ---------------------------------------------------------------------------
# integration — the harness prefers the B signal under a real momentum effect
# ---------------------------------------------------------------------------


def test_integration_harness_ranks_buy_above_sell_on_momentum_panel():
    """On a synthetic panel with a genuine, persistent momentum effect (high-mom
    names keep rising, low-mom names keep falling), harness.evaluate must assign
    the B signal a higher OOS/average alpha than the S signal."""
    from trade_modules.validation.harness import evaluate

    n = 900
    n_names = 24
    rng = np.random.default_rng(7)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")

    # Assign each name a persistent drift; winners drift up, losers drift down.
    # Momentum_12_1 will rank them by that drift, and forward returns follow the
    # same drift => a genuine momentum effect the signal should capture.
    data = {}
    for i in range(n_names):
        drift = (i - n_names / 2) / n_names * 0.004  # spread of daily drifts
        noise = rng.normal(0, 0.001, n)
        logret = drift + noise
        price = 100.0 * np.exp(np.cumsum(logret))
        data[f"N{i:02d}"] = price
    panel = pd.DataFrame(data, index=idx)
    spy = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.0005, n))), index=idx, name="SPY")

    rows = build_signal_rows(panel, spy, freq_days=30, horizon_days=30, buy_pct=0.25, sell_pct=0.25)
    assert rows

    report = evaluate(rows, family_key="signal", n_trials=5, min_obs=5)
    fams = report["families"]
    assert "B" in fams and "S" in fams

    b_alpha = fams["B"].get("mu_alpha")
    s_alpha = fams["S"].get("mu_alpha")
    assert b_alpha is not None and s_alpha is not None
    assert b_alpha > s_alpha, f"B alpha {b_alpha} should exceed S alpha {s_alpha}"


# ---------------------------------------------------------------------------
# exit_bucket_alpha — long-only exit-rule validation helper (PURE)
# ---------------------------------------------------------------------------


def _make_rows_with_known_exit_alpha(exit_alpha: float, universe_alpha: float) -> list[dict]:
    """Synthetic harness rows where EXIT bucket and universe have known mean alphas.

    EXIT rows (signal='EXIT') carry exit_alpha; HOLD rows carry universe_alpha.
    This lets us test exit_bucket_alpha's arithmetic in isolation.
    """
    rows = []
    # 20 EXIT rows
    for i in range(20):
        rows.append(
            {
                "ticker": f"EXIT_{i}",
                "signal": "EXIT",
                "alpha": exit_alpha,
                "net_alpha": exit_alpha - 0.002,
                "horizon": 30,
                "tier": "NA",
                "signal_date": "2022-01-01",
            }
        )
    # 40 HOLD rows (stand-in for universe)
    for i in range(40):
        rows.append(
            {
                "ticker": f"HOLD_{i}",
                "signal": "HOLD",
                "alpha": universe_alpha,
                "net_alpha": universe_alpha - 0.002,
                "horizon": 30,
                "tier": "NA",
                "signal_date": "2022-01-01",
            }
        )
    return rows


class TestExitBucketAlpha:
    """Tests for exit_bucket_alpha(rows) — pure helper for the long-only exit rule.

    exit_bucket_alpha(rows) -> dict with keys:
      exit_mean_alpha (float), universe_mean_alpha (float),
      exit_justified (bool),  # exit_mean_alpha < universe_mean_alpha
      exit_n (int), universe_n (int)
    """

    def test_exit_lags_universe_justified_true(self):
        """When EXIT bucket underperforms universe, exit_justified=True."""
        rows = _make_rows_with_known_exit_alpha(exit_alpha=-0.01, universe_alpha=0.02)
        result = exit_bucket_alpha(rows)
        assert result["exit_justified"] is True

    def test_exit_outperforms_universe_justified_false(self):
        """When EXIT bucket outperforms universe, exit_justified=False (honest: no exit edge)."""
        rows = _make_rows_with_known_exit_alpha(exit_alpha=0.05, universe_alpha=0.01)
        result = exit_bucket_alpha(rows)
        assert result["exit_justified"] is False

    def test_exit_equal_to_universe_justified_false(self):
        """When EXIT == universe mean alpha exactly, exit_justified=False (no edge)."""
        rows = _make_rows_with_known_exit_alpha(exit_alpha=0.02, universe_alpha=0.02)
        result = exit_bucket_alpha(rows)
        assert result["exit_justified"] is False

    def test_correct_mean_alphas(self):
        """exit_mean_alpha and universe_mean_alpha match the known synthetic values.

        universe_mean_alpha is the mean over ALL rows (EXIT + HOLD combined).
        With 20 EXIT at -0.03 and 40 HOLD at 0.01:
          universe_mean = (20*-0.03 + 40*0.01) / 60 = (-0.6 + 0.4) / 60 ≈ -0.00333
        """
        rows = _make_rows_with_known_exit_alpha(exit_alpha=-0.03, universe_alpha=0.01)
        result = exit_bucket_alpha(rows)
        assert result["exit_mean_alpha"] == pytest.approx(-0.03, abs=1e-10)
        expected_universe = (20 * -0.03 + 40 * 0.01) / 60
        assert result["universe_mean_alpha"] == pytest.approx(expected_universe, abs=1e-10)

    def test_correct_counts(self):
        """exit_n and universe_n match the synthetic row counts."""
        rows = _make_rows_with_known_exit_alpha(exit_alpha=0.0, universe_alpha=0.0)
        result = exit_bucket_alpha(rows)
        assert result["exit_n"] == 20
        assert result["universe_n"] == 60  # all rows

    def test_no_exit_rows_returns_none_justified(self):
        """When there are no EXIT rows, exit_justified is None (cannot determine)."""
        rows = [
            {
                "ticker": "A",
                "signal": "HOLD",
                "alpha": 0.01,
                "net_alpha": 0.008,
                "horizon": 30,
                "tier": "NA",
                "signal_date": "2022-01-01",
            }
        ]
        result = exit_bucket_alpha(rows)
        assert result["exit_justified"] is None
        assert result["exit_n"] == 0

    def test_empty_rows_returns_none_justified(self):
        """Empty row list → exit_justified=None, counts=0."""
        result = exit_bucket_alpha([])
        assert result["exit_justified"] is None
        assert result["exit_n"] == 0
        assert result["universe_n"] == 0
