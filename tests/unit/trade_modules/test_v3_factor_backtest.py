"""TDD — panel-history per-factor backtest helpers (which parameters have alpha)."""

from __future__ import annotations

import pandas as pd

from trade_modules.v3.factor_backtest import (
    BACKTEST_PANEL_FACTORS,
    factor_zscore,
    forward_return_at,
    is_active,
    summarize_ic,
)


def test_factor_zscore_respects_direction():
    idx = ["a", "b", "c", "d"]
    # low P/E is good (direction -1): the cheapest name gets the HIGHEST z.
    zp = factor_zscore(pd.Series([10, 20, 30, 40], index=idx), "pe_trailing", sector_neutral=False)
    assert zp["a"] > zp["d"]
    # high ROE is good (direction +1): the highest ROE gets the highest z.
    zr = factor_zscore(pd.Series([10, 20, 30, 40], index=idx), "roe", sector_neutral=False)
    assert zr["d"] > zr["a"]


def test_forward_return_at():
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    px = pd.DataFrame({"AAA": [100, 101, 102, 103, 110], "BBB": [50, 50, 50, 50, 45]}, index=idx)
    fr = forward_return_at(px, idx[0], horizon=4)
    assert round(fr["AAA"], 4) == 0.10  # 100 -> 110
    assert round(fr["BBB"], 4) == -0.10  # 50 -> 45


def test_forward_return_at_missing_or_out_of_range():
    idx = pd.date_range("2026-01-01", periods=3, freq="B")
    px = pd.DataFrame({"AAA": [100, 101, 102]}, index=idx)
    assert forward_return_at(px, idx[0], horizon=9).empty  # horizon runs past data
    assert forward_return_at(px, "2099-01-01", horizon=1).empty  # as_of absent


def test_summarize_ic():
    s = summarize_ic(pd.Series([0.1, 0.2, 0.3]))
    assert s["n"] == 3
    assert round(s["mean_ic"], 4) == 0.2
    assert s["hit_rate"] == 1.0
    assert s["t_stat"] > 0


def test_is_active_flags_discarded_factors():
    assert is_active("pe_trailing") is True  # in the value cluster
    assert is_active("roe") is True
    assert is_active("upside") is False  # discarded
    assert is_active("buy_pct") is False  # discarded


def test_backtest_factor_set_covers_active_and_discarded():
    feats = set(BACKTEST_PANEL_FACTORS.values())
    assert {"pe_trailing", "roe", "analyst_mom"} <= feats  # active
    assert {"upside", "buy_pct", "pe_forward"} <= feats  # discarded, tested for alpha
