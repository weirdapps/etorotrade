"""TDD — panel-history per-factor backtest helpers (which parameters have alpha)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_modules.v3.factor_backtest import (
    BACKTEST_PANEL_FACTORS,
    beta_neutralize,
    factor_zscore,
    fama_macbeth,
    forward_return_at,
    hac_tstat,
    ic_by_sector,
    is_active,
    sector_neutralize,
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


def test_beta_neutralize_removes_pure_beta():
    idx = ["a", "b", "c", "d"]
    beta = pd.Series([0.5, 1.0, 1.5, 2.0], index=idx)
    r = 0.01 + 0.02 * beta  # pure beta exposure, zero alpha
    resid = beta_neutralize(r, beta)
    assert resid.abs().max() < 1e-9  # the market-beta component is fully removed


def test_beta_neutralize_keeps_alpha():
    idx = ["a", "b", "c", "d", "e"]
    beta = pd.Series([0.5, 1.0, 1.5, 2.0, 1.0], index=idx)
    r = 0.01 + 0.02 * beta
    r["e"] = r["e"] + 0.10  # e has genuine +10% alpha on top of its beta
    resid = beta_neutralize(r, beta)
    assert resid["e"] > 0.05  # alpha survives neutralization


def test_sector_neutralize_demeans_within_sector():
    idx = ["a", "b", "c", "d"]
    r = pd.Series([0.1, 0.3, 0.0, 0.2], index=idx)
    sec = pd.Series(["X", "X", "Y", "Y"], index=idx)
    resid = sector_neutralize(r, sec)
    assert round(resid["a"], 4) == -0.1 and round(resid["b"], 4) == 0.1  # X mean 0.2
    assert round(resid["c"], 4) == -0.1 and round(resid["d"], 4) == 0.1  # Y mean 0.1


def test_ic_by_sector_splits_and_signs():
    # sector X: z predicts fwd POSITIVELY; sector Y: z predicts NEGATIVELY.
    rows = []
    for d in ("2026-01-01", "2026-01-08", "2026-01-15"):
        for i in range(6):
            rows.append(
                {"date": d, "ticker": f"X{i}", "z": float(i), "fwd": i * 0.01, "sector": "X"}
            )
            rows.append(
                {"date": d, "ticker": f"Y{i}", "z": float(i), "fwd": -i * 0.01, "sector": "Y"}
            )
    df = pd.DataFrame(rows)
    res = ic_by_sector(df, "z", "fwd", "sector", date_col="date", min_names_per_date=4)
    assert res["X"]["mean_ic"] > 0.9  # perfectly rank-aligned within X
    assert res["Y"]["mean_ic"] < -0.9  # perfectly rank-inverted within Y
    assert res["X"]["n_dates"] == 3


def test_ic_by_sector_drops_thin_sectors():
    # sector Z has only 2 names/date -> below min_names_per_date, so it is dropped.
    rows = []
    for d in ("2026-01-01", "2026-01-08"):
        for i in range(2):
            rows.append(
                {"date": d, "ticker": f"Z{i}", "z": float(i), "fwd": i * 0.01, "sector": "Z"}
            )
    df = pd.DataFrame(rows)
    res = ic_by_sector(df, "z", "fwd", "sector", date_col="date", min_names_per_date=5)
    assert "Z" not in res


def test_ic_by_sector_min_dates_drops_short_history():
    # A sector present on only ONE date is dropped when min_dates=2 (too few
    # independent observations for a meaningful mean IC).
    rows = [
        {"date": "2026-01-01", "ticker": f"X{i}", "z": float(i), "fwd": i * 0.01, "sector": "X"}
        for i in range(8)
    ]
    df = pd.DataFrame(rows)
    res = ic_by_sector(df, "z", "fwd", "sector", date_col="date", min_names_per_date=4, min_dates=2)
    assert "X" not in res


def test_hac_tstat_deflates_autocorrelated_series():
    # A blocky, positively-autocorrelated IC series: the naive t treats the 32
    # points as independent and OVERSTATES significance; HAC (Newey-West) inflates
    # the SE, so the corrected |t| is smaller (but still positive here).
    x = pd.Series([0.10] * 8 + [0.06] * 8 + [0.08] * 8 + [0.05] * 8)
    naive = summarize_ic(x)["t_stat"]
    hac = hac_tstat(x, lags=6)
    assert 0 < hac < naive


def test_hac_tstat_close_to_naive_without_lags():
    # lags=0 applies no autocorrelation correction, so HAC ~ the naive t-stat.
    x = pd.Series([0.1, -0.05, 0.08, -0.02, 0.06, -0.01, 0.04, 0.02])
    naive = summarize_ic(x)["t_stat"]
    assert hac_tstat(x, lags=0) == pytest.approx(naive, rel=0.15)


def test_fama_macbeth_recovers_slope():
    # y = intercept_d + slope_d * x with slope alternating 1.98/2.02 (mean 2.0).
    # The simultaneous per-date regression should recover a mean slope ~2.0 with a
    # large t-stat (the slope is highly consistent across dates).
    frames = []
    for d in range(6):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        slope = 2.02 if d % 2 else 1.98
        frames.append(pd.DataFrame({"date": f"d{d}", "x": x, "y": 1.0 * d + slope * x}))
    df = pd.concat(frames, ignore_index=True)
    res = fama_macbeth(df, y_col="y", x_cols=["x"], date_col="date")
    assert res["x"]["coef"] == pytest.approx(2.0, abs=1e-6)
    assert res["x"]["t"] > 3
    assert res["x"]["n_dates"] == 6


def test_fama_macbeth_multivariate_isolates_each_factor():
    # y = 3*a - 1*b per date; FM should recover coef(a)~3, coef(b)~-1 SIMULTANEOUSLY
    # (this is the point vs sequential residualization).
    frames = []
    for d in range(5):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        frames.append(pd.DataFrame({"date": f"d{d}", "a": a, "b": b, "y": 3.0 * a - 1.0 * b}))
    df = pd.concat(frames, ignore_index=True)
    res = fama_macbeth(df, y_col="y", x_cols=["a", "b"], date_col="date")
    assert res["a"]["coef"] == pytest.approx(3.0, abs=1e-6)
    assert res["b"]["coef"] == pytest.approx(-1.0, abs=1e-6)
