"""
trade_modules/v3/validate_spine.py — Gate runner (cross-sectional IC + harness DSR)

Public API
----------
classify_regimes(index_close) -> dict[date, str]
    Map each trading date to RISK_ON / NEUTRAL / RISK_OFF (point-in-time).

build_rows(scores, fwd, horizons, top_q=0.2, regime=None) -> list[dict]
    Top-quantile long book → harness rows with net_alpha.

run_gate(scores, fwd, horizons, n_trials=2, min_obs=10, ic_min=0.02,
         t_min=3.0, hit_min=0.55) -> dict
    Calls the REAL validate harness + IC decay + v3 labels IC funcs.
    Returns verdict with primary_ic_pass/dsr_pass/gate_pass.
"""

from __future__ import annotations

import pandas as pd

from trade_modules.v3.constants import V3_SIGNAL_HORIZON
from trade_modules.v3.labels import cross_sectional_ic, demean_by_date, ic_summary
from trade_modules.validation.harness import evaluate
from trade_modules.validation.ic_decay import compute_ic_decay


def classify_regimes(index_close: pd.Series) -> dict:
    """Map each trading date to RISK_ON / NEUTRAL / RISK_OFF (point-in-time).

    Logic (mirrors compute_regime in report.py, applied across the full series):
      RISK_ON  = index above its 200-day MA AND expanding vol-percentile < 60
      RISK_OFF = index below its 200-day MA AND expanding vol-percentile > 60
      else NEUTRAL

    Uses expanding-window vol-percentile rank so the classification is purely
    PIT (no look-ahead). Dates with < 200 bars of history are excluded.

    Args:
        index_close: Daily close prices for a broad market index (e.g. S&P 500).

    Returns:
        Dict mapping each eligible date to its regime label string.
    """
    s = pd.to_numeric(pd.Series(index_close), errors="coerce").dropna()
    ma200 = s.rolling(200).mean()
    rv = s.pct_change(fill_method=None).rolling(21).std()
    # Expanding-window percentile rank: fraction of past vol readings ≤ current.
    rv_pctile = rv.expanding().rank(pct=True) * 100

    result: dict = {}
    for date in s.index:
        ma = ma200.get(date)
        pct = rv_pctile.get(date)
        if pd.isna(ma) or pd.isna(pct):
            continue
        above = float(s[date]) > float(ma)
        if above and float(pct) < 60:
            label = "RISK_ON"
        elif not above and float(pct) > 60:
            label = "RISK_OFF"
        else:
            label = "NEUTRAL"
        result[date] = label
    return result


def build_rows(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    horizons: list[int],
    top_q: float = 0.2,
    regime: dict | None = None,
) -> list[dict]:
    """Build harness result rows from the top-quantile long book.

    Args:
        scores: Long-form DataFrame with columns [as_of, ticker, score].
        fwd:    Long-form DataFrame with columns [as_of, ticker, horizon, fwd_ret].
        horizons: List of forecast horizons to include.
        top_q:  Top quantile fraction (e.g. 0.2 = top 20% by score).
        regime: Optional mapping of as_of date → regime label.

    Returns:
        List of dicts with keys: ticker, signal_date, horizon, net_alpha, signal,
        and optionally regime.
    """
    fwd = demean_by_date(fwd)
    rows: list[dict] = []
    for h in horizons:
        f = fwd[fwd["horizon"] == h].merge(scores, on=["as_of", "ticker"], how="inner")
        for asof, g in f.groupby("as_of"):
            thr = g["score"].quantile(1.0 - top_q)
            longs = g[g["score"] >= thr]
            for _, r in longs.iterrows():
                row: dict = {
                    "ticker": str(r["ticker"]),
                    "signal_date": str(pd.Timestamp(asof).date()),
                    "horizon": int(h),
                    "net_alpha": float(r["net_alpha"]),
                    "signal": "spine",
                }
                if regime is not None:
                    row["regime"] = regime.get(asof, "NA")
                rows.append(row)
    return rows


def run_gate(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    horizons: list[int],
    *,
    n_trials: int = 2,
    min_obs: int = 10,
    ic_min: float = 0.02,
    t_min: float = 3.0,
    hit_min: float = 0.55,
    primary_horizon: int = V3_SIGNAL_HORIZON,
    regime: dict | None = None,
) -> dict:
    """Run the price-spine gate: cross-sectional IC check + harness DSR.

    Args:
        scores:   Long-form DataFrame with columns [as_of, ticker, score].
        fwd:      Long-form DataFrame with columns [as_of, ticker, horizon, fwd_ret].
        horizons: List of forecast horizons to grade (the measurement grid).
        primary_horizon: The horizon at which the IC screen AND the harness DSR
            are graded (default V3_SIGNAL_HORIZON = 21). Falls back to the nearest
            member of ``horizons`` when the exact value is absent.
        n_trials: Number of tuned trials for DSR deflation.
        min_obs:  Minimum observations required for the harness to consider a family
                  material.
        ic_min:   Minimum mean IC for primary_ic_pass.
        t_min:    Minimum t-stat for primary_ic_pass.
        hit_min:  Minimum hit rate for primary_ic_pass.
        regime:   Optional as_of→regime mapping forwarded to build_rows.

    Returns:
        Dict with keys:
          - ic:               {horizon: ic_summary_dict}
          - ic_decay:         compute_ic_decay result
          - primary_horizon:  the horizon graded (member of ``horizons``)
          - harness:          raw evaluate() VerdictReport
          - primary_ic_pass:  bool — IC gate on primary horizon
          - dsr_pass:         bool — harness overall passed
          - gate_pass:        bool — primary_ic_pass AND dsr_pass
    """
    rows = build_rows(scores, fwd, horizons, regime=regime)
    report = evaluate(
        rows,
        family_key="signal",
        n_trials=n_trials,
        min_obs=min_obs,
        horizons=tuple(horizons),
        primary_horizon=primary_horizon,
    )
    ic = {h: ic_summary(cross_sectional_ic(scores, fwd, h)) for h in horizons}
    decay = compute_ic_decay({h: ic[h]["mean_ic"] for h in horizons})
    ph = (
        primary_horizon
        if primary_horizon in horizons
        else min(horizons, key=lambda h: abs(h - primary_horizon))
    )
    primary_ic_pass = bool(
        ic[ph]["mean_ic"] >= ic_min and ic[ph]["t_stat"] >= t_min and ic[ph]["hit_rate"] >= hit_min
    )
    dsr_pass = bool(report.get("overall", {}).get("passed", False))
    return {
        "ic": ic,
        "ic_decay": decay,
        "primary_horizon": ph,
        "harness": report,
        "primary_ic_pass": primary_ic_pass,
        "dsr_pass": dsr_pass,
        "gate_pass": bool(primary_ic_pass and dsr_pass),
    }
