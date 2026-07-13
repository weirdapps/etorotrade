"""v3 portfolio construction bridge + report-only risk gate.

Turns the v3 combiner output (a scored frame carrying ``conviction`` /
``eligible`` / ``sector`` / ``price`` / ``beta``) into a risk-first book by
delegating the heavy lifting to :mod:`trade_modules.riskfirst` — ERC weights,
vol targeting, single-name / sector / USD-bloc caps, and the regime dial — then
layering:

  * an empirical shrunk-covariance estimate from the supplied price history,
    falling back to the engine's single-factor beta covariance whenever a
    selected name lacks usable price history;
  * a report-only risk gate — parametric-normal CVaR(95%), net beta, and the
    effective number of bets — whose only hard action is a soft *proportional
    gross shrink* when CVaR exceeds its budget (it never raises); and
  * fractional-Kelly gross sizing as the final capital-deployment dial.

Pure orchestration: no network access and no ``yahoofinance.core.config``
import (module-level or otherwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.covariance import single_factor_cov
from trade_modules.riskfirst.engine import select_and_construct
from trade_modules.riskfirst.fx import USD_BLOC, currency_of
from trade_modules.riskfirst.prices import daily_returns, shrunk_cov

# Parametric-normal 95% Expected Shortfall multiplier: E[Z | Z > z_.95] = φ(1.645)/0.05.
_Z_ES_95 = 2.063
# Report-only risk-gate thresholds.
_BETA_BAND = (0.3, 1.1)
_MIN_EFFECTIVE_BETS = 12
# Minimum overlapping return observations before the empirical cov is trusted.
_MIN_COV_OBS = 60
# Single-factor fallback assumptions (mirror engine.select_and_construct defaults).
_MARKET_VOL = 0.18
_IDIO_VOL = 0.25


def _make_cov_fn(prices: pd.DataFrame, scored: pd.DataFrame):
    """Build ``cov_fn(selected) -> cov`` for the engine.

    Prefers the empirical shrunk covariance from ``prices``; falls back to the
    single-factor beta covariance whenever any selected name lacks usable price
    history (missing column, empty frame, or too few overlapping observations).
    The returned matrix is aligned to the order of ``selected``.
    """

    def cov_fn(selected) -> np.ndarray:
        sel = list(selected)
        have_cols = (
            prices is not None and not prices.empty and all(t in prices.columns for t in sel)
        )
        if have_cols:
            rets = daily_returns(prices[sel]).dropna(how="any")
            if len(rets) >= _MIN_COV_OBS:
                return shrunk_cov(rets)
        betas = pd.to_numeric(scored.reindex(sel)["beta"], errors="coerce").fillna(1.0).to_numpy()
        return single_factor_cov(betas, _MARKET_VOL, _IDIO_VOL)

    return cov_fn


def _sector_exposures(weights: pd.Series, sector_labels: dict) -> dict:
    """Aggregate (absolute) portfolio weight per uppercased sector label."""
    out: dict[str, float] = {}
    for tkr, w in weights.items():
        if w <= 1e-12:
            continue
        lab = sector_labels.get(tkr, "UNKNOWN")
        out[lab] = out.get(lab, 0.0) + float(w)
    return out


def _empty_result(cvar_budget: float) -> dict:
    """A well-formed all-cash result for a degenerate (no eligible names) input."""
    return {
        "weights": pd.Series(dtype=float),
        "gross": 0.0,
        "cash": 1.0,
        "usd_bloc": 0.0,
        "sector_exposures": {},
        "selected": [],
        "diagnostics": {
            "cvar_95": 0.0,
            "cvar_budget": cvar_budget,
            "net_beta": 0.0,
            "net_beta_band": _BETA_BAND,
            "effective_bets": 0.0,
            "port_vol": 0.0,
            "gross_risk": 0.0,
            "binding": {"cvar": False, "net_beta": False, "effective_bets": False},
        },
    }


def build_portfolio(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_n: int = 20,
    target_vol: float = 0.12,
    name_cap: float = 0.08,
    sector_cap: float = 0.25,
    usd_bloc_cap: float = 0.60,
    regime_multiplier: float = 1.0,
    kelly_fraction: float = 0.25,
    cvar_budget: float = 0.25,
) -> dict:
    """Construct a risk-first book from a v3 scored frame + price history.

    Args:
        scored: Output of :func:`trade_modules.v3.combine.compute_scores` — must
            carry ``conviction``, ``eligible``, ``sector``, ``price`` and
            ``beta`` columns, indexed by ticker.
        prices: Daily closes (dates x tickers) for the covariance estimate.
        top_n: Number of top-conviction names to select.
        target_vol: Annualized portfolio volatility target (ERC + vol scaling).
        name_cap / sector_cap / usd_bloc_cap: Binding concentration caps handed
            to the riskfirst engine (sector cap uses an uppercased ``SECTOR``).
        regime_multiplier: Gross dial applied by the engine (<1.0 de-risks).
        kelly_fraction: Fractional-Kelly gross scaler applied last; the final
            gross is then capped at 1.0 (no leverage).
        cvar_budget: Annual parametric-normal CVaR(95%) ceiling for the risk
            book; a breach triggers a soft proportional gross shrink (never a
            crash).

    Returns:
        ``{weights, gross, cash, usd_bloc, sector_exposures, selected,
        diagnostics}``. ``diagnostics`` holds ``cvar_95`` (annual fraction),
        ``net_beta``, ``effective_bets`` (= 1/Σwᵢ² on invested proportions),
        ``port_vol``, ``gross_risk`` (pre-Kelly), and per-metric ``binding``
        flags. CVaR / net-beta / effective-bets are reported for oversight; only
        CVaR acts (soft shrink). The shape metrics (net beta, effective bets)
        are computed on invested proportions so they stay meaningful regardless
        of the Kelly / regime gross dials.
    """
    if "eligible" in scored.columns:
        elig = scored[scored["eligible"].astype(bool)].copy()
    else:
        elig = scored.copy()
    if elig.empty:
        return _empty_result(cvar_budget)

    # Attach the columns the engine reads: uppercased SECTOR (activates the
    # sector cap) and CAP (carried through for downstream reporting).
    sub = elig.copy()
    sub["SECTOR"] = sub["sector"].astype("string").str.upper()
    sub["CAP"] = sub["cap"] if "cap" in sub.columns else np.nan
    sector_labels = sub["SECTOR"].astype(str).to_dict()

    # Conviction is the sole selection/sizing factor (a pass-through).
    factor_fns = [lambda df: df["conviction"].reindex(df.index)]

    res = select_and_construct(
        sub,
        factor_fns,
        top_n=top_n,
        target_vol=target_vol,
        name_cap=name_cap,
        usd_bloc_cap=usd_bloc_cap,
        sector_cap=sector_cap,
        cov_fn=_make_cov_fn(prices, sub),
        regime_multiplier=regime_multiplier,
    )
    selected = res["selected"]
    if not selected:
        return _empty_result(cvar_budget)

    w = res["weights"].copy()  # full-index Series, post-regime risk book
    cov = np.asarray(res["cov"], dtype=float)
    w_sel = w.loc[selected].to_numpy()

    # --- report-only risk gate on the risk book (pre-Kelly) ---
    port_var = float(w_sel @ cov @ w_sel)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    cvar_95 = _Z_ES_95 * port_vol
    gross_risk = float(w_sel.sum())

    cvar_binding = bool(cvar_95 > cvar_budget)
    if cvar_binding and cvar_95 > 0:  # soft proportional shrink to the budget
        shrink = cvar_budget / cvar_95
        w = w * shrink
        w_sel = w_sel * shrink
        port_vol *= shrink
        cvar_95 = _Z_ES_95 * port_vol

    # Shape metrics on invested proportions (scale-invariant across the dials).
    sw = float(w_sel.sum())
    if sw > 0:
        prop = w_sel / sw
        effective_bets = float(1.0 / np.sum(prop**2))
    else:
        prop = w_sel
        effective_bets = 0.0
    betas_sel = pd.to_numeric(scored.reindex(selected)["beta"], errors="coerce").to_numpy()
    net_beta = float(np.nansum(prop * betas_sel))

    diagnostics = {
        "cvar_95": cvar_95,
        "cvar_budget": cvar_budget,
        "net_beta": net_beta,
        "net_beta_band": _BETA_BAND,
        "effective_bets": effective_bets,
        "port_vol": port_vol,
        "gross_risk": gross_risk,
        "binding": {
            "cvar": cvar_binding,
            "net_beta": not (_BETA_BAND[0] <= net_beta <= _BETA_BAND[1]),
            "effective_bets": effective_bets < _MIN_EFFECTIVE_BETS,
        },
    }

    # --- fractional-Kelly final gross sizing (capped at no leverage) ---
    kelly = max(0.0, float(kelly_fraction))
    w = w * kelly
    gross = float(w.sum())
    if gross > 1.0:
        w = w / gross
        gross = 1.0

    usd_bloc = float(sum(v for t, v in w.items() if currency_of(t) in USD_BLOC))
    return {
        "weights": w,
        "gross": gross,
        "cash": max(0.0, 1.0 - gross),
        "usd_bloc": usd_bloc,
        "sector_exposures": _sector_exposures(w, sector_labels),
        "selected": selected,
        "diagnostics": diagnostics,
    }
