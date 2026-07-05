"""riskfirst engine — combine factors, select, construct (risk-first), recommend.

Factors are injected as callables ``df -> pd.Series`` so this core is testable on
its own. The actual wiring of the five factor modules + live data + the edge gate
lives in the shadow runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.analysis.tiers import _parse_market_cap

from .construct import apply_name_cap, cap_groups, erc_weights, vol_target_scale
from .covariance import single_factor_cov
from .fx import USD_BLOC, cap_bloc, currency_of
from .regime_overlay import scale_for_regime

_ELIGIBILITY_FACTOR_COLS = ("ROE", "PET", "PEF", "FCF", "EG", "%B")


def eligible_universe(
    df: pd.DataFrame,
    *,
    min_cap: float = 2e9,
    min_factors: int = 3,
    factor_cols=_ELIGIBILITY_FACTOR_COLS,
) -> pd.DataFrame:
    """Investability gate: drop names below ``min_cap`` market cap or with fewer
    than ``min_factors`` present fundamentals. Prevents the size factor from
    pulling the book into illiquid, data-less micro-caps."""
    if "CAP" in df.columns:
        caps = df["CAP"].map(_parse_market_cap)
        cap_ok = pd.to_numeric(caps, errors="coerce").fillna(0.0) >= min_cap
    else:
        cap_ok = pd.Series(False, index=df.index)
    present = [c for c in factor_cols if c in df.columns]
    if present:
        n_present = df[present].notna().sum(axis=1)
    else:
        n_present = pd.Series(0, index=df.index)
    return df[cap_ok & (n_present >= min_factors)]


def composite_score(df: pd.DataFrame, factor_fns, weights=None) -> pd.Series:
    """Mean (or weighted mean) of the per-factor z-scores, row-wise, NaN-skipping."""
    mat = pd.concat([fn(df) for fn in factor_fns], axis=1)
    if weights is None:
        comp = mat.mean(axis=1, skipna=True)
    else:
        w = np.asarray(weights, dtype=float)
        masked = mat.notna().to_numpy() * w
        num = (mat.fillna(0.0).to_numpy() * w).sum(axis=1)
        den = masked.sum(axis=1)
        comp = pd.Series(np.where(den > 0, num / den, np.nan), index=mat.index)
    return comp.reindex(df.index)


def select_and_construct(
    df: pd.DataFrame,
    factor_fns,
    *,
    top_n: int,
    market_vol: float = 0.18,
    idio_vol: float = 0.25,
    target_vol: float = 0.10,
    name_cap: float = 0.08,
    usd_bloc_cap: float = 0.60,
    sector_cap: float = 0.25,
    weights=None,
    cov_fn=None,
    regime_multiplier: float = 1.0,
) -> dict:
    """Score -> select top_n -> ERC -> vol-target -> name cap -> USD-bloc cap
    -> sector cap (if a SECTOR column is present).

    ``cov_fn(selected_tickers) -> cov matrix`` supplies an empirical covariance
    (aligned to the selected order); when None, a single-factor beta covariance is
    used as the fallback.

    ``regime_multiplier`` scales gross after all caps; default 1.0 is a no-op."""
    comp = composite_score(df, factor_fns, weights)
    ranked = comp.dropna().sort_values(ascending=False)
    selected = list(ranked.index[:top_n])

    sub = df.loc[selected]
    if cov_fn is not None:
        cov = np.asarray(cov_fn(selected), dtype=float)
    else:
        if "B" in sub.columns:
            betas = pd.to_numeric(sub["B"], errors="coerce").fillna(1.0).to_numpy()
        else:
            betas = np.ones(len(selected))
        cov = single_factor_cov(betas, market_vol, idio_vol)
    w = erc_weights(cov)
    w = vol_target_scale(w, cov, target_vol, max_gross=1.0)
    w = apply_name_cap(w, name_cap)
    is_bloc = np.array([currency_of(t) in USD_BLOC for t in selected])
    w = cap_bloc(w, is_bloc, usd_bloc_cap)
    w = apply_name_cap(w, name_cap)  # second pass: keep single-name cap after bloc redistribution
    if "SECTOR" in sub.columns:  # sector cap is dormant until sector labels exist
        w = cap_groups(w, sub["SECTOR"].astype(str).to_numpy(), sector_cap)
        w = apply_name_cap(w, name_cap)

    if regime_multiplier < 1.0:
        w = scale_for_regime(w, regime_multiplier)

    full = pd.Series(0.0, index=df.index)
    full.loc[selected] = w
    gross = float(full.sum())
    return {
        "weights": full,
        "composite": comp,
        "selected": selected,
        "cov": cov,
        "gross": gross,
        "cash": max(0.0, 1.0 - gross),
        "usd_bloc": float(full[[currency_of(t) in USD_BLOC for t in full.index]].sum()),
    }


def recommend(target: pd.Series, current: pd.Series, eps: float = 1e-4) -> pd.DataFrame:
    """Classify per-name weight deltas into BUY / ADD / TRIM / SELL / HOLD."""
    idx = target.index.union(current.index)
    t = target.reindex(idx, fill_value=0.0)
    c = current.reindex(idx, fill_value=0.0)
    rows = []
    for k in idx:
        delta = float(t[k] - c[k])
        if delta > eps:
            action = "BUY" if c[k] <= eps else "ADD"
        elif delta < -eps:
            action = "SELL" if t[k] <= eps else "TRIM"
        else:
            action = "HOLD"
        rows.append(
            {
                "ticker": k,
                "current": float(c[k]),
                "target": float(t[k]),
                "delta": delta,
                "action": action,
            }
        )
    return pd.DataFrame(rows)
