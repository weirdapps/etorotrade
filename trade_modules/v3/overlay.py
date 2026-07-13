"""v3 OVERLAY construction — keep the live book, sell only the genuinely weak.

The from-scratch top-N constructor (:func:`trade_modules.v3.construct.build_portfolio`)
rebuilds the book from zero and therefore recommends selling most of a 44-name
live book purely as a 44->20 structural artifact, not because the tilt says so.
This module instead treats the CURRENT book as the anchor and applies a minimal,
conviction-driven overlay:

  * **SELL** only holdings that are genuinely weak — ineligible / dataless
    (conviction NaN) OR whose conviction sits at/below the ``sell_pctile``
    percentile of the eligible universe. Their weight is freed.
  * **KEEP** every other holding at its current weight (the anchor).
  * **BUY** up to ``max_new`` of the highest-conviction NON-held names whose
    conviction clears the ``buy_pctile`` percentile, funded from the freed
    weight plus cash toward ``gross_target``.
  * then run the combined book (keeps + buys) through the SAME Phase 5A hard risk
    gate (:func:`trade_modules.v3.risk_gate.apply_risk_gate`) that
    :func:`build_portfolio` uses — vol ceiling, concentration caps, redistribution
    to convergence. The returned book is the GATED book.

Turnover (``sum(|Δw|)/2``) is REPORTED, never capped — the owner sees the churn
the overlay implies and decides. Pure orchestration: no network access and no
``yahoofinance.core.config`` import (module-level or otherwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.covariance import single_factor_cov
from trade_modules.riskfirst.fx import currency_of
from trade_modules.riskfirst.prices import daily_returns, shrunk_cov
from trade_modules.v3.risk_gate import apply_risk_gate

# A holding/target weight at or below this is treated as flat (not held).
_EPS = 1e-9
# Minimum overlapping return observations before the empirical cov is trusted
# (mirrors construct._MIN_COV_OBS); below it, fall back to single-factor beta cov.
_MIN_COV_OBS = 60
# Single-factor fallback assumptions (mirror construct._MARKET_VOL / _IDIO_VOL).
_MARKET_VOL = 0.18
_IDIO_VOL = 0.25


def _as_weight_series(weights) -> pd.Series:
    """Coerce a current-weights input (Series / dict) to a clean float Series.

    Non-numeric entries are dropped, the index is stringified, and duplicate
    tickers are summed so downstream membership tests are unambiguous.
    """
    if weights is None:
        return pd.Series(dtype=float)
    s = weights if isinstance(weights, pd.Series) else pd.Series(weights)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s):
        s.index = s.index.map(str)
        if not s.index.is_unique:
            s = s.groupby(level=0).sum()
    return s.astype(float)


def _beta_array(names: list[str], scored: pd.DataFrame, betas) -> np.ndarray:
    """Per-name betas aligned to ``names`` (explicit ``betas`` -> scored -> 1.0)."""
    bser = None
    if betas is not None:
        bser = betas if isinstance(betas, pd.Series) else pd.Series(betas)
        bser = pd.to_numeric(bser, errors="coerce")
        bser.index = bser.index.map(str)
    have_col = scored is not None and "beta" in scored.columns
    out: list[float] = []
    for t in names:
        v = np.nan
        if bser is not None and t in bser.index:
            v = float(bser.loc[t]) if pd.notna(bser.loc[t]) else np.nan
        if not np.isfinite(v) and have_col and t in scored.index:
            v = pd.to_numeric(scored.loc[t, "beta"], errors="coerce")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = np.nan
        out.append(fv if np.isfinite(fv) else 1.0)
    return np.asarray(out, dtype=float)


def _cov_matrix(prices: pd.DataFrame, names: list[str], betas_arr: np.ndarray) -> np.ndarray:
    """Shrunk empirical cov from ``prices`` when usable, else single-factor beta cov."""
    if not names:
        return np.zeros((0, 0), dtype=float)
    have = prices is not None and not prices.empty and all(t in prices.columns for t in names)
    if have:
        rets = daily_returns(prices[names]).dropna(how="any")
        if len(rets) >= _MIN_COV_OBS:
            return np.asarray(shrunk_cov(rets), dtype=float)
    return np.asarray(single_factor_cov(betas_arr, _MARKET_VOL, _IDIO_VOL), dtype=float)


def _sector_labels(names: list[str], scored: pd.DataFrame) -> list[str]:
    """Uppercased sector label per name ("UNKNOWN" when missing)."""
    if scored is None or "sector" not in scored.columns:
        return ["UNKNOWN"] * len(names)
    ser = scored.reindex(names)["sector"].astype("string").str.upper().fillna("UNKNOWN")
    return [s if s else "UNKNOWN" for s in ser.tolist()]


def build_overlay(
    scored: pd.DataFrame,
    current_weights,
    prices: pd.DataFrame,
    *,
    sell_pctile: float = 0.15,
    buy_pctile: float = 0.85,
    max_new: int = 8,
    gross_target: float = 0.95,
    name_cap: float = 0.08,
    sector_cap: float = 0.25,
    usd_bloc_cap: float = 0.60,
    vol_ceiling: float = 0.18,
    betas=None,
    core_list=None,
) -> dict:
    """Build a minimal keep/sell/buy overlay on the live book, then risk-gate it.

    Args:
        scored: Output of :func:`trade_modules.v3.combine.compute_scores` — must
            carry ``conviction`` and (ideally) ``eligible`` / ``sector`` / ``beta``
            columns, indexed by ticker. Ineligible names carry a NaN conviction.
        current_weights: Live book weights (Series or dict, fractions of NAV)
            indexed by ticker — the anchor the overlay preserves.
        prices: Daily closes (dates x tickers) for the covariance estimate; pass an
            empty frame to force the single-factor (beta) covariance fallback.
        sell_pctile: A held name is SOLD when its conviction is at or below this
            percentile of the eligible-universe convictions (default 0.15 = the
            weakest ~15%), in addition to always selling ineligible/dataless names.
        buy_pctile: A non-held name is BUY-eligible only when its conviction is at
            or above this percentile (default 0.85 = the strongest ~15%).
        max_new: Cap on the number of new names bought (strongest first).
        gross_target: Fraction of capital the book is funded toward — buys draw
            from the freed weight + cash to bring gross up to this target.
        name_cap / sector_cap / usd_bloc_cap / vol_ceiling: Passed to the Phase 5A
            risk gate applied to the combined keeps+buys book.
        betas: Optional per-name betas (Series/dict keyed by ticker). ``None`` ->
            read ``scored["beta"]`` (missing -> 1.0). Report-only in the gate (net
            beta) and used for the single-factor covariance fallback.
        core_list: Optional list of "core" tickers to report kept/sold status for
            (e.g. the mega-cap anchors), surfaced under ``core_retention``.

    Returns:
        ``{"weights", "diagnostics"}``:

        * ``weights``: the GATED target book as a ``pd.Series`` over keeps+buys
          (sold names are absent, i.e. weight 0).
        * ``diagnostics``: ``n_sell`` / ``n_buy`` / ``n_keep``, ``turnover``
          (= ``sum(|Δw|)/2`` vs the current book, REPORTED not capped),
          ``freed_weight``, ``keep_weight``, ``buy_weight``, ``sell_threshold``,
          ``buy_threshold``, ``gross_target``, the ``sold`` / ``kept`` / ``bought``
          ticker lists, ``core_retention`` (when ``core_list`` is given: per-name
          kept/sold/not_held + ``n_kept`` / ``n_sold``), and ``gate`` (the
          apply_risk_gate diagnostics on the final book).
    """
    cur = _as_weight_series(current_weights)
    held = [t for t in cur.index if cur[t] > _EPS]
    held_set = set(held)

    # Conviction lookup + eligibility (ineligible names carry a NaN conviction).
    if scored is not None and "conviction" in scored.columns:
        conv_all = pd.to_numeric(scored["conviction"], errors="coerce")
    else:
        conv_all = pd.Series(dtype=float)
    if scored is not None and "eligible" in scored.columns:
        elig_mask = scored["eligible"].fillna(False).astype(bool)
    elif scored is not None:
        elig_mask = pd.Series(True, index=scored.index)
    else:
        elig_mask = pd.Series(dtype=bool)

    elig_conv = conv_all[elig_mask].dropna() if len(conv_all) else pd.Series(dtype=float)
    if len(elig_conv):
        sell_threshold = float(elig_conv.quantile(sell_pctile))
        buy_threshold = float(elig_conv.quantile(buy_pctile))
    else:  # no eligible universe -> nothing to keep against / buy
        sell_threshold = float("nan")
        buy_threshold = float("nan")

    def _is_weak(t: str) -> bool:
        if t not in conv_all.index:  # dataless (absent from scored)
            return True
        c = conv_all.loc[t]
        if pd.isna(c):  # ineligible names have a NaN conviction
            return True
        if t in elig_mask.index and not bool(elig_mask.loc[t]):
            return True
        if not np.isnan(sell_threshold) and float(c) <= sell_threshold:
            return True  # genuinely weak by percentile
        return False

    sold = [t for t in held if _is_weak(t)]
    sold_set = set(sold)
    kept = [t for t in held if t not in sold_set]
    freed_weight = float(sum(cur[t] for t in sold))
    keep_sum = float(sum(cur[t] for t in kept))

    # BUY: strongest non-held eligible names clearing the buy percentile.
    bought: list[str] = []
    if not np.isnan(buy_threshold) and max_new > 0:
        cand = elig_conv[[t for t in elig_conv.index if t not in held_set]]
        cand = cand[cand >= buy_threshold].sort_values(ascending=False)
        bought = list(cand.index[: int(max_new)])

    # Pre-gate target: keeps anchored at current weight; buys funded from the freed
    # weight + cash toward gross_target (conviction-proportional across buys).
    target: dict[str, float] = {t: float(cur[t]) for t in kept}
    buy_budget = max(0.0, float(gross_target) - keep_sum)
    if bought and buy_budget > 0.0:
        bconv = conv_all.reindex(bought).astype(float).clip(lower=0.0).to_numpy()
        tot = float(bconv.sum())
        frac = bconv / tot if tot > 0 else np.full(len(bought), 1.0 / len(bought))
        for t, f in zip(bought, frac, strict=True):
            target[t] = buy_budget * float(f)

    target_names = list(target.keys())

    # Phase 5A hard risk gate on the combined book (keeps + buys).
    if target_names:
        betas_arr = _beta_array(target_names, scored, betas)
        cov = _cov_matrix(prices, target_names, betas_arr)
        tw = pd.Series([target[t] for t in target_names], index=target_names, dtype=float)
        final, gate_diag = apply_risk_gate(
            tw,
            cov,
            sectors=_sector_labels(target_names, scored),
            currencies=[currency_of(t) for t in target_names],
            betas=betas_arr,
            vol_ceiling=vol_ceiling,
            name_cap=name_cap,
            sector_cap=sector_cap,
            usd_bloc_cap=usd_bloc_cap,
        )
    else:  # everything sold / nothing to hold -> all cash
        final = pd.Series(dtype=float)
        gate_diag = {}

    # Turnover = sum(|Δw|)/2 over the union of tickers (reported, not capped).
    idx = final.index.union(cur.index)
    turnover = 0.5 * float(
        (final.reindex(idx).fillna(0.0) - cur.reindex(idx).fillna(0.0)).abs().sum()
    )

    diagnostics: dict = {
        "n_sell": len(sold),
        "n_buy": len(bought),
        "n_keep": len(kept),
        "turnover": turnover,
        "freed_weight": freed_weight,
        "keep_weight": keep_sum,
        "buy_weight": float(sum(target[t] for t in bought)) if bought else 0.0,
        "sell_threshold": sell_threshold,
        "buy_threshold": buy_threshold,
        "gross_target": float(gross_target),
        "sold": sold,
        "kept": kept,
        "bought": bought,
        "gate": gate_diag,
    }

    if core_list is not None:
        per: dict[str, str] = {}
        for c in core_list:
            cc = str(c)
            per[cc] = "kept" if cc in set(kept) else ("sold" if cc in sold_set else "not_held")
        diagnostics["core_retention"] = {
            "per_name": per,
            "n_kept": sum(1 for v in per.values() if v == "kept"),
            "n_sold": sum(1 for v in per.values() if v == "sold"),
        }

    return {"weights": final, "diagnostics": diagnostics}
