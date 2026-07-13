"""
validation/xsection_ic.py — Cross-sectional rank Information Coefficient (Phase 2B)

Two pure primitives (DataFrame in, dict out), independent of the harness row
producer:

  cross_sectional_ic  Per-date Spearman rank IC between a signal and a forward
                      return, then the mean IC, its dispersion, a t-stat, and the
                      hit rate across dates.  This is the *true* cross-sectional
                      IC — a rank correlation computed WITHIN each date's cross
                      section, not pooled across dates/horizons.

  incremental_ic      The marginal cross-sectional rank information of a new
                      signal BEYOND a set of incumbent signals: per date the new
                      signal is cross-sectionally residualised on the incumbents
                      (OLS with intercept) and the Spearman IC of the residual vs
                      the forward return is taken.  Answers "does this signal add
                      rank-ordering power the incumbents don't already have?".

Both reuse ``_spearman_rho`` from the harness (NaN-safe, len<3 / constant-guarded)
and pairwise-drop NaNs before calling it (``_spearman_rho`` returns None if ANY
NaN is present, so cleaning first is what makes these NaN-safe).

No I/O.  Never raises on thin or degenerate data — returns explicit None fields.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trade_modules.validation.harness import _spearman_rho

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _series_t_stat(ics: np.ndarray) -> float | None:
    """One-sample t-stat of an IC series vs 0: mean / (std/sqrt(n)).

    Returns None when n < 2 or the dispersion is non-finite / zero (a constant
    IC series has no sampling variance — t is undefined, never +/-inf).
    """
    n = len(ics)
    if n < 2:
        return None
    sd = float(np.std(ics, ddof=1))
    if not math.isfinite(sd) or sd <= 0.0:
        return None
    return float(np.mean(ics)) / (sd / math.sqrt(n))


def _clean_pair(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to numeric and pairwise-drop NaN across two aligned series."""
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    return x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)


def _residualize(y: np.ndarray, x_mat: np.ndarray) -> np.ndarray:
    """OLS residual of ``y`` on ``x_mat`` (with an intercept), via numpy lstsq."""
    n = x_mat.shape[0]
    design = np.column_stack([np.ones(n), x_mat])  # (n, k+1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _residual_is_negligible(resid: np.ndarray, reference: np.ndarray) -> bool:
    """True when the residual has no material variance relative to the signal.

    A perfectly-spanned signal (e.g. signal == incumbent) leaves a residual that
    is floating-point noise (~1e-15).  ``harness._is_degenerate_sigma`` uses a
    mean-relative threshold that does NOT catch a near-zero-mean residual, so we
    compare the residual scale to the ORIGINAL signal scale instead — a residual
    negligible vs the signal means zero incremental information (IC = 0), not an
    undefined one.
    """
    ref_scale = float(np.std(reference))
    resid_scale = float(np.std(resid))
    return resid_scale < 1e-8 * (ref_scale + 1e-12)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cross_sectional_ic(
    panel: pd.DataFrame,
    signal_col: str,
    forward_col: str,
    date_col: str = "signal_date",
) -> dict[str, Any]:
    """Per-date Spearman rank IC of ``signal_col`` vs ``forward_col``.

    Args:
        panel: long-format DataFrame (one row per name-date).
        signal_col: pre-trade signal / score column.
        forward_col: realised forward return (or alpha) column.
        date_col: cross-section key (default "signal_date").

    Returns dict:
        mean_ic     mean of the per-date ICs (None when no date qualifies)
        ic_std      sample std (ddof=1) of the per-date ICs (None when <2 dates)
        t_stat      mean_ic / (ic_std / sqrt(n_dates)) (None when undefined)
        hit_rate    fraction of qualifying dates with IC > 0 (None when 0 dates)
        n_dates     number of qualifying dates (>= 3 clean names each)
        ic_by_date  {date: ic} for every qualifying date

    NaN-safe; a date is skipped when fewer than 3 clean name-pairs remain, or the
    signal / forward is constant across the cross section (Spearman undefined).
    """
    ic_by_date: dict[Any, float] = {}

    if panel is not None and len(panel) > 0 and date_col in panel.columns:
        for date, group in panel.groupby(date_col, sort=True):
            x, y = _clean_pair(group[signal_col], group[forward_col])
            if len(x) < 3:
                continue
            rho = _spearman_rho(x.tolist(), y.tolist())
            if rho is not None:
                ic_by_date[date] = rho

    n_dates = len(ic_by_date)
    if n_dates == 0:
        return {
            "mean_ic": None,
            "ic_std": None,
            "t_stat": None,
            "hit_rate": None,
            "n_dates": 0,
            "ic_by_date": {},
        }

    ics = np.array(list(ic_by_date.values()), dtype=float)
    ic_std = float(np.std(ics, ddof=1)) if n_dates > 1 else None
    return {
        "mean_ic": float(np.mean(ics)),
        "ic_std": ic_std,
        "t_stat": _series_t_stat(ics),
        "hit_rate": float(np.sum(ics > 0)) / n_dates,
        "n_dates": n_dates,
        "ic_by_date": ic_by_date,
    }


def incremental_ic(
    panel: pd.DataFrame,
    signal_col: str,
    incumbent_cols: list[str],
    forward_col: str,
    date_col: str = "signal_date",
) -> dict[str, Any]:
    """Incumbent-partialled marginal cross-sectional rank IC.

    Per date the cross section of ``signal_col`` is residualised on
    ``incumbent_cols`` (OLS with intercept) and the Spearman IC of the residual
    vs ``forward_col`` is taken.  Raw IC (signal vs forward) is computed on the
    SAME cleaned subset so the two are directly comparable, and a date counts
    only when both are computable.

    Args:
        panel: long-format DataFrame (one row per name-date).
        signal_col: candidate signal column.
        incumbent_cols: incumbent signal columns to partial out.
        forward_col: realised forward return (or alpha) column.
        date_col: cross-section key (default "signal_date").

    Returns dict:
        raw_ic          mean per-date Spearman IC(signal, forward)
        incremental_ic  mean per-date Spearman IC(residual, forward);
                        exactly 0 for a date whose residual is negligible
                        (signal fully spanned by incumbents that date)
        ratio           incremental_ic / raw_ic (None when raw_ic == 0)
        t_stat          one-sample t-stat of the incremental IC series
        n_dates         qualifying dates

    A date needs >= max(3, k+2) clean names (k = #incumbents) for the OLS to
    leave residual degrees of freedom.  All-None when no date qualifies.
    """
    raw_list: list[float] = []
    incr_list: list[float] = []
    min_names = max(3, len(incumbent_cols) + 2)
    cols = [signal_col, *incumbent_cols, forward_col]

    if panel is not None and len(panel) > 0 and date_col in panel.columns:
        for _date, group in panel.groupby(date_col, sort=True):
            sub = group[cols].apply(lambda c: pd.to_numeric(c, errors="coerce")).dropna()
            if len(sub) < min_names:
                continue
            sig = sub[signal_col].to_numpy(dtype=float)
            fwd = sub[forward_col].to_numpy(dtype=float)
            x_mat = sub[list(incumbent_cols)].to_numpy(dtype=float)

            raw_rho = _spearman_rho(sig.tolist(), fwd.tolist())
            if raw_rho is None:
                continue  # constant signal or forward this date — cannot assess

            resid = _residualize(sig, x_mat)
            if _residual_is_negligible(resid, sig):
                incr_rho: float | None = 0.0  # fully spanned → zero marginal info
            else:
                incr_rho = _spearman_rho(resid.tolist(), fwd.tolist())
                if incr_rho is None:
                    incr_rho = 0.0

            raw_list.append(raw_rho)
            incr_list.append(incr_rho)

    n_dates = len(incr_list)
    if n_dates == 0:
        return {
            "raw_ic": None,
            "incremental_ic": None,
            "ratio": None,
            "t_stat": None,
            "n_dates": 0,
        }

    raw_ic = float(np.mean(raw_list))
    incr_arr = np.array(incr_list, dtype=float)
    incremental = float(np.mean(incr_arr))
    ratio = (incremental / raw_ic) if raw_ic != 0.0 else None
    return {
        "raw_ic": raw_ic,
        "incremental_ic": incremental,
        "ratio": ratio,
        "t_stat": _series_t_stat(incr_arr),
        "n_dates": n_dates,
    }
