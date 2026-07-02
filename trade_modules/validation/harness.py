"""
validation/harness.py — S0 Validation Harness Orchestrator

Pure orchestrator that aggregates all validation primitives into a single
VerdictReport dict.  No I/O.  Never crashes on thin or missing data.

Public API
----------
evaluate(results_rows, action_records=None, *, n_trials, var_sr, horizons,
         family_key, min_obs) -> dict
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew
from scipy.stats import spearmanr

from trade_modules.backtest_stats import rolling_walk_forward
from trade_modules.riskfirst.edgegate import (
    deflated_sharpe_ratio,
    gate_verdict,
    pbo_cscv,
)
from trade_modules.validation.ic_decay import compute_ic_decay
from trade_modules.validation.perf_matrix import build_perf_matrix
from trade_modules.validation.turnover import compute_turnover

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _spearman_rho(x: list[float], y: list[float]) -> float | None:
    """NaN-safe Spearman rank correlation.  Returns None when len < 3."""
    if len(x) < 3 or len(y) < 3:
        return None
    try:
        result = spearmanr(x, y, nan_policy="omit")
        rho = float(result.statistic)
        if math.isnan(rho):
            return None
        return rho
    except Exception:
        return None


def _safe_float(v: Any) -> float | None:
    """Convert to float; return None on failure."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _alpha_value(row: dict) -> float | None:
    """Prefer net_alpha over alpha.  Return None if neither is available."""
    na = _safe_float(row.get("net_alpha"))
    if na is not None:
        return na
    return _safe_float(row.get("alpha"))


def _has_future_price(row: dict) -> bool:
    """Return True if future_price is a valid finite float."""
    fp = row.get("future_price")
    if fp is None:
        return False
    try:
        f = float(fp)
        return not (math.isnan(f) or math.isinf(f))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Per-family analysis
# ---------------------------------------------------------------------------


def _analyse_family(
    rows: list[dict],
    family_name: str,
    *,
    n_trials: int,
    var_sr: float,
    horizons: tuple[int, ...],
    min_obs: int,
    primary_horizon: int = 30,
) -> dict:
    """Compute all stats for one signal family.  Never raises."""

    # Filter to primary horizon (nearest if exact not present)
    h30_rows = [r for r in rows if _safe_float(r.get("horizon")) == primary_horizon]
    if not h30_rows:
        # Fall back to closest available horizon
        available = sorted(
            {_safe_float(r.get("horizon")) for r in rows if r.get("horizon") is not None}
        )
        if not available:
            return {"insufficient_data": True, "n": len(rows)}
        closest = min(available, key=lambda h: abs(h - primary_horizon))
        h30_rows = [r for r in rows if _safe_float(r.get("horizon")) == closest]

    n = len(h30_rows)

    if n < min_obs:
        return {"insufficient_data": True, "n": n}

    # Collect alpha values (prefer net_alpha)
    alphas = [_alpha_value(r) for r in h30_rows]
    alphas = [a for a in alphas if a is not None]

    # Gross alpha (raw alpha, not net) for independent comparison
    gross_alphas = [_safe_float(r.get("alpha")) for r in h30_rows]
    gross_alphas = [a for a in gross_alphas if a is not None]

    if len(alphas) < min_obs:
        return {"insufficient_data": True, "n": n}

    arr = np.array(alphas, dtype=float)

    # Annualised Sharpe (30-day horizon)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if sigma == 0.0 or math.isnan(sigma):
        sharpe = None
        dsr = None
    else:
        sharpe = mu / sigma * math.sqrt(252.0 / primary_horizon)
        skew = float(scipy_skew(arr))
        # kurtosis: Fisher=False gives non-excess (normal=3); we want excess so Fisher=True
        kurt_excess = float(scipy_kurtosis(arr, fisher=True))
        kurt_normal = kurt_excess + 3.0  # PSR formula uses non-excess kurt
        dsr = deflated_sharpe_ratio(sharpe, len(arr), n_trials, var_sr, skew, kurt_normal)

    # PBO via CSCV performance matrix
    pbo: float | None = None
    try:
        # Use net_alpha if present; filter rows where the chosen value is None
        use_net = any("net_alpha" in r and r["net_alpha"] is not None for r in h30_rows)
        pbo_value_key = "net_alpha" if use_net else "alpha"
        pbo_rows = [r for r in h30_rows if r.get(pbo_value_key) is not None]
        if pbo_rows:
            _, col_labels, matrix = build_perf_matrix(
                pbo_rows,
                period_key="signal_date",
                config_key="ticker",
                value_key=pbo_value_key,
            )
            T, N = matrix.shape
            if T >= 10 and N >= 2:
                pbo = pbo_cscv(matrix)["pbo"]
    except Exception:
        pbo = None

    # OOS hit rate and alpha via rolling walk-forward
    oos_hit: float | None = None
    oos_alpha: float | None = None
    try:
        max_embargo = max(horizons) if horizons else 250
        folds = rolling_walk_forward(h30_rows, n_folds=5, embargo_days=max_embargo)
        oos_rows_all: list[dict] = []
        for _, oos in folds:
            oos_rows_all.extend(oos)
        if oos_rows_all:
            oos_alphas = [_alpha_value(r) for r in oos_rows_all]
            oos_alphas = [a for a in oos_alphas if a is not None]
            if oos_alphas:
                oos_hit = float(sum(1 for a in oos_alphas if a > 0)) / len(oos_alphas)
                oos_alpha = float(np.mean(oos_alphas))
    except Exception:
        oos_hit = None
        oos_alpha = None

    # IC by horizon (Spearman of abs-alpha-at-h30 vs alpha-at-each-horizon)
    # We need tickers at h30 to get their abs_alpha as the "signal strength" proxy
    ticker_alpha_h30: dict[str, float] = {}
    for r in h30_rows:
        t = r.get("ticker")
        a = _alpha_value(r)
        if t and a is not None:
            ticker_alpha_h30[t] = a

    ic_by_horizon: dict[int, float] = {}
    for h in horizons:
        h_rows = [r for r in rows if _safe_float(r.get("horizon")) == h]
        if len(h_rows) < 5:
            continue
        # Align tickers: only those present at h30
        aligned_x: list[float] = []
        aligned_y: list[float] = []
        for r in h_rows:
            t = r.get("ticker")
            a = _alpha_value(r)
            if t and t in ticker_alpha_h30 and a is not None:
                aligned_x.append(abs(ticker_alpha_h30[t]))
                aligned_y.append(a)
        rho = _spearman_rho(aligned_x, aligned_y)
        if rho is not None:
            ic_by_horizon[h] = rho

    ic_decay_result = (
        compute_ic_decay(ic_by_horizon)
        if ic_by_horizon
        else {"half_life_days": None, "ic0": None, "curve": {}, "note": "no horizon pairs"}
    )

    return {
        "n": n,
        "sharpe": sharpe,
        "dsr": dsr,
        "pbo": pbo,
        "oos_hit": oos_hit,
        "oos_alpha": oos_alpha,
        "alpha_gross": float(np.mean(gross_alphas)) if gross_alphas else None,
        "ic_decay": ic_decay_result,
        "insufficient_data": False,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    results_rows: list[dict],
    action_records: list[dict] | None = None,
    *,
    n_trials: int = 100,
    var_sr: float = 0.5,
    horizons: tuple[int, ...] = (7, 30, 60, 90, 180, 250),
    family_key: str = "signal",
    min_obs: int = 30,
) -> dict:
    """Pure orchestrator.

    results_rows: list of dicts from backtest_results.csv.
    action_records: list of dicts for turnover calculation (optional).
        Schema note: the real action_log.jsonl uses ``committee_date`` (not
        ``date``) for the date field and ``size`` (not ``weight_change``) for
        the position-size delta.  ``size`` is ``null`` in all current records,
        so meaningful turnover cannot be computed from the live log.  To obtain
        a real turnover figure, supply records that contain a numeric
        ``weight_change`` key (fraction of portfolio, e.g. 0.05 = 5 %).
        When only the action_log schema is available and ``weight_change`` is
        absent or all-null, turnover will be a dict with a ``note`` key
        explaining why the computation was skipped — NOT bare None.

    Returns VerdictReport dict.  Never crashes.
    """
    primary_horizon = 30

    # -----------------------------------------------------------------------
    # Survivorship bias accounting (done over ALL rows)
    # -----------------------------------------------------------------------
    total_rows = len(results_rows)
    no_forward_price = sum(1 for r in results_rows if not _has_future_price(r))
    pct_dropped = (no_forward_price / total_rows * 100.0) if total_rows > 0 else 0.0
    survivorship = {
        "total_rows": total_rows,
        "no_forward_price": no_forward_price,
        "pct_dropped": pct_dropped,
    }

    # -----------------------------------------------------------------------
    # Group rows by family
    # -----------------------------------------------------------------------
    family_rows: dict[str, list[dict]] = defaultdict(list)
    for r in results_rows:
        fam = str(r.get(family_key, "UNKNOWN"))
        family_rows[fam].append(r)

    # -----------------------------------------------------------------------
    # Per-family analysis
    # -----------------------------------------------------------------------
    families: dict[str, dict] = {}
    for fam, rows in family_rows.items():
        families[fam] = _analyse_family(
            rows,
            fam,
            n_trials=n_trials,
            var_sr=var_sr,
            horizons=horizons,
            min_obs=min_obs,
            primary_horizon=primary_horizon,
        )

    # -----------------------------------------------------------------------
    # Overall aggregate (across non-insufficient families at primary horizon)
    # -----------------------------------------------------------------------
    all_alphas: list[float] = []
    pbo_values: list[float] = []

    for fam, stats in families.items():
        if stats.get("insufficient_data"):
            continue
        # Collect h30 alphas for this family
        h30_rows = [r for r in family_rows[fam] if _safe_float(r.get("horizon")) == primary_horizon]
        for r in h30_rows:
            a = _alpha_value(r)
            if a is not None:
                all_alphas.append(a)
        if stats.get("pbo") is not None:
            pbo_values.append(stats["pbo"])

    # Count regimes
    regime_counts: set[str] = set()
    for r in results_rows:
        regime = r.get("regime")
        if regime:
            regime_counts.add(str(regime))
    n_regimes = max(1, len(regime_counts))

    aggregate_pbo: float | None = float(np.mean(pbo_values)) if pbo_values else None

    if len(all_alphas) >= 2:
        agg_arr = np.array(all_alphas, dtype=float)
        agg_mu = float(np.mean(agg_arr))
        agg_sigma = float(np.std(agg_arr, ddof=1))
        if agg_sigma == 0.0 or math.isnan(agg_sigma):
            overall_verdict = {
                "passed": False,
                "dsr": None,
                "pbo": aggregate_pbo,
                "reasons": ["zero variance in aggregate alpha"],
            }
        else:
            agg_sr = agg_mu / agg_sigma * math.sqrt(252.0 / primary_horizon)
            agg_skew = float(scipy_skew(agg_arr))
            agg_kurt_excess = float(scipy_kurtosis(agg_arr, fisher=True))
            agg_kurt = agg_kurt_excess + 3.0
            overall_verdict = gate_verdict(
                sr=agg_sr,
                n_obs=len(agg_arr),
                n_trials=n_trials,
                var_sr=var_sr,
                skew=agg_skew,
                kurt=agg_kurt,
                n_regimes=n_regimes,
                pbo=aggregate_pbo,
            )
    else:
        overall_verdict = {
            "passed": False,
            "dsr": None,
            "pbo": aggregate_pbo,
            "reasons": ["insufficient aggregate data for evaluation"],
        }

    # -----------------------------------------------------------------------
    # Regime-stratified stats
    # -----------------------------------------------------------------------
    regime_stratified: dict[str, dict] = {}
    for r in results_rows:
        regime = r.get("regime")
        if not regime:
            continue
        regime = str(regime)
        a = _alpha_value(r)
        if a is None:
            continue
        if regime not in regime_stratified:
            regime_stratified[regime] = {"_alphas": []}
        regime_stratified[regime]["_alphas"].append(a)

    # Convert raw alphas to summary stats
    regime_out: dict[str, dict] = {}
    for regime, data in regime_stratified.items():
        alphas_r = data["_alphas"]
        regime_out[regime] = {
            "n": len(alphas_r),
            "hit": float(sum(1 for a in alphas_r if a > 0)) / len(alphas_r),
            "avg_alpha": float(np.mean(alphas_r)),
        }

    # -----------------------------------------------------------------------
    # Turnover
    # -----------------------------------------------------------------------
    turnover_result: dict | None = None
    if action_records is not None:
        # Normalise real action_log schema → harness schema.
        # Real log uses "committee_date" (not "date") and "size" (not
        # "weight_change").  If "weight_change" is absent but "size" is
        # present, map it — treating null size as 0.0.
        normalised: list[dict] = []
        for r in action_records:
            rec = dict(r)
            # Date field: prefer "date", fall back to "committee_date"
            if "date" not in rec or rec["date"] is None:
                rec["date"] = rec.get("committee_date")
            # Weight field: use "weight_change" if present and non-null,
            # otherwise fall back to "size" (null → 0.0)
            if rec.get("weight_change") is None:
                size_val = rec.get("size")
                rec["weight_change"] = float(size_val) if size_val is not None else None
            normalised.append(rec)

        # Check whether weight_change data is available for computation.
        has_weight = any(r.get("weight_change") is not None for r in normalised)

        if not has_weight:
            # Honest reporting: log was found but lacks the data needed.
            turnover_result = {
                "note": (
                    "action_log found but lacks weight_change; "
                    "size is null in all records — turnover not computed"
                ),
                "n_records": len(normalised),
            }
        else:
            try:
                # Estimate window from date range of normalised records
                dates = [r.get("date", "") for r in normalised if r.get("date")]
                if dates:
                    dates_sorted = sorted(dates)
                    from datetime import date as date_cls

                    d0 = date_cls.fromisoformat(str(dates_sorted[0])[:10])
                    d1 = date_cls.fromisoformat(str(dates_sorted[-1])[:10])
                    window_days = max(1, (d1 - d0).days)
                else:
                    window_days = 365
                # Filter to records with a usable weight_change value
                computable = [r for r in normalised if r.get("weight_change") is not None]
                turnover_result = compute_turnover(computable, window_days)
            except KeyError as exc:
                turnover_result = {
                    "note": f"turnover computation failed — missing key: {exc}",
                    "n_records": len(normalised),
                }
            except Exception as exc:
                turnover_result = {
                    "note": f"turnover computation failed: {exc}",
                    "n_records": len(normalised),
                }

    # -----------------------------------------------------------------------
    # Assemble report
    # -----------------------------------------------------------------------
    return {
        "overall": overall_verdict,
        "families": families,
        "regime_stratified": regime_out,
        "turnover": turnover_result,
        "survivorship": survivorship,
        "n_trials": n_trials,
    }
