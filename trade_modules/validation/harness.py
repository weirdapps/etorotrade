"""
validation/harness.py — S0 Validation Harness Orchestrator ("the referee")

Pure orchestrator that aggregates all validation primitives into a single
VerdictReport dict.  No I/O.  Never crashes on thin or missing data.

This is the most safety-critical code in the pipeline: a false PASS greenlights a
broken strategy with real capital.  The correctness invariants enforced here were
established by adversarial review:

  C1  A zero- / near-zero-variance alpha series must NEVER produce a large Sharpe
      or DSR 1.0.  Guarded with a RELATIVE variance test, not ``sigma == 0``.
  C4  Out-of-sample walk-forward uses an embargo derived from the ACTUAL max
      horizon in the data, and records ``oos={"computed": False, "reason": ...}``
      when 0 folds result — never a silent None.
  C5/I2/I3  DSR is computed from a PER-PERIOD (non-annualized) Sharpe with an
      EFFECTIVE observation count (distinct signal_dates, to deflate for
      cross-sectional duplication and overlapping horizons).  ``n_trials`` and
      ``var_sr`` are explicit, documented, and logged in the report.
  C2/C3/I5  PBO/CSCV only means something across candidate CONFIGURATIONS
      (rulesets), not tickers.  With a single ruleset (S0) per-family PBO is set
      to None with reason ``pbo_not_applicable_single_ruleset``.  A real PBO path
      is available via ``evaluate(..., config_perf=...)``.
  I1  The overall PASS decision requires EVERY material family to clear the gate
      individually (worst-material-family gate).  Pooled stats are kept for
      reporting only — a strong family must not mask a broken one.
  I4  The horizon-persistence field is named ``alpha_persistence`` (it correlates
      |alpha@primary| with alpha@h — NOT a true information coefficient).

Public API
----------
evaluate(results_rows, action_records=None, *, n_trials, var_sr, horizons,
         family_key, min_obs, config_perf=None) -> dict
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
from trade_modules.validation.turnover import compute_turnover

# ---------------------------------------------------------------------------
# Calibration constants (documented, explicit — see module docstring C5/I2/I3)
# ---------------------------------------------------------------------------

# Real tuned-parameter count for the signal engine.  The senior-PM review (see
# ``trade_modules/riskfirst/edgegate.py`` docstring) found ~220 parameters were
# tuned on ~60 observations in a single bull regime.  DSR must deflate for that
# many trials, so 220 is the honest default — not the old 100.
DEFAULT_N_TRIALS = 220

# Conservative floor for var_sr (variance of the per-period Sharpe estimates
# across trials/variants) when it cannot be estimated from the data.  A LOWER
# var_sr makes DSR EASIER to pass, so the default is deliberately generous-but-
# bounded; when >1 family Sharpe is available we estimate var_sr from their
# dispersion and floor it here so a lucky 2-family near-tie cannot drive var_sr
# to ~0 and inflate DSR.
DEFAULT_VAR_SR = 0.5
VAR_SR_FLOOR = 0.01

# Relative variance guard (C1).  A Sharpe is only computed when the standard
# deviation is materially non-zero relative to the mean.  ``sigma == 0.0`` is
# insufficient because catastrophic cancellation in float std of a constant
# series yields a tiny non-zero (~1e-17), which then blows Sharpe up to ~1e15.
_SIGMA_REL_EPS = 1e-8

_TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_degenerate_sigma(mu: float, sigma: float) -> bool:
    """C1 guard: True when sigma is non-finite or negligibly small relative to mu.

    Treat such a series as having NO edge (Sharpe/DSR undefined), never a blow-up.
    """
    if not math.isfinite(sigma):
        return True
    return sigma < _SIGMA_REL_EPS * (abs(mu) + 1e-12)


def _spearman_rho(x: list[float], y: list[float]) -> float | None:
    """NaN-safe Spearman rank correlation.  Returns None when len < 3 or constant."""
    if len(x) < 3 or len(y) < 3:
        return None
    # spearmanr is undefined for a (near-)constant input; guard to avoid a
    # warning + NaN.  Use a relative check so float cancellation on a constant
    # series (std ~1e-17, not exactly 0) is still treated as constant.
    ax, ay = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if _is_degenerate_sigma(float(np.mean(ax)), float(np.std(ax))) or _is_degenerate_sigma(
        float(np.mean(ay)), float(np.std(ay))
    ):
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


def _distinct_signal_dates(rows: list[dict]) -> int:
    """Effective independent observation count = number of distinct signal_dates.

    Deflates for cross-sectional duplication (many tickers share a date) and for
    overlapping horizons (the same date appears at every horizon).  Used as
    ``n_obs`` for DSR/PSR so significance is not overstated by the raw row count.
    """
    dates = {str(r.get("signal_date")) for r in rows if r.get("signal_date")}
    return len(dates)


def _horizons_present(rows: list[dict]) -> list[int]:
    """Distinct integer horizons actually present in the rows."""
    hs: set[int] = set()
    for r in rows:
        h = _safe_float(r.get("horizon"))
        if h is not None:
            hs.add(int(h))
    return sorted(hs)


# ---------------------------------------------------------------------------
# Per-family analysis (pass 1: stats without DSR; DSR filled in pass 2)
# ---------------------------------------------------------------------------


def _analyse_family(
    rows: list[dict],
    family_name: str,
    *,
    horizons: tuple[int, ...],
    min_obs: int,
    primary_horizon: int = 30,
) -> dict:
    """Compute all per-family stats EXCEPT the DSR (which needs the cross-family
    var_sr estimate).  Never raises.

    Returns a dict that always contains ``per_period_sharpe`` (float | None) and
    ``n_eff`` so the second pass can compute DSR consistently.
    """

    # Filter to primary horizon (nearest if exact not present)
    h30_rows = [r for r in rows if _safe_float(r.get("horizon")) == primary_horizon]
    if not h30_rows:
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

    alphas = [a for a in (_alpha_value(r) for r in h30_rows) if a is not None]
    gross_alphas = [a for a in (_safe_float(r.get("alpha")) for r in h30_rows) if a is not None]

    if len(alphas) < min_obs:
        return {"insufficient_data": True, "n": n}

    arr = np.array(alphas, dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    # Effective independent observation count for DSR/PSR.
    n_eff = _distinct_signal_dates(h30_rows)

    # C1 relative-variance guard: degenerate sigma => no edge.
    if _is_degenerate_sigma(mu, sigma):
        per_period_sharpe: float | None = None
        sharpe_annual: float | None = None
        skew = None
        kurt_normal = None
    else:
        per_period_sharpe = mu / sigma
        # Annualized Sharpe is for HUMANS only (report), not for the DSR call.
        sharpe_annual = per_period_sharpe * math.sqrt(_TRADING_DAYS / primary_horizon)
        skew = float(scipy_skew(arr))
        kurt_normal = float(scipy_kurtosis(arr, fisher=True)) + 3.0  # PSR uses non-excess

    # PBO: NOT applicable per-ticker with a single ruleset (C2/C3).  Recorded as
    # None with an explicit reason so it is never silently dropped from the gate.
    pbo: float | None = None
    pbo_reason = "pbo_not_applicable_single_ruleset"

    # OOS walk-forward (C4): embargo = actual max horizon present in THIS family.
    hs_present = _horizons_present(rows)
    embargo_days = max(hs_present) if hs_present else (max(horizons) if horizons else 250)
    oos = _oos_walk_forward(h30_rows, embargo_days=embargo_days)

    # alpha_persistence (I4): |alpha@primary| vs alpha@h — NOT a true IC.
    persistence = _alpha_persistence(rows, h30_rows, horizons)

    return {
        "n": n,
        "n_eff": n_eff,
        "per_period_sharpe": per_period_sharpe,
        "sharpe_annual": sharpe_annual,
        "sharpe": sharpe_annual,  # back-compat alias (annualized, human-facing)
        "skew": skew,
        "kurt": kurt_normal,
        "mu_alpha": mu,
        "sigma_alpha": sigma,
        "pbo": pbo,
        "pbo_reason": pbo_reason,
        "oos": oos,
        # Back-compat mirrors (populated only when OOS actually computed).
        "oos_hit": oos.get("oos_hit") if oos.get("computed") else None,
        "oos_alpha": oos.get("oos_alpha") if oos.get("computed") else None,
        "alpha_gross": float(np.mean(gross_alphas)) if gross_alphas else None,
        "alpha_persistence": persistence,
        "insufficient_data": False,
        # dsr / passed / reasons are filled in pass 2 (need var_sr + gate).
    }


def _oos_walk_forward(h30_rows: list[dict], *, embargo_days: int) -> dict:
    """Rolling walk-forward OOS.  Returns an explicit structured result.

    Keys: computed (bool), reason (str, when not computed), n_folds (int),
    oos_hit (float|None), oos_alpha (float|None), embargo_days (int).
    """
    try:
        folds = rolling_walk_forward(h30_rows, n_folds=5, embargo_days=embargo_days)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "computed": False,
            "reason": f"walk_forward_error:{exc}",
            "n_folds": 0,
            "oos_hit": None,
            "oos_alpha": None,
            "embargo_days": embargo_days,
        }

    if not folds:
        return {
            "computed": False,
            "reason": "embargo_exceeds_span",
            "n_folds": 0,
            "oos_hit": None,
            "oos_alpha": None,
            "embargo_days": embargo_days,
        }

    oos_rows_all: list[dict] = []
    for _, oos_part in folds:
        oos_rows_all.extend(oos_part)
    oos_alphas = [a for a in (_alpha_value(r) for r in oos_rows_all) if a is not None]

    if not oos_alphas:
        return {
            "computed": False,
            "reason": "no_oos_alpha_values",
            "n_folds": len(folds),
            "oos_hit": None,
            "oos_alpha": None,
            "embargo_days": embargo_days,
        }

    return {
        "computed": True,
        "reason": "",
        "n_folds": len(folds),
        "oos_hit": float(sum(1 for a in oos_alphas if a > 0)) / len(oos_alphas),
        "oos_alpha": float(np.mean(oos_alphas)),
        "embargo_days": embargo_days,
    }


def _alpha_persistence(rows: list[dict], h30_rows: list[dict], horizons: tuple[int, ...]) -> dict:
    """Horizon-persistence curve (I4).

    Correlates |alpha@primary| (a realized-outcome magnitude, NOT a pre-trade
    signal score) against alpha@h for each horizon.  This is deliberately NOT
    labelled an information coefficient — see ``note``.  If a per-row pre-trade
    signal-score column existed, a real IC could be computed instead; none does.
    """
    ticker_alpha_primary: dict[str, float] = {}
    for r in h30_rows:
        t = r.get("ticker")
        a = _alpha_value(r)
        if t and a is not None:
            ticker_alpha_primary[t] = a

    by_horizon: dict[int, float] = {}
    for h in horizons:
        h_rows = [r for r in rows if _safe_float(r.get("horizon")) == h]
        if len(h_rows) < 5:
            continue
        aligned_x: list[float] = []
        aligned_y: list[float] = []
        for r in h_rows:
            t = r.get("ticker")
            a = _alpha_value(r)
            if t and t in ticker_alpha_primary and a is not None:
                aligned_x.append(abs(ticker_alpha_primary[t]))
                aligned_y.append(a)
        rho = _spearman_rho(aligned_x, aligned_y)
        if rho is not None:
            by_horizon[h] = rho

    decay = (
        compute_ic_decay(by_horizon)
        if by_horizon
        else {"half_life_days": None, "ic0": None, "curve": {}, "note": "no horizon pairs"}
    )
    return {
        "by_horizon": by_horizon,
        "decay": decay,
        "note": (
            "alpha_persistence: Spearman(|alpha@primary|, alpha@h) — a realized-"
            "outcome persistence measure, NOT an information coefficient."
        ),
    }


def _estimate_var_sr(per_period_sharpes: list[float]) -> tuple[float, str]:
    """Estimate var_sr from the cross-family per-period Sharpe dispersion.

    When >1 family Sharpe is available, use their sample variance, floored at
    ``VAR_SR_FLOOR`` (so a near-tie cannot drive var_sr → 0 and inflate DSR).
    Otherwise fall back to ``DEFAULT_VAR_SR``.  Returns (var_sr, source_note).
    """
    finite = [s for s in per_period_sharpes if s is not None and math.isfinite(s)]
    if len(finite) > 1:
        v = float(np.var(np.array(finite, dtype=float), ddof=1))
        var_sr = max(v, VAR_SR_FLOOR)
        return var_sr, (
            f"estimated from {len(finite)} family Sharpes "
            f"(raw var={v:.4f}, floored at {VAR_SR_FLOOR})"
        )
    return DEFAULT_VAR_SR, f"default (only {len(finite)} family Sharpe available)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    results_rows: list[dict],
    action_records: list[dict] | None = None,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    var_sr: float | None = None,
    horizons: tuple[int, ...] = (7, 30, 60, 90, 180, 250),
    family_key: str = "signal",
    min_obs: int = 30,
    config_perf: np.ndarray | None = None,
) -> dict:
    """Pure orchestrator — the validation referee.

    Args:
        results_rows: list of dicts from backtest_results.csv.
        action_records: optional list of dicts for turnover (see note below).
        n_trials: number of tuned trials for DSR deflation.  Defaults to
            DEFAULT_N_TRIALS (~220, per the charter / edgegate docstring).
        var_sr: prior variance of the per-period Sharpe under the null.  When
            None (default), it is ESTIMATED from the cross-family per-period
            Sharpe dispersion (floored), else falls back to DEFAULT_VAR_SR.  Pass
            a float to override.
        horizons: candidate horizons for the persistence curve.
        family_key: row key that identifies the signal family.
        min_obs: minimum primary-horizon obs for a family to be "material".
        config_perf: OPTIONAL T×N matrix of per-period performance across genuine
            candidate CONFIGURATIONS (rulesets/parameter sets).  When supplied,
            a real PBO is computed on it (C2/C3).  When None (S0, single ruleset)
            PBO is reported as None with an explicit reason.

    action_records schema note: the live action_log.jsonl uses ``committee_date``
    (not ``date``) and ``size`` (not ``weight_change``); ``size`` is null in all
    current records, so turnover is reported as a note dict (not bare None) when
    no usable ``weight_change`` is present.

    Returns a VerdictReport dict.  Never crashes.
    """
    primary_horizon = 30

    # -----------------------------------------------------------------------
    # Survivorship bias accounting (over ALL rows)
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
    # Pass 1: per-family stats (no DSR yet)
    # -----------------------------------------------------------------------
    families: dict[str, dict] = {}
    for fam, rows in family_rows.items():
        families[fam] = _analyse_family(
            rows,
            fam,
            horizons=horizons,
            min_obs=min_obs,
            primary_horizon=primary_horizon,
        )

    # -----------------------------------------------------------------------
    # var_sr calibration (C5/I2/I3): estimate from cross-family Sharpe variance
    # -----------------------------------------------------------------------
    material_families = {f: s for f, s in families.items() if not s.get("insufficient_data")}
    per_period_sharpes = [s.get("per_period_sharpe") for s in material_families.values()]
    if var_sr is None:
        var_sr_used, var_sr_source = _estimate_var_sr(per_period_sharpes)
    else:
        var_sr_used, var_sr_source = float(var_sr), "caller-supplied"

    # Count regimes (shared across families for the gate).
    regime_counts: set[str] = {str(r.get("regime")) for r in results_rows if r.get("regime")}
    n_regimes = max(1, len(regime_counts))

    # -----------------------------------------------------------------------
    # Optional REAL PBO across candidate configurations (C2/C3)
    # -----------------------------------------------------------------------
    config_pbo: float | None = None
    config_pbo_reason = "config_perf_not_supplied_single_ruleset"
    if config_perf is not None:
        pbo_res = pbo_cscv(config_perf)
        config_pbo = pbo_res.get("pbo")
        if config_pbo is None:
            config_pbo_reason = pbo_res.get("reason", "pbo_none")
        else:
            config_pbo_reason = ""

    # -----------------------------------------------------------------------
    # Pass 2: DSR + per-family gate verdict
    # -----------------------------------------------------------------------
    for fam, s in families.items():
        if s.get("insufficient_data"):
            continue
        pp_sharpe = s.get("per_period_sharpe")
        n_eff = s.get("n_eff", 0)
        if pp_sharpe is None or n_eff < 2:
            # Degenerate (C1) or too few effective obs: no edge, DSR undefined.
            s["dsr"] = None
            reasons = []
            if pp_sharpe is None:
                reasons.append("degenerate variance (no edge; Sharpe undefined)")
            if n_eff < 2:
                reasons.append(f"insufficient effective obs (distinct dates={n_eff})")
            if n_regimes < 2:
                reasons.append("single-regime sample (need >= 2; no bear/stress)")
            s["passed"] = False
            s["reasons"] = reasons
            continue

        # Per-family PBO for the gate: prefer the real config PBO if supplied,
        # else None (per-ticker PBO is meaningless — C2/C3, do not gate on it).
        gate_pbo = config_pbo if config_perf is not None else None

        verdict = gate_verdict(
            sr=pp_sharpe,
            n_obs=n_eff,
            n_trials=n_trials,
            var_sr=var_sr_used,
            skew=s.get("skew") or 0.0,
            kurt=s.get("kurt") or 3.0,
            n_regimes=n_regimes,
            pbo=gate_pbo,
            min_obs=min_obs,
        )
        s["dsr"] = verdict["dsr"]
        s["passed"] = verdict["passed"]
        s["reasons"] = verdict["reasons"]
        # PBO n/a (single ruleset) is NOT gated on and NOT a failure reason —
        # record the note for transparency only.
        if config_perf is None:
            s["pbo_note"] = "pbo_not_applicable_single_ruleset (not gated)"

    # -----------------------------------------------------------------------
    # Overall verdict (I1): PASS iff EVERY material family passes its own gate.
    # Pooled stats are computed for REPORTING ONLY — never for the decision.
    # -----------------------------------------------------------------------
    overall = _overall_verdict(
        families=families,
        family_rows=family_rows,
        material_families=material_families,
        primary_horizon=primary_horizon,
        n_trials=n_trials,
        var_sr_used=var_sr_used,
        n_regimes=n_regimes,
        min_obs=min_obs,
        config_pbo=config_pbo,
        config_pbo_reason=config_pbo_reason,
        config_perf_supplied=config_perf is not None,
    )

    # -----------------------------------------------------------------------
    # Regime-stratified stats
    # -----------------------------------------------------------------------
    regime_out = _regime_stratified(results_rows)

    # -----------------------------------------------------------------------
    # Turnover
    # -----------------------------------------------------------------------
    turnover_result = _turnover(action_records)

    # -----------------------------------------------------------------------
    # Assemble report
    # -----------------------------------------------------------------------
    return {
        "overall": overall,
        "families": families,
        "regime_stratified": regime_out,
        "turnover": turnover_result,
        "survivorship": survivorship,
        "n_trials": n_trials,
        "var_sr": var_sr_used,
        "dsr_assumptions": {
            "n_trials": n_trials,
            "var_sr": var_sr_used,
            "var_sr_source": var_sr_source,
            "n_obs_definition": "distinct signal_dates per family (effective N)",
            "sharpe_units": "per-period (non-annualized) for DSR; annualized reported separately",
        },
    }


def _overall_verdict(
    *,
    families: dict[str, dict],
    family_rows: dict[str, list[dict]],
    material_families: dict[str, dict],
    primary_horizon: int,
    n_trials: int,
    var_sr_used: float,
    n_regimes: int,
    min_obs: int,
    config_pbo: float | None,
    config_pbo_reason: str,
    config_perf_supplied: bool,
) -> dict:
    """Aggregate verdict.  Overall PASS iff all material families pass (I1)."""
    if not material_families:
        return {
            "passed": False,
            "dsr": None,
            "pbo": config_pbo,
            "pbo_reason": config_pbo_reason,
            "reasons": ["insufficient data: no material families to evaluate"],
            "material_families": [],
            "failing_families": [],
        }

    failing: list[str] = []
    reasons: list[str] = []
    for fam, s in material_families.items():
        if not s.get("passed", False):
            failing.append(fam)
            fam_reasons = s.get("reasons") or ["did not pass gate"]
            reasons.append(f"family {fam}: " + "; ".join(fam_reasons))

    passed = len(failing) == 0

    # Pooled DSR — REPORTING ONLY (does not drive the decision).
    pooled_alphas: list[float] = []
    for fam in material_families:
        h30_rows = [r for r in family_rows[fam] if _safe_float(r.get("horizon")) == primary_horizon]
        for r in h30_rows:
            a = _alpha_value(r)
            if a is not None:
                pooled_alphas.append(a)

    pooled_dsr: float | None = None
    pooled_n_eff = _distinct_signal_dates(
        [r for fam in material_families for r in family_rows[fam]]
    )
    if len(pooled_alphas) >= 2:
        arr = np.array(pooled_alphas, dtype=float)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        if not _is_degenerate_sigma(mu, sigma) and pooled_n_eff >= 2:
            pp_sr = mu / sigma
            skew = float(scipy_skew(arr))
            kurt = float(scipy_kurtosis(arr, fisher=True)) + 3.0
            pooled_dsr = deflated_sharpe_ratio(
                pp_sr, pooled_n_eff, n_trials, var_sr_used, skew, kurt
            )

    if not passed and not reasons:  # pragma: no cover - defensive
        reasons = ["one or more material families failed the gate"]

    return {
        "passed": passed,
        "dsr": pooled_dsr,  # pooled, reporting-only
        "pbo": config_pbo,
        "pbo_reason": config_pbo_reason if not config_perf_supplied else (config_pbo_reason or ""),
        "reasons": reasons,
        "material_families": sorted(material_families.keys()),
        "failing_families": sorted(failing),
        "decision_basis": "worst-material-family gate (no pooling of the pass decision)",
    }


def _regime_stratified(results_rows: list[dict]) -> dict[str, dict]:
    """Per-regime hit rate and average alpha."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in results_rows:
        regime = r.get("regime")
        if not regime:
            continue
        a = _alpha_value(r)
        if a is None:
            continue
        buckets[str(regime)].append(a)

    out: dict[str, dict] = {}
    for regime, alphas_r in buckets.items():
        out[regime] = {
            "n": len(alphas_r),
            "hit": float(sum(1 for a in alphas_r if a > 0)) / len(alphas_r),
            "avg_alpha": float(np.mean(alphas_r)),
        }
    return out


def _turnover(action_records: list[dict] | None) -> dict | None:
    """Turnover with honest reporting (see action_records schema note in evaluate)."""
    if action_records is None:
        return None

    normalised: list[dict] = []
    for r in action_records:
        rec = dict(r)
        if "date" not in rec or rec["date"] is None:
            rec["date"] = rec.get("committee_date")
        if rec.get("weight_change") is None:
            size_val = rec.get("size")
            rec["weight_change"] = float(size_val) if size_val is not None else None
        normalised.append(rec)

    has_weight = any(r.get("weight_change") is not None for r in normalised)
    if not has_weight:
        return {
            "note": (
                "action_log found but lacks weight_change; "
                "size is null in all records — turnover not computed"
            ),
            "n_records": len(normalised),
        }

    try:
        dates = [r.get("date", "") for r in normalised if r.get("date")]
        if dates:
            from datetime import date as date_cls

            dates_sorted = sorted(dates)
            d0 = date_cls.fromisoformat(str(dates_sorted[0])[:10])
            d1 = date_cls.fromisoformat(str(dates_sorted[-1])[:10])
            window_days = max(1, (d1 - d0).days)
        else:
            window_days = 365
        computable = [r for r in normalised if r.get("weight_change") is not None]
        return compute_turnover(computable, window_days)
    except KeyError as exc:
        return {
            "note": f"turnover computation failed — missing key: {exc}",
            "n_records": len(normalised),
        }
    except Exception as exc:
        return {
            "note": f"turnover computation failed: {exc}",
            "n_records": len(normalised),
        }
