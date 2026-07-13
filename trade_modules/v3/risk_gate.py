"""Phase 5A HARD risk gate (blocking enforcement) for the v3 book.

:func:`build_portfolio` produces a conviction-tilted, capped book scaled to a
``gross_target`` fraction of capital, with a REPORT-ONLY risk assessment. This
module turns that assessment into ENFORCEMENT, using the owner's HYBRID rule —
keep deployment high (~85-95%), raise the vol ceiling to ~18%, de-weight the
worst tail-risk names FIRST, and only cut gross below the deployment target if
the book is still over budget after de-weighting:

  1. **Caps to convergence.** Iterate {name-cap -> USD-bloc-cap -> sector-cap}
     until no cap is left breached (each cap redistributes its excess to under-cap
     names, so the invested sum is preserved whenever a receiver exists). A single
     interleaved pass can leave a residual breach (e.g. the sector cap re-breaches
     the name cap); iterating to a fixed point removes it.

  2. **Vol ceiling via tail de-weighting (lever 1).** While the portfolio vol
     exceeds the ceiling, shrink the highest risk-contribution name
     (``RC_i = w_i * (cov @ w)_i``) by a fixed factor and redistribute the freed
     weight to the lowest-risk in-book names (inverse-vol), then re-run the caps.
     This keeps the invested sum constant (deployment is preserved) and only
     accepts a step that strictly reduces vol, so it terminates.

  3. **Gross cut (lever 2, fallback).** If de-weighting is exhausted and vol is
     still over the ceiling, uniformly scale the whole book so ``vol == ceiling``.
     Uniform scaling preserves every cap fraction; only deployment drops.

Net beta is computed on INVESTED PROPORTIONS (scale-invariant; comparable to the
pre-gate value in construct.py) and is report-only (a breach only sets a flag).
Final diagnostics are computed on the GATED book. Pure numpy/pandas + the
riskfirst primitives; no ``yahoofinance.core.config`` import (module-level or
otherwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.construct import apply_name_cap, cap_groups, portfolio_vol
from trade_modules.riskfirst.fx import USD_BLOC, cap_bloc

# Parametric-normal 95% Expected Shortfall multiplier: E[Z | Z > z_.95].
_Z_ES_95 = 2.063
# Tail-name shrink factor per de-weight step (×0.85 => frees 15% of the name).
_SHRINK = 0.85
# Tolerances.
_CAP_TOL = 1e-6  # cap-compliance check
_STABLE_TOL = 1e-9  # caps fixed-point / weight-stability
_VOL_TOL = 1e-12  # vol-improvement / ceiling slack


def _clamp_caps(
    w: np.ndarray,
    sec_arr: np.ndarray,
    is_bloc: np.ndarray,
    *,
    name_thr: float,
    bloc_thr: float,
    sector_thr: float,
) -> np.ndarray:
    """Force compliance by clamping DOWN (excess -> cash, no redistribution).

    Applied once after the redistributing loop as a terminator: for a feasible
    book the loop has already converged so this is a no-op, but when the caps are
    mutually infeasible at the current gross (e.g. an over-cap sector with too few
    non-zero receivers) the loop can oscillate — this pass guarantees a compliant
    book by dropping the irreducible excess to cash. Order name -> sector -> bloc,
    all downward, so every later clamp preserves the earlier ones.
    """
    w = np.minimum(np.asarray(w, dtype=float).copy(), name_thr)  # per-name -> cash
    for lab in dict.fromkeys(sec_arr.tolist()):
        mask = sec_arr == lab
        s = float(w[mask].sum())
        if s > sector_thr + _STABLE_TOL:
            w[mask] *= sector_thr / s
    s = float(w[is_bloc].sum())
    if s > bloc_thr + _STABLE_TOL:
        w[is_bloc] *= bloc_thr / s
    return w


def _caps_to_convergence(
    w: np.ndarray,
    sec_arr: np.ndarray,
    is_bloc: np.ndarray,
    *,
    name_thr: float,
    bloc_thr: float,
    sector_thr: float,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    """Iterate {name -> USD-bloc -> sector} caps until the weights stop moving.

    Thresholds are ABSOLUTE (``cap * gross``), which for a book whose gross equals
    the caller's deployment target is exactly the proportional cap, and — unlike
    renormalising each pass — gives a stable fixed point (no re-inflation spiral)
    when a cap cannot be satisfied without dropping gross to cash. The
    redistributing loop preserves the invested sum whenever a receiver exists; a
    final :func:`_clamp_caps` guarantees compliance (dropping to cash) if the caps
    are infeasible and the loop would otherwise oscillate.
    """
    w = np.asarray(w, dtype=float).copy()
    iters = 0
    for _ in range(max_iter):
        iters += 1
        prev = w.copy()
        w = apply_name_cap(w, name_thr)
        w = cap_bloc(w, is_bloc, bloc_thr)
        w = cap_groups(w, sec_arr, sector_thr)
        if np.max(np.abs(w - prev)) < _STABLE_TOL:
            break
    w = _clamp_caps(
        w, sec_arr, is_bloc, name_thr=name_thr, bloc_thr=bloc_thr, sector_thr=sector_thr
    )
    return w, iters


def apply_risk_gate(
    weights: pd.Series,
    cov: np.ndarray,
    *,
    sectors,
    currencies,
    betas=None,
    conviction=None,
    vol_ceiling: float = 0.18,
    name_cap: float = 0.08,
    sector_cap: float = 0.25,
    usd_bloc_cap: float = 0.60,
    net_beta_band: tuple[float, float] = (0.3, 1.1),
    min_effective_bets: float = 10.0,
    max_iter: int = 100,
) -> tuple[pd.Series, dict]:
    """Enforce the vol ceiling + concentration caps on a constructed book.

    Args:
        weights: Deployed book (long-only), indexed by ticker, summing to the
            deployment target. ``cov`` / ``sectors`` / ``currencies`` / ``betas``
            are aligned POSITIONALLY to ``weights``.
        cov: Annualised covariance matrix aligned to ``weights`` order.
        sectors: Per-name sector labels aligned to ``weights``.
        currencies: Per-name listing currencies aligned to ``weights`` (a name is
            USD-bloc when its currency is in :data:`USD_BLOC`).
        betas: Per-name betas aligned to ``weights`` (NaN -> 1.0); ``None`` -> all
            ones.
        conviction: Optional per-name conviction (a ``pd.Series`` aligned by index
            to ``weights``, or an array-like aligned positionally). When supplied,
            the lever-1 tail-deweight redistribution only ADDS freed weight to
            names the model likes (``conviction > 0``); a name with
            ``conviction <= 0`` (or NaN / missing) is NEVER increased by the gate.
            If no positive-conviction receiver can pull vol under the ceiling, the
            gate falls through to the lever-2 gross cut rather than piling weight
            into disliked names. ``None`` (the default) restores the legacy
            behavior (redistribute to the lowest-vol names regardless of
            conviction), so every prior call is unchanged.
        vol_ceiling: Hard annualised vol ceiling enforced by levers 1 and 2.
        name_cap / sector_cap / usd_bloc_cap: Concentration caps (fraction of the
            invested book).
        net_beta_band: Report-only acceptable net-beta band.
        min_effective_bets: Report-only diversification floor.
        max_iter: Iteration cap for BOTH the caps fixed-point loop and the
            tail-deweight loop (guarantees termination).

    Returns:
        ``(gated_weights, gate_diag)``. ``gate_diag`` keys: ``levers_fired``,
        ``gross_before``, ``gross_after``, ``vol_before``, ``vol_after``,
        ``vol_ceiling``, ``cvar_after``, ``net_beta``, ``net_beta_band``,
        ``net_beta_out``, ``gross_cut``, ``effective_bets``,
        ``min_effective_bets``, ``caps_ok``, ``max_name``, ``max_sector``,
        ``usd_bloc``, ``iterations`` (tail-deweight steps), ``caps_iterations``.
    """
    index = weights.index
    w = np.asarray(weights.to_numpy(), dtype=float).copy()
    cov = np.asarray(cov, dtype=float)
    sec_arr = np.asarray(list(sectors), dtype=object)
    ccy = list(currencies)
    is_bloc = np.array([c in USD_BLOC for c in ccy], dtype=bool)
    n = int(w.shape[0])

    if betas is None:
        b = np.ones(n, dtype=float)
    else:
        b = pd.to_numeric(pd.Series(list(betas)), errors="coerce").fillna(1.0).to_numpy()

    # Conviction-aware redistribution (optional). ``pos_mask`` flags the names
    # lever 1 is allowed to ADD weight to (conviction > 0); NaN / missing ->
    # False (never increased). ``None`` -> no restriction (legacy behavior).
    if conviction is None:
        pos_mask = None
    else:
        conv_ser = (
            conviction
            if isinstance(conviction, pd.Series)
            else pd.Series(list(conviction), index=index)
        )
        conv_arr = pd.to_numeric(conv_ser.reindex(index), errors="coerce").to_numpy()
        pos_mask = np.isfinite(conv_arr) & (conv_arr > 0.0)

    gross_before = float(w.sum())
    vol_before = portfolio_vol(w, cov) if n else 0.0
    lo_band, hi_band = float(net_beta_band[0]), float(net_beta_band[1])

    def _diag(
        levers,
        gross_after,
        port_vol,
        net_beta,
        net_beta_out,
        gross_cut,
        eff,
        caps_ok,
        max_name,
        max_sector,
        usd_bloc,
        l1_iter,
        caps_iter,
    ) -> dict:
        return {
            "levers_fired": levers,
            "gross_before": gross_before,
            "gross_after": float(gross_after),
            "vol_before": float(vol_before),
            "vol_after": float(port_vol),
            "vol_ceiling": float(vol_ceiling),
            "cvar_after": _Z_ES_95 * float(port_vol),
            "net_beta": float(net_beta),
            "net_beta_band": (lo_band, hi_band),
            "net_beta_out": bool(net_beta_out),
            "gross_cut": bool(gross_cut),
            "effective_bets": float(eff),
            "min_effective_bets": float(min_effective_bets),
            "caps_ok": bool(caps_ok),
            "max_name": float(max_name),
            "max_sector": float(max_sector),
            "usd_bloc": float(usd_bloc),
            "iterations": int(l1_iter),
            "caps_iterations": int(caps_iter),
        }

    # Degenerate: empty / all-cash book — nothing to enforce.
    if n == 0 or gross_before <= 0:
        return pd.Series(w, index=index), _diag(
            [],
            gross_before,
            vol_before,
            0.0,
            False,
            False,
            0.0,
            True,
            0.0,
            0.0,
            0.0,
            0,
            0,
        )

    g_target = gross_before
    name_thr = name_cap * g_target
    bloc_thr = usd_bloc_cap * g_target
    sector_thr = sector_cap * g_target

    levers_fired: list[str] = []
    caps_iter_total = 0

    # --- 1. caps to convergence -------------------------------------------- #
    w_pre = w.copy()
    w, ci = _caps_to_convergence(
        w,
        sec_arr,
        is_bloc,
        name_thr=name_thr,
        bloc_thr=bloc_thr,
        sector_thr=sector_thr,
        max_iter=max_iter,
    )
    caps_iter_total += ci
    if np.max(np.abs(w - w_pre)) > _STABLE_TOL:
        levers_fired.append("caps")

    # --- 2. vol ceiling via tail de-weighting (lever 1) -------------------- #
    l1_iter = 0
    if n > 1:
        inv_vol = 1.0 / np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        cur_vol = portfolio_vol(w, cov)
        while cur_vol > vol_ceiling + _VOL_TOL and l1_iter < max_iter:
            in_book = w > 1e-12
            if int(in_book.sum()) <= 1:
                break
            rc = w * (cov @ w)  # per-name risk contribution
            hi = int(np.argmax(np.where(in_book, rc, -np.inf)))
            recv = in_book.copy()
            recv[hi] = False
            if pos_mask is not None:
                recv &= pos_mask  # only ADD to names the model likes (conviction>0)
            if not recv.any():
                break  # no eligible receiver -> fall through to the gross cut
            iv_recv = np.where(recv, inv_vol, 0.0)
            iv_sum = float(iv_recv.sum())
            if iv_sum <= 0:
                break
            # Shrink the worst tail-risk name; redistribute freed weight to the
            # lowest-risk in-book names (inverse-vol) so the invested sum holds.
            trial = w.copy()
            freed = trial[hi] * (1.0 - _SHRINK)
            trial[hi] *= _SHRINK
            trial += freed * (iv_recv / iv_sum)
            trial, ci = _caps_to_convergence(
                trial,
                sec_arr,
                is_bloc,
                name_thr=name_thr,
                bloc_thr=bloc_thr,
                sector_thr=sector_thr,
                max_iter=max_iter,
            )
            caps_iter_total += ci
            new_vol = portfolio_vol(trial, cov)
            accept = new_vol < cur_vol - _VOL_TOL  # accept only a strict improvement
            if accept and pos_mask is not None:
                # Guard: never let a disliked (conviction<=0) name grow, even
                # indirectly via the cap redistribution inside this step.
                disliked = ~pos_mask
                if np.any(trial[disliked] > w[disliked] + _STABLE_TOL):
                    accept = False
            if accept:
                w, cur_vol = trial, new_vol
                l1_iter += 1
            else:  # de-weighting can no longer help — lever 1 exhausted
                break
        if l1_iter > 0:
            levers_fired.append("tail_deweight")

    # --- 3. gross cut (lever 2, fallback) ---------------------------------- #
    gross_cut = False
    pv = portfolio_vol(w, cov)
    if pv > vol_ceiling + _VOL_TOL:
        w = w * (vol_ceiling / pv)  # uniform -> vol == ceiling, caps preserved
        gross_cut = True
        levers_fired.append("gross_cut")

    # --- 4. net beta on invested PROPORTIONS (scale-invariant; matches pre-gate) #
    _gross_for_beta = float(w.sum())
    _p_beta = w / _gross_for_beta if _gross_for_beta > 0 else w
    net_beta = float(np.dot(_p_beta, b))
    net_beta_out = not (lo_band <= net_beta <= hi_band)

    # --- 5. final diagnostics on the GATED book ---------------------------- #
    gross_after = float(w.sum())
    port_vol = portfolio_vol(w, cov)
    if gross_after > 0:
        p = w / gross_after
    else:
        p = w
    ss = float(np.sum(p**2))
    effective_bets = float(1.0 / ss) if ss > 0 else 0.0
    max_name = float(p.max()) if n else 0.0
    sec_tot: dict = {}
    for lab, wt in zip(sec_arr.tolist(), p.tolist(), strict=True):
        sec_tot[lab] = sec_tot.get(lab, 0.0) + float(wt)
    max_sector = max(sec_tot.values(), default=0.0)
    usd_bloc = float(np.sum(p[is_bloc])) if is_bloc.any() else 0.0
    caps_ok = (
        max_name <= name_cap + _CAP_TOL
        and max_sector <= sector_cap + _CAP_TOL
        and usd_bloc <= usd_bloc_cap + _CAP_TOL
    )

    gate_diag = _diag(
        levers_fired,
        gross_after,
        port_vol,
        net_beta,
        net_beta_out,
        gross_cut,
        effective_bets,
        caps_ok,
        max_name,
        max_sector,
        usd_bloc,
        l1_iter,
        caps_iter_total,
    )
    return pd.Series(w, index=index), gate_diag
