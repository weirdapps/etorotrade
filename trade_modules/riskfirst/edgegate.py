"""Edge-gate: multiple-testing-aware significance for a strategy's Sharpe.

The senior-PM review found NO Deflated Sharpe / PSR / PBO anywhere in the system,
while ~220 parameters were tuned on ~60 independent observations in a single bull
regime — a textbook overfitting setup. This module implements:

- Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) per
  Bailey & Lopez de Prado (2014). DSR deflates the observed Sharpe by the Sharpe
  you'd expect as the maximum across N trials under the null.
- A gate_verdict() that ALSO refuses to pass without enough observations and at
  least one bear/stress regime in the sample.

Conviction stays clamped (sizing flat) until this gate passes. Uses stdlib
NormalDist — no scipy dependency.
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329

# Defaults for the gate.
DSR_THRESHOLD = 0.95  # conventional significance hurdle
MIN_OBSERVATIONS = 252  # ~1 year of daily obs as an absolute floor
MIN_REGIMES = 2  # must include at least one bear/stress regime


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """P(true SR > benchmark) given the observed per-period SR and its moments.

    PSR = Phi( (SR - SR*) * sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) ).
    """
    denom = math.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(max(n_obs - 1, 1)) / denom
    return _N.cdf(z)


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent trials under the
    null (the DSR deflation benchmark SR0)."""
    if n_trials < 2:
        return 0.0
    z = (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1.0 / n_trials) + EULER_GAMMA * _N.inv_cdf(
        1 - 1.0 / (n_trials * math.e)
    )
    return math.sqrt(max(var_sr, 0.0)) * z


def deflated_sharpe_ratio(
    sr: float, n_obs: int, n_trials: int, var_sr: float, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """DSR = PSR with the benchmark set to the expected max Sharpe across trials."""
    sr0 = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurt)


def pbo_cscv(perf, n_splits: int = 10) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    Args:
        perf: a (T periods x N configurations) matrix of per-period performance.
        n_splits: number of disjoint time blocks S (forced even). All C(S, S/2)
            in-sample/out-of-sample partitions are evaluated.

    For each partition: pick the config with the best IN-sample mean, then measure
    its OUT-of-sample relative rank. PBO is the fraction of partitions where that
    IS-best config lands at or below the OOS median (an overfit selection).
    """
    M = np.asarray(perf, dtype=float)
    T, N = M.shape
    S = n_splits - (n_splits % 2)
    S = max(S, 2)
    groups = np.array_split(np.arange(T), S)
    all_g = list(range(S))
    half = S // 2

    logits, n_overfit, n_total = [], 0, 0
    for is_groups in combinations(all_g, half):
        is_rows = np.concatenate([groups[g] for g in is_groups])
        oos_rows = np.concatenate([groups[g] for g in all_g if g not in is_groups])
        is_best = int(np.argmax(M[is_rows].mean(axis=0)))
        oos_perf = M[oos_rows].mean(axis=0)
        rank = int(np.argsort(np.argsort(oos_perf))[is_best])  # 0..N-1, higher = better
        omega = (rank + 1) / (N + 1)
        omega = min(max(omega, 1e-9), 1 - 1e-9)
        logits.append(math.log(omega / (1 - omega)))
        n_total += 1
        if omega <= 0.5:
            n_overfit += 1
    return {
        "pbo": (n_overfit / n_total) if n_total else float("nan"),
        "logits": logits,
        "n_combinations": n_total,
        "n_configs": N,
    }


def gate_verdict(
    *,
    sr: float,
    n_obs: int,
    n_trials: int,
    var_sr: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_regimes: int = 1,
    pbo: float | None = None,
    dsr_threshold: float = DSR_THRESHOLD,
    min_obs: int = MIN_OBSERVATIONS,
    min_regimes: int = MIN_REGIMES,
    pbo_threshold: float = 0.5,
) -> dict:
    """Combined edge gate. Passes only when the strategy clears the deflated
    Sharpe hurdle AND has enough observations AND spans >= one bear/stress regime.
    """
    dsr = deflated_sharpe_ratio(sr, n_obs, n_trials, var_sr, skew, kurt)
    reasons = []
    if dsr < dsr_threshold:
        reasons.append(f"deflated Sharpe {dsr:.3f} < {dsr_threshold}")
    if n_obs < min_obs:
        reasons.append(f"insufficient observations {n_obs} < {min_obs}")
    if n_regimes < min_regimes:
        reasons.append(f"single-regime sample (need >= {min_regimes}; no bear/stress)")
    if pbo is not None and pbo >= pbo_threshold:
        reasons.append(f"PBO {pbo:.2f} >= {pbo_threshold} (selection likely overfit)")
    return {"passed": len(reasons) == 0, "dsr": dsr, "pbo": pbo, "reasons": reasons}
