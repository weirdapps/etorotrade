"""
validation/spanning.py — Spanning / factor-redundancy tests (Phase 2B)

Two pure primitives (arrays in, dict out) for asking "does this candidate return
stream add anything beyond a set of factors?":

  spanning_test  Time-series OLS of a candidate return series on a factor set
                 (with intercept).  The intercept (alpha) is the return the
                 candidate earns that the factors do NOT explain; a t-stat below
                 threshold means the candidate is *spanned* (redundant).

  grs_test       Gibbons-Ross-Shanken joint test that the alphas of MULTIPLE
                 candidates are all zero (an exact F-test), the multivariate
                 generalisation of the single-asset spanning t-test.

Inference uses PLAIN (homoskedastic, iid) OLS standard errors — NOT
Newey-West/HAC.  For monthly-or-coarser rebalanced return series with little
autocorrelation this is adequate; if the candidate stream has strong serial
correlation the alpha t-stat here is anti-conservative.  Documented, not hidden.

scipy.stats (a declared dependency) supplies the t / F tail probabilities.

No I/O.  Never raises on thin or degenerate data — returns explicit None fields.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import f as _f_dist
from scipy.stats import t as _t_dist

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_factors(
    factor_returns: np.ndarray | dict[str, Any] | list,
) -> tuple[np.ndarray, list[str]]:
    """Normalise factor input to a (T, K) float matrix + ordered names."""
    if isinstance(factor_returns, dict):
        names = list(factor_returns.keys())
        cols = [np.asarray(factor_returns[k], dtype=float).ravel() for k in names]
        mat = np.column_stack(cols) if cols else np.empty((0, 0), dtype=float)
        return mat, names
    mat = np.asarray(factor_returns, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    names = [f"factor_{i}" for i in range(mat.shape[1])]
    return mat, names


def _as_matrix(x: Any, n: int) -> np.ndarray:
    """Coerce a covariance-like argument to an (n, n) matrix (scalars → diagonal)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.array([[float(arr)]]) if n == 1 else np.eye(n) * float(arr)
    return np.atleast_2d(arr)


def _safe_inv(mat: np.ndarray) -> np.ndarray:
    """Invert, falling back to the Moore-Penrose pseudo-inverse when singular."""
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spanning_test(
    candidate_returns: np.ndarray | list,
    factor_returns: np.ndarray | dict[str, Any] | list,
    *,
    alpha_threshold_t: float = 2.0,
) -> dict[str, Any]:
    """Time-series spanning regression: candidate ~ alpha + betas . factors.

    Args:
        candidate_returns: 1D candidate return series (length T).
        factor_returns: (T, K) matrix, a 1D series, or a dict {name: series}.
        alpha_threshold_t: |alpha t-stat| below which the candidate is spanned.

    Returns dict:
        alpha         intercept (return unexplained by the factors)
        alpha_tstat   alpha / plain-OLS SE(alpha) (None when dof <= 0 / degenerate)
        alpha_pvalue  two-sided t p-value (None when tstat undefined)
        betas         {factor_name: loading}
        r2            regression R^2 (None when the candidate is constant)
        n_obs         observations after pairwise NaN drop
        spanned       abs(alpha_tstat) < alpha_threshold_t (None when undefined)
    """
    y = np.asarray(candidate_returns, dtype=float).ravel()
    factors, names = _coerce_factors(factor_returns)
    k = factors.shape[1] if factors.ndim == 2 else 0

    # Pairwise-drop rows with NaN/inf in the candidate or any factor.
    if factors.size:
        finite = np.isfinite(y) & np.all(np.isfinite(factors), axis=1)
        y = y[finite]
        factors = factors[finite]
    else:
        y = y[np.isfinite(y)]

    n = len(y)
    betas = dict.fromkeys(names)
    degenerate = {
        "alpha": None,
        "alpha_tstat": None,
        "alpha_pvalue": None,
        "betas": betas,
        "r2": None,
        "n_obs": n,
        "spanned": None,
    }

    if n == 0 or n <= k + 1:
        # dof = n - (k+1) <= 0 → SEs undefined.  Still recover point estimates
        # when the system is at least determined (n == k+1), else bail entirely.
        if n >= k + 1 and n > 0:
            design = np.column_stack([np.ones(n), factors]) if k else np.ones((n, 1))
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            degenerate["alpha"] = float(coef[0])
            degenerate["betas"] = {name: float(coef[i + 1]) for i, name in enumerate(names)}
        return degenerate

    design = np.column_stack([np.ones(n), factors]) if k else np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = n - design.shape[1]
    ssr = float(resid @ resid)
    sigma2 = ssr / dof

    xtx_inv = _safe_inv(design.T @ design)
    var_alpha = sigma2 * float(xtx_inv[0, 0])
    alpha = float(coef[0])
    betas = {name: float(coef[i + 1]) for i, name in enumerate(names)}

    # R^2 vs the candidate's own mean.
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2: float | None = (1.0 - ssr / sst) if sst > 0 else None

    # Degenerate residual variance (candidate exactly spanned): plain-OLS
    # inference is undefined — report point estimates, leave inference None.
    if not math.isfinite(var_alpha) or var_alpha <= 0.0:
        return {
            "alpha": alpha,
            "alpha_tstat": None,
            "alpha_pvalue": None,
            "betas": betas,
            "r2": r2,
            "n_obs": n,
            "spanned": None,
        }

    alpha_tstat = alpha / math.sqrt(var_alpha)
    alpha_pvalue = float(2.0 * _t_dist.sf(abs(alpha_tstat), dof))
    return {
        "alpha": alpha,
        "alpha_tstat": float(alpha_tstat),
        "alpha_pvalue": alpha_pvalue,
        "betas": betas,
        "r2": r2,
        "n_obs": n,
        "spanned": bool(abs(alpha_tstat) < alpha_threshold_t),
    }


def grs_test(
    alphas: np.ndarray | list,
    resid_cov: np.ndarray | float,
    factor_means: np.ndarray | list | float,
    factor_cov: np.ndarray | float,
    n_obs: int,
) -> dict[str, Any]:
    """Gibbons-Ross-Shanken joint-alpha F-test.

    Tests H0: every asset's alpha is zero (the factors span all N candidates).

        F = ((T - N - K) / N) * (a' Sigma^-1 a) / (1 + mu' Omega^-1 mu)

    with F ~ F(N, T - N - K) under H0.  Sigma is the residual covariance, mu /
    Omega the factor means / covariance, T = n_obs.

    Args:
        alphas: length-N vector of regression intercepts.
        resid_cov: (N, N) residual covariance (scalar → diagonal).
        factor_means: length-K factor mean returns (scalar for K=1).
        factor_cov: (K, K) factor covariance (scalar for K=1).
        n_obs: number of time periods T.

    Returns dict:
        f_stat    GRS F statistic (None when T - N - K <= 0)
        p_value   upper-tail F p-value (None when f_stat undefined)
        n_assets  N
        n_obs     T
    """
    alpha = np.asarray(alphas, dtype=float).ravel()
    mu = np.atleast_1d(np.asarray(factor_means, dtype=float).ravel())
    n_assets = int(alpha.shape[0])
    n_factors = int(mu.shape[0])
    t_obs = int(n_obs)

    dof2 = t_obs - n_assets - n_factors
    if n_assets == 0 or dof2 <= 0:
        return {"f_stat": None, "p_value": None, "n_assets": n_assets, "n_obs": t_obs}

    sigma_inv = _safe_inv(_as_matrix(resid_cov, n_assets))
    omega_inv = _safe_inv(_as_matrix(factor_cov, n_factors))

    sharpe_sq = float(mu @ omega_inv @ mu)
    alpha_term = float(alpha @ sigma_inv @ alpha)
    f_stat = (dof2 / n_assets) * (alpha_term / (1.0 + sharpe_sq))
    p_value = float(_f_dist.sf(f_stat, n_assets, dof2))
    return {
        "f_stat": float(f_stat),
        "p_value": p_value,
        "n_assets": n_assets,
        "n_obs": t_obs,
    }
