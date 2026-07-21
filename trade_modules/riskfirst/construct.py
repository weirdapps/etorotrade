"""Risk-first portfolio construction: ERC weights, vol targeting, binding caps.

All functions operate on plain numpy arrays (a covariance matrix and weight
vectors); the integrator wraps them with tickers/sectors. Long-only.
"""

from __future__ import annotations

import numpy as np


def erc_weights(cov: np.ndarray, tol: float = 1e-12, max_iter: int = 100_000) -> np.ndarray:
    """Equal-Risk-Contribution (risk-parity) weights, long-only, summing to 1.

    Fixed-point iteration w_i <- 1 / (Sigma w)_i, renormalised — the standard
    converging scheme for the long-only ERC problem. For uncorrelated assets it
    reduces to inverse-volatility weights. Seeded from inverse-vol for speed.
    """
    cov = np.asarray(cov, dtype=float)
    vol = np.sqrt(np.diag(cov))
    w = np.where(vol > 0, 1.0 / vol, 1.0)
    w = w / w.sum()
    for _ in range(max_iter):
        sigma_w = cov @ w
        sigma_w = np.where(sigma_w <= 0, 1e-18, sigma_w)
        w_new = 1.0 / sigma_w
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Annualised (same units as cov) portfolio volatility sqrt(w' Σ w)."""
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(max(w @ np.asarray(cov, dtype=float) @ w, 0.0)))


def vol_target_scale(
    w: np.ndarray,
    cov: np.ndarray,
    target_vol: float,
    max_gross: float = 1.0,
) -> np.ndarray:
    """Scale weights so portfolio vol == target_vol, capped at gross ``max_gross``.

    Scales DOWN (to cash) when the book is hotter than target; will NOT lever
    beyond ``max_gross`` to reach target when the book is too quiet.
    """
    w = np.asarray(w, dtype=float)
    pv = portfolio_vol(w, cov)
    if pv <= 0:
        return w
    scale = target_vol / pv
    gross = scale * np.abs(w).sum()
    if gross > max_gross:
        scale = max_gross / np.abs(w).sum()
    return w * scale


def cap_groups(w: np.ndarray, labels, cap: float, max_iter: int = 1000) -> np.ndarray:
    """Cap the aggregate weight of ANY group (sector / cluster / region) at ``cap``.

    Iteratively finds an over-cap group, scales it to the cap, and redistributes
    the freed weight to names in groups that are still under the cap. Generalises
    :func:`~trade_modules.riskfirst.fx.cap_bloc`; plug a sector label array in for
    sector concentration control. If no under-cap group can absorb the excess, the
    freed weight becomes cash (gross drops)."""
    w = np.asarray(w, dtype=float).copy()
    labels = np.asarray(labels)
    uniq = list(dict.fromkeys(labels.tolist()))
    for _ in range(max_iter):
        over_mask = None
        over_sum = 0.0
        for lab in uniq:
            mask = labels == lab
            s = float(w[mask].sum())
            if s > cap + 1e-9:
                over_mask, over_sum = mask, s
                break
        if over_mask is None:
            break
        excess = over_sum - cap
        w[over_mask] *= cap / over_sum
        recv = np.zeros(len(w), dtype=bool)
        for lab in uniq:
            mask = labels == lab
            if float(w[mask].sum()) < cap - 1e-9:
                recv |= mask
        recv &= w > 0
        recv_sum = float(w[recv].sum())
        if not recv.any() or recv_sum <= 0:
            break
        w[recv] += excess * (w[recv] / recv_sum)
    return w


def apply_name_cap(w: np.ndarray, cap: float, max_iter: int = 1000) -> np.ndarray:
    """Cap each weight at ``cap``, redistributing the excess proportionally to the
    remaining under-cap names. Preserves the total invested weight."""
    w = np.asarray(w, dtype=float).copy()
    for _ in range(max_iter):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = (~over) & (w > 0)
        under_sum = float(w[under].sum())
        if not under.any() or under_sum <= 0:
            break
        w[under] += excess * (w[under] / under_sum)
    return w


def apply_name_cap_vec(w: np.ndarray, caps: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Per-name cap: cap each weight at ``caps[i]`` (e.g. a market-cap-tier schedule),
    redistributing the excess proportionally to the remaining HEADROOM of under-cap names.
    Preserves the total invested weight (headroom-proportional avoids re-exceeding small caps)."""
    w = np.asarray(w, dtype=float).copy()
    caps = np.asarray(caps, dtype=float)
    for _ in range(max_iter):
        over = w > caps + 1e-12
        if not over.any():
            break
        excess = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        head = np.where(~over, caps - w, 0.0)
        head_sum = float(head.sum())
        if head_sum <= 1e-15:
            break
        w = w + excess * (head / head_sum)
    return w
