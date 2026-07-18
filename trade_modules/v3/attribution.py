"""FIX-NOW 2026-07-18 (D21): measure the owner-core sleeve + classify holdings by
attribution layer.

"Owned discretion" (protect_core / core_floor) is a DECISION right, not an exemption
from MEASUREMENT. These helpers quantify the core sleeve's share of book RISK (not
just weight) and tag each holding by attribution layer, so a dominant, unmeasured
owner bet cannot hide behind the 6-cluster forecast's IC. Pure numpy/pandas; no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def core_attribution(weights, cov, core_list, betas=None) -> dict:
    """Attribution of a deployed book to its owner-core sleeve.

    Args:
        weights: deployed book weights (Series / dict, fractions of NAV), by ticker.
        cov: annualised covariance aligned POSITIONALLY to ``weights`` order.
        core_list: the owner-core tickers (the protect_core / core_floor names).
        betas: optional per-name betas aligned to ``weights`` (NaN -> 1.0).

    Returns (all on invested proportions; an all-cash book -> zeros, never a
    divide-by-zero):
        ``core_weight`` — core share of the invested book;
        ``core_variance_share`` — core share of ex-ante book VARIANCE (the number
            that reveals a small-weight / high-vol core dominating book risk);
        ``core_net_beta_contribution`` / ``book_net_beta``;
        ``n_core_held``.
    """
    ws = weights if isinstance(weights, pd.Series) else pd.Series(weights, dtype=float)
    idx = [str(t) for t in ws.index]
    w = pd.to_numeric(ws, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gross = float(w.sum())
    if gross <= 0.0 or len(idx) == 0:
        return {
            "core_weight": 0.0,
            "core_variance_share": 0.0,
            "core_net_beta_contribution": float("nan"),
            "book_net_beta": float("nan"),
            "n_core_held": 0,
        }

    p = w / gross
    core = {str(t) for t in (core_list or [])}
    core_mask = np.array([t in core for t in idx], dtype=bool)
    core_weight = float(p[core_mask].sum())

    core_var_share = 0.0
    cov = np.asarray(cov, dtype=float)
    if cov.shape == (len(idx), len(idx)):
        port_var = float(p @ cov @ p)
        if port_var > 1e-18:
            rc = p * (cov @ p)  # per-name risk contribution (sums to port_var)
            core_var_share = float(rc[core_mask].sum() / port_var)

    if betas is not None:
        b = pd.to_numeric(pd.Series(list(betas)), errors="coerce").fillna(1.0).to_numpy(dtype=float)
        book_net_beta = float(p @ b)
        core_net_beta = float((p * core_mask) @ b)
    else:
        book_net_beta = core_net_beta = float("nan")

    return {
        "core_weight": core_weight,
        "core_variance_share": core_var_share,
        "core_net_beta_contribution": core_net_beta,
        "book_net_beta": book_net_beta,
        "n_core_held": int(core_mask.sum()),
    }


def attribution_layer(ticker, *, managed_sleeves, core_list) -> dict:
    """3-layer attribution taxonomy at the ticker level.

    * ``SAA`` — owner-managed sleeves (gold / vol / Greece): owned policy, never
      scored as security-selection alpha.
    * ``forecast`` — a scored equity driven by the 6-cluster IC-gated model.

    (The ``process`` layer — gates / caps / deadband / deployment — is not per-ticker
    and is judged on Sharpe / turnover / drawdown, not IC.)

    ``owner_override`` flags a name held under protect_core / core_floor: a MEASURED
    discretionary override (see :func:`core_attribution`), not process, not exempt.
    """
    t = str(ticker).upper()
    managed = {str(s).upper() for s in (managed_sleeves or [])}
    core = {str(s).upper() for s in (core_list or [])}
    return {"layer": "SAA" if t in managed else "forecast", "owner_override": t in core}
