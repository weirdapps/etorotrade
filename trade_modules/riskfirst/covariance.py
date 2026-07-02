"""Single-factor (market-beta) covariance estimate.

Cov = beta beta' * market_var + diag(idio_var)

A transparent one-factor approximation that needs only betas plus two
assumptions (market vol, idiosyncratic vol). It is always symmetric PSD and
needs no per-name price history. A full sample covariance with Ledoit-Wolf
shrinkage (from a returns matrix) is the documented upgrade once price history
is wired into the engine.
"""

from __future__ import annotations

import numpy as np


def single_factor_cov(betas, market_vol: float, idio_vol) -> np.ndarray:
    """Build a one-factor covariance matrix from betas.

    Args:
        betas: array of market betas, one per name.
        market_vol: market (factor) volatility, e.g. 0.18 for 18%/yr.
        idio_vol: idiosyncratic volatility — a scalar applied to all names, or an
            array of per-name idio vols.
    """
    b = np.asarray(betas, dtype=float)
    cov = np.outer(b, b) * (float(market_vol) ** 2)
    idio_var = np.asarray(idio_vol, dtype=float) ** 2
    if idio_var.ndim == 0:
        idio_var = np.full(b.shape[0], float(idio_var))
    return cov + np.diag(idio_var)
