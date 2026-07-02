"""
IC Decay Estimator

Fits an exponential decay model IC(h) = IC0 * exp(-h/τ) to an
information-coefficient-by-horizon dictionary via log-linear regression.
Returns the half-life and the intercept IC0.

Pure / no I/O.
"""

import math

import numpy as np


def compute_ic_decay(ic_by_horizon: dict[int, float]) -> dict:
    """
    Estimate IC decay half-life from a horizon→IC mapping.

    Filters to horizons where IC > 0 (log regression requires positive values),
    then fits log(IC) = log(IC0) - h/τ via numpy polyfit.

    Args:
        ic_by_horizon: Dict mapping horizon_days (int) to IC value (float).

    Returns:
        Dict with keys:
          - half_life_days: float | None
          - ic0:            float | None
          - curve:          dict sorted by horizon (original values)
          - note:           str — empty on success, explanatory on failure
    """
    # Sorted curve (original values, all horizons)
    curve = dict(sorted(ic_by_horizon.items()))

    # Filter to positive IC for log regression
    positive = {h: ic for h, ic in ic_by_horizon.items() if ic > 0}

    if len(positive) < 2:
        return {
            "half_life_days": None,
            "ic0": None,
            "curve": curve,
            "note": (
                "Insufficient positive IC data points for regression "
                f"(need >= 2, got {len(positive)})."
            ),
        }

    horizons = np.array(sorted(positive.keys()), dtype=float)
    log_ic = np.log(np.array([positive[int(h)] for h in horizons]))

    # log(IC) = log(IC0) - h/τ  →  slope = -1/τ, intercept = log(IC0)
    slope, intercept = np.polyfit(horizons, log_ic, deg=1)

    # A valid decay requires a negative slope (IC decreasing with horizon)
    if slope >= 0:
        return {
            "half_life_days": None,
            "ic0": None,
            "curve": curve,
            "note": (
                f"IC is not decaying (slope={slope:.4f} >= 0); "
                "half-life undefined for flat or rising IC."
            ),
        }

    tau = -1.0 / slope  # decay constant in days
    half_life = tau * math.log(2)
    ic0 = math.exp(intercept)

    return {
        "half_life_days": half_life,
        "ic0": ic0,
        "curve": curve,
        "note": "",
    }
