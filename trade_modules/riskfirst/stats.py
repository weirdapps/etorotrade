"""Cross-sectional scoring primitives: winsorize + NaN-safe z-score.

Factors are expressed as winsorized cross-sectional z-scores where HIGHER means
MORE ATTRACTIVE, so they can be averaged into a composite on a common scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series to its [lower, upper] cross-sectional quantiles."""
    if s is None or len(s) == 0:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def zscore(
    s: pd.Series,
    winsor: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """NaN-safe cross-sectional z-score (population std, ddof=0).

    NaN inputs stay NaN. A constant (zero-variance) series returns all zeros
    rather than inf/NaN, so a factor with no cross-sectional spread contributes
    nothing instead of blowing up the composite.
    """
    x = winsorize(s, lower, upper) if winsor else s
    mean = x.mean(skipna=True)
    std = x.std(ddof=0, skipna=True)
    if not np.isfinite(std) or std == 0:
        # zero cross-sectional spread -> no signal; return 0 for present names but
        # keep NaN where the input was missing (don't manufacture a score).
        return pd.Series(0.0, index=s.index).where(x.notna())
    return (x - mean) / std
