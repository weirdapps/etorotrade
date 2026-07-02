"""Cross-sectional scoring primitives: winsorize + NaN-safe z-score.

Factors are expressed as winsorized cross-sectional z-scores where HIGHER means
MORE ATTRACTIVE, so they can be averaged into a composite on a common scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """Clip a series to its [lower, upper] cross-sectional quantiles."""
    if len(s) == 0:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def zscore(
    s: pd.Series,
    winsor: bool = True,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.Series:
    """NaN-safe cross-sectional z-score (population std, ddof=0).

    NaN inputs stay NaN. A constant (zero-variance) series returns all zeros
    rather than inf/NaN, so a factor with no cross-sectional spread doesn't
    blow up the composite.
    """
    x = winsorize(s, lower, upper) if winsor else s
    mean = x.mean()
    std = x.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.where(s.notna(), 0.0, np.nan), index=s.index)
    result = (x - mean) / std
    return result.where(s.notna(), other=np.nan)
