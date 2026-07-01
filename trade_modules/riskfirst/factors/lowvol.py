"""Low-volatility factor — low-vol anomaly (lower risk is rewarded).

PROXY NOTE
----------
This implementation uses beta (column 'B') as a *snapshot proxy* for
realized volatility.  Beta is readily available in the processed universe
(etoro.csv) and is correlated with trailing volatility, but it is an
imperfect substitute.

TODO: replace with trailing realized volatility computed from a rolling
      window of daily returns (e.g. 252-day annualised vol from price
      history).  That is the standard Frazzini & Pedersen (2014) measure.
      Beta inherits market-direction skew and can differ materially for
      low-correlation stocks.

FACTOR DIRECTION
----------------
Lower beta -> more attractive (lower risk -> higher score).
We therefore pass -B to zscore so that lower beta produces a HIGHER z-score,
consistent with the factor contract (HIGHER = MORE ATTRACTIVE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore


def compute(df: pd.DataFrame) -> pd.Series:
    """Return cross-sectional low-vol z-scores aligned to df.index.

    Parameters
    ----------
    df : pd.DataFrame
        Universe DataFrame indexed by ticker.  Must contain column 'B' (beta).
        All other columns are ignored.

    Returns
    -------
    pd.Series
        NaN-safe cross-sectional z-scores where HIGHER = lower beta = more
        attractive.  Tickers with NaN beta receive NaN.  If column 'B' is
        absent the function returns an all-NaN series aligned to df.index
        (never raises).
    """
    if "B" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    beta = df["B"].astype(float)

    # If every value is NaN there is no cross-sectional signal; return NaN
    # rather than the all-zeros that zscore's constant-series guard would emit.
    if beta.isna().all():
        return pd.Series(np.nan, index=df.index)

    # Negate: lower beta -> higher raw value -> higher z-score
    return zscore(-beta)
