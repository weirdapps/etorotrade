"""Low-volatility factor — low-vol anomaly (lower risk is rewarded).

PROXY NOTE
----------
This implementation uses beta (column 'B') as a *snapshot proxy* for
realized volatility.  Beta is readily available in etoro.csv; realized vol
from a price series is available via ``riskfirst.prices.price_lowvol_factor``.

Higher z-score = LOWER beta = lower risk = more attractive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore


def compute(df: pd.DataFrame) -> pd.Series:
    """Low-vol factor: z-score of NEGATIVE beta (so lower beta → higher score)."""
    idx = df.index

    for col in ("B", "BETA"):
        if col in df.columns:
            return zscore(-df[col])

    return pd.Series(np.nan, index=idx)
