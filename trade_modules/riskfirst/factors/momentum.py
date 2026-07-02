"""Momentum factor (snapshot proxy).

LIMITATION — this is a *proxy* implementation that operates on the snapshot
columns available in etoro.csv.  The canonical momentum factor would be the
12-month minus 1-month (Jegadeesh-Titman) return computed from a price series;
that is available via ``riskfirst.prices.price_momentum_factor`` when a full
price history is at hand.

Snapshot proxy:
    momentum_12_1 proxy ← '52W' column (52-week return) where available,
    minus '1M' (1-month return) to approximate the skip-month convention.
    Falls back to '52W' alone if '1M' is absent.

Higher z-score = stronger momentum = more attractive.

Column aliases: '52W' / 'RETURN_52W', '1M' / 'RETURN_1M', 'AM' / 'ANALYST_MOM'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore

_COL_52W = ("52W", "RETURN_52W")
_COL_AM = ("AM", "ANALYST_MOM", "ANALYST_MOMENTUM")


def compute(df: pd.DataFrame) -> pd.Series:
    """Momentum factor: z-scored 12-1 proxy + optional analyst momentum."""
    idx = df.index
    parts = []

    def _col(*candidates: str) -> pd.Series | None:
        for c in candidates:
            if c in df.columns:
                return df[c]
        return None

    ret_52w = _col(*_COL_52W)
    ret_1m = _col("1M", "RETURN_1M")
    analyst_mom = _col(*_COL_AM)

    if ret_52w is not None:
        mom = ret_52w if ret_1m is None else (ret_52w - ret_1m)
        parts.append(zscore(mom))

    if analyst_mom is not None:
        parts.append(zscore(analyst_mom))

    if not parts:
        return pd.Series(np.nan, index=idx)

    mat = pd.concat(parts, axis=1)
    return mat.mean(axis=1, skipna=True)
