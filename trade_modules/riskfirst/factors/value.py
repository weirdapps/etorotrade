"""VALUE factor — cross-sectional cheapness score.

Sub-metrics:
    earnings_yield_trailing  = 1 / PET    (only when PET > 0, else NaN)
    earnings_yield_forward   = 1 / PEF    (only when PEF > 0, else NaN)
    book_yield               = 1 / PB     (only when PB  > 0, else NaN)

Composite = row-wise mean (skipna) of the three z-scored sub-metrics.
Higher = cheaper = more attractive.

Column aliases (etoro.csv):
    PET: trailing P/E  ('PE' or 'PET')
    PEF: forward  P/E  ('PEF' or 'PE_FWD')
    PB:  price/book    ('PB')
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore


def compute(df: pd.DataFrame) -> pd.Series:
    """Value factor: row-wise mean of z-scored earnings yields + book yield."""
    idx = df.index
    parts = []

    def _col(*candidates: str) -> pd.Series | None:
        for c in candidates:
            if c in df.columns:
                return df[c]
        return None

    pet = _col("PE", "PET")
    pef = _col("PEF", "PE_FWD")
    pb = _col("PB")

    def _yield(price_ratio: pd.Series | None) -> pd.Series | None:
        if price_ratio is None:
            return None
        ey = pd.Series(np.where(price_ratio > 0, 1.0 / price_ratio, np.nan), index=idx)
        return zscore(ey)

    for raw in (pet, pef, pb):
        z = _yield(raw)
        if z is not None:
            parts.append(z)

    if not parts:
        return pd.Series(np.nan, index=idx)

    mat = pd.concat(parts, axis=1)
    return mat.mean(axis=1, skipna=True)
