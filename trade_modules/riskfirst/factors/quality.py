"""QUALITY factor — profitable, low-leverage, cash-generative.

Composite = row-wise mean (skipna) of z-scored sub-metrics:

    zscore(ROE)   return on equity        (higher = better)
    zscore(-DE)   inverse debt-to-equity  (lower leverage = better)
    zscore(FCF)   FCF yield               (higher = better)

Column aliases (etoro.csv): ROE, DE, FCFY / FCF_YIELD / FCF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore

_SUB_METRICS: list[tuple[str, bool]] = [
    # (column_candidate_list, negate)
]


def compute(df: pd.DataFrame) -> pd.Series:
    """Quality factor: row-wise mean of z-scored ROE, inverse D/E, FCF yield."""
    idx = df.index
    parts = []

    def _col(*candidates: str) -> pd.Series | None:
        for c in candidates:
            if c in df.columns:
                return df[c]
        return None

    roe = _col("ROE")
    de = _col("DE", "DEBT_EQUITY", "D/E")
    fcf = _col("FCFY", "FCF_YIELD", "FCF")

    if roe is not None:
        parts.append(zscore(roe))
    if de is not None:
        # negate: lower leverage = higher quality score
        parts.append(zscore(-de))
    if fcf is not None:
        parts.append(zscore(fcf))

    if not parts:
        return pd.Series(np.nan, index=idx)

    mat = pd.concat(parts, axis=1)
    return mat.mean(axis=1, skipna=True)
