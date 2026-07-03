"""QUALITY factor — profitable, low-leverage, cash-generative.

Composite = row-wise mean (skipna) of z-scored sub-metrics:

    zscore(ROE)   return on equity        (higher = better)
    zscore(-DE)   debt/equity inverted    (lower DE = higher score)
    zscore(FCF)   free-cash-flow yield    (higher = better)
    zscore(EG)    earnings growth %       (higher = better)

Each sub-metric is z-scored independently (NaN-safe, winsorized) before
averaging, so a ticker missing one metric is scored on the remaining ones.
Missing columns are silently skipped. Result is HIGHER = HIGHER QUALITY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore

# Sub-metrics: (column_name, negate_before_zscore)
_SUB_METRICS: list[tuple[str, bool]] = [
    ("ROE", False),
    ("DE", True),  # invert: lower leverage = better
    ("FCF", False),
    ("EG", False),
]


def compute(df: pd.DataFrame) -> pd.Series:
    """Compute QUALITY factor scores aligned to df.index.

    Args:
        df: DataFrame indexed by ticker; expected columns: ROE, DE, FCF, EG.

    Returns:
        pd.Series indexed like df, float64. Higher = higher quality.
        NaN where all sub-metrics are missing for a ticker.
        Never raises on missing or partial columns.
    """
    if df.empty:
        return pd.Series(dtype=float)

    z_cols: list[pd.Series] = []
    for col, negate in _SUB_METRICS:
        if col not in df.columns:
            continue
        raw = df[col].astype(float)
        scored = zscore(-raw if negate else raw)
        # Re-apply the original NaN mask: zscore() turns an all-NaN column into
        # all-zeros (constant-series guard), which would falsely contribute 0 to
        # a ticker that had no data for this sub-metric.
        scored = scored.where(raw.notna(), other=np.nan)
        z_cols.append(scored)

    if not z_cols:
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Stack z-scores into a frame and take row-wise mean (skipna=True so
    # tickers missing some sub-metrics use only the ones they have).
    z_frame = pd.concat(z_cols, axis=1)
    return z_frame.mean(axis=1, skipna=True)
