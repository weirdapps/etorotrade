"""VALUE factor — cross-sectional cheapness score.

Sub-metrics:
    earnings_yield_trailing  = 1 / PET    (only when PET > 0, else NaN)
    earnings_yield_forward   = 1 / PEF    (only when PEF > 0, else NaN)
    fcf_yield                = FCF        (already a yield %; higher = better)
    sales_yield              = 1 / (P/S)  (only when P/S > 0, else NaN)

Composite: row-wise mean of zscore(sub_metric) across available sub-metrics.
Higher composite = cheaper = more attractive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore


def compute(df: pd.DataFrame) -> pd.Series:
    """Return a cross-sectional value z-score aligned to df.index.

    Parameters
    ----------
    df:
        DataFrame indexed by ticker; expected columns PET, PEF, FCF, P/S.
        Any missing column is silently treated as an all-NaN sub-metric.

    Returns
    -------
    pd.Series
        Cross-sectional z-score (higher = cheaper = more attractive).
        NaN for a name if ALL sub-metrics are NaN for that name.
    """
    sub_metrics: list[pd.Series] = []

    # earnings yield trailing = 1 / PET  (PET > 0 only)
    if "PET" in df.columns:
        pet = df["PET"].copy().astype(float)
        ey_trailing = pd.Series(np.where(pet > 0, 1.0 / pet, np.nan), index=df.index)
        sub_metrics.append(zscore(ey_trailing))

    # earnings yield forward = 1 / PEF  (PEF > 0 only)
    if "PEF" in df.columns:
        pef = df["PEF"].copy().astype(float)
        ey_forward = pd.Series(np.where(pef > 0, 1.0 / pef, np.nan), index=df.index)
        sub_metrics.append(zscore(ey_forward))

    # FCF yield — already a yield %; higher = better, no transformation needed
    if "FCF" in df.columns:
        sub_metrics.append(zscore(df["FCF"].astype(float)))

    # sales yield = 1 / (P/S)  (P/S > 0 only)
    if "P/S" in df.columns:
        ps = df["P/S"].copy().astype(float)
        sales_yield = pd.Series(np.where(ps > 0, 1.0 / ps, np.nan), index=df.index)
        sub_metrics.append(zscore(sales_yield))

    # No value columns present at all → all-NaN
    if not sub_metrics:
        return pd.Series(np.nan, index=df.index)

    # Stack sub-metric z-scores as columns and average row-wise (skipna=True)
    stacked = pd.concat(sub_metrics, axis=1)
    return stacked.mean(axis=1, skipna=True)
