"""Momentum factor (snapshot proxy).

LIMITATION — this is a *proxy* implementation that operates on the snapshot
columns available in etoro.csv.  The canonical momentum factor would be the
12-month minus 1-month (skip-month) cumulative return, scaled by realised
volatility over the same window (vol-scaled 12-1 momentum).  That requires a
price-history matrix that is not available in the snapshot.

TODO: wire in price-history (OHLCV from yahoofinance or eToro market-data API)
and replace the proxy with the proper vol-scaled 12-1 momentum signal.

Proxy inputs (from contract in factors/__init__.py):
    52W  — % of 52-week high (higher = stronger recent price momentum)
    AM   — analyst momentum in pp (higher = improving analyst sentiment)

Composite = mean of zscore(52W) and zscore(AM), computed row-wise and skipping
NaN so that a row missing one component still gets a score from the other.
Higher score = more momentum (consistent with the HIGHER-IS-BETTER convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore

# Columns used as momentum proxies
_COL_52W = "52W"
_COL_AM = "AM"


def compute(df: pd.DataFrame) -> pd.Series:
    """Return a cross-sectional momentum z-score aligned to ``df.index``.

    Parameters
    ----------
    df:
        Universe DataFrame indexed by ticker.  Must contain at least one of
        ``52W`` (% of 52-week high) or ``AM`` (analyst momentum in pp).
        Any other columns are ignored.

    Returns
    -------
    pd.Series
        Z-score indexed identically to ``df``.  Rows where all proxy inputs are
        NaN receive NaN.  Never raises on missing columns.
    """
    components: list[pd.Series] = []

    if _COL_52W in df.columns:
        raw = pd.to_numeric(df[_COL_52W], errors="coerce")
        z = zscore(raw)
        # Restore NaN for rows that were NaN in the original input; zscore()
        # returns 0.0 on zero-variance / all-NaN inputs to avoid inf, but those
        # 0.0s should not masquerade as valid momentum scores.
        components.append(z.where(raw.notna()))

    if _COL_AM in df.columns:
        raw = pd.to_numeric(df[_COL_AM], errors="coerce")
        z = zscore(raw)
        components.append(z.where(raw.notna()))

    if not components:
        # No proxy columns present — return all-NaN aligned to the index
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Stack into a DataFrame and average row-wise, skipping NaN so a single
    # valid component is not penalised when the other is missing.
    stacked = pd.concat(components, axis=1)
    return stacked.mean(axis=1, skipna=True).where(stacked.notna().any(axis=1))
