"""SIZE factor — small-cap tilt (size premium).

Higher z-score means SMALLER market cap (more attractive under the size premium).

compute() contract:
- Input df must have a 'CAP' column with market-cap strings ("1.2T"/"500B"/"3.4M").
- Returns a pd.Series aligned to df.index.
- market_cap <= 0 or NaN -> NaN for that row (no raises).
- Missing 'CAP' column -> all-NaN Series aligned to df.index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.analysis.tiers import _parse_market_cap
from trade_modules.riskfirst.stats import zscore


def compute(df: pd.DataFrame) -> pd.Series:
    """Compute the SIZE factor z-score for each row in df.

    Smaller market cap -> higher z-score (size premium convention).
    """
    nan_series = pd.Series(np.nan, index=df.index)

    if "CAP" not in df.columns:
        return nan_series

    # Parse each CAP string to a float; _parse_market_cap returns 0.0 for invalid entries.
    caps = df["CAP"].apply(_parse_market_cap)

    # Guard: non-positive or NaN -> NaN (log undefined for <= 0).
    valid_mask = caps > 0  # False for 0.0 (invalid parse result) and NaN
    caps = caps.where(valid_mask, other=np.nan)

    # Small cap premium: negate log so smaller cap yields a larger value before z-scoring.
    raw = -np.log(caps)

    # zscore is NaN-safe for inputs, but its constant-series fallback returns 0.0 for all
    # rows (including originally-NaN ones). Re-apply the mask so invalid rows stay NaN.
    result = zscore(raw)
    result = result.where(valid_mask, other=np.nan)
    return result
