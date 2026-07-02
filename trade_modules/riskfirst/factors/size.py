"""SIZE factor — small-cap tilt (size premium).

Higher z-score means SMALLER market cap (more attractive under the size premium).

compute() contract:
- Input df must have a 'CAP' column with market-cap values (numeric or string
  like '$5.2B').  Falls back to a NaN series when absent.
- Negates log(cap) so that smaller caps score higher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.stats import zscore


def _parse_market_cap(s: pd.Series) -> pd.Series:
    """Coerce market-cap column to float (handles '$5.2B', '1.3T', plain floats)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    out = []
    for v in s:
        if pd.isna(v):
            out.append(float("nan"))
            continue
        v = str(v).strip().replace("$", "").replace(",", "")
        if v and v[-1].upper() in multipliers:
            try:
                out.append(float(v[:-1]) * multipliers[v[-1].upper()])
            except ValueError:
                out.append(float("nan"))
        else:
            try:
                out.append(float(v))
            except ValueError:
                out.append(float("nan"))
    return pd.Series(out, index=s.index)


def compute(df: pd.DataFrame) -> pd.Series:
    """Size factor: z-score of NEGATIVE log market cap (smaller = higher score)."""
    idx = df.index

    for col in ("CAP", "MARKET_CAP", "MARKETCAP"):
        if col in df.columns:
            cap = _parse_market_cap(df[col])
            log_cap = np.log(cap.clip(lower=1.0))  # clip to avoid log(0)
            return zscore(-pd.Series(log_cap, index=idx))

    return pd.Series(np.nan, index=idx)
