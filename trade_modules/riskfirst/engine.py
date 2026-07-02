"""riskfirst engine — combine factors, select, construct (risk-first), recommend.

Factors are injected as callables ``df -> pd.Series`` so this core is testable on
its own. The actual wiring of the five factor modules + live data + the edge gate
lives in the shadow runner.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def composite_score(
    df: pd.DataFrame,
    factor_fns: list[Callable[[pd.DataFrame], pd.Series]],
    weights: list[float] | None = None,
) -> pd.Series:
    """Mean (or weighted mean) of the per-factor z-scores, row-wise, NaN-skipping.

    Each factor_fn is called as ``fn(df) -> pd.Series`` aligned to df.index.
    When weights is None, equal weights are used. Weights are normalized to sum=1.
    NaN from individual factors is skipped in the mean (a name with at least one
    valid factor gets a composite; all-NaN → NaN).
    """
    if not factor_fns:
        return pd.Series(np.nan, index=df.index)

    scores = [fn(df).reindex(df.index) for fn in factor_fns]

    if weights is not None:
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        # Weighted mean, NaN-skipping: accumulate weighted values and weight sums
        num = pd.Series(0.0, index=df.index)
        den = pd.Series(0.0, index=df.index)
        for s, wi in zip(scores, w, strict=True):
            mask = s.notna()
            num = num.where(~mask, num + wi * s)
            den = den.where(~mask, den + wi)
        result = num / den
        result = result.where(den > 0, other=np.nan)
    else:
        mat = pd.concat(scores, axis=1)
        result = mat.mean(axis=1, skipna=True)
        # Rows where ALL factors are NaN must be NaN, not 0.0
        all_nan = mat.isna().all(axis=1)
        result = result.where(~all_nan, other=np.nan)

    return result
