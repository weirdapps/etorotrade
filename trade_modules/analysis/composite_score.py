"""Multi-factor composite score for cross-sectional stock ranking (CIO v44).

Combines five factor dimensions into a single percentile-ranked score.
Designed for the 1M-1Y holding horizon where momentum and revisions
dominate (Jegadeesh-Titman 1993, Bernard-Thomas 1989).

SHADOW MODE: This score is captured in the signal log for forward
validation. It does NOT change the BS column or any live signal.
Promotion to primary signal requires out-of-sample evidence that
composite_score predicts T+60/90/180 returns better than BS.

Default weights reflect academic consensus on factor premia magnitude:
- Momentum (30%): strongest 3-12m predictor (Asness et al. 2013)
- Revisions (20%): earnings estimate drift / PEAD
- Quality (20%): ROE + FCF + low leverage (QMJ, Asness et al. 2019)
- Value (15%): inverse PE + FCF yield
- Analyst upside (15%): capped to prevent mean-reversion dominance
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "momentum": 0.30,
    "revision": 0.20,
    "quality": 0.20,
    "value": 0.15,
    "analyst": 0.15,
}

# Cap analyst upside at 50% to prevent extreme targets from dominating
ANALYST_UPSIDE_CAP = 50.0


def compute_composite_scores(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute cross-sectional composite factor scores.

    Each factor is percentile-ranked (0-100) across the universe,
    then weighted-summed into a composite score. Higher = better.

    Required columns (gracefully handles missing with NaN):
        momentum_12_1m: 12-1 month return (skip-month)
        analyst_momentum: 3-month change in %BUY (AM field)
        return_on_equity: ROE %
        debt_to_equity: D/E %
        fcf_yield: Free cash flow yield %
        pe_forward: Forward P/E
        upside: Analyst price target upside %

    Returns:
        DataFrame with added columns:
        - r_momentum, r_revision, r_quality, r_value, r_analyst (percentile ranks)
        - composite_score (0-100 weighted sum)
        - composite_quintile (1=top, 5=bottom)
    """
    w = weights or DEFAULT_WEIGHTS
    result = df.copy()
    n = len(result)

    if n < 5:
        result["composite_score"] = np.nan
        result["composite_quintile"] = np.nan
        return result

    # Factor 1: Momentum (higher = better)
    result["r_momentum"] = _safe_rank(result, "momentum_12_1m")

    # Factor 2: Revisions / earnings drift (higher AM = better)
    result["r_revision"] = _safe_rank(result, "analyst_momentum")

    # Factor 3: Quality (composite of ROE, low leverage)
    roe_rank = _safe_rank(result, "return_on_equity")
    de_rank = _safe_rank(result, "debt_to_equity", ascending=True)  # lower D/E = better
    fcf_rank = _safe_rank(result, "fcf_yield")
    result["r_quality"] = (roe_rank + de_rank + fcf_rank) / 3

    # Factor 4: Value (inverse PE + FCF yield)
    result["_inv_pe"] = 1 / result["pe_forward"].clip(lower=0.5)
    pe_rank = _safe_rank(result, "_inv_pe")
    fcf_val_rank = _safe_rank(result, "fcf_yield")
    result["r_value"] = (pe_rank + fcf_val_rank) / 2
    result.drop(columns=["_inv_pe"], inplace=True)

    # Factor 5: Analyst upside (capped to prevent mean-reversion dominance)
    result["_capped_upside"] = result.get("upside", pd.Series(dtype=float)).clip(
        upper=ANALYST_UPSIDE_CAP
    )
    result["r_analyst"] = _safe_rank(result, "_capped_upside")
    result.drop(columns=["_capped_upside"], inplace=True)

    # Weighted composite
    result["composite_score"] = (
        w.get("momentum", 0) * result["r_momentum"].fillna(50)
        + w.get("revision", 0) * result["r_revision"].fillna(50)
        + w.get("quality", 0) * result["r_quality"].fillna(50)
        + w.get("value", 0) * result["r_value"].fillna(50)
        + w.get("analyst", 0) * result["r_analyst"].fillna(50)
    )

    # Quintile (1=top 20%, 5=bottom 20%)
    result["composite_quintile"] = pd.qcut(
        result["composite_score"],
        q=5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop",
    ).astype(float)

    return result


def _safe_rank(df: pd.DataFrame, col: str, ascending: bool = False) -> pd.Series:
    """Percentile rank (0-100) that gracefully handles missing columns/values."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.rank(pct=True, ascending=not ascending, na_option="bottom") * 100
