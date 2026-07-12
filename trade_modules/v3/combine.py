"""v3 factor combiner.

Turns the enriched feature frame into cluster z-scores and a single
conviction score:

  raw metric -> winsorized (1/99) cross-sectional z
             -> directional sign (low-is-good metrics negated)
             -> optional sector-neutral demeaning
             -> cluster z (mean of member metric-z, skipping NaN)
             -> weighted conviction (Value + Quality jointly ~55%)
             -> re-z cross-sectionally + integer rank (1 = best)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Metric direction: +1 = high is good (keep), -1 = low is good (negate).
DIRECTION = {
    # value (low is good)
    "pe_trailing": -1,
    "pe_forward": -1,
    "ps_sector": -1,
    "pb": -1,
    "ev_ebitda": -1,
    "peg": -1,
    # quality (high is good; leverage negated)
    "roe": +1,
    "roa": +1,
    "gross_margin": +1,
    "op_margin": +1,
    "fcf": +1,
    "current_ratio": +1,
    "de": -1,
    # momentum (high is good)
    "mom_12_1": +1,
    "price_perf": +1,
    "pct_52w_high": +1,
    # low-vol (low is good)
    "beta": -1,
    "realized_vol": -1,
    # strength / analyst (high is good; short interest & target dispersion negated)
    "analyst_mom": +1,
    "upside": +1,
    "buy_pct": +1,
    "short_interest": -1,
    "target_dispersion": -1,
}

# Scoring clusters -> member metrics.
CLUSTERS = {
    "value_z": ["pe_trailing", "pe_forward", "ps_sector", "pb", "ev_ebitda", "peg"],
    "quality_z": ["roe", "roa", "gross_margin", "op_margin", "fcf", "current_ratio", "de"],
    "momentum_z": ["mom_12_1", "price_perf", "pct_52w_high"],
    "lowvol_z": ["beta", "realized_vol"],
    "strength_z": ["analyst_mom", "upside", "buy_pct", "short_interest", "target_dispersion"],
}

# Cluster weights: Value + Quality = 0.55 (the ~55% joint cap), rest sum to 0.45.
CLUSTER_WEIGHTS = {
    "value_z": 0.275,
    "quality_z": 0.275,
    "momentum_z": 0.20,
    "lowvol_z": 0.15,
    "strength_z": 0.10,
}


def _winsor_z(s: pd.Series) -> pd.Series:
    """Winsorize to the 1st/99th pct, then cross-sectional z (ddof=0)."""
    x = pd.to_numeric(s, errors="coerce").astype(float)
    valid = x.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    lo, hi = valid.quantile(0.01), valid.quantile(0.99)
    clipped = x.clip(lo, hi)
    mu = clipped.mean()
    sd = clipped.std(ddof=0)
    if not sd or sd == 0 or np.isnan(sd):
        # No spread: everyone is average (0) where present, NaN where absent.
        return pd.Series(0.0, index=s.index).where(x.notna())
    return (clipped - mu) / sd


def _z_plain(s: pd.Series) -> pd.Series:
    """Plain cross-sectional z (no winsorization) — preserves extreme ordering."""
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not sd or sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index).where(x.notna())
    return (x - mu) / sd


def _sector_demean(z: pd.Series, sector: pd.Series) -> pd.Series:
    """Subtract the within-sector mean of z. Missing sectors form one fallback
    group (their own subset mean), leaving the global ~0-mean z ~unchanged."""
    grp = sector.fillna("__NA__").astype(str)
    return z - z.groupby(grp).transform("mean")


def compute_scores(features: pd.DataFrame, sector_neutral: bool = True) -> pd.DataFrame:
    """Add metric-z, cluster-z, conviction and rank columns to ``features``.

    Args:
        features: Per-ticker feature frame (from ``enrich_features``).
        sector_neutral: If True, demean each metric-z within its sector.

    Returns:
        The input frame plus ``{metric}_z``, the five cluster columns,
        ``conviction`` and ``rank`` (1 = best; NaN conviction -> NaN rank).
    """
    out = features.copy()
    if "sector" in out.columns:
        sector = out["sector"]
    else:
        sector = pd.Series("__NA__", index=out.index)

    # Per-metric directional z (+ optional sector demeaning).
    cluster_zcols: dict[str, list[str]] = {c: [] for c in CLUSTERS}
    for cluster, members in CLUSTERS.items():
        for m in members:
            if m not in out.columns:
                continue
            z = _winsor_z(out[m]) * DIRECTION[m]
            if sector_neutral:
                z = _sector_demean(z, sector)
            zcol = f"{m}_z"
            out[zcol] = z
            cluster_zcols[cluster].append(zcol)

    # Cluster z = mean of member metric-z, skipping NaN.
    for cluster, zcols in cluster_zcols.items():
        out[cluster] = out[zcols].mean(axis=1, skipna=True) if zcols else np.nan

    # Weighted conviction, renormalized per-row over the clusters actually present.
    cluster_names = list(CLUSTERS.keys())
    zmat = out[cluster_names]
    weights = pd.Series(CLUSTER_WEIGHTS)[cluster_names]
    wmat = pd.DataFrame(
        np.tile(weights.to_numpy(), (len(out), 1)),
        index=out.index,
        columns=cluster_names,
    ).where(zmat.notna())
    wsum = wmat.sum(axis=1)
    conv_raw = (zmat * wmat).sum(axis=1) / wsum.where(wsum > 0)

    out["conviction"] = _z_plain(conv_raw)
    out["rank"] = out["conviction"].rank(ascending=False, method="min").astype("Int64")
    return out
