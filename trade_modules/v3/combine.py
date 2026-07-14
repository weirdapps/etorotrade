"""v3 factor combiner.

Turns the enriched feature frame into cluster z-scores and a single
conviction score:

  raw metric -> rank-normal (van der Waerden) cross-sectional score
             -> directional sign (low-is-good metrics negated)
             -> optional sector-neutral demeaning
             -> cluster z (mean of member metric-z, skipping NaN)
             -> weighted conviction (Value + Quality jointly ~55%)
             -> re-z cross-sectionally + integer rank (1 = best)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Metric direction: +1 = high is good (keep), -1 = low is good (negate).
DIRECTION = {
    # value (low is good)
    "pe_trailing": -1,
    "pe_forward": -1,
    "ps_sector": -1,
    "pb": -1,
    "ev_ebitda": -1,
    # quality (high is good; leverage and accruals negated)
    "roe": +1,
    "roa": +1,
    "gross_margin": +1,
    "op_margin": +1,
    "fcf": +1,
    "current_ratio": +1,
    "de": -1,
    "accruals": -1,  # high accruals = lower earnings quality = BAD
    # momentum (high is good)
    "mom_12_1": +1,
    "price_perf": +1,
    "pct_52w_high": +1,
    # growth (high is good)
    "earn_growth": +1,
    "rev_growth": +1,
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
    # pe_forward + ps_sector pruned 2026-07-14: near-duplicates of pe_trailing
    # (rho 0.75/0.77) and of each other (pe_forward~ps_sector 0.98) -> double-counted
    # valuation. Kept a diverse trio: earnings (pe_trailing), book (pb), enterprise
    # (ev_ebitda). Pruned metrics still SHOWN as raw context in the card, not scored.
    "value_z": ["pe_trailing", "pb", "ev_ebitda"],
    "quality_z": [
        "roe",
        "roa",
        "gross_margin",
        "op_margin",
        "fcf",
        "current_ratio",
        "de",
        "accruals",
    ],
    # price_perf pruned 2026-07-14: rho 0.95 with mom_12_1 (same price move counted
    # twice). Kept the academic 12-1 skip-month + 52w-high proximity (distinct).
    "momentum_z": ["mom_12_1", "pct_52w_high"],
    "growth_z": ["earn_growth", "rev_growth"],
    "lowvol_z": ["beta", "realized_vol"],
    "strength_z": ["analyst_mom", "upside", "buy_pct", "short_interest", "target_dispersion"],
}

# Cluster weights: Value + Quality = 0.55 (the ~55% joint cap), rest sum to 0.45.
# Growth carries ZERO weight by default — it exists as a 6th cluster but is OFF
# in the default / IC path (backward-compat: the default model is unchanged).
# The growth rebalance is exposed only via the explicit ``cluster_weights`` arg.
CLUSTER_WEIGHTS = {
    "value_z": 0.275,
    "quality_z": 0.275,
    "momentum_z": 0.20,
    "growth_z": 0.0,
    "lowvol_z": 0.15,
    "strength_z": 0.10,
}

# The Value+Quality joint weight is capped here regardless of the weighting scheme.
_VALUE_QUALITY = ("value_z", "quality_z")
_VQ_CAP = 0.55

# The five "evidence" clusters the IC / default weighting operates over. Growth
# is deliberately EXCLUDED here: it is off (weight 0) in the default + IC paths
# and only activated through the explicit ``cluster_weights`` arg, so the IC
# redistribution math (and its unit tests) are unchanged.
_IC_CLUSTERS = ["value_z", "quality_z", "momentum_z", "lowvol_z", "strength_z"]

# Short-name aliases for the explicit ``cluster_weights`` arg: "value" -> "value_z".
_CLUSTER_ALIASES = {c[:-2]: c for c in CLUSTERS}


def _canonical_cluster(key: str) -> str:
    """Map a cluster key to its canonical ``{name}_z`` form.

    Accepts both the z-column name (``"value_z"``) and the short name
    (``"value"``); an unknown key is returned unchanged (it then resolves to a
    zero weight because it matches no cluster).
    """
    key = str(key)
    if key in CLUSTERS:
        return key
    return _CLUSTER_ALIASES.get(key, key)


def resolve_cluster_weights(
    cluster_weights: dict | None,
    ic_weights: dict | None,
    cluster_names: list[str],
) -> dict[str, float]:
    """Resolve the per-cluster weight vector used for conviction.

    Precedence: an explicit ``cluster_weights`` dict wins over ``ic_weights``
    and the default. Explicit weights are read DIRECTLY — keys may be short
    (``"growth"``) or canonical (``"growth_z"``), negatives are clamped to 0,
    missing clusters get 0, and the vector is renormalized to sum 1. A degenerate
    all-zero (or empty) explicit dict falls back to the IC / default resolution.
    When ``cluster_weights`` is ``None`` the fixed / IC weights from
    :func:`derive_cluster_weights` are used (the unchanged default model).
    """
    if cluster_weights:
        provided: dict[str, float] = {}
        for k, v in cluster_weights.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            provided[_canonical_cluster(k)] = max(fv, 0.0)
        resolved = {c: provided.get(c, 0.0) for c in cluster_names}
        total = sum(resolved.values())
        if total > 0:
            return {c: resolved[c] / total for c in cluster_names}
        # degenerate (all-zero / unrecognized) -> fall through to the default.
    return {c: derive_cluster_weights(ic_weights).get(c, 0.0) for c in cluster_names}


def derive_cluster_weights(ic_weights: dict | None = None) -> dict[str, float]:
    """Resolve the per-cluster conviction weights.

    Args:
        ic_weights: Optional ``{cluster -> information coefficient}`` map. When
            ``None`` (the default until IC is measured on-panel) the fixed
            near-equal :data:`CLUSTER_WEIGHTS` are returned unchanged.

    When ``ic_weights`` is supplied, cluster weights are derived ∝ ``max(IC, 0)``
    and renormalized to sum to 1, while **preserving the Value+Quality ≤ 0.55
    cap**: if the IC-implied Value+Quality weight exceeds the cap it is scaled
    back to exactly 0.55 (split within the bucket by IC) and the freed weight is
    redistributed across the other three clusters (by IC). A degenerate vector
    with no positive IC falls back to the fixed :data:`CLUSTER_WEIGHTS`.
    """
    if ic_weights is None:
        return dict(CLUSTER_WEIGHTS)

    names = list(_IC_CLUSTERS)  # growth excluded from IC weighting (off by default)
    raw = {c: max(float(ic_weights.get(c, 0.0)), 0.0) for c in names}
    total = sum(raw.values())
    if total <= 0:  # no positive IC anywhere -> keep the fixed baseline
        return dict(CLUSTER_WEIGHTS)

    w = {c: raw[c] / total for c in names}
    vq = w["value_z"] + w["quality_z"]
    if vq > _VQ_CAP:
        rest = [c for c in names if c not in _VALUE_QUALITY]
        rest_sum = sum(w[c] for c in rest)
        for c in _VALUE_QUALITY:  # scale the VQ bucket down to the cap, by IC
            w[c] = _VQ_CAP * (w[c] / vq)
        target_rest = 1.0 - _VQ_CAP
        if rest_sum > 0:
            for c in rest:
                w[c] = target_rest * (w[c] / rest_sum)
        else:  # all remaining IC non-positive -> spread the freed weight equally
            for c in rest:
                w[c] = target_rest / len(rest)
    w["growth_z"] = 0.0  # growth stays off in the IC path
    return w


# Eligibility: cluster columns whose availability is counted (need >= 3 of 6).
# Growth is included so a name rich in growth data counts toward the minimum;
# the threshold stays at 3 (a name never becomes ineligible by adding a cluster).
_ELIG_CLUSTERS = ["value_z", "quality_z", "momentum_z", "growth_z", "lowvol_z", "strength_z"]
_MIN_CLUSTERS = 3


def _rank_z(s: pd.Series) -> pd.Series:
    """Rank-normal (van der Waerden) cross-sectional score.

    Ranks the non-NaN values (average ranks for ties), maps ranks to (0, 1) via
    k/(n+1), then applies the inverse normal CDF. A name's contribution is bounded
    by its RANK, not its raw magnitude, so a single fat-tailed outlier — a
    small-base +900% earnings-growth artifact or a one-off momentum spike — cannot
    dominate the composite. NaNs are preserved; fewer than two valid points -> NaN.
    """
    x = pd.to_numeric(s, errors="coerce").astype(float)
    valid = x.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    u = valid.rank(method="average") / (len(valid) + 1.0)
    z = pd.Series(norm.ppf(u.to_numpy()), index=valid.index, dtype=float)
    return z.reindex(s.index)


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


def _eligibility(out: pd.DataFrame) -> pd.Series:
    """Boolean per-ticker eligibility for ranking / display.

    A name is eligible when it is an equity with the core data present:
    ``quote_type == "EQUITY"`` AND price present (>0) AND ``mom_12_1`` present
    AND ``realized_vol`` present AND at least three of the five cluster z's are
    non-NaN. Non-equities (ETFs, funds) and dataless names are excluded so
    their equity-factor conviction never leaks into the ranking.

    Frames without a ``quote_type`` column (abstract combiner inputs) are all
    eligible, so the numeric scoring contract is unchanged for callers that do
    not run the yfinance enrichment.
    """
    idx = out.index
    if "quote_type" not in out.columns:
        return pd.Series(True, index=idx)

    qt = out["quote_type"].astype("string").str.upper()
    ok = qt.eq("EQUITY").fillna(False)
    if "price" in out.columns:
        price = pd.to_numeric(out["price"], errors="coerce")
        ok &= price.notna() & (price > 0)
    for col in ("mom_12_1", "realized_vol"):
        if col in out.columns:
            ok &= pd.to_numeric(out[col], errors="coerce").notna()
    clusters_present = out[_ELIG_CLUSTERS].notna().sum(axis=1)
    ok &= clusters_present >= _MIN_CLUSTERS
    return ok.fillna(False).astype(bool)


def compute_scores(
    features: pd.DataFrame,
    sector_neutral: bool = True,
    ic_weights: dict | None = None,
    cluster_weights: dict | None = None,
) -> pd.DataFrame:
    """Add metric-z, cluster-z, conviction and rank columns to ``features``.

    Args:
        features: Per-ticker feature frame (from ``enrich_features``).
        sector_neutral: If True, demean each metric-z within its sector.
        ic_weights: Optional ``{cluster -> information coefficient}`` map. When
            supplied, cluster weights are derived ∝ ``max(IC, 0)`` (renormalized,
            still honoring the Value+Quality ≤ 0.55 cap) via
            :func:`derive_cluster_weights`. Leave ``None`` (the default) to use
            the fixed near-equal :data:`CLUSTER_WEIGHTS`; IC-weighting only
            activates once the ICs have actually been measured on-panel.
        cluster_weights: Optional explicit ``{cluster -> weight}`` map (keys may
            be short like ``"growth"`` or canonical like ``"growth_z"``). When
            given it is used DIRECTLY — normalized to sum 1, missing clusters →
            0 — taking precedence over both ``ic_weights`` and the default. This
            is the ONLY path that activates the Growth cluster; leaving it
            ``None`` reproduces the current model exactly (growth weight 0).

    Returns:
        The input frame plus ``{metric}_z``, the six cluster columns,
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
            z = _rank_z(out[m]) * DIRECTION[m]
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
    weights = pd.Series(resolve_cluster_weights(cluster_weights, ic_weights, cluster_names))[
        cluster_names
    ]
    wmat = pd.DataFrame(
        np.tile(weights.to_numpy(), (len(out), 1)),
        index=out.index,
        columns=cluster_names,
    ).where(zmat.notna())
    wsum = wmat.sum(axis=1)
    conv_raw = (zmat * wmat).sum(axis=1) / wsum.where(wsum > 0)

    # Eligibility gate: ineligible names never enter the ranked cross-section,
    # so their conviction is NaN (which yields a NaN rank) and the re-z baseline
    # is the eligible equity universe only.
    eligible = _eligibility(out)
    out["eligible"] = eligible
    out["conviction"] = _z_plain(conv_raw.where(eligible))
    out["rank"] = out["conviction"].rank(ascending=False, method="min").astype("Int64")
    return out
