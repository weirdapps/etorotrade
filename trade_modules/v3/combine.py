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
    "gp_assets": +1,  # gross profitability / assets (Novy-Marx) — added 2026-07-19
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
    # earnings surprise / PEAD + investment (2026-07-19)
    "sue": +1,  # standardized unexpected earnings (post-earnings drift; high is good)
    "asset_growth": -1,  # CMA / investment (low is good) — monitored, not scored
    # earnings trajectory (2026-07-23, owner: PEF/PET) = forward/trailing P/E ratio (LOW =
    # forward cheaper = earnings expected to RISE; high = value-trap). Renamed+inverted from
    # PET/PEF so "smaller is better" -> DIRECTION -1. Strongest survivorship-clean signal.
    "earn_trajectory": -1,
    # net share issuance (2026-07-21): dilution bad / buyback good (Pontiff-Woodgate).
    "net_issuance": -1,
    # earnings stability (2026-07-21): low earnings coefficient-of-variation = QMJ 'safety'.
    "earn_stability": +1,
}

# Scoring clusters -> member metrics.
CLUSTERS = {
    # pe_forward + ps_sector pruned 2026-07-14: near-duplicates of pe_trailing
    # (rho 0.75/0.77) and of each other (pe_forward~ps_sector 0.98) -> double-counted
    # valuation. Kept a diverse trio: earnings (pe_trailing), book (pb), enterprise
    # (ev_ebitda). Pruned metrics still SHOWN as raw context in the card, not scored.
    # FIX-NOW 2026-07-18 (D7): raw P/B dropped (degrades in an intangible-heavy
    # mega-cap-tech universe — shorts the R&D/brand compounders). Anchor on
    # EV/EBITDA + trailing earnings yield. Intangibles-adjusted book -> BUILD.
    "value_z": ["ps_sector", "pe_forward", "pb"],
    # quality = ROE + FCF + GP/assets. GP/assets (Novy-Marx) added 2026-07-19 once the
    # Sharadar SF1 PIT data was in — it was the best survivorship-clean raw IC (t 3.5)
    # AND the documented profitability anchor. de + current_ratio stay a distress
    # filter; accruals shadow-only (no clean alpha).
    "quality_z": ["roe", "fcf", "gp_assets", "net_issuance"],
    # price_perf pruned 2026-07-14: rho 0.95 with mom_12_1 (same price move counted
    # twice). Kept the academic 12-1 skip-month + 52w-high proximity (distinct).
    "momentum_z": ["pct_52w_high"],
    "growth_z": ["earn_growth", "earn_stability"],
    "lowvol_z": ["beta", "realized_vol"],
    # FIX-NOW 2026-07-18 (agreed setup D2-D6): strength collapses to analyst_mom
    # alone. upside (contaminated/contrarian), buy_pct (zero-IC level),
    # short_interest (→ squeeze-exclude gate) and target_dispersion (non-PIT,
    # discarded) are removed from scoring. Raw columns stay in the feature frame.
    "strength_z": ["analyst_mom"],
    # PEAD / earnings surprise (2026-07-19): the most consistent survivorship-clean
    # signal (beta-neutral IC t 2.06, hit 68%) + robust literature (Bernard-Thomas).
    # SUE = seasonal-random-walk standardized unexpected earnings, from SF1 actuals.
    "pead_z": ["sue"],
    # Earnings trajectory (owner 2026-07-23): PEF/PET = forward/trailing P/E ratio (LOW =
    # forward cheaper = earnings expected to RISE; DIRECTION -1). Recovers the forward-looking
    # information dropped when raw pe_forward was pruned (the LEVEL was a near-dup of trailing,
    # zero IC; the SPREAD is not). Best clean signal on the panel: beta-neutral forward IC
    # t 2.17, hit 80%, incremental to value+growth (FM t 1.71). Panel-only (Sharadar has no
    # forward estimates) -> earn-in via the adaptive mechanism as forward IC accrues.
    "trajectory_z": ["earn_trajectory"],
}

# ---------------------------------------------------------------------------
# P1 (2026-07-19): sector-conditional VALUE cluster. The right valuation metric
# differs by sector (banks->P/B, REITs->cash-flow, tech->sales — confirmed by the
# per-sector backtest and Damodaran / Fama-French). Rather than an over-fit
# 55-cell per-sector lookup, we use THREE theory-motivated groups, each with its
# own value recipe. Non-value clusters are UNCHANGED. Guardrails: value is never
# dropped and its sign is never flipped — Group C only DOWN-WEIGHTS it.
# ---------------------------------------------------------------------------
SECTOR_GROUPS = {
    # Group B — asset / cash-flow (earnings distorted; book / cash-flow is the anchor)
    "Financial Services": "B",
    "Real Estate": "B",
    # Group C — growth / intangible (book meaningless, trailing earnings mislead)
    "Technology": "C",
    "Healthcare": "C",
    # Group A — conventional earnings (classic value behaves)
    "Industrials": "A",
    "Consumer Cyclical": "A",
    "Consumer Defensive": "A",
    "Basic Materials": "A",
    "Energy": "A",
    "Communication Services": "A",
    "Utilities": "A",
}
_DEFAULT_GROUP = "A"  # unknown / missing sector -> conventional recipe

# Per-group value recipe: {metric -> weight within the value cluster}. Weights are
# applied over the metrics PRESENT for a row and renormalized, so one present
# metric reproduces its own z (a name is never penalized for a missing metric).
VALUE_GROUP_RECIPES = {
    # 2026-07-21 (owner): P/S default value lead, P/B for financials/REITs, pe_forward
    # PROMOTED in place of pe_trailing (earn-in — yfinance forward EPS, no 10yr PIT);
    # ev_ebitda + pe_trailing dropped (redundant / dead this regime).
    "A": {"ps_sector": 1.0, "pe_forward": 1.0},  # sales + forward-earnings (default)
    "B": {"pb": 1.0, "pe_forward": 0.5},  # book primary; banks/REITs
    "C": {"ps_sector": 1.0, "pe_forward": 0.5},  # sales primary; growth/intangible
}
# Per-group multiplier on the value cluster's WEIGHT in conviction (value is weak /
# sign-inverted in growth/intangible sectors -> halved there, never dropped).
VALUE_WEIGHT_MULT = {"A": 1.0, "B": 1.0, "C": 0.5}

# Every metric that can appear in a value recipe needs a metric-z computed.
_VALUE_CANDIDATES = sorted({m for r in VALUE_GROUP_RECIPES.values() for m in r})

# ---------------------------------------------------------------------------
# Owner taxonomy (2026-07-23): per-metric conviction weights are the SINGLE SOURCE OF
# TRUTH. Every scored metric carries an explicit GLOBAL weight here; the cluster weights
# below are DERIVED as the sum of their members' weights, and the intra-cluster split (see
# compute_scores) is these weights renormalized over the members PRESENT for a row. This
# replaces the old equal-mean-within-cluster approximation, so a declared weight (e.g. ROE
# at 0.10 = 2x the other quality metrics) is what the engine ACTUALLY applies.
#
# Value is the one COMPOSITE cluster: it carries a single cluster weight (_VALUE_WEIGHT) and
# its internal split is sector-conditional (VALUE_GROUP_RECIPES), so it is not listed per
# metric here. Low-vol (beta/realized_vol) is SIZING-only -> weight 0 (BAB: the low-vol
# premium is a beta artifact — size on it, do not score alpha on it).
#
# History: quality-led best-bet prior (2026-07-19/21; GP/assets + net_issuance + PEAD +
# trajectory the survivorship-clean leaders, low-vol -> sizing). 2026-07-23 owner rebalance:
# quality 0.42->0.25 (ROE 0.10 kept as the quality lead, 2x the FCF/GP/net_issuance 0.05
# each); SUE + PEF/PET raised to 0.15 (the two most robust expectation signals); analyst_mom
# 0.10, momentum 0.10, growth 0.10, value 0.15. The adaptive shrinkage mechanism
# (v3/weight_proposal.py) proposes drift toward measured forward IC as data accrues; these
# remain a PRIOR, applied only by human decision.
# ---------------------------------------------------------------------------
METRIC_WEIGHTS: dict[str, float] = {
    # quality (0.25): ROE leads at 2x the others
    "roe": 0.10,
    "fcf": 0.05,
    "gp_assets": 0.05,
    "net_issuance": 0.05,
    # growth (0.10)
    "earn_growth": 0.05,
    "earn_stability": 0.05,
    # momentum (0.10)
    "pct_52w_high": 0.10,
    # PEAD / earnings surprise (0.15)
    "sue": 0.15,
    # earnings trajectory PEF/PET (0.15)
    "earn_trajectory": 0.15,
    # analyst momentum (0.10)
    "analyst_mom": 0.10,
}
_VALUE_WEIGHT = 0.15  # the composite value cluster (sector-conditional recipe)

# Cluster weights are DERIVED: value = _VALUE_WEIGHT; every other cluster = the sum of its
# members' METRIC_WEIGHTS (metrics absent from METRIC_WEIGHTS contribute 0 -> low-vol = 0).
CLUSTER_WEIGHTS = {
    cluster: (
        _VALUE_WEIGHT
        if cluster == "value_z"
        else round(sum(METRIC_WEIGHTS.get(m, 0.0) for m in members), 10)
    )
    for cluster, members in CLUSTERS.items()
}

# The Value+Quality joint weight is capped here regardless of the weighting scheme.
_VALUE_QUALITY = ("value_z", "quality_z")
_VQ_CAP = 0.60  # raised 0.55->0.60 (2026-07-21): the quality-led set puts value+quality at 0.58

# The five "evidence" clusters the IC / default weighting operates over. Growth
# is deliberately EXCLUDED here: it is off (weight 0) in the default + IC paths
# and only activated through the explicit ``cluster_weights`` arg, so the IC
# redistribution math (and its unit tests) are unchanged.
_IC_CLUSTERS = [
    "value_z",
    "quality_z",
    "momentum_z",
    "lowvol_z",
    "strength_z",
    "pead_z",
    "trajectory_z",
]

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
_ELIG_CLUSTERS = [
    "value_z",
    "quality_z",
    "momentum_z",
    "growth_z",
    "lowvol_z",
    "strength_z",
    "pead_z",
    "trajectory_z",
]
_MIN_CLUSTERS = 3

# Value-trap gate (owner rule): a forward P/E materially above trailing means earnings are
# expected to FALL (a value trap / cyclical peak) — the trailing-cheap look is a mirage.
# earn_trajectory = PEF/PET, so "forward > 1.10x trailing" is earn_trajectory > 1.10. Trap
# names are made INELIGIBLE: they never surface as a new BUY, and a held trap trips the
# overlay's weak-name SELL — so a collapsing-earnings name is never recommended as attractive.
# Names with a missing PET or PEF (NaN trajectory) are never gated.
_TRAP_MAX_FWD_TTM = 1.10  # forward P/E may not exceed 1.10x trailing (= PEF/PET ceiling)

# Valuation multiples where a NON-positive value is meaningless as "cheap": a negative or
# zero P/E / P/B / EV-EBITDA is a loss-maker or negative book, NOT a bargain. Cleaned to NaN
# before scoring so their low-is-good direction can never rank them as the cheapest. Owner
# 2026-07-23 (the PEF/PET rework surfaced negative forward P/Es scoring as ultra-cheap).
_POSITIVE_MULTIPLES = frozenset({"pe_trailing", "pe_forward", "ps_sector", "pb", "ev_ebitda"})


def _fwd_loss_trap(out: pd.DataFrame) -> pd.Series:
    """True where trailing earnings are POSITIVE but forward earnings are NON-positive
    (profitable today, a loss expected). The extreme value-trap the PEF/PET ratio cannot
    express — ``earn_trajectory`` is NaN when PEF<=0 — so it is read from the raw signs.
    Owner 2026-07-23: forward earnings turning negative is the opposite of 'price lagging
    RISING earnings', so such a name is made ineligible (and a held one is sold)."""
    if "pe_trailing" not in out.columns or "pe_forward" not in out.columns:
        return pd.Series(False, index=out.index)
    pet = pd.to_numeric(out["pe_trailing"], errors="coerce")
    pef = pd.to_numeric(out["pe_forward"], errors="coerce")
    return (pet > 0) & (pef <= 0)  # NaN on either side -> False (not a trap)


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


def _equal_vol_standardize(zmat: pd.DataFrame) -> pd.DataFrame:
    """Scale each cluster-z column to ~unit cross-sectional std (population, ddof=0).

    FIX-NOW 2026-07-18 (D14): "true equal-vol". Cluster-z's are means of member
    rank-z's, so a cluster with more (or more-correlated) members has a lower
    cross-sectional dispersion and would UNDER-contribute under equal nominal
    weights. Dividing each cluster by its own cross-sectional std makes the
    equal-risk nominal weights genuinely equal-RISK. A zero-/near-zero-std or
    all-NaN column is left unchanged (no divide-by-zero); NaNs are preserved.
    Computed per-snapshot on the current cross-section only -> no look-ahead.
    """
    std = zmat.std(axis=0, ddof=0)
    denom = std.where(std > 1e-12, 1.0)
    return zmat.divide(denom, axis=1)


def _sector_demean(z: pd.Series, sector: pd.Series) -> pd.Series:
    """Subtract the within-sector mean of z. Missing sectors form one fallback
    group (their own subset mean), leaving the global ~0-mean z ~unchanged."""
    grp = sector.fillna("__NA__").astype(str)
    return z - z.groupby(grp).transform("mean")


def _sector_group(sector: pd.Series) -> pd.Series:
    """Map each ticker's sector to its value group (A/B/C); unknown/missing -> A."""
    return sector.astype(str).map(SECTOR_GROUPS).fillna(_DEFAULT_GROUP)


def _conditional_value_z(zframe: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """Per-row ``value_z`` = the group's recipe weighted-mean of present value-z's.

    ``zframe`` holds the directional (already sector-demeaned) ``{metric}_z``
    columns for the value candidates. For each group the recipe weights are applied
    over the metrics PRESENT for that row and renormalized (so a single present
    metric reproduces its own z, and a missing metric is not penalized). Rows whose
    group has no present recipe metric get NaN.
    """
    result = pd.Series(np.nan, index=zframe.index)
    for g, recipe in VALUE_GROUP_RECIPES.items():
        mask = groups == g
        if not mask.any():
            continue
        cols = [f"{m}_z" for m in recipe if f"{m}_z" in zframe.columns]
        if not cols:
            continue
        ws = np.array([recipe[m] for m in recipe if f"{m}_z" in zframe.columns], dtype=float)
        sub = zframe.loc[mask, cols]
        wmat = pd.DataFrame(np.tile(ws, (int(mask.sum()), 1)), index=sub.index, columns=cols).where(
            sub.notna()
        )
        wsum = wmat.sum(axis=1)
        result.loc[mask] = (sub * wmat).sum(axis=1) / wsum.where(wsum > 0)
    return result


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
    # Value-trap gate: forward P/E > 1.10x trailing (earn_trajectory = PEF/PET above 1.10).
    if "earn_trajectory" in out.columns:
        traj = pd.to_numeric(out["earn_trajectory"], errors="coerce")
        ok &= ~(traj > _TRAP_MAX_FWD_TTM)  # NaN trajectory (missing PET/PEF) never gated
    # Forward-loss trap: profitable now but forward earnings expected NON-positive (owner
    # 2026-07-23) — the ratio is NaN there, so gate on the raw PET>0 & PEF<=0 signs.
    ok &= ~_fwd_loss_trap(out)
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

    # Per-metric directional z (+ optional sector demeaning), for every cluster
    # member PLUS the sector-conditional value candidates (pb, ps_sector, ...).
    metrics_needing_z = {m for members in CLUSTERS.values() for m in members}
    metrics_needing_z.update(_VALUE_CANDIDATES)
    for m in metrics_needing_z:
        if m not in out.columns:
            continue
        raw = out[m]
        if m in _POSITIVE_MULTIPLES:
            # A non-positive valuation multiple = a loss-maker / negative book, NOT cheap.
            # Exclude it so DIRECTION -1 can't flip it into the best (cheapest) score.
            raw = pd.to_numeric(raw, errors="coerce")
            raw = raw.where(raw > 0)
        z = _rank_z(raw) * DIRECTION.get(m, 1)
        if sector_neutral:
            z = _sector_demean(z, sector)
        out[f"{m}_z"] = z

    # Non-value clusters: cluster z = METRIC_WEIGHTS-weighted mean of member metric-z,
    # renormalized over the members PRESENT for each row (a missing metric is never a
    # penalty; equal weights reproduce the plain mean). This is what makes the declared
    # per-metric weights — e.g. ROE at 2x the other quality metrics — actually apply.
    for cluster, members in CLUSTERS.items():
        if cluster == "value_z":
            continue
        zcols = [f"{m}_z" for m in members if f"{m}_z" in out.columns]
        if not zcols:
            out[cluster] = np.nan
            continue
        ws = np.array([METRIC_WEIGHTS.get(c[:-2], 1.0) for c in zcols], dtype=float)
        sub = out[zcols]
        wmat = pd.DataFrame(np.tile(ws, (len(sub), 1)), index=sub.index, columns=zcols).where(
            sub.notna()
        )
        wsum = wmat.sum(axis=1)
        out[cluster] = (sub * wmat).sum(axis=1) / wsum.where(wsum > 0)

    # Value cluster: sector-conditional recipe (banks->P/B, REITs->cash-flow,
    # tech->sales) instead of one uniform P/E + EV/EBITDA mean.
    groups = _sector_group(sector)
    out["value_z"] = _conditional_value_z(out, groups)

    # Weighted conviction, renormalized per-row over the clusters actually present.
    cluster_names = list(CLUSTERS.keys())
    zmat = out[cluster_names]
    # Equal-VOL standardize the cluster-z's so the equal-risk nominal weights are
    # truly equal-RISK (the stored {cluster}_z columns stay RAW; only conviction
    # uses the standardized matrix).
    zmat_ev = _equal_vol_standardize(zmat)
    weights = pd.Series(resolve_cluster_weights(cluster_weights, ic_weights, cluster_names))[
        cluster_names
    ]
    wmat = pd.DataFrame(
        np.tile(weights.to_numpy(), (len(out), 1)),
        index=out.index,
        columns=cluster_names,
    ).where(zmat_ev.notna())
    # P1: down-weight the value cluster per-row in growth/intangible sectors
    # (Group C x0.5). NaN value weights (value_z absent) stay NaN.
    wmat["value_z"] = wmat["value_z"] * groups.map(VALUE_WEIGHT_MULT).fillna(1.0)
    wsum = wmat.sum(axis=1)
    conv_raw = (zmat_ev * wmat).sum(axis=1) / wsum.where(wsum > 0)

    # Eligibility gate: ineligible names never enter the ranked cross-section,
    # so their conviction is NaN (which yields a NaN rank) and the re-z baseline
    # is the eligible equity universe only.
    eligible = _eligibility(out)
    out["eligible"] = eligible
    out["conviction"] = _z_plain(conv_raw.where(eligible))
    out["rank"] = out["conviction"].rank(ascending=False, method="min").astype("Int64")
    return out
