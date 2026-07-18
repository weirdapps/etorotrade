"""BUILD ⑥a (2026-07-18): tiered, copier-aware ADV liquidity gate for v3.

v3 previously gated liquidity with a single flat $1M floor applied inline in the
report. After ④ expanded the scored universe ~25x (admitting many smaller / less-
liquid names), the floor should scale with market-cap tier and — for a copied book —
with a copier multiplier. This gate uses v3's own ``adv_usd`` (avg_volume x price x
fx, already computed in features) so it costs NO extra network fetch, unlike
``trade_modules.liquidity_filter`` whose ``get_adv`` fetches per ticker.

Scope: the copier multiplier defaults to 1.0 (single-account = the tiered floor).
Deriving a real multiplier from copier AUM is a proper capacity-model question,
deferred; the parameter is the hook. Held names and unknown-ADV names are never
dropped.
"""

from __future__ import annotations

import math

import pandas as pd

from trade_modules.liquidity_filter import TIER_MIN_ADV

# Market-cap tier thresholds (USD). ``load_universe`` floors cap at $500M, so in
# practice only SMALL..MEGA appear; MICRO is kept for completeness / held names.
_CAP_TIERS: tuple[tuple[float, str], ...] = (
    (2e11, "MEGA"),  # >= $200B
    (1e10, "LARGE"),  # >= $10B
    (2e9, "MID"),  # >= $2B
    (3e8, "SMALL"),  # >= $300M
)


def cap_tier(cap_usd) -> str:
    """Market-cap tier for a USD market cap. Unknown/NaN cap -> neutral ``MID``."""
    try:
        c = float(cap_usd)
    except (TypeError, ValueError):
        return "MID"
    if math.isnan(c):
        return "MID"
    for threshold, name in _CAP_TIERS:
        if c >= threshold:
            return name
    return "MICRO"


def required_adv(cap_usd, *, copier_multiplier: float = 1.0) -> float:
    """Minimum acceptable ADV (USD/day): the tier floor scaled by the copier
    multiplier (clamped at >= 1.0 — copiers only ever raise the liquidity bar)."""
    return TIER_MIN_ADV[cap_tier(cap_usd)] * max(1.0, float(copier_multiplier))


def liquidity_gate(
    scores: pd.DataFrame,
    *,
    copier_multiplier: float = 1.0,
    held_col: str = "is_portfolio",
    adv_col: str = "adv_usd",
    cap_col: str = "cap",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``scores`` into ``(kept, dropped)`` on the tiered ADV floor.

    A row is DROPPED only when it is (a) not held, (b) has a KNOWN ADV, and (c) that
    ADV is below its tier's (copier-scaled) floor. Held names, unknown-ADV names, and
    (defensively) a frame lacking the ADV/cap columns are always kept.
    """
    if adv_col not in scores.columns or cap_col not in scores.columns:
        return scores, scores.iloc[0:0]

    adv = pd.to_numeric(scores[adv_col], errors="coerce")
    held = (
        scores[held_col].fillna(False).astype(bool)
        if held_col in scores.columns
        else pd.Series(False, index=scores.index)
    )
    floor = scores[cap_col].map(lambda c: required_adv(c, copier_multiplier=copier_multiplier))
    drop = (~held) & adv.notna() & (adv < floor)
    return scores[~drop].copy(), scores[drop].copy()
