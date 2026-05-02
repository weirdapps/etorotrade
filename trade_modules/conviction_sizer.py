"""
Conviction-Based Position Sizer

Modulates position size based on:
1. BUY conviction score (0-100) from signal scoring
2. VIX regime (reduce size in high volatility)
3. Cost-adjusted expected return (skip if costs exceed alpha)
4. Correlation clusters (reduce size for correlated groups)
5. Sector rotation (adjust conviction for rotating sectors)
6. Data freshness (reduce size for stale data)
7. Market impact (skip if trading costs exceed alpha)
8. Holding-period-adjusted costs (eToro overnight financing)
9. Portfolio-level sector constraints (max sector exposure)

CIO Review Findings:
- E1: BUY conviction score should feed position sizing
- M2: VIX regime adjustments don't touch position sizing
- Task #4: Continuous conviction function
- Task #6: Correlation cluster sizing
- Task #1: Sector rotation integration
- Task #13: Data freshness integration
- Task #2: Market impact modeling
- F2 (v4): Holding-period-adjusted cost model
- F8 (v4): Portfolio-level sizing constraint
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# VIX regime → position size multiplier
# In high volatility, reduce position sizes
REGIME_POSITION_MULTIPLIERS = {
    "low": 1.00,       # Low VIX: normal sizing
    "normal": 1.00,    # Normal VIX: normal sizing
    "elevated": 0.75,  # Elevated VIX: reduce 25%
    "high": 0.50,      # High VIX: reduce 50%
}


def get_conviction_multiplier(conviction_score: float) -> float:
    """
    Get position size multiplier based on BUY conviction score.

    Uses continuous function instead of discrete buckets:
    - 0 score: 0.35 multiplier (35% position)
    - 100 score: 1.00 multiplier (full position)
    - Linear interpolation in between

    Task #4: Replaced discrete CONVICTION_MULTIPLIERS dict with continuous function.

    Args:
        conviction_score: BUY conviction score (0-100)

    Returns:
        Multiplier between 0.35 and 1.0
    """
    score = max(0, min(100, conviction_score))
    return 0.35 + (score / 100) * 0.65


def kelly_fraction(
    expected_return_pct: float,
    realized_vol_pct: float,
    risk_free_pct: float = 0.0,
) -> float:
    """
    CIO v17 R3: Classical Kelly fraction for a single risky asset vs cash.

        f* = (μ - r_f) / σ²

    All inputs are returns expressed as decimals (e.g. 0.05 = 5%).
    Returns 0 when vol is zero or μ ≤ r_f (no positive expectancy).
    """
    if realized_vol_pct <= 0:
        return 0.0
    mu = expected_return_pct / 100.0
    rf = risk_free_pct / 100.0
    sigma = realized_vol_pct / 100.0
    if sigma <= 0:
        return 0.0
    edge = mu - rf
    if edge <= 0:
        return 0.0
    return edge / (sigma * sigma)


def quarter_kelly_size_pct(
    conviction: float,
    atr_pct_daily: float,
    *,
    expected_return_table: Optional[Dict[int, float]] = None,
    horizon_days: int = 30,
    risk_free_apr: float = 0.04,
    fraction: float = 0.25,
    cap_pct: float = 5.0,
    floor_pct: float = 0.0,
) -> Dict[str, float]:
    """
    CIO v17 R3: Convert (conviction, daily ATR%) into a Kelly-sized %.

    Practitioners (Carver "Systematic Trading", Thorp "A Man for All Markets")
    converge on quarter-Kelly because raw Kelly is dominated by σ estimation
    error and produces drawdowns retail accounts can't survive.

    Pipeline:
      μ_30  = expected_return_table[conviction_bucket]   (table-driven)
      σ_30  = atr_pct_daily * sqrt(horizon_days)         (vol scaling)
      f*    = (μ - r_f_30) / σ²
      size  = clamp(fraction * f* * 100, floor, cap)

    The expected_return_table is a conviction-bucket → expected α(T+30) map.
    Defaults are calibrated from the v17 backtest (n=219 T+30):
      ≥70 → 4.74%, 60-69 → 2.14%, 55-59 → 1.51%, 45-54 → 0.04%, <45 → 0.0
    These come straight from the BUY/ADD T+30 alpha table in the v17 review.

    Returns dict with `size_pct`, `kelly_f_full`, `expected_alpha`,
    `realized_vol_horizon`, `cap_active` so the caller can audit the sizing.
    """
    table = expected_return_table or {
        70: 4.74,   # ≥70 conviction
        60: 2.14,   # 60-69
        55: 1.51,   # 55-59
        45: 0.04,   # 45-54
        0: 0.0,     # <45
    }

    # Look up expected α from the table (largest threshold ≤ conviction).
    keys_desc = sorted(table.keys(), reverse=True)
    mu = 0.0
    for k in keys_desc:
        if conviction >= k:
            mu = table[k]
            break

    sigma = max(0.01, atr_pct_daily * math.sqrt(horizon_days))
    rf_horizon = risk_free_apr * 100 * (horizon_days / 365)

    f_full = kelly_fraction(mu, sigma, rf_horizon)
    size_pct = max(floor_pct, min(cap_pct, fraction * f_full * 100))
    cap_active = (fraction * f_full * 100) > cap_pct

    return {
        "size_pct": round(size_pct, 3),
        "kelly_f_full": round(f_full, 4),
        "kelly_fraction": fraction,
        "expected_alpha_horizon": round(mu, 2),
        "realized_vol_horizon": round(sigma, 2),
        "horizon_days": horizon_days,
        "cap_active": cap_active,
    }


# CIO v17 N1: Portfolio-size-aware micro/small-cap allocation cap.
# For a €25K-€250K book, illiquidity is irrelevant and we can run up to 25%
# in small/micro. As the book scales, the same allocation in micro-caps
# becomes a market-impact and exit-liquidity problem. Tiers below mirror
# the institutional rule of thumb that a position should not exceed
# ~10× average daily $-volume for orderly exit.
SMALL_CAP_ALLOCATION_TIERS = (
    # (max_portfolio_value_eur, max_small_micro_pct)
    (100_000, 25.0),
    (500_000, 15.0),
    (float("inf"), 8.0),
)


def get_small_cap_cap(portfolio_value_eur: Optional[float]) -> float:
    """
    CIO v17 N1: Portfolio-size-aware cap on combined SMALL+MICRO allocation.

    Returns the maximum % of the portfolio that may sit in SMALL or MICRO
    cap tickers, given the portfolio value in EUR. None or 0 → 25% (assume
    new/small book).
    """
    if not portfolio_value_eur or portfolio_value_eur <= 0:
        return SMALL_CAP_ALLOCATION_TIERS[0][1]
    for ceiling, cap in SMALL_CAP_ALLOCATION_TIERS:
        if portfolio_value_eur < ceiling:
            return cap
    return SMALL_CAP_ALLOCATION_TIERS[-1][1]


def apply_small_cap_cap(
    new_positions: List[Dict[str, Any]],
    current_small_cap_pct: float,
    portfolio_value_eur: Optional[float],
) -> List[Dict[str, Any]]:
    """
    CIO v17 N1: Constrain new positions in SMALL/MICRO tiers to the
    portfolio-size-aware cap.

    Each position dict needs `cap_tier` ("MEGA"/"LARGE"/"MID"/"SMALL"/"MICRO")
    and `position_size` (treated as %). Positions in MEGA/LARGE/MID pass
    through unchanged. SMALL/MICRO positions are scaled down so that
    `current_small_cap_pct + Σ(new SMALL+MICRO)` ≤ cap. Returns the same
    list with possibly-scaled `position_size` and a `small_cap_cap_applied`
    boolean flag per position.
    """
    cap_pct = get_small_cap_cap(portfolio_value_eur)
    pending = [
        p for p in new_positions
        if str(p.get("cap_tier", "")).upper() in ("SMALL", "MICRO")
    ]
    pending_total = sum(p.get("position_size", 0.0) for p in pending)
    if pending_total <= 0:
        return new_positions

    available = max(0.0, cap_pct - max(0.0, current_small_cap_pct))
    if pending_total <= available:
        # Within the cap — flag but don't scale.
        for p in pending:
            p["small_cap_cap_applied"] = False
            p["small_cap_cap_pct"] = cap_pct
    else:
        scale = available / pending_total if pending_total else 0.0
        for p in pending:
            original = p.get("position_size", 0.0)
            p["position_size"] = round(original * scale, 4)
            p["small_cap_cap_applied"] = True
            p["small_cap_cap_pct"] = cap_pct
            p["small_cap_cap_scale"] = round(scale, 3)
    return new_positions


def get_regime_multiplier(regime: str = "normal") -> float:
    """
    Get position size multiplier based on VIX regime.

    Args:
        regime: VIX regime string (low, normal, elevated, high)

    Returns:
        Multiplier between 0.5 and 1.0
    """
    return REGIME_POSITION_MULTIPLIERS.get(regime.lower(), 1.0)


def get_portfolio_var_scaling(
    portfolio_var_95: Optional[float] = None,
    var_trigger: float = 2.5,
    var_max: float = 5.0,
) -> float:
    """
    Calculate position scaling based on portfolio VaR budget (CIO v3 F10).

    When portfolio VaR exceeds the trigger threshold, all NEW position sizes
    are scaled down proportionally. This prevents the portfolio from
    accumulating risk beyond acceptable limits.

    Args:
        portfolio_var_95: Current portfolio VaR at 95% (daily, as percentage).
            None means VaR is unknown — return 1.0 (no scaling).
        var_trigger: VaR level that triggers scaling (default 2.5%)
        var_max: VaR level that triggers maximum scaling (default 5.0%)

    Returns:
        Scaling factor: 1.0 (within budget) to 0.5 (at var_max).
        0.0 only if VaR exceeds 2× var_max (emergency).
    """
    if portfolio_var_95 is None:
        return 1.0

    if portfolio_var_95 <= var_trigger:
        return 1.0

    if portfolio_var_95 >= var_max * 2:
        logger.warning(
            f"Portfolio VaR {portfolio_var_95:.1f}% exceeds emergency threshold "
            f"{var_max * 2:.1f}% — blocking new positions"
        )
        return 0.0

    if portfolio_var_95 >= var_max:
        return 0.5

    # Linear scaling between trigger and max: 1.0 → 0.5
    fraction = (portfolio_var_95 - var_trigger) / (var_max - var_trigger)
    return round(1.0 - fraction * 0.5, 2)


def get_cluster_size_adjustment(
    ticker: str,
    clusters: List[Dict[str, Any]],
    conviction: float = 50,
) -> float:
    """
    Calculate position size adjustment for correlation clusters.

    CIO Legacy C4: Dampened adjustment that considers conviction and
    correlation strength. The original 1/sqrt(N) is mathematically sound
    for independent positions but too aggressive because:
    1. Correlated stocks may have different factor exposures
    2. Correlations change, especially in regime shifts
    3. High-conviction stocks deserve a smaller penalty

    Task #6: Integrate correlation clusters into position sizing.

    Args:
        ticker: Stock ticker symbol
        clusters: List of cluster dicts from portfolio_risk.flag_correlation_clusters()
                 Each cluster has 'tickers' key with list of ticker symbols
        conviction: Stock's conviction score (0-100), used to dampen the penalty

    Returns:
        Adjustment multiplier (1.0 if not in cluster, dampened 1/sqrt(N) otherwise)
    """
    for cluster in clusters:
        cluster_tickers = cluster.get("tickers", [])
        if ticker in cluster_tickers:
            cluster_size = len(cluster_tickers)
            # Base adjustment: 1/sqrt(N)
            base_adj = 1.0 / math.sqrt(cluster_size)
            # CIO Legacy C4: Dampen based on conviction — high-conviction
            # stocks in correlated clusters get a smaller penalty
            conviction_factor = min(conviction / 100.0, 0.8)
            adjustment = base_adj + (1.0 - base_adj) * conviction_factor * 0.3
            logger.debug(
                f"{ticker} in correlation cluster of {cluster_size} stocks, "
                f"conviction={conviction:.0f}, adjustment: {adjustment:.2f}"
            )
            return adjustment

    return 1.0  # Not in any cluster


def get_sector_rotation_adjustment(
    sector: str,
    rotation_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate conviction adjustment and blocking flag based on sector rotation.

    Task #1: Integrate sector rotation into conviction scoring.

    Args:
        sector: Stock's sector
        rotation_context: Dict from sector_rotation.detect_sector_rotation()
                         Contains 'gaining_sectors', 'losing_sectors' lists

    Returns:
        Dict with:
            - adjustment_points: int (-15 for losing, +10 for gaining, 0 otherwise)
            - sector_rotation_blocked: bool (True if >20pp BUY% collapse)
    """
    gaining_sectors = rotation_context.get("gaining_sectors", [])
    losing_sectors = rotation_context.get("losing_sectors", [])

    # Check if sector is in losing sectors with severe collapse
    for losing in losing_sectors:
        if losing.get("sector") == sector:
            change_pp = losing.get("change_pp", 0)
            # Block new positions if BUY% collapsed >20pp
            if change_pp <= -20:
                logger.warning(
                    f"Sector {sector} BUY% collapsed {abs(change_pp):.1f}pp - "
                    "blocking new positions"
                )
                return {
                    "adjustment_points": -15,
                    "sector_rotation_blocked": True,
                }
            # Regular losing sector penalty
            return {
                "adjustment_points": -15,
                "sector_rotation_blocked": False,
            }

    # Check if sector is gaining
    for gaining in gaining_sectors:
        if gaining.get("sector") == sector:
            return {
                "adjustment_points": 10,
                "sector_rotation_blocked": False,
            }

    # No rotation detected for this sector
    return {
        "adjustment_points": 0,
        "sector_rotation_blocked": False,
    }


def get_freshness_multiplier(staleness: str) -> float:
    """
    Get position size multiplier based on data freshness.

    Task #13: Integrate data freshness into conviction scoring.

    Args:
        staleness: Staleness category from DataFreshnessTracker
                  ('fresh', 'aging', 'stale', 'dead')

    Returns:
        Multiplier: 1.0 for fresh, 0.75 for aging, 0.50 for stale, 0.0 for dead
    """
    multipliers = {
        "fresh": 1.0,   # <30 days: no penalty
        "aging": 0.75,  # 30-60 days: 25% reduction
        "stale": 0.50,  # 60-90 days: 50% reduction
        "dead": 0.0,    # 90+ days: INCONCLUSIVE
    }
    return multipliers.get(staleness.lower(), 1.0)


def calculate_market_impact(
    position_usd: float,
    average_daily_volume_usd: float,
    tier: str,
) -> Dict[str, Any]:
    """
    Calculate market impact costs for position sizing.

    Task #2: Add market impact modeling to position sizing.

    Total cost = static spread + dynamic impact
    - Static spread from config (MEGA=2bps, LARGE=5bps, etc.)
    - Dynamic impact = (position_size / ADV) * 10000 basis points

    Args:
        position_usd: Proposed position size in USD
        average_daily_volume_usd: Average daily trading volume in USD
        tier: Stock tier (MEGA, LARGE, MID, SMALL, MICRO)

    Returns:
        Dict with:
            - spread_bps: Static spread cost in basis points
            - impact_bps: Dynamic impact cost in basis points
            - total_cost_bps: Total transaction cost
            - cost_exceeds_alpha: bool (True if cost is prohibitive)
    """
    # Static spreads from config
    SPREAD_BPS = {
        "MEGA": 2,
        "LARGE": 5,
        "MID": 10,
        "SMALL": 15,  # Updated from config (was 20)
        "MICRO": 20,  # Updated from config (was 40)
    }

    spread_bps = SPREAD_BPS.get(tier.upper(), 10)

    # Dynamic impact: larger positions relative to ADV have higher impact
    if average_daily_volume_usd > 0:
        impact_ratio = position_usd / average_daily_volume_usd
        impact_bps = impact_ratio * 10000  # Convert to basis points
    else:
        # No volume data - assume high impact
        impact_bps = 50

    total_cost_bps = spread_bps + impact_bps

    # Flag if total cost exceeds 15% of typical upside
    # (For a 20% upside stock, 3% cost eats 15% of alpha)
    cost_exceeds_alpha = total_cost_bps > 300  # 3% in basis points

    if cost_exceeds_alpha:
        logger.warning(
            f"Market impact too high: {total_cost_bps:.0f}bps "
            f"(spread: {spread_bps}bps, impact: {impact_bps:.0f}bps)"
        )

    return {
        "spread_bps": spread_bps,
        "impact_bps": impact_bps,
        "total_cost_bps": total_cost_bps,
        "cost_exceeds_alpha": cost_exceeds_alpha,
    }


# Constants shared with liquidity_filter.py
# eToro daily overnight fee rate (annualized ~6.4% for long positions)
ETORO_OVERNIGHT_ANNUAL_RATE = 0.064

# Estimated spread costs by tier (basis points) — round-trip (entry + exit)
HOLDING_SPREAD_BPS: Dict[str, float] = {
    "MEGA": 2.0,
    "LARGE": 5.0,
    "MID": 10.0,
    "SMALL": 20.0,
    "MICRO": 40.0,
}


def estimate_holding_cost_pct(
    holding_period_days: int = 90,
    tier: str = "MID",
) -> Dict[str, float]:
    """
    Estimate total holding cost as a percentage of position value.

    Combines:
    - Round-trip spread cost (entry + exit), tier-dependent
    - eToro overnight financing, prorated to holding period

    Uses the same constants as liquidity_filter.py (ETORO_OVERNIGHT_ANNUAL_RATE
    = 6.4% annualized, TIER_SPREAD_BPS for spread costs).

    CIO Review v4 Finding F2: Holding-period-adjusted cost model.

    Args:
        holding_period_days: Expected holding period in calendar days (default 90).
        tier: Market cap tier (MEGA, LARGE, MID, SMALL, MICRO).

    Returns:
        Dict with:
            - spread_pct: Round-trip spread cost as percentage
            - financing_pct: Prorated overnight financing as percentage
            - total_pct: Total estimated cost as percentage
            - holding_period_days: The holding period used
    """
    tier_upper = tier.upper() if tier else "MID"
    spread_bps = HOLDING_SPREAD_BPS.get(tier_upper, HOLDING_SPREAD_BPS["MID"])

    # Round-trip spread: entry + exit
    spread_pct = (spread_bps / 10000) * 2 * 100  # Convert bps to percentage

    # eToro overnight financing prorated to holding period
    financing_pct = ETORO_OVERNIGHT_ANNUAL_RATE * (holding_period_days / 365) * 100

    total_pct = spread_pct + financing_pct

    logger.debug(
        f"Holding cost estimate ({tier_upper}, {holding_period_days}d): "
        f"spread={spread_pct:.2f}%, financing={financing_pct:.2f}%, "
        f"total={total_pct:.2f}%"
    )

    return {
        "spread_pct": round(spread_pct, 4),
        "financing_pct": round(financing_pct, 4),
        "total_pct": round(total_pct, 4),
        "holding_period_days": holding_period_days,
    }


def calculate_conviction_size(
    base_position_size: float,
    tier_multiplier: float,
    conviction_score: float,
    regime: str = "normal",
    cost_adjusted_return: Optional[float] = None,
    min_cost_adjusted_return: float = 1.0,
    max_position_usd: float = 22500,
    max_position_pct: float = 5.0,
    portfolio_value: Optional[float] = None,
    cluster_adjustment: float = 1.0,
    sector_adjustment: int = 0,
    sector_rotation_blocked: bool = False,
    freshness_multiplier: float = 1.0,
    market_impact_blocked: bool = False,
    portfolio_var_scaling: float = 1.0,
    holding_period_days: int = 90,
    tier: str = "MID",
) -> Dict[str, Any]:
    """
    Calculate final position size with conviction and regime adjustments.

    Position size flow:
    1. Start with base_position_size * tier_multiplier (existing logic)
    2. Apply sector rotation adjustment to conviction score (Task #1)
    3. Apply conviction multiplier - continuous function (Task #4)
    4. Apply VIX regime multiplier (high vol = smaller positions)
    5. Apply correlation cluster adjustment (Task #6)
    6. Apply data freshness multiplier (Task #13)
    7. Apply portfolio VaR scaling (CIO v3 F10)
    8. Check cost-adjusted return and market impact (Task #2)
    9. Apply max position constraints

    Args:
        base_position_size: Base position size from config (e.g., $2500)
        tier_multiplier: Tier-based multiplier (MEGA=5, LARGE=4, etc.)
        conviction_score: BUY conviction score (0-100)
        regime: Current VIX regime
        cost_adjusted_return: Expected return minus transaction costs (%)
        min_cost_adjusted_return: Minimum cost-adjusted return to proceed (%)
        max_position_usd: Maximum position in USD
        max_position_pct: Maximum position as % of portfolio
        portfolio_value: Total portfolio value (for % constraint)
        cluster_adjustment: Multiplier from get_cluster_size_adjustment() (Task #6)
        sector_adjustment: Conviction points from get_sector_rotation_adjustment() (Task #1)
        sector_rotation_blocked: Block flag from sector rotation (Task #1)
        freshness_multiplier: Multiplier from get_freshness_multiplier() (Task #13)
        market_impact_blocked: Block flag from calculate_market_impact() (Task #2)
        portfolio_var_scaling: Scaling factor from portfolio VaR budget (CIO v3 F10).
            1.0 = within budget, 0.8 = VaR exceeds trigger (scale down 20%),
            calculated by get_portfolio_var_scaling().
        holding_period_days: Expected holding period in calendar days (default 90).
            Used to calculate prorated eToro overnight financing costs.
            CIO Review v4 Finding F2.
        tier: Market cap tier for holding cost estimation (default "MID").
            CIO Review v4 Finding F2.

    Returns:
        Dict with position_size, conviction_mult, regime_mult, adjustments,
        blocking info, and holding_cost (F2).
    """
    # Step 1: Base tier size
    tier_size = base_position_size * tier_multiplier

    # Step 2: Apply sector rotation adjustment to conviction score (Task #1)
    adjusted_conviction = conviction_score + sector_adjustment
    adjusted_conviction = max(0, min(100, adjusted_conviction))  # Clamp to 0-100

    # Step 3: Conviction multiplier (Task #4 - continuous function)
    conviction_mult = get_conviction_multiplier(adjusted_conviction)
    after_conviction = tier_size * conviction_mult

    # Step 4: Regime adjustment
    regime_mult = get_regime_multiplier(regime)
    after_regime = after_conviction * regime_mult

    # Step 5: Correlation cluster adjustment (Task #6)
    after_cluster = after_regime * cluster_adjustment

    # Step 6: Data freshness multiplier (Task #13)
    after_freshness = after_cluster * freshness_multiplier

    # Step 7: Portfolio VaR scaling (CIO v3 F10)
    after_var = after_freshness * portfolio_var_scaling

    # Step 8: Check blocking conditions
    skip_due_to_cost = False
    if cost_adjusted_return is not None and cost_adjusted_return < min_cost_adjusted_return:
        skip_due_to_cost = True
        logger.info(
            f"Cost-adjusted return {cost_adjusted_return:.1f}% below minimum "
            f"{min_cost_adjusted_return:.1f}% — position size zeroed"
        )

    # Combine all blocking conditions
    is_blocked = (
        skip_due_to_cost
        or sector_rotation_blocked
        or market_impact_blocked
        or math.isclose(freshness_multiplier, 0.0, abs_tol=1e-9)  # Dead data
    )

    # Step 9: Apply max constraints
    final_size = 0.0 if is_blocked else after_var
    final_size = min(final_size, max_position_usd)

    if portfolio_value and portfolio_value > 0:
        max_from_pct = portfolio_value * (max_position_pct / 100)
        final_size = min(final_size, max_from_pct)

    # Calculate holding cost estimate (CIO v4 F2)
    holding_cost = estimate_holding_cost_pct(holding_period_days, tier)

    return {
        "position_size": round(final_size, 2),
        "base_tier_size": round(tier_size, 2),
        "conviction_multiplier": conviction_mult,
        "regime_multiplier": regime_mult,
        "cluster_adjustment": cluster_adjustment,
        "freshness_multiplier": freshness_multiplier,
        "portfolio_var_scaling": portfolio_var_scaling,
        "original_conviction": conviction_score,
        "adjusted_conviction": adjusted_conviction,
        "sector_adjustment": sector_adjustment,
        "regime": regime,
        "cost_adjusted_return": cost_adjusted_return,
        "skip_due_to_cost": skip_due_to_cost,
        "sector_rotation_blocked": sector_rotation_blocked,
        "market_impact_blocked": market_impact_blocked,
        "is_blocked": is_blocked,
        "holding_cost": holding_cost,
        "adjustments": _describe_adjustments(
            conviction_mult,
            regime_mult,
            cluster_adjustment,
            freshness_multiplier,
            sector_adjustment,
            skip_due_to_cost,
            sector_rotation_blocked,
            market_impact_blocked,
        ),
    }


def _describe_adjustments(
    conviction_mult: float,
    regime_mult: float,
    cluster_adjustment: float,
    freshness_multiplier: float,
    sector_adjustment: int,
    skip_due_to_cost: bool,
    sector_rotation_blocked: bool,
    market_impact_blocked: bool,
) -> str:
    """Generate human-readable description of sizing adjustments."""
    parts = []

    # Sector rotation adjustment to conviction
    if sector_adjustment != 0:
        sign = "+" if sector_adjustment > 0 else ""
        parts.append(f"sector rotation {sign}{sector_adjustment} pts")

    # Conviction multiplier
    if conviction_mult < 1.0:
        pct = (1 - conviction_mult) * 100
        parts.append(f"conviction -{pct:.0f}%")
    elif math.isclose(conviction_mult, 1.0):
        parts.append("full conviction")

    # VIX regime
    if regime_mult < 1.0:
        pct = (1 - regime_mult) * 100
        parts.append(f"VIX regime -{pct:.0f}%")

    # Correlation cluster
    if cluster_adjustment < 1.0:
        pct = (1 - cluster_adjustment) * 100
        parts.append(f"cluster -{pct:.0f}%")

    # Data freshness
    if freshness_multiplier < 1.0:
        pct = (1 - freshness_multiplier) * 100
        if math.isclose(freshness_multiplier, 0.0, abs_tol=1e-9):
            parts.append("BLOCKED: dead data (90+ days)")
        else:
            parts.append(f"stale data -{pct:.0f}%")

    # Blocking conditions
    if skip_due_to_cost:
        parts.append("BLOCKED: costs exceed expected return")

    if sector_rotation_blocked:
        parts.append("BLOCKED: sector rotation collapse >20pp")

    if market_impact_blocked:
        parts.append("BLOCKED: market impact >300bps")

    return "; ".join(parts) if parts else "no adjustments"


def apply_portfolio_constraints(
    new_positions: List[Dict[str, Any]],
    current_sector_exposures: Dict[str, float],
    max_sector_pct: float = 40.0,
    max_single_stock_pct: float = 10.0,
    current_stock_exposures: Optional[Dict[str, float]] = None,
    portfolio_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Apply portfolio-level sector and single-stock concentration constraints.

    Two layers of constraint, applied in order:
    1. Single-stock cap: per-position size + current exposure ≤ max_single_stock_pct
    2. Sector cap: sector new total + current exposure ≤ max_sector_pct
       If projected sector exposure exceeds the cap, all new positions in
       that sector are proportionally reduced.

    CIO Review v4 Finding F8 (sector) + user-requested single-stock cap.

    Args:
        new_positions: List of proposed positions, each a dict with:
            - ticker: str (stock ticker)
            - sector: str (sector name)
            - position_size: float (proposed size — % if portfolio_value None, USD if provided)
        current_sector_exposures: {"Technology": 35.0, ...} — current % weights.
        max_sector_pct: Maximum sector exposure (default 40%).
        max_single_stock_pct: Maximum per-stock exposure (default 10%).
        current_stock_exposures: Optional {"MSFT": 10.84, ...} — current % weights.
            When provided, the single-stock cap is enforced against
            (current + new). When omitted, only the new size is checked.
        portfolio_value: Total portfolio value. If None, position_size is %.

    Returns:
        List of dicts: ticker, sector, original_size, constrained_size,
        was_constrained, constraint_reason.
    """
    if not new_positions:
        return []

    current_stock_exposures = current_stock_exposures or {}

    # Snapshot original requested sizes before any clipping
    _orig_inputs = [pos.get("position_size", 0.0) for pos in new_positions]

    # Layer 1: Single-stock cap. Cap each position so current + new ≤ max_single_stock_pct.
    # Track per-stock pre-clip and post-clip sizes for clear reasoning later.
    stock_clip_reasons: Dict[str, str] = {}
    for pos in new_positions:
        tkr = pos.get("ticker", "")
        size = pos.get("position_size", 0.0)
        if portfolio_value and portfolio_value > 0:
            new_pct = (size / portfolio_value) * 100
        else:
            new_pct = size
        current_pct = current_stock_exposures.get(tkr, 0.0)
        projected_pct = current_pct + new_pct
        if projected_pct > max_single_stock_pct and new_pct > 0:
            available_pct = max(0.0, max_single_stock_pct - current_pct)
            scale_stock = available_pct / new_pct
            scale_stock = max(0.0, min(1.0, scale_stock))
            pos["position_size"] = round(size * scale_stock, 4)
            stock_clip_reasons[tkr] = (
                f"{tkr} at {current_pct:.1f}%, "
                f"single-stock limit {max_single_stock_pct:.0f}% — "
                f"clipped to {scale_stock:.0%}"
            )

    # Layer 2: Group (already-clipped) positions by sector
    sector_new_totals: Dict[str, float] = {}
    for pos in new_positions:
        sector = pos.get("sector", "Unknown")
        size = pos.get("position_size", 0.0)
        sector_new_totals[sector] = sector_new_totals.get(sector, 0.0) + size

    # Calculate scaling factor per sector
    sector_scale: Dict[str, float] = {}
    for sector, new_total in sector_new_totals.items():
        current_pct = current_sector_exposures.get(sector, 0.0)

        # Convert new_total to percentage if portfolio_value is provided
        if portfolio_value and portfolio_value > 0:
            new_pct = (new_total / portfolio_value) * 100
        else:
            new_pct = new_total

        projected_pct = current_pct + new_pct

        if projected_pct > max_sector_pct:
            # How much room is left for new positions in this sector
            available_pct = max(0.0, max_sector_pct - current_pct)

            if new_pct > 0:
                scale = available_pct / new_pct
                scale = max(0.0, min(1.0, scale))  # Clamp to [0, 1]
            else:
                scale = 1.0

            sector_scale[sector] = scale
            logger.info(
                f"Sector constraint: {sector} projected {projected_pct:.1f}% "
                f"exceeds {max_sector_pct:.0f}% limit — scaling new positions "
                f"by {scale:.2f} (current: {current_pct:.1f}%, "
                f"new: {new_pct:.1f}%, available: {available_pct:.1f}%)"
            )
        else:
            sector_scale[sector] = 1.0

    # Apply sector scaling on top of stock-clip already done above.
    # NOTE: original_size here reflects the post-stock-clip size, not the
    # pre-clip request. We track the true pre-clip size separately so the
    # caller can show "X requested, Y after stock cap, Z after sector cap".
    results: List[Dict[str, Any]] = []
    for pos, original_input_size in zip(new_positions, _orig_inputs):
        ticker = pos.get("ticker", "")
        sector = pos.get("sector", "Unknown")
        post_stock_clip_size = pos.get("position_size", 0.0)
        scale = sector_scale.get(sector, 1.0)
        constrained_size = round(post_stock_clip_size * scale, 4)
        sector_constrained = scale < 1.0
        stock_constrained = ticker in stock_clip_reasons
        was_constrained = sector_constrained or stock_constrained

        reasons = []
        if stock_constrained:
            reasons.append(stock_clip_reasons[ticker])
        if sector_constrained:
            current_pct = current_sector_exposures.get(sector, 0.0)
            reasons.append(
                f"{sector} at {current_pct:.1f}%, "
                f"sector limit {max_sector_pct:.0f}% — "
                f"scaled to {scale:.0%}"
            )
        constraint_reason = "; ".join(reasons)

        results.append({
            "ticker": ticker,
            "sector": sector,
            "original_size": original_input_size,
            "constrained_size": constrained_size,
            "was_constrained": was_constrained,
            "constraint_reason": constraint_reason,
        })

    constrained_count = sum(1 for r in results if r["was_constrained"])
    if constrained_count > 0:
        logger.info(
            f"Portfolio constraints: {constrained_count}/{len(results)} "
            f"positions reduced due to sector limits"
        )

    return results


def adjust_sizes_for_opportunity_cost(
    positions: List[Dict[str, Any]],
    conviction_key: str = "conviction",
    size_key: str = "position_size",
    max_pct: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Redistribute position sizes from low-conviction to high-conviction stocks
    (CIO Legacy C3).

    Position sizing is a zero-sum game — sizing one position larger means
    another must be smaller. This function applies a relative adjustment
    that reduces below-average conviction positions by 10% and increases
    above-average ones by 10% (capped at max_pct).

    Args:
        positions: List of position dicts, each with conviction and size fields.
        conviction_key: Key for conviction score in each dict.
        size_key: Key for position size in each dict.
        max_pct: Maximum position size percentage (passed through for capping).

    Returns:
        The same list with adjusted position sizes and an 'opp_cost_adj' field.
    """
    if not positions or len(positions) < 2:
        return positions

    convictions = [p.get(conviction_key, 50) for p in positions]
    avg_conviction = sum(convictions) / len(convictions)

    for p in positions:
        conv = p.get(conviction_key, 50)
        original = p.get(size_key, 0)
        distance = conv - avg_conviction
        if distance < -10:
            # CIO v6.0: Proportional reduction — scale by distance from mean.
            # Distance -15 → 7.5% reduction, -30 → 15% reduction, capped at -20%.
            reduction = min(0.20, abs(distance) / 200)
            p[size_key] = round(original * (1 - reduction), 2)
            p["opp_cost_adj"] = round(-reduction, 4)
        elif distance > 10:
            # Proportional increase — capped at +15% and max_pct.
            increase = min(0.15, distance / 200)
            p[size_key] = round(min(original * (1 + increase), max_pct), 2)
            p["opp_cost_adj"] = round(increase, 4)
        else:
            p["opp_cost_adj"] = 0.0

    return positions
