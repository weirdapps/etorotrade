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

CIO Review Findings:
- E1: BUY conviction score should feed position sizing
- M2: VIX regime adjustments don't touch position sizing
- Task #4: Continuous conviction function
- Task #6: Correlation cluster sizing
- Task #1: Sector rotation integration
- Task #13: Data freshness integration
- Task #2: Market impact modeling
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
) -> float:
    """
    Calculate position size adjustment for correlation clusters.

    If a ticker is in a cluster of size N, reduce position by 1/sqrt(N)
    to account for non-diversified risk.

    Task #6: Integrate correlation clusters into position sizing.

    Args:
        ticker: Stock ticker symbol
        clusters: List of cluster dicts from portfolio_risk.flag_correlation_clusters()
                 Each cluster has 'tickers' key with list of ticker symbols

    Returns:
        Adjustment multiplier (1.0 if not in cluster, 1/sqrt(N) if in cluster of size N)
    """
    for cluster in clusters:
        cluster_tickers = cluster.get("tickers", [])
        if ticker in cluster_tickers:
            cluster_size = len(cluster_tickers)
            adjustment = 1.0 / math.sqrt(cluster_size)
            logger.debug(
                f"{ticker} in correlation cluster of {cluster_size} stocks, "
                f"adjustment: {adjustment:.2f}"
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

    Returns:
        Dict with position_size, conviction_mult, regime_mult, adjustments, and blocking info
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
        or freshness_multiplier == 0.0  # Dead data
    )

    # Step 9: Apply max constraints
    final_size = 0.0 if is_blocked else after_var
    final_size = min(final_size, max_position_usd)

    if portfolio_value and portfolio_value > 0:
        max_from_pct = portfolio_value * (max_position_pct / 100)
        final_size = min(final_size, max_from_pct)

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
    elif conviction_mult == 1.0:
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
        if freshness_multiplier == 0.0:
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
