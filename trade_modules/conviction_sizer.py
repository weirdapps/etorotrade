"""
Conviction-Based Position Sizer

Modulates position size based on:
1. BUY conviction score (0-100) from signal scoring
2. VIX regime (reduce size in high volatility)
3. Cost-adjusted expected return (skip if costs exceed alpha)

CIO Review Findings:
- E1: BUY conviction score should feed position sizing
- M2: VIX regime adjustments don't touch position sizing
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Conviction score → position size multiplier
# Higher conviction = full position; lower conviction = reduced position
CONVICTION_MULTIPLIERS = {
    (90, 100): 1.00,   # Very high conviction: full position
    (80, 89): 0.90,    # High conviction: 90%
    (70, 79): 0.80,    # Good conviction: 80%
    (60, 69): 0.65,    # Moderate conviction: 65%
    (50, 59): 0.50,    # Low conviction: 50%
    (0, 49): 0.35,     # Very low conviction: 35%
}

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

    Args:
        conviction_score: BUY conviction score (0-100)

    Returns:
        Multiplier between 0.35 and 1.0
    """
    score = max(0, min(100, conviction_score))

    for (low, high), multiplier in CONVICTION_MULTIPLIERS.items():
        if low <= score <= high:
            return multiplier

    return 0.50  # Default fallback


def get_regime_multiplier(regime: str = "normal") -> float:
    """
    Get position size multiplier based on VIX regime.

    Args:
        regime: VIX regime string (low, normal, elevated, high)

    Returns:
        Multiplier between 0.5 and 1.0
    """
    return REGIME_POSITION_MULTIPLIERS.get(regime.lower(), 1.0)


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
) -> Dict[str, Any]:
    """
    Calculate final position size with conviction and regime adjustments.

    Position size flow:
    1. Start with base_position_size * tier_multiplier (existing logic)
    2. Apply conviction multiplier (higher conviction = larger position)
    3. Apply VIX regime multiplier (high vol = smaller positions)
    4. Check cost-adjusted return (skip if costs > expected alpha)
    5. Apply max position constraints

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

    Returns:
        Dict with position_size, conviction_mult, regime_mult, adjustments
    """
    # Step 1: Base tier size
    tier_size = base_position_size * tier_multiplier

    # Step 2: Conviction adjustment
    conviction_mult = get_conviction_multiplier(conviction_score)
    after_conviction = tier_size * conviction_mult

    # Step 3: Regime adjustment
    regime_mult = get_regime_multiplier(regime)
    after_regime = after_conviction * regime_mult

    # Step 4: Cost-adjusted return check
    skip_due_to_cost = False
    if cost_adjusted_return is not None and cost_adjusted_return < min_cost_adjusted_return:
        skip_due_to_cost = True
        logger.info(
            f"Cost-adjusted return {cost_adjusted_return:.1f}% below minimum "
            f"{min_cost_adjusted_return:.1f}% — position size zeroed"
        )

    # Step 5: Apply max constraints
    final_size = 0.0 if skip_due_to_cost else after_regime
    final_size = min(final_size, max_position_usd)

    if portfolio_value and portfolio_value > 0:
        max_from_pct = portfolio_value * (max_position_pct / 100)
        final_size = min(final_size, max_from_pct)

    return {
        "position_size": round(final_size, 2),
        "base_tier_size": round(tier_size, 2),
        "conviction_multiplier": conviction_mult,
        "regime_multiplier": regime_mult,
        "conviction_score": conviction_score,
        "regime": regime,
        "cost_adjusted_return": cost_adjusted_return,
        "skip_due_to_cost": skip_due_to_cost,
        "adjustments": _describe_adjustments(conviction_mult, regime_mult, skip_due_to_cost),
    }


def _describe_adjustments(
    conviction_mult: float,
    regime_mult: float,
    skip_due_to_cost: bool,
) -> str:
    """Generate human-readable description of sizing adjustments."""
    parts = []

    if conviction_mult < 1.0:
        pct = (1 - conviction_mult) * 100
        parts.append(f"conviction -{pct:.0f}%")
    elif conviction_mult == 1.0:
        parts.append("full conviction")

    if regime_mult < 1.0:
        pct = (1 - regime_mult) * 100
        parts.append(f"VIX regime -{pct:.0f}%")

    if skip_due_to_cost:
        parts.append("BLOCKED: costs exceed expected return")

    return "; ".join(parts) if parts else "no adjustments"
