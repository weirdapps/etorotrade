"""
M9 / N9: Portfolio-level vol-targeting.

VaR-based scaling (existing in conviction_sizer.get_portfolio_var_scaling)
catches sudden spikes via 95th-percentile daily moves but misses slow
grinding losses. A portfolio losing 1% per day for 30 days has stable
daily VaR but realized 60d annualized vol is way over a 12% target.

This module:
- realized_portfolio_vol(returns) — annualized 60d std of daily returns
- vol_scale_factor(realized, target) — scaling factor ≤ 1 to keep vol on target
- Used by enrich_with_position_sizes via the vol_scale parameter

Target = 12% annualized — institutional standard for retail equity. Easy
to override for higher/lower risk tolerance.
"""

from __future__ import annotations

import math

DEFAULT_TARGET_VOL = 0.12  # 12% annualized
TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 5


def realized_portfolio_vol(
    daily_returns: list[float],
) -> float | None:
    """Annualized realized volatility from a daily-return series.

    Returns None if fewer than MIN_OBSERVATIONS observations. Returns are
    expected as decimals (0.01 = 1%, not 1.0%).
    """
    n = len(daily_returns)
    if n < MIN_OBSERVATIONS:
        return None
    mean = sum(daily_returns) / n
    var = sum((r - mean) ** 2 for r in daily_returns) / n
    if var <= 0:
        return 0.0
    daily_std = math.sqrt(var)
    return daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)


def vol_scale_factor(
    realized: float,
    target: float = DEFAULT_TARGET_VOL,
) -> float:
    """Position-size scaling factor to bring realized vol back to target.

    - If realized ≤ target: return 1.0 (no scaling, room to take more risk)
    - If realized > target: return target / realized (≤ 1)
    - Defensive: zero/negative realized returns 1.0 (no data, no scaling)
    """
    if not realized or realized <= 0:
        return 1.0
    if realized <= target:
        return 1.0
    return target / realized


def estimate_portfolio_vol_from_history(
    history_dir: str | None = None,
    benchmark: str = "SPY",
    days: int = 60,
) -> float | None:
    """Estimate portfolio realized vol from concordance history.

    Best-effort: walks recent concordance files, computes daily portfolio
    returns weighted by suggested_size_usd, returns annualized vol. Falls
    back to None when insufficient data.

    For now this is a stub the operator can call to feed vol_scale_factor.
    The full implementation would need price-cache-backed daily returns
    per-position, weighted by current portfolio composition.
    """
    # Stub for tomorrow's wiring — requires per-day portfolio composition
    # tracking that doesn't exist yet. Operator can pass realized vol
    # directly to vol_scale_factor() in the meantime.
    return None
