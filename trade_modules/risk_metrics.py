"""Canonical risk metrics — single source of truth for portfolio risk scoring."""

from __future__ import annotations


def canonical_risk_score(
    var_95_pct: float,
    max_drawdown_pct: float,
    portfolio_beta: float,
    sortino: float = 2.0,
) -> int:
    """Canonical risk score (0-100, higher = riskier).

    Unifies three previously inconsistent implementations into one formula.

    Components and their typical contribution ranges:
    - VaR 95% (daily):    abs(var_95_pct) * 10    → 0-30 typical
    - Max drawdown:       abs(max_dd_pct) * 2     → 0-30 typical
    - Beta excess:        max(0, beta-1) * 20     → 0-20 typical
    - Sortino deficit:    max(0, 2-sortino) * 10  → 0-20 typical

    Args:
        var_95_pct: VaR at 95% confidence as a percentage (e.g., 2.5 means 2.5%)
        max_drawdown_pct: Maximum drawdown as a percentage (e.g., 15.0 means 15%)
        portfolio_beta: Portfolio beta vs benchmark
        sortino: Sortino ratio (default 2.0 = neutral contribution)
    """
    return min(
        100,
        max(
            0,
            int(
                abs(var_95_pct) * 10
                + abs(max_drawdown_pct) * 2
                + max(0, portfolio_beta - 1) * 20
                + max(0, 2 - sortino) * 10
            ),
        ),
    )


def risk_level(score: int) -> str:
    """Human-readable risk level from canonical score."""
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MODERATE"
    return "LOW"
