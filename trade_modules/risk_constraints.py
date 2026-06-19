"""Executable risk constraints — enforce rules that were previously text-only.

These constraints are CHECK functions, not hard blocks. Each returns a structured
result that the CIO can override with documented justification. The key difference
from before: these are EVALUATED in code, not just printed as advisory strings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConstraintResult:
    """Result of a risk constraint check."""

    name: str
    passed: bool
    severity: str = "INFO"  # INFO, WARNING, BLOCK
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    override_allowed: bool = True


def check_cash_reserve(
    available_cash_pct: float,
    min_cash_pct: float = 15.0,
) -> ConstraintResult:
    """Block new positions if cash < 15% of equity."""
    passed = available_cash_pct >= min_cash_pct
    return ConstraintResult(
        name="cash_reserve",
        passed=passed,
        severity="BLOCK" if not passed else "INFO",
        message=f"Cash {available_cash_pct:.1f}% {'above' if passed else 'below'} {min_cash_pct}% minimum",
        current_value=available_cash_pct,
        threshold=min_cash_pct,
    )


def check_position_stoploss(
    current_pnl_pct: float,
    threshold: float = -20.0,
    above_200dma: bool = True,
) -> ConstraintResult:
    """Flag positions that have breached -20% and broken 200DMA."""
    breached = current_pnl_pct <= threshold and not above_200dma
    return ConstraintResult(
        name="position_stoploss",
        passed=not breached,
        severity="BLOCK" if breached else ("WARNING" if current_pnl_pct <= threshold else "INFO"),
        message=(
            f"Position at {current_pnl_pct:+.1f}% — "
            + (
                "STOP-LOSS TRIGGERED (below threshold AND below 200DMA)"
                if breached
                else f"{'below threshold but above 200DMA — thesis intact' if current_pnl_pct <= threshold else 'within limits'}"
            )
        ),
        current_value=current_pnl_pct,
        threshold=threshold,
    )


def check_portfolio_drawdown(
    current_drawdown_pct: float,
    threshold: float = -15.0,
) -> ConstraintResult:
    """If portfolio drawdown exceeds -15%, signal to reduce to 50% equity."""
    breached = current_drawdown_pct <= threshold
    return ConstraintResult(
        name="portfolio_drawdown",
        passed=not breached,
        severity="BLOCK" if breached else "INFO",
        message=(
            f"Portfolio drawdown {current_drawdown_pct:.1f}% — "
            + ("REDUCE TO 50% EQUITY" if breached else "within limits")
        ),
        current_value=current_drawdown_pct,
        threshold=threshold,
    )


def check_single_position_drift(
    position_pct: float,
    max_pct: float = 7.0,
) -> ConstraintResult:
    """Flag positions that have drifted above 7% due to appreciation."""
    breached = position_pct > max_pct
    return ConstraintResult(
        name="position_drift",
        passed=not breached,
        severity="WARNING" if breached else "INFO",
        message=(
            f"Position at {position_pct:.1f}% of portfolio — "
            + (f"REBALANCE (exceeds {max_pct}% drift threshold)" if breached else "within limits")
        ),
        current_value=position_pct,
        threshold=max_pct,
    )


def check_sector_concentration(
    sector_pct: float,
    max_pct: float = 25.0,
) -> ConstraintResult:
    """Block new positions in sectors already at 25% weight."""
    breached = sector_pct >= max_pct
    return ConstraintResult(
        name="sector_concentration",
        passed=not breached,
        severity="BLOCK" if breached else ("WARNING" if sector_pct >= max_pct * 0.8 else "INFO"),
        message=(
            f"Sector at {sector_pct:.1f}% — "
            + (
                f"BLOCKED (at or above {max_pct}% cap)"
                if breached
                else f"{'approaching cap' if sector_pct >= max_pct * 0.8 else 'within limits'}"
            )
        ),
        current_value=sector_pct,
        threshold=max_pct,
    )


def check_all_constraints(
    available_cash_pct: float = 100.0,
    portfolio_drawdown_pct: float = 0.0,
    min_cash_pct: float = 15.0,
    drawdown_threshold: float = -15.0,
) -> list[ConstraintResult]:
    """Run all portfolio-level constraints. Returns list of results."""
    return [
        check_cash_reserve(available_cash_pct, min_cash_pct),
        check_portfolio_drawdown(portfolio_drawdown_pct, drawdown_threshold),
    ]
