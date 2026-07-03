"""Deployable budget calculator.

Translates cash surplus above a target into a regime-adjusted deployment
fraction. Pure function: no I/O, no state.
"""

from __future__ import annotations


def deployable_budget(
    cash_pct: float,
    target_cash_pct: float = 0.07,
    regime_mult: float = 1.0,
) -> float:
    """Fraction of NAV to deploy into new longs.

    = max(0.0, cash_pct - target_cash_pct) * clamp(regime_mult, 0, 1)

    At or below target → 0.  All inputs/outputs are fractions (0.0 to 1.0).
    No nav parameter — caller passes fractions directly.

    Args:
        cash_pct: Current cash as a fraction of NAV (e.g. 0.29 = 29%).
        target_cash_pct: Minimum cash buffer to keep (default 0.07 = 7%).
        regime_mult: Regime dial in [0, 1]; clamped if outside that range.

    Returns:
        Fraction of NAV available to deploy (always >= 0.0).
    """
    surplus = max(0.0, cash_pct - target_cash_pct)
    mult_clamped = min(1.0, max(0.0, regime_mult))
    return surplus * mult_clamped
