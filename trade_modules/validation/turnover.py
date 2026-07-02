"""
Portfolio Turnover and Drag Calculator

Computes annualised portfolio turnover and the associated transaction-cost
drag from a list of trade actions over a measurement window.

Pure / no I/O.
"""

_DEFAULT_COST_BPS = 20.0


def compute_turnover(
    actions: list[dict],
    window_days: int,
    tier_cost_bps: dict[str, float] | None = None,
) -> dict:
    """
    Compute annualised turnover and estimated transaction-cost drag.

    Turnover convention — TWO-WAY counting:
        Each entry in ``actions`` represents ONE leg of a trade (either the
        opening buy or the closing sell).  Both legs are counted independently,
        so a full round trip (buy +5 %, then unwind -5 %) contributes 10 % to
        ``sum_abs_weight_change`` and therefore 10 % worth of annualised
        turnover — not 5 %.  This matches the standard industry convention for
        two-way (round-trip) turnover reporting.

        If the caller wants ONE-WAY turnover (only opening legs), pass only the
        opening-leg actions and divide the result by 2, or supply a pre-filtered
        ``actions`` list that contains at most one leg per trade.

    Args:
        actions:       List of dicts with keys:
                         - weight_change (float): signed weight delta as a
                           fraction (e.g. 0.05 = 5% of portfolio).
                         - tier (str, optional): cost tier identifier.
                         - date, ticker (informational, not used in math).
        window_days:   Length of the measurement window in calendar days.
        tier_cost_bps: Dict mapping tier name → round-trip cost in bps.
                       If None, flat 20 bps is used for all actions.
                       Actions with a missing or None tier default to 20 bps.

    Returns:
        Dict with keys:
          - turnover_annual_pct: float — annualised turnover as a percentage.
          - annualized_drag_bps: float — annualised cost drag in basis points.
          - n_trades:            int   — total number of actions.
    """
    if not actions:
        return {
            "turnover_annual_pct": 0.0,
            "annualized_drag_bps": 0.0,
            "n_trades": 0,
        }
    if window_days <= 0:
        raise ValueError(f"window_days must be positive; got {window_days}")

    sum_abs_weight_change = sum(abs(a["weight_change"]) for a in actions)
    turnover_annual_pct = (sum_abs_weight_change / window_days) * 365 * 100

    def _cost_bps(action: dict) -> float:
        if tier_cost_bps is None:
            return _DEFAULT_COST_BPS
        tier = action.get("tier")
        if tier is None:
            return _DEFAULT_COST_BPS
        return tier_cost_bps.get(tier, _DEFAULT_COST_BPS)

    annualized_drag_bps = sum(
        abs(a["weight_change"]) / window_days * 365 * _cost_bps(a) for a in actions
    )

    return {
        "turnover_annual_pct": turnover_annual_pct,
        "annualized_drag_bps": annualized_drag_bps,
        "n_trades": len(actions),
    }
