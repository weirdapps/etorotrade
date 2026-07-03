"""actions.py — long-only action assignment.

assign_action(conviction, is_held, in_universe, cfg=None) -> str

Decision logic (thresholds are >= comparisons):
  not is_held:
    conviction >= buy         → 'BUY'
    else                      → 'NONE'  (not selected; excluded from output)

  is_held and NOT in_universe (dropped from eligible):
    always                    → 'SELL'  (close position)

  is_held and in_universe:
    conviction >= add         → 'ADD'
    conviction >= hold        → 'HOLD'
    conviction >= trim        → 'TRIM'
    else                      → 'SELL'

Default thresholds:
  buy  = 65
  add  = 70
  hold = 40   (softened: widens the HOLD band so borderline held names are not trimmed on a mid score)
  trim = 30

Long-only: never emits 'SHORT' or any bearish direction label.
Config-overridable via cfg dict with same keys.
"""

from __future__ import annotations

_DEFAULT_CFG = {
    "buy": 65,
    "add": 70,
    "hold": 40,
    "trim": 30,
}

_VALID_ACTIONS = {"BUY", "ADD", "HOLD", "TRIM", "SELL", "NONE"}


def assign_action(
    conviction: float,
    is_held: bool,
    in_universe: bool,
    cfg: dict | None = None,
) -> str:
    """Assign a long-only action label.

    Args:
        conviction:   Float [0, 100] from score_conviction.
        is_held:      True if this ticker is in the current portfolio.
        in_universe:  True if this ticker is in the eligible candidate set.
        cfg:          Optional threshold overrides (buy, add, hold, trim).

    Returns:
        One of: 'BUY', 'ADD', 'HOLD', 'TRIM', 'SELL', 'NONE'.
        Never returns a short-direction label.
    """
    thresholds = {**_DEFAULT_CFG, **(cfg or {})}
    t_buy = thresholds["buy"]
    t_add = thresholds["add"]
    t_hold = thresholds["hold"]
    t_trim = thresholds["trim"]

    if not is_held:
        return "BUY" if conviction >= t_buy else "NONE"

    # held
    if not in_universe:
        return "SELL"

    # held and in_universe
    if conviction >= t_add:
        return "ADD"
    if conviction >= t_hold:
        return "HOLD"
    if conviction >= t_trim:
        return "TRIM"
    return "SELL"
