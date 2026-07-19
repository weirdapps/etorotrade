"""P2 governance invariants — the disciplined defaults must not silently drift.

Freeze-the-line stance: equal-risk cluster weights, within-sector z-scoring on by
default, and a strict multi-year DSR gate. The equal-risk weights / IC-default /
DSR thresholds themselves are locked in test_v3_combine.py and
test_v3_trial_register.py; here we lock the invariants that P1 (sector-conditional
value) must respect so a future edit cannot quietly break the discipline.
"""

from __future__ import annotations

import inspect

from trade_modules.v3.combine import (
    _DEFAULT_GROUP,
    VALUE_GROUP_RECIPES,
    VALUE_WEIGHT_MULT,
    compute_scores,
)


def test_compute_scores_defaults_to_sector_neutral():
    """Within-sector z-scoring stays ON by default (the pipeline default)."""
    assert inspect.signature(compute_scores).parameters["sector_neutral"].default is True


def test_value_is_never_dropped_or_sign_flipped():
    """P1 guardrail: every group keeps >=1 value metric with a POSITIVE weight.

    A negative/zero recipe weight would invert or delete the value signal in a
    sector (turning value into anti-value / momentum) — an explicitly forbidden
    regime bet. Value is only ever re-composed or down-weighted, never flipped.
    """
    for group, recipe in VALUE_GROUP_RECIPES.items():
        assert recipe, f"group {group}: value recipe is empty (value dropped)"
        assert all(w > 0 for w in recipe.values()), f"group {group}: non-positive metric weight"
    # The cluster multiplier only down-weights value (0 < m <= 1); never 0, never negative.
    assert all(0.0 < m <= 1.0 for m in VALUE_WEIGHT_MULT.values())


def test_conventional_sectors_are_a_noop():
    """P1 must not change conventional/unknown-sector conviction: multiplier == 1.0."""
    assert VALUE_WEIGHT_MULT[_DEFAULT_GROUP] == 1.0


def test_multi_year_dsr_gate_stays_strict():
    """The acceptance gate keeps a strict deflated-Sharpe hurdle (>= 0.95)."""
    from trade_modules.v3.trial_register import threshold

    assert threshold("dsr_min") >= 0.95
