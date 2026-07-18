"""TDD — BUILD ① (2026-07-18): a single declared V3_SIGNAL_HORIZON (21 trading
days, ~1 month) that the IC logger, the price-spine gate, AND the referee all
grade the model against.

Before this increment the model was graded at two horizons at once:
``run_gate`` screened IC at ``horizons[0]`` (= 5, one week) while the harness DSR
resolved to 21 via a nearest-30 fallback. These tests lock the alignment and the
v2 back-compat guarantee (the shared harness default stays 30).

Owner decision: primary = 21 td, co-report 5 & 63. Sources: Grinold-Kahn (grade
IC at the rebalance horizon); Jegadeesh-Titman (momentum 1-12mo).
"""

from __future__ import annotations

import inspect

import pandas as pd

# ---------------------------------------------------------------------------
# The declared constant is the single source of truth
# ---------------------------------------------------------------------------


def test_signal_horizon_declared_and_on_grid():
    from trade_modules.v3.constants import V3_IC_HORIZONS, V3_SIGNAL_HORIZON

    assert V3_SIGNAL_HORIZON == 21
    assert V3_IC_HORIZONS == (5, 21, 63)
    assert V3_SIGNAL_HORIZON in V3_IC_HORIZONS  # invariant: primary is on the grid


# ---------------------------------------------------------------------------
# Every measurement surface reads the same grid / primary
# ---------------------------------------------------------------------------


def test_ic_logger_default_grid_is_the_declared_grid():
    from scripts.v3_ic_logger import ic_from_log
    from trade_modules.v3.constants import V3_IC_HORIZONS

    default = inspect.signature(ic_from_log).parameters["horizons"].default
    assert default == V3_IC_HORIZONS  # no more stray 10-step horizon


def test_ic_report_grid_is_the_declared_grid():
    import scripts.v3_ic_report as rep
    from trade_modules.v3.constants import V3_IC_HORIZONS

    assert rep.HORIZONS == V3_IC_HORIZONS


def test_spine_gate_grid_is_the_declared_grid():
    import scripts.v3_spine_gate as gate
    from trade_modules.v3.constants import V3_IC_HORIZONS

    assert gate.HORIZONS == list(V3_IC_HORIZONS)


def test_evaluate_exposes_primary_horizon_defaulting_to_30_for_v2():
    """The shared harness default MUST stay 30 — v2 callers are byte-identical."""
    from trade_modules.validation.harness import evaluate

    p = inspect.signature(evaluate).parameters["primary_horizon"]
    assert p.default == 30


def test_run_gate_defaults_primary_horizon_to_declared_constant():
    from trade_modules.v3.constants import V3_SIGNAL_HORIZON
    from trade_modules.v3.validate_spine import run_gate

    p = inspect.signature(run_gate).parameters["primary_horizon"]
    assert p.default == V3_SIGNAL_HORIZON == 21


# ---------------------------------------------------------------------------
# The behavioural fix: run_gate grades the IC screen at the PRIMARY, not [0]
# ---------------------------------------------------------------------------


def _synthetic():
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    scores = pd.DataFrame(
        [{"as_of": d, "ticker": t, "score": float(i)} for d in dates for i, t in enumerate(tickers)]
    )
    fwd = pd.DataFrame(
        [
            {"as_of": d, "ticker": t, "horizon": h, "fwd_ret": 0.01 * i}
            for d in dates
            for i, t in enumerate(tickers)
            for h in (5, 21, 63)
        ]
    )
    return scores, fwd


def test_run_gate_grades_ic_at_primary_not_horizons_zero():
    """Default grades at 21 — NOT the old horizons[0] = 5 (one-week noise)."""
    from trade_modules.v3.validate_spine import run_gate

    scores, fwd = _synthetic()
    out = run_gate(scores, fwd, [5, 21, 63], n_trials=2, min_obs=5)
    assert out["primary_horizon"] == 21


def test_run_gate_primary_horizon_override_and_nearest_fallback():
    from trade_modules.v3.validate_spine import run_gate

    scores, fwd = _synthetic()
    # explicit member is honoured
    assert (
        run_gate(scores, fwd, [5, 21, 63], primary_horizon=5, n_trials=2, min_obs=5)[
            "primary_horizon"
        ]
        == 5
    )
    # off-grid primary -> nearest member (30 -> 21)
    assert (
        run_gate(scores, fwd, [5, 21, 63], primary_horizon=30, n_trials=2, min_obs=5)[
            "primary_horizon"
        ]
        == 21
    )
