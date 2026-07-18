"""TDD tests for trade_modules.v3.trial_register — FIX-NOW (D23/spec §7): a frozen,
auditable acceptance gate + trial register so DSR sees the true trial count and the
thresholds cannot be quietly relaxed to pass a favoured signal."""

import pytest


def test_acceptance_thresholds_frozen_and_correct():
    from trade_modules.v3.trial_register import ACCEPTANCE_THRESHOLDS

    assert ACCEPTANCE_THRESHOLDS["dsr_min"] == 0.95
    assert ACCEPTANCE_THRESHOLDS["min_regimes"] == 2
    assert ACCEPTANCE_THRESHOLDS["ic_t_min"] == 3.0
    assert ACCEPTANCE_THRESHOLDS["pbo_max"] == 0.05
    # read-only (frozen): the gate cannot be mutated in place.
    with pytest.raises(TypeError):
        ACCEPTANCE_THRESHOLDS["dsr_min"] = 0.5  # type: ignore[index]


def test_trial_register_counts_candidates_for_dsr():
    from trade_modules.v3.trial_register import CANDIDATE_TRIALS, trial_count

    assert trial_count() == len(CANDIDATE_TRIALS)
    assert trial_count() >= 10  # the whole shadow queue is honestly counted for DSR
    assert "investment_cma" in CANDIDATE_TRIALS
    assert "pead_sue" in CANDIDATE_TRIALS
    assert "analyst_eps_revision" in CANDIDATE_TRIALS


def test_is_relaxation_detects_loosening_both_directions():
    from trade_modules.v3.trial_register import is_relaxation

    assert is_relaxation("dsr_min", 0.90) is True  # lowering a _min gate = loosening
    assert is_relaxation("dsr_min", 0.99) is False
    assert is_relaxation("pbo_max", 0.10) is True  # raising a _max gate = loosening
    assert is_relaxation("pbo_max", 0.02) is False
    assert is_relaxation("min_regimes", 1) is True  # fewer regimes = loosening
