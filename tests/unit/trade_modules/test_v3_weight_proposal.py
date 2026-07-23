"""TDD — adaptive weight-proposal mechanism (Bayesian shrinkage prior->IC)."""

from __future__ import annotations

import pytest

from trade_modules.v3.weight_proposal import proposal_table, propose_weights

# Guardrail-valid prior (each ≤ cap 0.30, core ≥ floor 0.10), like the real CLUSTER_WEIGHTS.
PRIOR = {"value_z": 0.30, "quality_z": 0.25, "momentum_z": 0.25, "lowvol_z": 0.20}


def test_no_data_returns_prior():
    p = propose_weights(PRIOR, ic={}, n={})
    for c in PRIOR:
        assert p[c] == pytest.approx(PRIOR[c])


@pytest.mark.xfail(
    reason=(
        "Real (pre-existing) inconsistency surfaced by the 2026-07-23 review: the deployed "
        "CLUSTER_WEIGHTS put quality at 0.42, above the adaptive mechanism's 0.30 cap (and "
        "several small clusters below its 0.10 floor), so the deployed prior is NOT "
        "guardrail-valid and does not round-trip. This is the quality-over-concentration "
        "finding; it resolves when the factor-set / cap decision is made — not a stale test."
    ),
    strict=True,
)
def test_real_cluster_weights_roundtrip_at_zero_data():
    # The deployed best-bet prior must be guardrail-valid: at n=0 the mechanism returns
    # it unchanged (so deploying it == the mechanism's current proposal).
    from trade_modules.v3.combine import CLUSTER_WEIGHTS

    p = propose_weights(dict(CLUSTER_WEIGHTS), ic={}, n={})
    for c in CLUSTER_WEIGHTS:
        assert p[c] == pytest.approx(CLUSTER_WEIGHTS[c])


def test_sums_to_one_and_nonneg():
    p = propose_weights(PRIOR, ic={"momentum_z": 0.05}, n={"momentum_z": 100})
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in p.values())


def test_promoted_factor_with_high_ic_rises_above_prior():
    ic = {"momentum_z": 0.10, "value_z": 0.0, "quality_z": 0.0}
    n = dict.fromkeys(PRIOR, 1000)  # high confidence
    p = propose_weights(PRIOR, ic, n, promoted={"momentum_z"})
    assert p["momentum_z"] > PRIOR["momentum_z"]


def test_dsr_gate_limits_unpromoted_gain():
    # momentum prior (0.2) sits below the cap so the gate (min with prior) actually bites.
    prior = {"value_z": 0.40, "quality_z": 0.40, "momentum_z": 0.20}
    ic = {"momentum_z": 0.10}
    n = dict.fromkeys(prior, 1000)
    gated = propose_weights(prior, ic, n, promoted=set())
    promoted = propose_weights(prior, ic, n, promoted={"momentum_z"})
    assert promoted["momentum_z"] > gated["momentum_z"]


def test_floors_never_zero_the_durable_core():
    ic = {"momentum_z": 0.5}  # momentum dominates the IC
    n = dict.fromkeys(PRIOR, 1000)
    p = propose_weights(PRIOR, ic, n, promoted={"momentum_z"})
    assert p["value_z"] > 0 and p["quality_z"] > 0  # durable core never zeroed


def test_proposal_table_sorted_by_abs_delta():
    t = proposal_table({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.7, "b": 0.2, "c": 0.1})
    assert t[0]["cluster"] == "a"  # largest |delta| (+0.2 vs -0.1, -0.1)
    assert t[0]["delta"] == pytest.approx(0.2)
