"""Tests for riskfirst.edgegate — multiple-testing-aware significance.

Implements the Probabilistic / Deflated Sharpe Ratio (Bailey & Lopez de Prado),
which the senior-PM review found ABSENT from the system. The gate also refuses to
pass on too few observations or a single-regime sample (no bear/stress).
"""

import pytest

from trade_modules.riskfirst.edgegate import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    gate_verdict,
    probabilistic_sharpe_ratio,
)

# ---- PSR ----


def test_psr_is_half_at_benchmark():
    assert probabilistic_sharpe_ratio(0.1, 0.1, n_obs=100) == pytest.approx(0.5, abs=1e-9)


def test_psr_increases_with_sharpe():
    lo = probabilistic_sharpe_ratio(0.05, 0.0, n_obs=250)
    hi = probabilistic_sharpe_ratio(0.20, 0.0, n_obs=250)
    assert hi > lo


def test_psr_approaches_one_for_strong_sharpe_long_sample():
    assert probabilistic_sharpe_ratio(0.3, 0.0, n_obs=2000) > 0.99


# ---- expected max Sharpe under the null (deflation benchmark) ----


def test_expected_max_sharpe_increases_with_trials():
    assert expected_max_sharpe(100, var_sr=1.0) > expected_max_sharpe(2, var_sr=1.0)


def test_expected_max_sharpe_increases_with_variance():
    assert expected_max_sharpe(50, var_sr=2.0) > expected_max_sharpe(50, var_sr=1.0)


# ---- DSR: more trials -> harder to clear ----


def test_more_trials_lowers_deflated_sharpe():
    few = deflated_sharpe_ratio(0.15, n_obs=250, n_trials=1, var_sr=0.01)
    many = deflated_sharpe_ratio(0.15, n_obs=250, n_trials=200, var_sr=0.01)
    assert many < few
    assert 0.0 <= many <= 1.0


# ---- the gate verdict ----


def test_gate_fails_on_single_regime_even_with_strong_stats():
    v = gate_verdict(sr=0.5, n_obs=2000, n_trials=1, var_sr=0.0001, n_regimes=1)
    assert v["passed"] is False
    assert any("regime" in r for r in v["reasons"])


def test_gate_fails_on_insufficient_observations():
    v = gate_verdict(sr=0.5, n_obs=60, n_trials=1, var_sr=0.0001, n_regimes=2)
    assert v["passed"] is False
    assert any("observ" in r.lower() for r in v["reasons"])


def test_gate_passes_only_when_all_conditions_met():
    v = gate_verdict(sr=0.5, n_obs=2000, n_trials=1, var_sr=0.0001, n_regimes=2)
    assert v["passed"] is True
    assert v["reasons"] == []


def test_current_data_regime_fails_gate():
    # ~5.5 months daily, ~220 configs tried, single bull regime, ~flat SR.
    v = gate_verdict(sr=0.0, n_obs=120, n_trials=220, var_sr=0.02, n_regimes=1)
    assert v["passed"] is False
    assert len(v["reasons"]) >= 2  # DSR + regime (+ maybe obs)
