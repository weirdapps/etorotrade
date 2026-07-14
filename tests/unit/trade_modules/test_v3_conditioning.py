"""Unit tests for trade_modules.v3.conditioning."""

from __future__ import annotations

import pytest

from trade_modules.v3.conditioning import (
    DEPLOYMENT_BY_REGIME,
    polymarket_adjustment,
    regime_deployment,
    resolve_deployment,
)

# ---------------------------------------------------------------------------
# DEPLOYMENT_BY_REGIME constant
# ---------------------------------------------------------------------------


def test_deployment_by_regime_values() -> None:
    assert DEPLOYMENT_BY_REGIME["risk_off"] == 0.78
    assert DEPLOYMENT_BY_REGIME["neutral"] == 0.88
    assert DEPLOYMENT_BY_REGIME["risk_on"] == 0.98


# ---------------------------------------------------------------------------
# regime_deployment
# ---------------------------------------------------------------------------


class TestRegimeDeployment:
    def test_risk_off(self) -> None:
        assert regime_deployment("risk_off") == pytest.approx(0.78)

    def test_neutral(self) -> None:
        assert regime_deployment("neutral") == pytest.approx(0.88)

    def test_risk_on(self) -> None:
        assert regime_deployment("risk_on") == pytest.approx(0.98)

    def test_unknown_returns_neutral(self) -> None:
        assert regime_deployment("unknown") == pytest.approx(0.88)

    def test_empty_string_returns_neutral(self) -> None:
        assert regime_deployment("") == pytest.approx(0.88)

    def test_none_as_string_returns_neutral(self) -> None:
        # None is not in the dict; falls back to neutral.
        assert regime_deployment("None") == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# polymarket_adjustment
# ---------------------------------------------------------------------------


class TestPolymarketAdjustment:
    # Inert cases — must return exactly 0.0
    def test_signal_none_returns_zero(self) -> None:
        assert polymarket_adjustment(None) == 0.0

    def test_max_tilt_zero_returns_zero(self) -> None:
        assert polymarket_adjustment(0.8) == 0.0

    def test_both_none_and_zero_tilt(self) -> None:
        assert polymarket_adjustment(None, max_tilt=0.0) == 0.0

    def test_positive_signal_zero_tilt(self) -> None:
        assert polymarket_adjustment(1.0, max_tilt=0.0) == 0.0

    # Enabled cases — nonzero max_tilt
    def test_full_positive_signal(self) -> None:
        adj = polymarket_adjustment(1.0, max_tilt=0.05)
        assert adj == pytest.approx(0.05)

    def test_full_negative_signal(self) -> None:
        adj = polymarket_adjustment(-1.0, max_tilt=0.05)
        assert adj == pytest.approx(-0.05)

    def test_partial_signal(self) -> None:
        adj = polymarket_adjustment(0.5, max_tilt=0.04)
        assert adj == pytest.approx(0.02)

    def test_signal_clipped_above(self) -> None:
        adj = polymarket_adjustment(2.0, max_tilt=0.05)
        assert abs(adj) <= 0.05

    def test_signal_clipped_below(self) -> None:
        adj = polymarket_adjustment(-3.0, max_tilt=0.05)
        assert abs(adj) <= 0.05

    def test_magnitude_bounded_by_max_tilt(self) -> None:
        for sig in (-1.5, -1.0, 0.0, 0.75, 2.0):
            adj = polymarket_adjustment(sig, max_tilt=0.03)
            assert abs(adj) <= 0.03 + 1e-9


# ---------------------------------------------------------------------------
# resolve_deployment
# ---------------------------------------------------------------------------


class TestResolveDeployment:
    def test_known_regime_no_pm(self) -> None:
        dep, diag = resolve_deployment("risk_on")
        assert dep == pytest.approx(0.98)
        assert diag["final_deployment"] == pytest.approx(0.98)

    def test_neutral_regime(self) -> None:
        dep, diag = resolve_deployment("neutral")
        assert dep == pytest.approx(0.88)

    def test_risk_off_regime(self) -> None:
        dep, diag = resolve_deployment("risk_off")
        assert dep == pytest.approx(0.78)

    def test_unknown_regime_neutral_fallback(self) -> None:
        dep, _ = resolve_deployment("garbage")
        assert dep == pytest.approx(0.88)

    # Polymarket inert by default
    def test_polymarket_inactive_by_default(self) -> None:
        _, diag = resolve_deployment("neutral", polymarket_signal=0.9)
        assert diag["polymarket_active"] is False
        assert diag["polymarket_tilt"] == pytest.approx(0.0)
        assert diag["final_deployment"] == pytest.approx(0.88)

    def test_polymarket_active_flag(self) -> None:
        _, diag = resolve_deployment("neutral", polymarket_signal=1.0, max_pm_tilt=0.03)
        assert diag["polymarket_active"] is True

    # Band clamping
    def test_clamped_to_upper_band(self) -> None:
        # risk_on (0.98) + large positive tilt should not exceed 0.98 (hi of default band)
        dep, diag = resolve_deployment("risk_on", polymarket_signal=1.0, max_pm_tilt=0.10)
        assert dep <= 0.98 + 1e-9

    def test_clamped_to_lower_band(self) -> None:
        # risk_off (0.78) + large negative tilt should not go below 0.78 (lo of default band)
        dep, diag = resolve_deployment("risk_off", polymarket_signal=-1.0, max_pm_tilt=0.10)
        assert dep >= 0.78 - 1e-9

    def test_custom_band(self) -> None:
        dep, _ = resolve_deployment("risk_on", band=(0.70, 0.80))
        assert dep == pytest.approx(0.80)

    # dial_diag keys
    def test_dial_diag_keys_present(self) -> None:
        _, diag = resolve_deployment("neutral", polymarket_signal=0.5, max_pm_tilt=0.02)
        required = {
            "regime",
            "base_deployment",
            "polymarket_signal",
            "polymarket_tilt",
            "polymarket_active",
            "final_deployment",
        }
        assert required <= set(diag.keys())

    def test_dial_diag_signal_preserved(self) -> None:
        _, diag = resolve_deployment("neutral", polymarket_signal=0.5)
        assert diag["polymarket_signal"] == 0.5

    def test_dial_diag_regime_preserved(self) -> None:
        _, diag = resolve_deployment("risk_off")
        assert diag["regime"] == "risk_off"

    # Consistency: base_deployment matches regime_deployment
    def test_base_deployment_matches_regime_fn(self) -> None:
        for regime in ("risk_off", "neutral", "risk_on"):
            _, diag = resolve_deployment(regime)
            assert diag["base_deployment"] == pytest.approx(regime_deployment(regime))
