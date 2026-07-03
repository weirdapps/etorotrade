import numpy as np

from trade_modules.riskfirst.regime_overlay import (
    confirm_regime,
    exposure_for_regime,
    scale_for_regime,
)


def test_exposure_known_regimes():
    assert exposure_for_regime("risk_on") == 1.00
    assert exposure_for_regime("neutral") == 0.90
    assert exposure_for_regime("risk_off") == 0.65
    assert exposure_for_regime("crisis") == 0.40


def test_exposure_unknown_or_none_falls_back_to_neutral():
    assert exposure_for_regime(None) == 0.90
    assert exposure_for_regime("garbage") == 0.90


def test_confirm_switches_only_after_persistence():
    # risk_off held 2 days (>= persistence) -> confirmed; trailing single risk_on ignored
    hist = ["neutral", "risk_off", "risk_off", "risk_on"]
    assert confirm_regime(hist, persistence_days=2) == "risk_off"


def test_confirm_keeps_most_recent_qualifying_run():
    hist = ["risk_off", "risk_off", "risk_on", "risk_on", "crisis"]
    assert confirm_regime(hist, persistence_days=2) == "risk_on"


def test_confirm_oscillation_no_qualifying_run_falls_back():
    assert confirm_regime(["risk_on", "risk_off", "risk_on"], persistence_days=2) == "neutral"


def test_confirm_cold_start_too_short_falls_back():
    assert confirm_regime(["risk_off"], persistence_days=2) == "neutral"


def test_scale_reduces_gross_and_clamps():
    w = np.array([0.4, 0.4, 0.2])
    np.testing.assert_allclose(scale_for_regime(w, 0.5), np.array([0.2, 0.2, 0.1]))
    np.testing.assert_allclose(scale_for_regime(w, 1.0), w)  # no-op
    np.testing.assert_allclose(scale_for_regime(w, 2.0), w)  # clamp up to 1.0
    np.testing.assert_allclose(scale_for_regime(w, -1.0), w * 0.0)  # clamp down to 0.0
