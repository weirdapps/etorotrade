"""Tests for trade_modules/modifier_governance.py

Covers all public constants and policy values defined in the
modifier governance module (CIO v44.0).
"""

from trade_modules.modifier_governance import (
    CALIBRATION_EXPIRY_DAYS,
    DEFAULT_STATE,
    GOVERNANCE_VERSION,
    PROMOTION_CRITERIA,
)


class TestGovernanceVersion:
    """Verify the governance version constant."""

    def test_version_string(self):
        assert GOVERNANCE_VERSION == "v44.0"

    def test_version_is_string_type(self):
        assert isinstance(GOVERNANCE_VERSION, str)


class TestDefaultState:
    """Verify the default modifier state."""

    def test_default_state_is_shadow(self):
        assert DEFAULT_STATE == "SHADOW"

    def test_default_state_is_uppercase(self):
        assert DEFAULT_STATE == DEFAULT_STATE.upper()


class TestCalibrationExpiryDays:
    """Verify the calibration expiry window."""

    def test_expiry_is_90_days(self):
        assert CALIBRATION_EXPIRY_DAYS == 90

    def test_expiry_is_int(self):
        assert isinstance(CALIBRATION_EXPIRY_DAYS, int)

    def test_expiry_is_positive(self):
        assert CALIBRATION_EXPIRY_DAYS > 0


class TestPromotionCriteria:
    """Verify all four promotion criteria are present and well-formed."""

    def test_has_four_criteria(self):
        assert len(PROMOTION_CRITERIA) == 4

    def test_required_keys_present(self):
        expected_keys = {
            "bh_significance",
            "out_of_sample",
            "net_of_cost",
            "correct_sign",
        }
        assert set(PROMOTION_CRITERIA.keys()) == expected_keys

    def test_all_values_are_nonempty_strings(self):
        for key, value in PROMOTION_CRITERIA.items():
            assert isinstance(value, str), f"{key} value is not a string"
            assert len(value) > 0, f"{key} has empty description"

    def test_bh_significance_mentions_p_value(self):
        assert "p" in PROMOTION_CRITERIA["bh_significance"].lower()

    def test_out_of_sample_mentions_after(self):
        assert "after" in PROMOTION_CRITERIA["out_of_sample"].lower()

    def test_net_of_cost_mentions_spread(self):
        assert "spread" in PROMOTION_CRITERIA["net_of_cost"].lower()

    def test_correct_sign_mentions_rho(self):
        assert "rho" in PROMOTION_CRITERIA["correct_sign"].lower()

    def test_criteria_is_dict(self):
        assert isinstance(PROMOTION_CRITERIA, dict)
