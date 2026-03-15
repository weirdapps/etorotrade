"""Tests for VIX regime provider with comprehensive threshold adjustments."""

import pytest
from unittest.mock import patch
from trade_modules.vix_regime_provider import (
    VixRegime,
    REGIME_ADJUSTMENTS,
    adjust_buy_criteria,
    adjust_sell_criteria,
    get_adjusted_thresholds,
    get_regime_context,
    get_vix_regime,
)


class TestRegimeAdjustments:
    """Test that regime adjustments are comprehensive and consistent."""

    def test_all_regimes_have_all_keys(self):
        """Every regime must define the same set of adjustment keys."""
        expected_keys = {
            "min_upside_multiplier",
            "min_buy_pct_multiplier",
            "min_exret_multiplier",
            "min_analysts_offset",
            "max_upside_sell_offset",
            "max_pct_52w_buy_multiplier",
            "max_pe_multiplier",
        }
        for regime in VixRegime:
            assert set(REGIME_ADJUSTMENTS[regime].keys()) == expected_keys, (
                f"{regime.value} missing keys: "
                f"{expected_keys - set(REGIME_ADJUSTMENTS[regime].keys())}"
            )

    def test_normal_regime_is_neutral(self):
        """Normal regime should not change anything."""
        adj = REGIME_ADJUSTMENTS[VixRegime.NORMAL]
        assert adj["min_upside_multiplier"] == 1.0
        assert adj["min_buy_pct_multiplier"] == 1.0
        assert adj["min_exret_multiplier"] == 1.0
        assert adj["min_analysts_offset"] == 0
        assert adj["max_upside_sell_offset"] == 0.0
        assert adj["max_pct_52w_buy_multiplier"] == 1.0
        assert adj["max_pe_multiplier"] == 1.0

    def test_low_vix_tightens_buy(self):
        """Low VIX (risk-on) should make BUY criteria stricter."""
        adj = REGIME_ADJUSTMENTS[VixRegime.LOW]
        # Higher multiplier = require more upside
        assert adj["min_upside_multiplier"] > 1.0
        assert adj["min_buy_pct_multiplier"] >= 1.0
        assert adj["min_exret_multiplier"] > 1.0
        # Tighter PE caps
        assert adj["max_pe_multiplier"] < 1.0

    def test_high_vix_loosens_buy(self):
        """High VIX (risk-off) should make BUY criteria more forgiving."""
        adj = REGIME_ADJUSTMENTS[VixRegime.HIGH]
        # Lower multiplier = require less upside
        assert adj["min_upside_multiplier"] < 1.0
        assert adj["min_buy_pct_multiplier"] < 1.0
        assert adj["min_exret_multiplier"] < 1.0
        # Looser PE caps
        assert adj["max_pe_multiplier"] > 1.0
        # Less aggressive sells
        assert adj["max_upside_sell_offset"] > 0

    def test_elevated_is_between_normal_and_high(self):
        """Elevated should be moderate risk-off adjustments."""
        normal = REGIME_ADJUSTMENTS[VixRegime.NORMAL]
        elevated = REGIME_ADJUSTMENTS[VixRegime.ELEVATED]
        high = REGIME_ADJUSTMENTS[VixRegime.HIGH]

        # Elevated should be between normal and high for sell offset
        assert normal["max_upside_sell_offset"] < elevated["max_upside_sell_offset"]
        assert elevated["max_upside_sell_offset"] < high["max_upside_sell_offset"]


class TestAdjustBuyCriteria:
    """Test buy criteria adjustment function."""

    SAMPLE_BUY_CONFIG = {
        "min_upside": 10,
        "min_buy_percentage": 75,
        "min_exret": 6,
        "min_analysts": 8,
        "min_pct_from_52w_high": 45,
        "max_forward_pe": 60,
        "max_trailing_pe": 90,
    }

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_normal_regime_no_change(self, mock_adj):
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.NORMAL].copy()
        result = adjust_buy_criteria(self.SAMPLE_BUY_CONFIG)
        assert result["min_upside"] == 10
        assert result["min_buy_percentage"] == 75
        assert result["min_exret"] == 6
        assert result["min_analysts"] == 8

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_high_vix_loosens_thresholds(self, mock_adj):
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.HIGH].copy()
        result = adjust_buy_criteria(self.SAMPLE_BUY_CONFIG)
        # Should be lower than original (more forgiving)
        assert result["min_upside"] < 10
        assert result["min_buy_percentage"] < 75
        assert result["min_exret"] < 6
        assert result["min_analysts"] == 7  # -1 offset, min 4

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_low_vix_tightens_thresholds(self, mock_adj):
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.LOW].copy()
        result = adjust_buy_criteria(self.SAMPLE_BUY_CONFIG)
        assert result["min_upside"] > 10
        assert result["max_forward_pe"] < 60
        assert result["max_trailing_pe"] < 90

    def test_no_adjustment_when_disabled(self):
        result = adjust_buy_criteria(self.SAMPLE_BUY_CONFIG, apply_adjustments=False)
        assert result == self.SAMPLE_BUY_CONFIG

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_min_analysts_floor(self, mock_adj):
        """min_analysts should never go below 4."""
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.HIGH].copy()
        config = {"min_analysts": 4}  # Already at minimum
        result = adjust_buy_criteria(config)
        assert result["min_analysts"] >= 4

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_buy_pct_cap_at_95(self, mock_adj):
        """min_buy_percentage should never exceed 95%."""
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.LOW].copy()
        config = {"min_buy_percentage": 93}
        result = adjust_buy_criteria(config)
        assert result["min_buy_percentage"] <= 95.0


class TestAdjustSellCriteria:
    """Test sell criteria adjustment function."""

    SAMPLE_SELL_CONFIG = {
        "max_upside": 0,
        "max_exret": 2,
    }

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_high_vix_relaxes_sells(self, mock_adj):
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.HIGH].copy()
        result = adjust_sell_criteria(self.SAMPLE_SELL_CONFIG)
        # Should add buffer to sell triggers (harder to trigger sell)
        assert result["max_upside"] > 0
        assert result["max_exret"] > 2


class TestGetAdjustedThresholds:
    """Test the unified threshold adjustment interface."""

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_buy_type(self, mock_adj):
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.NORMAL].copy()
        config = {"min_upside": 10}
        result = get_adjusted_thresholds(config, "buy")
        assert result["min_upside"] == 10

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_sell_type(self, mock_adj):
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.NORMAL].copy()
        config = {"max_upside": 0}
        result = get_adjusted_thresholds(config, "sell")
        assert result["max_upside"] == 0


class TestGetRegimeContext:
    """Test regime context generation for committee reports."""

    @patch("trade_modules.vix_regime_provider.get_regime_status")
    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_context_structure(self, mock_adj, mock_status):
        mock_status.return_value = (VixRegime.NORMAL, 18.5, "Normal volatility")
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.NORMAL].copy()

        ctx = get_regime_context()
        assert ctx["regime"] == "normal"
        assert ctx["vix"] == 18.5
        assert ctx["description"] == "Normal volatility"
        assert "adjustments" in ctx
        assert "implications" in ctx

    @patch("trade_modules.vix_regime_provider.get_regime_status")
    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_high_vix_has_implications(self, mock_adj, mock_status):
        mock_status.return_value = (VixRegime.HIGH, 42.0, "High volatility - defensive mode")
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.HIGH].copy()

        ctx = get_regime_context()
        assert len(ctx["implications"]) > 0
        assert ctx["regime"] == "high"


class TestVixRegimeClassification:
    """Test VIX level to regime mapping."""

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_low_vix(self, mock_vix):
        mock_vix.return_value = 12.0
        assert get_vix_regime() == VixRegime.LOW

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_normal_vix(self, mock_vix):
        mock_vix.return_value = 20.0
        assert get_vix_regime() == VixRegime.NORMAL

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_elevated_vix(self, mock_vix):
        mock_vix.return_value = 30.0
        assert get_vix_regime() == VixRegime.ELEVATED

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_high_vix(self, mock_vix):
        mock_vix.return_value = 40.0
        assert get_vix_regime() == VixRegime.HIGH

    @patch("trade_modules.vix_regime_provider.get_current_vix")
    def test_none_defaults_to_normal(self, mock_vix):
        mock_vix.return_value = None
        assert get_vix_regime() == VixRegime.NORMAL
