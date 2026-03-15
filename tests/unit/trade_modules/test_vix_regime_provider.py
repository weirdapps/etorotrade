"""Tests for VIX regime provider.

CIO Review v2: Threshold adjustments neutralized. Signal criteria held constant;
risk is managed through position sizing only. All multipliers are 1.0, all offsets are 0.
"""

import pytest
from unittest.mock import patch
from trade_modules.vix_regime_provider import (
    VixRegime,
    REGIME_ADJUSTMENTS,
    REGIME_POSITION_MULTIPLIERS,
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

    def test_all_regimes_are_neutral(self):
        """CIO v2: All regimes should be neutral (no threshold adjustments)."""
        for regime in VixRegime:
            adj = REGIME_ADJUSTMENTS[regime]
            assert adj["min_upside_multiplier"] == 1.0, f"{regime.value} upside not neutral"
            assert adj["min_buy_pct_multiplier"] == 1.0, f"{regime.value} buy_pct not neutral"
            assert adj["min_exret_multiplier"] == 1.0, f"{regime.value} exret not neutral"
            assert adj["min_analysts_offset"] == 0, f"{regime.value} analysts not neutral"
            assert adj["max_upside_sell_offset"] == 0.0, f"{regime.value} sell offset not neutral"
            assert adj["max_pct_52w_buy_multiplier"] == 1.0, f"{regime.value} 52w not neutral"
            assert adj["max_pe_multiplier"] == 1.0, f"{regime.value} PE not neutral"

    def test_position_sizing_still_varies_by_regime(self):
        """CIO v2: Position sizing multipliers should still vary by regime."""
        assert REGIME_POSITION_MULTIPLIERS[VixRegime.LOW] == 1.00
        assert REGIME_POSITION_MULTIPLIERS[VixRegime.NORMAL] == 1.00
        assert REGIME_POSITION_MULTIPLIERS[VixRegime.ELEVATED] == 0.75
        assert REGIME_POSITION_MULTIPLIERS[VixRegime.HIGH] == 0.50


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
    def test_all_regimes_no_change(self, mock_adj):
        """CIO v2: All regimes should produce no threshold changes."""
        for regime in VixRegime:
            mock_adj.return_value = REGIME_ADJUSTMENTS[regime].copy()
            result = adjust_buy_criteria(self.SAMPLE_BUY_CONFIG)
            assert result["min_upside"] == 10
            assert result["min_buy_percentage"] == 75
            assert result["min_exret"] == 6
            assert result["min_analysts"] == 8
            assert result["max_forward_pe"] == 60
            assert result["max_trailing_pe"] == 90

    def test_no_adjustment_when_disabled(self):
        result = adjust_buy_criteria(self.SAMPLE_BUY_CONFIG, apply_adjustments=False)
        assert result == self.SAMPLE_BUY_CONFIG

    @patch("trade_modules.vix_regime_provider.get_regime_adjustments")
    def test_min_analysts_floor(self, mock_adj):
        """min_analysts should never go below 4."""
        mock_adj.return_value = REGIME_ADJUSTMENTS[VixRegime.HIGH].copy()
        config = {"min_analysts": 4}
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
    def test_all_regimes_no_change(self, mock_adj):
        """CIO v2: All regimes should produce no sell threshold changes."""
        for regime in VixRegime:
            mock_adj.return_value = REGIME_ADJUSTMENTS[regime].copy()
            result = adjust_sell_criteria(self.SAMPLE_SELL_CONFIG)
            assert result["max_upside"] == 0
            assert result["max_exret"] == 2


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
