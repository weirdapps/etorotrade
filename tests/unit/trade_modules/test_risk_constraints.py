"""Tests for risk_constraints.py — executable risk constraint checks."""

import pytest

from trade_modules.risk_constraints import (
    ConstraintResult,
    check_all_constraints,
    check_cash_reserve,
    check_portfolio_drawdown,
    check_position_stoploss,
    check_sector_concentration,
    check_single_position_drift,
)


class TestCheckCashReserve:
    """Tests for check_cash_reserve."""

    def test_cash_above_minimum_passes(self):
        result = check_cash_reserve(20.0)
        assert result.passed is True
        assert result.severity == "INFO"
        assert "above" in result.message

    def test_cash_below_minimum_blocks(self):
        result = check_cash_reserve(10.0)
        assert result.passed is False
        assert result.severity == "BLOCK"
        assert "below" in result.message

    def test_cash_at_exact_minimum_passes(self):
        result = check_cash_reserve(15.0)
        assert result.passed is True

    def test_custom_threshold(self):
        result = check_cash_reserve(12.0, min_cash_pct=10.0)
        assert result.passed is True

    def test_result_structure(self):
        result = check_cash_reserve(20.0)
        assert result.name == "cash_reserve"
        assert result.current_value == pytest.approx(20.0)
        assert result.threshold == pytest.approx(15.0)
        assert result.override_allowed is True


class TestCheckPositionStoploss:
    """Tests for check_position_stoploss."""

    def test_within_limits_passes(self):
        result = check_position_stoploss(-5.0, above_200dma=True)
        assert result.passed is True
        assert result.severity == "INFO"

    def test_below_threshold_but_above_200dma_passes(self):
        """Thesis intact if above 200DMA even when below threshold."""
        result = check_position_stoploss(-25.0, above_200dma=True)
        assert result.passed is True
        assert result.severity == "WARNING"
        assert "thesis intact" in result.message

    def test_below_threshold_and_below_200dma_blocks(self):
        """Both conditions met = stop-loss triggered."""
        result = check_position_stoploss(-25.0, above_200dma=False)
        assert result.passed is False
        assert result.severity == "BLOCK"
        assert "STOP-LOSS TRIGGERED" in result.message

    def test_at_exact_threshold_and_below_200dma(self):
        result = check_position_stoploss(-20.0, above_200dma=False)
        assert result.passed is False
        assert result.severity == "BLOCK"


class TestCheckPortfolioDrawdown:
    """Tests for check_portfolio_drawdown."""

    def test_no_drawdown_passes(self):
        result = check_portfolio_drawdown(0.0)
        assert result.passed is True
        assert result.severity == "INFO"

    def test_mild_drawdown_passes(self):
        result = check_portfolio_drawdown(-10.0)
        assert result.passed is True

    def test_severe_drawdown_blocks(self):
        result = check_portfolio_drawdown(-20.0)
        assert result.passed is False
        assert result.severity == "BLOCK"
        assert "REDUCE TO 50% EQUITY" in result.message

    def test_at_exact_threshold_blocks(self):
        result = check_portfolio_drawdown(-15.0)
        assert result.passed is False
        assert result.severity == "BLOCK"


class TestCheckSinglePositionDrift:
    """Tests for check_single_position_drift."""

    def test_within_limits(self):
        result = check_single_position_drift(5.0)
        assert result.passed is True
        assert result.severity == "INFO"

    def test_above_drift_threshold(self):
        result = check_single_position_drift(8.5)
        assert result.passed is False
        assert result.severity == "WARNING"
        assert "REBALANCE" in result.message

    def test_at_exact_threshold_passes(self):
        """7.0 is not > 7.0, so it passes."""
        result = check_single_position_drift(7.0)
        assert result.passed is True

    def test_custom_threshold(self):
        result = check_single_position_drift(6.0, max_pct=5.0)
        assert result.passed is False


class TestCheckSectorConcentration:
    """Tests for check_sector_concentration."""

    def test_low_concentration_passes(self):
        result = check_sector_concentration(15.0)
        assert result.passed is True
        assert result.severity == "INFO"

    def test_approaching_cap_warns(self):
        """80% of 25% = 20%, so 21% should warn."""
        result = check_sector_concentration(21.0)
        assert result.passed is True
        assert result.severity == "WARNING"
        assert "approaching cap" in result.message

    def test_at_cap_blocks(self):
        result = check_sector_concentration(25.0)
        assert result.passed is False
        assert result.severity == "BLOCK"
        assert "BLOCKED" in result.message

    def test_above_cap_blocks(self):
        result = check_sector_concentration(30.0)
        assert result.passed is False
        assert result.severity == "BLOCK"


class TestCheckAllConstraints:
    """Tests for check_all_constraints aggregation."""

    def test_all_passing(self):
        results = check_all_constraints(available_cash_pct=20.0, portfolio_drawdown_pct=-5.0)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_cash_failing(self):
        results = check_all_constraints(available_cash_pct=5.0, portfolio_drawdown_pct=0.0)
        assert not results[0].passed  # cash_reserve
        assert results[1].passed  # portfolio_drawdown

    def test_drawdown_failing(self):
        results = check_all_constraints(available_cash_pct=20.0, portfolio_drawdown_pct=-20.0)
        assert results[0].passed  # cash_reserve
        assert not results[1].passed  # portfolio_drawdown

    def test_both_failing(self):
        results = check_all_constraints(available_cash_pct=5.0, portfolio_drawdown_pct=-20.0)
        assert not results[0].passed
        assert not results[1].passed

    def test_returns_constraint_result_instances(self):
        results = check_all_constraints()
        for r in results:
            assert isinstance(r, ConstraintResult)
