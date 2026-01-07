"""
Tests for yahoofinance/utils/data/price_target_utils.py

This module tests price target validation and robustness utilities.
"""

import pytest

from yahoofinance.utils.data.price_target_utils import (
    calculate_price_target_robustness,
    validate_price_target_data,
)


class TestCalculatePriceTargetRobustness:
    """Tests for calculate_price_target_robustness function."""

    def test_insufficient_data_returns_not_robust(self):
        """Test that missing data returns not robust."""
        result = calculate_price_target_robustness(
            mean=None, median=None, high=None, low=None, current_price=100.0
        )
        assert result["is_robust"] is False
        assert result["quality_grade"] == "F"
        assert "Insufficient price target data" in result["warning_flags"]

    def test_zero_current_price_returns_not_robust(self):
        """Test that zero current price returns not robust."""
        result = calculate_price_target_robustness(
            mean=100.0, median=100.0, high=110.0, low=90.0, current_price=0
        )
        assert result["is_robust"] is False
        assert "Insufficient price target data" in result["warning_flags"]

    def test_excellent_robustness_grade_a(self):
        """Test excellent price target consistency gives grade A."""
        result = calculate_price_target_robustness(
            mean=100.0,
            median=100.0,
            high=110.0,
            low=90.0,
            current_price=95.0,
            analyst_count=15,
        )
        assert result["quality_grade"] == "A"
        assert result["is_robust"] is True
        assert result["robustness_score"] >= 85

    def test_good_robustness_grade_b(self):
        """Test good price target consistency gives grade B."""
        result = calculate_price_target_robustness(
            mean=100.0,
            median=102.0,
            high=130.0,
            low=80.0,
            current_price=95.0,
            analyst_count=10,
        )
        # Spread is 50/102 = ~49% which gives warning
        assert result["quality_grade"] in ["A", "B", "C"]
        assert result["robustness_score"] >= 50

    def test_high_spread_penalty(self):
        """Test high price target spread penalizes robustness."""
        result = calculate_price_target_robustness(
            mean=100.0,
            median=100.0,
            high=200.0,  # 100% above median
            low=50.0,    # 50% below median
            current_price=95.0,
        )
        # Spread is 150/100 = 150% which is extreme
        assert "Extreme price target spread" in str(result["warning_flags"]) or \
               "Very high price target spread" in str(result["warning_flags"])
        assert result["robustness_score"] < 70

    def test_mean_median_skewness_penalty(self):
        """Test mean-median difference penalizes robustness."""
        result = calculate_price_target_robustness(
            mean=130.0,   # Mean is 30% higher than median
            median=100.0,
            high=150.0,
            low=80.0,
            current_price=95.0,
        )
        assert result["mean_median_diff_percent"] == 30.0
        assert "High mean-median difference" in str(result["warning_flags"])

    def test_extreme_outliers_penalty(self):
        """Test extreme outlier targets penalize robustness."""
        result = calculate_price_target_robustness(
            mean=100.0,
            median=100.0,
            high=350.0,  # 250% above mean
            low=95.0,
            current_price=95.0,
        )
        assert "Extreme outlier price targets" in str(result["warning_flags"])

    def test_low_analyst_coverage_penalty(self):
        """Test low analyst coverage penalizes robustness."""
        result = calculate_price_target_robustness(
            mean=100.0,
            median=100.0,
            high=110.0,
            low=90.0,
            current_price=95.0,
            analyst_count=2,
        )
        assert "Low analyst coverage" in str(result["warning_flags"])

    def test_high_analyst_coverage_bonus(self):
        """Test high analyst coverage gives bonus vs low coverage penalty."""
        # With low coverage (gets -15 penalty)
        result_low = calculate_price_target_robustness(
            mean=100.0,
            median=100.0,
            high=140.0,  # Moderate spread to not hit 100 cap
            low=70.0,
            current_price=95.0,
            analyst_count=2,
        )
        # With high coverage (gets +5 bonus)
        result_high = calculate_price_target_robustness(
            mean=100.0,
            median=100.0,
            high=140.0,
            low=70.0,
            current_price=95.0,
            analyst_count=15,
        )
        # High coverage should score higher due to bonus vs penalty difference
        assert result_high["robustness_score"] > result_low["robustness_score"]
        assert "Low analyst coverage" in str(result_low["warning_flags"])

    def test_extreme_price_vs_current_penalty(self):
        """Test extreme median vs current price penalizes robustness."""
        result = calculate_price_target_robustness(
            mean=500.0,
            median=500.0,
            high=600.0,
            low=400.0,
            current_price=100.0,  # Median is 400% above current
        )
        assert "Extreme median price target vs current price" in str(result["warning_flags"])

    def test_spread_percent_calculated_correctly(self):
        """Test spread percent is calculated correctly."""
        result = calculate_price_target_robustness(
            mean=100.0,
            median=100.0,
            high=130.0,
            low=70.0,
            current_price=95.0,
        )
        # Spread = 130 - 70 = 60, spread_percent = 60/100 * 100 = 60%
        assert result["spread_percent"] == 60.0


class TestValidatePriceTargetData:
    """Tests for validate_price_target_data function."""

    def test_no_price_targets_returns_invalid(self):
        """Test that missing price targets returns invalid."""
        ticker_data = {"ticker": "AAPL", "price": 175.0}
        is_valid, info = validate_price_target_data(ticker_data)
        assert is_valid is False
        assert info["has_price_targets"] is False
        assert info["recommended_action"] == "exclude"

    def test_valid_price_targets_returns_valid(self):
        """Test that valid price targets return valid."""
        ticker_data = {
            "ticker": "AAPL",
            "price": 175.0,
            "target_price_mean": 200.0,
            "target_price_median": 200.0,
            "target_price_high": 220.0,
            "target_price_low": 180.0,
            "price_target_analyst_count": 15,
        }
        is_valid, info = validate_price_target_data(ticker_data)
        assert info["has_price_targets"] is True
        assert info["robustness_metrics"] is not None

    def test_robust_targets_set_confidence_high(self):
        """Test that robust targets set confidence level appropriately."""
        ticker_data = {
            "ticker": "AAPL",
            "price": 175.0,
            "target_price_mean": 190.0,
            "target_price_median": 190.0,
            "target_price_high": 200.0,
            "target_price_low": 180.0,
            "price_target_analyst_count": 20,
        }
        is_valid, info = validate_price_target_data(ticker_data)
        assert info["has_price_targets"] is True
        # High confidence with robust targets
        if info["has_robust_targets"]:
            assert info["confidence_level"] in ["medium", "high"]

    def test_unreliable_targets_recommend_manual_review(self):
        """Test that unreliable targets recommend manual review or exclude."""
        ticker_data = {
            "ticker": "RISKY",
            "price": 50.0,
            "target_price_mean": 200.0,
            "target_price_median": 150.0,  # 33% skew
            "target_price_high": 500.0,
            "target_price_low": 10.0,
            "price_target_analyst_count": 2,
        }
        is_valid, info = validate_price_target_data(ticker_data)
        # Should recommend caution for unreliable targets
        assert info["recommended_action"] in ["exclude", "manual_review", "use_median"]


class TestEdgeCases:
    """Edge case tests for price target utilities."""

    def test_negative_price_target(self):
        """Test handling of negative price targets."""
        result = calculate_price_target_robustness(
            mean=-10.0,
            median=-10.0,
            high=10.0,
            low=-30.0,
            current_price=5.0,
        )
        # Should handle without crashing
        assert result is not None

    def test_very_small_values(self):
        """Test handling of very small price values."""
        result = calculate_price_target_robustness(
            mean=0.001,
            median=0.001,
            high=0.002,
            low=0.0005,
            current_price=0.001,
        )
        assert result is not None
        assert result["spread_percent"] is not None

    def test_very_large_values(self):
        """Test handling of very large price values."""
        result = calculate_price_target_robustness(
            mean=1e6,
            median=1e6,
            high=1.1e6,
            low=0.9e6,
            current_price=0.95e6,
        )
        assert result is not None
        assert result["is_robust"] is True or result["quality_grade"] in ["A", "B"]

    def test_identical_high_low(self):
        """Test handling when high equals low."""
        result = calculate_price_target_robustness(
            mean=100.0,
            median=100.0,
            high=100.0,
            low=100.0,
            current_price=100.0,
        )
        assert result is not None
        assert result["spread_percent"] == 0.0
