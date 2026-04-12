"""
Tests for conviction_sizer.py v2 enhancements.

Tests the five new features:
- Task #4: Continuous conviction function
- Task #6: Correlation cluster adjustment
- Task #1: Sector rotation integration
- Task #13: Data freshness integration
- Task #2: Market impact modeling
"""

import pytest

from trade_modules.conviction_sizer import (
    calculate_conviction_size,
    calculate_market_impact,
    get_cluster_size_adjustment,
    get_conviction_multiplier,
    get_freshness_multiplier,
    get_portfolio_var_scaling,
    get_sector_rotation_adjustment,
)

class TestContinuousConviction:
    """Test Task #4: Continuous conviction function."""

    def test_conviction_at_zero(self):
        """Conviction score of 0 should give 0.35 multiplier."""
        result = get_conviction_multiplier(0)
        assert result == pytest.approx(0.35)

    def test_conviction_at_fifty(self):
        """Conviction score of 50 should give 0.675 multiplier."""
        result = get_conviction_multiplier(50)
        assert result == pytest.approx(0.675, abs=0.001)

    def test_conviction_at_boundary_89_9(self):
        """Conviction score of 89.9 should be continuous (not bucket)."""
        result = get_conviction_multiplier(89.9)
        expected = 0.35 + (89.9 / 100) * 0.65
        assert result == pytest.approx(expected, abs=0.001)

    def test_conviction_at_boundary_90_1(self):
        """Conviction score of 90.1 should be continuous (not bucket)."""
        result = get_conviction_multiplier(90.1)
        expected = 0.35 + (90.1 / 100) * 0.65
        assert result == pytest.approx(expected, abs=0.001)

    def test_conviction_at_hundred(self):
        """Conviction score of 100 should give 1.0 multiplier."""
        result = get_conviction_multiplier(100)
        assert result == pytest.approx(1.0)

    def test_conviction_clamping_above(self):
        """Conviction score above 100 should clamp to 1.0."""
        result = get_conviction_multiplier(150)
        assert result == pytest.approx(1.0)

    def test_conviction_clamping_below(self):
        """Conviction score below 0 should clamp to 0.35."""
        result = get_conviction_multiplier(-10)
        assert result == pytest.approx(0.35)

class TestClusterAdjustment:
    """Test Task #6: Correlation cluster sizing."""

    def test_not_in_cluster(self):
        """Ticker not in any cluster should have 1.0 adjustment."""
        clusters = [
            {"tickers": ["AAPL", "MSFT", "GOOGL"]},
            {"tickers": ["JPM", "BAC", "WFC"]},
        ]
        result = get_cluster_size_adjustment("NVDA", clusters)
        assert result == pytest.approx(1.0)

    def test_in_cluster_of_three(self):
        """Ticker in cluster of 3 should have dampened 1/sqrt(3) adjustment.

        CIO Legacy C4: with default conviction=50, conviction_factor=0.5,
        adjustment = base + (1-base)*0.5*0.3 = dampened value > raw 1/sqrt(N).
        """
        import math
        clusters = [
            {"tickers": ["AAPL", "MSFT", "GOOGL"]},
        ]
        result = get_cluster_size_adjustment("AAPL", clusters)
        base = 1.0 / math.sqrt(3)
        conv_factor = min(50 / 100.0, 0.8)
        expected = base + (1.0 - base) * conv_factor * 0.3
        assert result == pytest.approx(expected, abs=0.001)

    def test_in_cluster_of_four(self):
        """Ticker in cluster of 4 should have dampened adjustment."""
        import math
        clusters = [
            {"tickers": ["JPM", "BAC", "WFC", "C"]},
        ]
        result = get_cluster_size_adjustment("JPM", clusters)
        base = 1.0 / math.sqrt(4)
        conv_factor = min(50 / 100.0, 0.8)
        expected = base + (1.0 - base) * conv_factor * 0.3
        assert result == pytest.approx(expected, abs=0.001)

    def test_in_cluster_of_nine(self):
        """Ticker in cluster of 9 should have dampened adjustment."""
        import math
        clusters = [
            {"tickers": ["A", "B", "C", "D", "E", "F", "G", "H", "I"]},
        ]
        result = get_cluster_size_adjustment("E", clusters)
        base = 1.0 / math.sqrt(9)
        conv_factor = min(50 / 100.0, 0.8)
        expected = base + (1.0 - base) * conv_factor * 0.3
        assert result == pytest.approx(expected, abs=0.001)

    def test_empty_clusters(self):
        """Empty clusters list should return 1.0."""
        result = get_cluster_size_adjustment("AAPL", [])
        assert result == pytest.approx(1.0)

class TestSectorRotation:
    """Test Task #1: Sector rotation integration."""

    def test_gaining_sector(self):
        """Gaining sector should get +10 adjustment."""
        rotation_context = {
            "gaining_sectors": [
                {"sector": "Technology", "change_pp": 20},
            ],
            "losing_sectors": [],
        }
        result = get_sector_rotation_adjustment("Technology", rotation_context)
        assert result["adjustment_points"] == 10
        assert result["sector_rotation_blocked"] is False

    def test_losing_sector_mild(self):
        """Losing sector <20pp should get -15 adjustment without block."""
        rotation_context = {
            "gaining_sectors": [],
            "losing_sectors": [
                {"sector": "Energy", "change_pp": -15},
            ],
        }
        result = get_sector_rotation_adjustment("Energy", rotation_context)
        assert result["adjustment_points"] == -15
        assert result["sector_rotation_blocked"] is False

    def test_losing_sector_collapse(self):
        """Losing sector >20pp should get -15 adjustment WITH block."""
        rotation_context = {
            "gaining_sectors": [],
            "losing_sectors": [
                {"sector": "Consumer Cyclical", "change_pp": -25},
            ],
        }
        result = get_sector_rotation_adjustment("Consumer Cyclical", rotation_context)
        assert result["adjustment_points"] == -15
        assert result["sector_rotation_blocked"] is True

    def test_neutral_sector(self):
        """Sector not in rotation should have 0 adjustment."""
        rotation_context = {
            "gaining_sectors": [{"sector": "Technology", "change_pp": 20}],
            "losing_sectors": [{"sector": "Energy", "change_pp": -15}],
        }
        result = get_sector_rotation_adjustment("Healthcare", rotation_context)
        assert result["adjustment_points"] == 0
        assert result["sector_rotation_blocked"] is False

    def test_empty_rotation_context(self):
        """Empty rotation context should return neutral."""
        rotation_context = {
            "gaining_sectors": [],
            "losing_sectors": [],
        }
        result = get_sector_rotation_adjustment("Technology", rotation_context)
        assert result["adjustment_points"] == 0
        assert result["sector_rotation_blocked"] is False

class TestFreshnessMultiplier:
    """Test Task #13: Data freshness integration."""

    def test_fresh_data(self):
        """Fresh data should have 1.0 multiplier."""
        result = get_freshness_multiplier("fresh")
        assert result == pytest.approx(1.0)

    def test_aging_data(self):
        """Aging data should have 0.75 multiplier."""
        result = get_freshness_multiplier("aging")
        assert result == pytest.approx(0.75)

    def test_stale_data(self):
        """Stale data should have 0.50 multiplier."""
        result = get_freshness_multiplier("stale")
        assert result == pytest.approx(0.50)

    def test_dead_data(self):
        """Dead data should have 0.0 multiplier (INCONCLUSIVE)."""
        result = get_freshness_multiplier("dead")
        assert result == pytest.approx(0.0)

    def test_unknown_staleness(self):
        """Unknown staleness should default to 1.0."""
        result = get_freshness_multiplier("unknown")
        assert result == pytest.approx(1.0)

class TestMarketImpact:
    """Test Task #2: Market impact modeling."""

    def test_mega_cap_low_impact(self):
        """MEGA cap with small position vs ADV should have low impact."""
        result = calculate_market_impact(
            position_usd=10000,
            average_daily_volume_usd=100_000_000,  # $100M ADV
            tier="MEGA",
        )
        assert result["spread_bps"] == 2
        assert result["impact_bps"] == pytest.approx(1.0, abs=0.1)  # 10k/100M * 10000
        assert result["total_cost_bps"] == pytest.approx(3.0, abs=0.1)
        assert result["cost_exceeds_alpha"] is False

    def test_small_cap_high_impact(self):
        """SMALL cap with large position vs ADV should have high impact."""
        result = calculate_market_impact(
            position_usd=50000,
            average_daily_volume_usd=500_000,  # $500k ADV
            tier="SMALL",
        )
        assert result["spread_bps"] == 15
        # 50k/500k = 0.1 = 10% of ADV -> 1000bps impact
        assert result["impact_bps"] == pytest.approx(1000, abs=1)
        assert result["total_cost_bps"] > 300
        assert result["cost_exceeds_alpha"] is True

    def test_zero_volume(self):
        """Zero ADV should assume high impact."""
        result = calculate_market_impact(
            position_usd=10000,
            average_daily_volume_usd=0,
            tier="MID",
        )
        assert result["spread_bps"] == 10
        assert result["impact_bps"] == 50  # Assumed high impact
        assert result["total_cost_bps"] == 60

    def test_tier_spreads(self):
        """Test spread_bps for each tier."""
        tiers_and_spreads = [
            ("MEGA", 2),
            ("LARGE", 5),
            ("MID", 10),
            ("SMALL", 15),
            ("MICRO", 20),
        ]
        for tier, expected_spread in tiers_and_spreads:
            result = calculate_market_impact(
                position_usd=1000,
                average_daily_volume_usd=1_000_000,
                tier=tier,
            )
            assert result["spread_bps"] == expected_spread

class TestIntegratedSizing:
    """Test integrated calculate_conviction_size with all new features."""

    def test_basic_size_no_adjustments(self):
        """Basic size with no adjustments should match existing behavior."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,  # MEGA
            conviction_score=90,
        )
        # 2500 * 5 = 12500 base
        # conviction 90 -> 0.35 + 0.9*0.65 = 0.935 multiplier
        expected_size = 12500 * 0.935
        assert result["position_size"] == pytest.approx(expected_size, abs=1)
        assert result["is_blocked"] is False

    def test_cluster_reduces_size(self):
        """Cluster adjustment should reduce position size."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=4,
            conviction_score=80,
            cluster_adjustment=0.5,  # In cluster of 4
        )
        # Without cluster: 2500 * 4 * 0.87 = 8700
        # With cluster: 8700 * 0.5 = 4350
        assert result["position_size"] < 5000
        assert result["cluster_adjustment"] == pytest.approx(0.5)

    def test_sector_rotation_adjusts_conviction(self):
        """Sector rotation should adjust conviction score."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=4,
            conviction_score=70,
            sector_adjustment=10,  # Gaining sector
        )
        # Conviction adjusted from 70 to 80
        assert result["adjusted_conviction"] == 80
        assert result["original_conviction"] == 70
        assert result["sector_adjustment"] == 10

    def test_sector_rotation_blocks(self):
        """Severe sector rotation should block position."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=4,
            conviction_score=80,
            sector_rotation_blocked=True,
        )
        assert result["position_size"] == pytest.approx(0.0)
        assert result["is_blocked"] is True
        assert result["sector_rotation_blocked"] is True

    def test_stale_data_reduces_size(self):
        """Stale data should reduce position size."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=4,
            conviction_score=80,
            freshness_multiplier=0.5,  # Stale
        )
        # Should be reduced by 50%
        assert result["freshness_multiplier"] == pytest.approx(0.5)
        assert result["position_size"] < 6000

    def test_dead_data_blocks(self):
        """Dead data should block position."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=4,
            conviction_score=80,
            freshness_multiplier=0.0,  # Dead
        )
        assert result["position_size"] == pytest.approx(0.0)
        assert result["is_blocked"] is True

    def test_market_impact_blocks(self):
        """High market impact should block position."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=4,
            conviction_score=80,
            market_impact_blocked=True,
        )
        assert result["position_size"] == pytest.approx(0.0)
        assert result["is_blocked"] is True
        assert result["market_impact_blocked"] is True

    def test_all_adjustments_combined(self):
        """Test all adjustments working together."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,
            conviction_score=70,
            regime="elevated",  # 0.75 multiplier
            cluster_adjustment=0.577,  # Cluster of 3
            sector_adjustment=10,  # Gaining sector
            freshness_multiplier=0.75,  # Aging data
        )
        # Base: 2500 * 5 = 12500
        # Conviction: 70 + 10 = 80 -> 0.87 multiplier -> 10875
        # Regime: 10875 * 0.75 = 8156.25
        # Cluster: 8156.25 * 0.577 = 4706
        # Freshness: 4706 * 0.75 = 3529.5
        assert result["position_size"] == pytest.approx(3529.5, abs=1)
        assert result["adjusted_conviction"] == 80
        assert result["is_blocked"] is False

class TestPortfolioVarScaling:
    """Tests for portfolio VaR-based position scaling (CIO v3 F10)."""

    def test_var_within_budget_no_scaling(self):
        """VaR within budget returns 1.0."""
        assert get_portfolio_var_scaling(1.5) == pytest.approx(1.0)
        assert get_portfolio_var_scaling(2.5) == pytest.approx(1.0)

    def test_var_unknown_no_scaling(self):
        """Unknown VaR returns 1.0."""
        assert get_portfolio_var_scaling(None) == pytest.approx(1.0)

    def test_var_exceeds_trigger(self):
        """VaR above trigger but below max scales linearly."""
        # Midpoint between 2.5 and 5.0 should give 0.75
        result = get_portfolio_var_scaling(3.75)
        assert result == pytest.approx(0.75, abs=0.01)

    def test_var_at_max(self):
        """VaR at max threshold gives 0.5."""
        assert get_portfolio_var_scaling(5.0) == pytest.approx(0.5)

    def test_var_emergency_blocks(self):
        """VaR above 2× max blocks all new positions."""
        assert get_portfolio_var_scaling(10.0) == pytest.approx(0.0)
        assert get_portfolio_var_scaling(15.0) == pytest.approx(0.0)

    def test_var_scaling_in_position_size(self):
        """VaR scaling reduces position size correctly."""
        # Without VaR scaling
        result_normal = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,
            conviction_score=70,
        )

        # With VaR scaling at 0.8
        result_scaled = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,
            conviction_score=70,
            portfolio_var_scaling=0.8,
        )

        assert result_scaled["position_size"] == pytest.approx(
            result_normal["position_size"] * 0.8, abs=1
        )
        assert result_scaled["portfolio_var_scaling"] == pytest.approx(0.8)
