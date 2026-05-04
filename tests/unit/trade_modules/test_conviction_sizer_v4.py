"""
Tests for conviction_sizer.py CIO v4 enhancements.

Tests the two new features:
- F2: Holding-period-adjusted cost model (estimate_holding_cost_pct)
- F8: Portfolio-level sizing constraint (apply_portfolio_constraints)
"""

import pytest

from trade_modules.conviction_sizer import (
    apply_portfolio_constraints,
    calculate_conviction_size,
    estimate_holding_cost_pct,
)


class TestEstimateHoldingCostPct:
    """Test F2: Holding-period-adjusted cost model."""

    def test_default_params(self):
        """Default (90 days, MID tier) should return reasonable costs."""
        result = estimate_holding_cost_pct()
        assert result["holding_period_days"] == 90
        # MID spread: 10bps * 2 (round trip) = 0.20%
        assert result["spread_pct"] == pytest.approx(0.20, abs=0.01)
        # Financing: 6.4% * (90/365) = ~1.578%
        assert result["financing_pct"] == pytest.approx(1.578, abs=0.01)
        # Total ~1.778%
        assert result["total_pct"] == pytest.approx(1.778, abs=0.01)

    def test_mega_tier_30_days(self):
        """MEGA tier with 30-day hold should have minimal costs."""
        result = estimate_holding_cost_pct(holding_period_days=30, tier="MEGA")
        # MEGA spread: 2bps * 2 = 0.04%
        assert result["spread_pct"] == pytest.approx(0.04, abs=0.001)
        # Financing: 6.4% * (30/365) = ~0.526%
        assert result["financing_pct"] == pytest.approx(0.526, abs=0.01)
        assert result["total_pct"] < 1.0

    def test_micro_tier_365_days(self):
        """MICRO tier with 1-year hold should have highest costs."""
        result = estimate_holding_cost_pct(holding_period_days=365, tier="MICRO")
        # MICRO spread: 40bps * 2 = 0.80%
        assert result["spread_pct"] == pytest.approx(0.80, abs=0.01)
        # Financing: 6.4% * (365/365) = 6.4%
        assert result["financing_pct"] == pytest.approx(6.4, abs=0.01)
        # Total = 7.2%
        assert result["total_pct"] == pytest.approx(7.2, abs=0.01)

    def test_zero_holding_period(self):
        """Zero-day hold should have only spread costs, no financing."""
        result = estimate_holding_cost_pct(holding_period_days=0, tier="LARGE")
        assert result["financing_pct"] == pytest.approx(0.0)
        # LARGE spread: 5bps * 2 = 0.10%
        assert result["spread_pct"] == pytest.approx(0.10, abs=0.001)
        assert result["total_pct"] == result["spread_pct"]

    def test_all_tiers(self):
        """Costs should increase from MEGA to MICRO."""
        tiers = ["MEGA", "LARGE", "MID", "SMALL", "MICRO"]
        costs = [estimate_holding_cost_pct(tier=t)["total_pct"] for t in tiers]
        for i in range(len(costs) - 1):
            assert costs[i] < costs[i + 1], (
                f"{tiers[i]} cost {costs[i]} should be less than "
                f"{tiers[i + 1]} cost {costs[i + 1]}"
            )

    def test_longer_hold_higher_cost(self):
        """Longer holding period should always increase costs."""
        cost_30 = estimate_holding_cost_pct(holding_period_days=30)["total_pct"]
        cost_90 = estimate_holding_cost_pct(holding_period_days=90)["total_pct"]
        cost_180 = estimate_holding_cost_pct(holding_period_days=180)["total_pct"]
        assert cost_30 < cost_90 < cost_180

    def test_case_insensitive_tier(self):
        """Tier should be case-insensitive."""
        upper = estimate_holding_cost_pct(tier="MEGA")
        lower = estimate_holding_cost_pct(tier="mega")
        mixed = estimate_holding_cost_pct(tier="Mega")
        assert upper["total_pct"] == lower["total_pct"] == mixed["total_pct"]

    def test_unknown_tier_defaults_to_mid(self):
        """Unknown tier should default to MID costs."""
        unknown = estimate_holding_cost_pct(tier="UNKNOWN")
        mid = estimate_holding_cost_pct(tier="MID")
        assert unknown["total_pct"] == mid["total_pct"]

    def test_return_keys(self):
        """Result dict should contain all expected keys."""
        result = estimate_holding_cost_pct()
        assert "spread_pct" in result
        assert "financing_pct" in result
        assert "total_pct" in result
        assert "holding_period_days" in result

    def test_consistency_with_liquidity_filter(self):
        """Cost model should use same 6.4% annual rate as liquidity_filter."""
        from trade_modules.conviction_sizer import ETORO_OVERNIGHT_ANNUAL_RATE as SIZER_RATE
        from trade_modules.liquidity_filter import ETORO_OVERNIGHT_ANNUAL_RATE

        assert SIZER_RATE == ETORO_OVERNIGHT_ANNUAL_RATE


class TestHoldingCostInConvictionSize:
    """Test that holding cost flows through calculate_conviction_size."""

    def test_default_holding_cost_in_result(self):
        """Default call should include holding_cost in result."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,
            conviction_score=90,
        )
        assert "holding_cost" in result
        assert result["holding_cost"]["holding_period_days"] == 90
        assert result["holding_cost"]["total_pct"] > 0

    def test_custom_holding_period(self):
        """Custom holding_period_days should be reflected in result."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,
            conviction_score=90,
            holding_period_days=30,
            tier="MEGA",
        )
        assert result["holding_cost"]["holding_period_days"] == 30
        # MEGA 30-day cost should be well under 1%
        assert result["holding_cost"]["total_pct"] < 1.0

    def test_holding_cost_does_not_change_position_size(self):
        """Holding cost is informational; position_size should be unchanged."""
        result_90 = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,
            conviction_score=80,
            holding_period_days=90,
        )
        result_180 = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,
            conviction_score=80,
            holding_period_days=180,
        )
        # Position size should be the same (holding cost is informational)
        assert result_90["position_size"] == result_180["position_size"]
        # But holding cost should differ
        assert result_90["holding_cost"]["total_pct"] < result_180["holding_cost"]["total_pct"]

    def test_backward_compatibility(self):
        """Existing calls without holding_period_days should still work."""
        result = calculate_conviction_size(
            base_position_size=2500,
            tier_multiplier=5,
            conviction_score=90,
            regime="elevated",
            cluster_adjustment=0.577,
            sector_adjustment=10,
            freshness_multiplier=0.75,
        )
        # Verify all existing fields still present
        assert "position_size" in result
        assert "conviction_multiplier" in result
        assert "regime_multiplier" in result
        assert "cluster_adjustment" in result
        assert "freshness_multiplier" in result
        assert "is_blocked" in result
        # And new field is present with defaults
        assert "holding_cost" in result
        assert result["holding_cost"]["holding_period_days"] == 90


class TestApplyPortfolioConstraints:
    """Test F8: Portfolio-level sizing constraint."""

    def test_no_constraint_needed(self):
        """Positions that don't breach sector limit should pass through."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 5000},
            {"ticker": "JNJ", "sector": "Healthcare", "position_size": 3000},
        ]
        current_exposures = {"Technology": 15.0, "Healthcare": 10.0}
        result = apply_portfolio_constraints(
            new_positions,
            current_exposures,
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        assert len(result) == 2
        for r in result:
            assert r["was_constrained"] is False
            assert r["constrained_size"] == r["original_size"]
            assert r["constraint_reason"] == ""

    def test_single_sector_breach(self):
        """Positions that breach one sector should be constrained."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 10000},
            {"ticker": "MSFT", "sector": "Technology", "position_size": 5000},
            {"ticker": "JNJ", "sector": "Healthcare", "position_size": 3000},
        ]
        # Technology is at 22%, adding 15% (15k/100k) would push to 37%
        current_exposures = {"Technology": 22.0, "Healthcare": 10.0}
        result = apply_portfolio_constraints(
            new_positions,
            current_exposures,
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        assert len(result) == 3

        # Tech positions should be constrained
        tech_results = [r for r in result if r["sector"] == "Technology"]
        for r in tech_results:
            assert r["was_constrained"] is True
            assert r["constrained_size"] < r["original_size"]

        # Healthcare should be untouched
        hc_result = [r for r in result if r["sector"] == "Healthcare"][0]
        assert hc_result["was_constrained"] is False
        assert hc_result["constrained_size"] == 3000

    def test_proportional_reduction(self):
        """All positions in a breached sector should be reduced by the same ratio."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 10000},
            {"ticker": "MSFT", "sector": "Technology", "position_size": 5000},
        ]
        # Technology at 20%, adding 15% would push to 35%, limit is 25%
        # Available room: 25% - 20% = 5%. New total: 15%. Scale = 5/15 = 1/3
        current_exposures = {"Technology": 20.0}
        result = apply_portfolio_constraints(
            new_positions,
            current_exposures,
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        # AAPL should be scaled to 10000 * (1/3) = 3333.33
        aapl = [r for r in result if r["ticker"] == "AAPL"][0]
        assert aapl["constrained_size"] == pytest.approx(3333.33, abs=1)

        # MSFT should be scaled to 5000 * (1/3) = 1666.67
        msft = [r for r in result if r["ticker"] == "MSFT"][0]
        assert msft["constrained_size"] == pytest.approx(1666.67, abs=1)

        # Ratio should be the same for both
        aapl_ratio = aapl["constrained_size"] / aapl["original_size"]
        msft_ratio = msft["constrained_size"] / msft["original_size"]
        assert aapl_ratio == pytest.approx(msft_ratio, abs=0.001)

    def test_sector_already_at_limit(self):
        """Sector already at limit should zero out all new positions."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 5000},
        ]
        current_exposures = {"Technology": 25.0}  # Already at limit
        result = apply_portfolio_constraints(
            new_positions,
            current_exposures,
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        assert result[0]["constrained_size"] == pytest.approx(0.0)
        assert result[0]["was_constrained"] is True

    def test_sector_over_limit(self):
        """Sector already over limit should zero out all new positions."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 5000},
        ]
        current_exposures = {"Technology": 30.0}  # Already over limit
        result = apply_portfolio_constraints(
            new_positions,
            current_exposures,
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        assert result[0]["constrained_size"] == pytest.approx(0.0)
        assert result[0]["was_constrained"] is True

    def test_empty_positions(self):
        """Empty positions list should return empty list."""
        result = apply_portfolio_constraints(
            [],
            {"Technology": 20.0},
        )
        assert result == []

    def test_no_current_exposures(self):
        """No current exposures should allow positions within limit."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 20000},
        ]
        # max_single_stock_pct=100 isolates this test to sector-cap behavior;
        # the layered single-stock + sector interaction is covered separately below.
        result = apply_portfolio_constraints(
            new_positions,
            {},
            max_sector_pct=25.0,
            max_single_stock_pct=100.0,
            portfolio_value=100_000,
        )
        # 20% < 25% limit, should pass
        assert result[0]["was_constrained"] is False
        assert result[0]["constrained_size"] == 20000

    def test_new_sector_breach(self):
        """New sector with no current exposure can still breach if too large."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 30000},
        ]
        result = apply_portfolio_constraints(
            new_positions,
            {},
            max_sector_pct=25.0,
            max_single_stock_pct=100.0,
            portfolio_value=100_000,
        )
        # 30% > 25% limit, should be constrained
        # Available: 25%, new: 30%, scale = 25/30
        assert result[0]["was_constrained"] is True
        expected_size = 30000 * (25.0 / 30.0)
        assert result[0]["constrained_size"] == pytest.approx(expected_size, abs=1)

    def test_single_stock_cap_fires_before_sector(self):
        """Single-stock cap is applied first; sector cap then sees the clipped size.

        A 20% AAPL request with default 10% single-stock cap clips to 10%.
        The sector cap (15%) then sees 10% and approves — only single-stock
        constraint is recorded, not sector.
        """
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 20000},
        ]
        result = apply_portfolio_constraints(
            new_positions,
            {},
            max_sector_pct=15.0,
            max_single_stock_pct=10.0,
            portfolio_value=100_000,
        )
        assert result[0]["was_constrained"] is True
        assert result[0]["constrained_size"] == 10000  # 10% cap
        # Reason should mention single-stock, NOT sector (sector saw clipped 10%)
        assert "single-stock" in result[0]["constraint_reason"]
        assert "sector limit" not in result[0]["constraint_reason"]

    def test_both_caps_combine(self):
        """Both caps can clip the same position in sequence.

        30% AAPL request → single-stock cap (10%) clips to 10% → sector cap
        (25%, with 20% already used) leaves 5% room → final size = 5%.
        Both reasons should appear in constraint_reason.
        """
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 30000},
        ]
        result = apply_portfolio_constraints(
            new_positions,
            current_sector_exposures={"Technology": 20.0},
            max_sector_pct=25.0,
            max_single_stock_pct=10.0,
            portfolio_value=100_000,
        )
        assert result[0]["was_constrained"] is True
        # Single-stock clips 30% → 10% (size 10000), sector then scales by 5/10 = 0.5
        assert result[0]["constrained_size"] == pytest.approx(5000, abs=1)
        assert "single-stock" in result[0]["constraint_reason"]
        assert "sector limit" in result[0]["constraint_reason"]

    def test_multiple_sectors_constrained(self):
        """Multiple sectors can be independently constrained."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 15000},
            {"ticker": "XOM", "sector": "Energy", "position_size": 12000},
            {"ticker": "JNJ", "sector": "Healthcare", "position_size": 3000},
        ]
        current_exposures = {
            "Technology": 20.0,  # 20 + 15 = 35 > 25
            "Energy": 18.0,  # 18 + 12 = 30 > 25
            "Healthcare": 5.0,  # 5 + 3 = 8 < 25
        }
        result = apply_portfolio_constraints(
            new_positions,
            current_exposures,
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        tech = [r for r in result if r["ticker"] == "AAPL"][0]
        energy = [r for r in result if r["ticker"] == "XOM"][0]
        healthcare = [r for r in result if r["ticker"] == "JNJ"][0]

        assert tech["was_constrained"] is True
        assert energy["was_constrained"] is True
        assert healthcare["was_constrained"] is False

    def test_custom_max_sector_pct(self):
        """Custom max_sector_pct should be respected."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 5000},
        ]
        # With 30% limit, 20% + 5% = 25% is fine
        result_30 = apply_portfolio_constraints(
            new_positions,
            {"Technology": 20.0},
            max_sector_pct=30.0,
            portfolio_value=100_000,
        )
        assert result_30[0]["was_constrained"] is False

        # With 22% limit, 20% + 5% = 25% breaches
        result_22 = apply_portfolio_constraints(
            new_positions,
            {"Technology": 20.0},
            max_sector_pct=22.0,
            portfolio_value=100_000,
        )
        assert result_22[0]["was_constrained"] is True

    def test_without_portfolio_value(self):
        """Without portfolio_value, position_size treated as percentage."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 10.0},
        ]
        # 15 + 10 = 25, just at the limit
        result_at = apply_portfolio_constraints(
            new_positions,
            {"Technology": 15.0},
            max_sector_pct=25.0,
            portfolio_value=None,
        )
        assert result_at[0]["was_constrained"] is False

        # 15 + 11 = 26, just over the limit
        new_positions_over = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 11.0},
        ]
        result_over = apply_portfolio_constraints(
            new_positions_over,
            {"Technology": 15.0},
            max_sector_pct=25.0,
            portfolio_value=None,
        )
        assert result_over[0]["was_constrained"] is True

    def test_result_structure(self):
        """Result dicts should contain all expected keys."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 5000},
        ]
        result = apply_portfolio_constraints(
            new_positions,
            {"Technology": 20.0},
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        assert len(result) == 1
        r = result[0]
        assert "ticker" in r
        assert "sector" in r
        assert "original_size" in r
        assert "constrained_size" in r
        assert "was_constrained" in r
        assert "constraint_reason" in r

    def test_constraint_reason_populated_when_constrained(self):
        """constraint_reason should describe the constraint when applied."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 20000},
        ]
        result = apply_portfolio_constraints(
            new_positions,
            {"Technology": 22.0},
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        r = result[0]
        assert r["was_constrained"] is True
        assert "Technology" in r["constraint_reason"]
        assert "22.0%" in r["constraint_reason"]
        assert "25%" in r["constraint_reason"]

    def test_unknown_sector_treated_as_zero(self):
        """Sector not in current_exposures should be treated as 0%."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 5000},
        ]
        # Technology not in exposures, so current = 0%, new = 5%, total = 5%
        result = apply_portfolio_constraints(
            new_positions,
            {"Healthcare": 20.0},
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        assert result[0]["was_constrained"] is False

    def test_missing_sector_in_position_defaults_to_unknown(self):
        """Position without sector key should default to 'Unknown'."""
        new_positions = [
            {"ticker": "XYZ", "position_size": 5000},
        ]
        result = apply_portfolio_constraints(
            new_positions,
            {},
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        assert result[0]["sector"] == "Unknown"

    def test_zero_position_size(self):
        """Zero-size position should not be constrained."""
        new_positions = [
            {"ticker": "AAPL", "sector": "Technology", "position_size": 0.0},
        ]
        result = apply_portfolio_constraints(
            new_positions,
            {"Technology": 24.0},
            max_sector_pct=25.0,
            portfolio_value=100_000,
        )
        assert result[0]["was_constrained"] is False
        assert result[0]["constrained_size"] == pytest.approx(0.0)


# ============================================================
# CIO Legacy Review: C4 — Dampened Cluster Sizing
# ============================================================


class TestDampenedClusterSizing:
    """Tests for CIO Legacy C4: conviction-aware cluster adjustment."""

    def test_not_in_cluster_returns_1(self):
        """Ticker not in any cluster should get 1.0 (no adjustment)."""
        from trade_modules.conviction_sizer import get_cluster_size_adjustment

        clusters = [{"tickers": ["MSFT", "GOOG"]}]
        assert get_cluster_size_adjustment("AAPL", clusters) == pytest.approx(1.0)

    def test_in_cluster_of_2(self):
        """Cluster of 2: base = 1/sqrt(2) ≈ 0.707, dampened higher."""
        import math

        from trade_modules.conviction_sizer import get_cluster_size_adjustment

        clusters = [{"tickers": ["AAPL", "MSFT"]}]
        adj = get_cluster_size_adjustment("AAPL", clusters, conviction=50)
        base = 1.0 / math.sqrt(2)
        # Should be between base and 1.0
        assert base < adj < 1.0

    def test_high_conviction_less_penalty(self):
        """High conviction should reduce cluster penalty."""
        from trade_modules.conviction_sizer import get_cluster_size_adjustment

        clusters = [{"tickers": ["AAPL", "MSFT", "GOOG"]}]
        adj_low = get_cluster_size_adjustment("AAPL", clusters, conviction=30)
        adj_high = get_cluster_size_adjustment("AAPL", clusters, conviction=80)
        assert adj_high > adj_low

    def test_low_conviction_more_penalty(self):
        """Low conviction should keep cluster penalty closer to raw 1/sqrt(N)."""
        import math

        from trade_modules.conviction_sizer import get_cluster_size_adjustment

        clusters = [{"tickers": ["AAPL", "MSFT", "GOOG", "AMZN"]}]
        adj = get_cluster_size_adjustment("AAPL", clusters, conviction=0)
        base = 1.0 / math.sqrt(4)
        # With conviction=0, factor=0, adjustment = base + 0 = base
        assert adj == pytest.approx(base, abs=0.01)

    def test_empty_clusters(self):
        """Empty clusters list should return 1.0."""
        from trade_modules.conviction_sizer import get_cluster_size_adjustment

        assert get_cluster_size_adjustment("AAPL", []) == pytest.approx(1.0)

    def test_conviction_capped_at_80_pct(self):
        """Conviction factor should cap at 0.8 (80%)."""
        from trade_modules.conviction_sizer import get_cluster_size_adjustment

        clusters = [{"tickers": ["AAPL", "MSFT"]}]
        adj_90 = get_cluster_size_adjustment("AAPL", clusters, conviction=90)
        adj_100 = get_cluster_size_adjustment("AAPL", clusters, conviction=100)
        # Both should produce the same result because factor caps at 0.8
        assert adj_90 == pytest.approx(adj_100, abs=0.001)


# ============================================================
# CIO Legacy Review: C3 — Opportunity Cost Sizing
# ============================================================


class TestOpportunityCostSizing:
    """Tests for CIO Legacy C3: opportunity cost redistribution."""

    def test_below_avg_reduced(self):
        """Positions with conviction > 10 below average should be reduced 10%."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost

        positions = [
            {"conviction": 40, "position_size": 3.0},
            {"conviction": 60, "position_size": 3.0},
            {"conviction": 80, "position_size": 3.0},
        ]
        # avg = 60, so conviction=40 is 20 below → reduce
        result = adjust_sizes_for_opportunity_cost(positions)
        assert result[0]["position_size"] == pytest.approx(2.7, abs=0.01)
        assert result[0]["opp_cost_adj"] == pytest.approx(-0.10)

    def test_above_avg_increased(self):
        """Positions with conviction > 10 above average should be increased 10%."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost

        positions = [
            {"conviction": 40, "position_size": 3.0},
            {"conviction": 60, "position_size": 3.0},
            {"conviction": 80, "position_size": 3.0},
        ]
        # avg = 60, conviction=80 is 20 above → increase
        result = adjust_sizes_for_opportunity_cost(positions)
        assert result[2]["position_size"] == pytest.approx(3.3, abs=0.01)
        assert result[2]["opp_cost_adj"] == pytest.approx(0.10)

    def test_near_avg_unchanged(self):
        """Positions near average should not be adjusted."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost

        positions = [
            {"conviction": 55, "position_size": 3.0},
            {"conviction": 60, "position_size": 3.0},
            {"conviction": 65, "position_size": 3.0},
        ]
        # avg = 60, all within ±10 → no adjustment
        result = adjust_sizes_for_opportunity_cost(positions)
        for p in result:
            assert p["opp_cost_adj"] == pytest.approx(0.0)
            assert p["position_size"] == pytest.approx(3.0)

    def test_max_cap_enforced(self):
        """Position size should not exceed max_pct."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost

        positions = [
            {"conviction": 30, "position_size": 3.0},
            {"conviction": 90, "position_size": 4.8},
        ]
        # avg = 60, conviction=90 would increase to 5.28 but cap at 5.0
        result = adjust_sizes_for_opportunity_cost(positions, max_pct=5.0)
        assert result[1]["position_size"] <= 5.0

    def test_single_position_unchanged(self):
        """Single position should be returned as-is."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost

        positions = [{"conviction": 80, "position_size": 3.0}]
        result = adjust_sizes_for_opportunity_cost(positions)
        assert result[0]["position_size"] == pytest.approx(3.0)

    def test_empty_list(self):
        """Empty list should return empty list."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost

        assert adjust_sizes_for_opportunity_cost([]) == []
