"""TDD tests for portfolio_manager — budget.py and sizer.py.

Write tests FIRST so they fail, then implement to make them pass.
"""

from __future__ import annotations

import copy

import pytest

from trade_modules.portfolio_manager.budget import deployable_budget
from trade_modules.portfolio_manager.sizer import size_book

# ---------------------------------------------------------------------------
# deployable_budget
# ---------------------------------------------------------------------------


class TestDeployableBudget:
    def test_basic_calculation(self):
        # (0.29, 0.07, 0.8) → 0.176  [= (0.29-0.07)*0.8 = 0.22*0.8]
        assert abs(deployable_budget(0.29, 0.07, 0.8) - 0.176) < 1e-9

    def test_cash_below_target_returns_zero(self):
        # cash 0.05 <= target 0.07 → 0
        assert deployable_budget(0.05, 0.07) == 0.0

    def test_regime_mult_shrinks_deployment(self):
        result = deployable_budget(0.20, 0.07, 0.2)
        assert result < deployable_budget(0.20, 0.07, 1.0)
        assert abs(result - (0.20 - 0.07) * 0.2) < 1e-9

    def test_regime_mult_clamps_negative_to_zero(self):
        assert deployable_budget(0.20, 0.07, -0.5) == 0.0

    def test_regime_mult_clamps_over_one_to_one(self):
        assert abs(deployable_budget(0.20, 0.07, 1.5) - (0.20 - 0.07) * 1.0) < 1e-9

    def test_at_exact_target_returns_zero(self):
        assert deployable_budget(0.07, 0.07) == 0.0


# ---------------------------------------------------------------------------
# size_book
# ---------------------------------------------------------------------------


class TestSizeBookErcBudget:
    def test_buy_add_erc_additions_sum_approx_budget(self):
        """BUY+ADD ERC additions sum ≈ budget_frac before caps."""
        final_universe = [
            {"ticker": "AAPL", "action": "BUY", "conviction": 7, "beta": 1.0},
            {"ticker": "MSFT", "action": "BUY", "conviction": 6, "beta": 1.2},
        ]
        current_weights = {}
        result = size_book(final_universe, current_weights, budget_frac=0.10)
        total_delta = sum(r["delta_pct"] for r in result if r["action"] == "BUY")
        assert abs(total_delta - 0.10) < 0.01  # within 1% of budget_frac


class TestSizeBookNameCap:
    def test_name_cap_clips_buy(self):
        """Single BUY with budget > name_cap is clipped at name_cap."""
        result = size_book(
            [{"ticker": "AAPL", "action": "BUY", "conviction": 8}],
            {},
            budget_frac=0.20,
            cfg={"name_cap": 0.12},
        )
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["target_pct"] <= 0.12

    def test_add_near_cap_tops_up_at_most_remaining_cap(self):
        """ADD on name already at 0.11, name_cap=0.12 → tops up ≤ 0.01."""
        result = size_book(
            [{"ticker": "AAPL", "action": "ADD", "conviction": 7}],
            {"AAPL": 0.11},
            budget_frac=0.20,
            cfg={"name_cap": 0.12},
        )
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["target_pct"] <= 0.12
        assert aapl["delta_pct"] <= 0.01 + 1e-9


class TestSizeBookSectorCap:
    def test_sector_cap_scales_additions_not_hold_core(self):
        """Sector cap scales BUY/ADD additions, HOLD core is untouched."""
        final_universe = [
            {"ticker": "AAPL", "action": "BUY", "conviction": 7, "sector": "tech"},
            {"ticker": "MSFT", "action": "BUY", "conviction": 7, "sector": "tech"},
            {"ticker": "GOOGL", "action": "HOLD", "conviction": 5, "sector": "tech"},
        ]
        current_weights = {"GOOGL": 0.30}  # large tech hold
        result = size_book(
            final_universe,
            current_weights,
            budget_frac=0.20,
            cfg={"sector_cap": 0.35},
        )
        # HOLD core not changed
        googl = next(r for r in result if r["ticker"] == "GOOGL")
        assert googl["target_pct"] == pytest.approx(0.30)
        # tech BUY additions don't push total tech above 35%
        tech_buy_add = [
            r for r in result if r["action"] in ("BUY", "ADD") and r.get("sector") == "tech"
        ]
        buy_tech_total = sum(r["delta_pct"] for r in tech_buy_add)
        assert 0.30 + buy_tech_total <= 0.35 + 1e-9


class TestSizeBookActions:
    def test_trim_target_is_trim_to_fraction_of_current(self):
        """TRIM → 80% of current (default trim_to=0.80, softened haircut)."""
        result = size_book(
            [{"ticker": "TSLA", "action": "TRIM", "conviction": 4}],
            {"TSLA": 0.10},
            budget_frac=0.0,
        )
        tsla = next(r for r in result if r["ticker"] == "TSLA")
        assert abs(tsla["target_pct"] - 0.08) < 1e-9

    def test_sell_target_is_zero(self):
        """SELL → 0."""
        result = size_book(
            [{"ticker": "TSLA", "action": "SELL", "conviction": 2}],
            {"TSLA": 0.10},
            budget_frac=0.0,
        )
        tsla = next(r for r in result if r["ticker"] == "TSLA")
        assert tsla["target_pct"] == 0.0

    def test_hold_target_is_current_weight(self):
        """HOLD → unchanged."""
        result = size_book(
            [{"ticker": "AAPL", "action": "HOLD", "conviction": 5}],
            {"AAPL": 0.08},
            budget_frac=0.0,
        )
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["target_pct"] == pytest.approx(0.08)

    def test_no_negative_targets(self):
        """All target_pct values are non-negative."""
        final_universe = [
            {"ticker": "TSLA", "action": "SELL", "conviction": 1},
            {"ticker": "AAPL", "action": "BUY", "conviction": 8},
        ]
        current_weights = {"TSLA": 0.05}
        result = size_book(final_universe, current_weights, budget_frac=0.10)
        assert all(r["target_pct"] >= 0.0 for r in result)


class TestSizeBookMinPosition:
    def test_sub_min_position_buy_dropped(self):
        """BUY names with final target < min_position are dropped."""
        result = size_book(
            [
                {"ticker": "AAPL", "action": "BUY", "conviction": 8},
                {"ticker": "MSFT", "action": "BUY", "conviction": 7},
                {"ticker": "GOOGL", "action": "BUY", "conviction": 3},
                {"ticker": "AMZN", "action": "BUY", "conviction": 2},
                {"ticker": "META", "action": "BUY", "conviction": 1},
                {"ticker": "NVDA", "action": "BUY", "conviction": 1},
                {"ticker": "TSLA", "action": "BUY", "conviction": 1},
                {"ticker": "NFLX", "action": "BUY", "conviction": 1},
                {"ticker": "AMGN", "action": "BUY", "conviction": 1},
                {"ticker": "INTC", "action": "BUY", "conviction": 1},
                {"ticker": "QCOM", "action": "BUY", "conviction": 1},
            ],
            {},
            budget_frac=0.02,  # small budget → many BUY names will be sub-1%
            cfg={"min_position": 0.01},
        )
        buy_results = [r for r in result if r["action"] == "BUY"]
        assert all(r["target_pct"] >= 0.01 for r in buy_results)


class TestSizeBookDeltas:
    def test_delta_equals_target_minus_current(self):
        """delta_pct == target_pct - current_pct."""
        result = size_book(
            [{"ticker": "AAPL", "action": "ADD", "conviction": 7}],
            {"AAPL": 0.05},
            budget_frac=0.05,
        )
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert abs(aapl["delta_pct"] - (aapl["target_pct"] - aapl["current_pct"])) < 1e-9


class TestSizeBookImmutability:
    def test_inputs_not_mutated(self):
        """size_book must not mutate its inputs."""
        universe = [{"ticker": "AAPL", "action": "BUY", "conviction": 7}]
        weights = {"CASH": 0.30}
        universe_copy = copy.deepcopy(universe)
        weights_copy = copy.deepcopy(weights)
        size_book(universe, weights, budget_frac=0.10)
        assert universe == universe_copy
        assert weights == weights_copy


class TestSizeBookOutputStructure:
    def test_all_output_dicts_have_required_keys(self):
        """All output dicts have the required keys. delta_usd is None when nav not provided."""
        result = size_book(
            [{"ticker": "AAPL", "action": "BUY", "conviction": 7}],
            {},
            budget_frac=0.10,
        )
        assert result
        required = {
            "ticker",
            "action",
            "current_pct",
            "target_pct",
            "delta_pct",
            "delta_usd",
            "conviction",
        }
        for row in result:
            assert set(row.keys()) >= required
            # delta_usd is None when nav is not provided
            assert row["delta_usd"] is None

    def test_delta_usd_computed_when_nav_provided(self):
        """delta_usd = delta_pct * nav when nav is given."""
        result = size_book(
            [{"ticker": "AAPL", "action": "BUY", "conviction": 7}],
            {},
            budget_frac=0.10,
            nav=10_000.0,
        )
        assert result
        for row in result:
            assert row["delta_usd"] is not None
            assert abs(row["delta_usd"] - row["delta_pct"] * 10_000.0) < 1e-6


class TestSizeBookBudgetLeak:
    def test_budget_leak_when_name_capped(self):
        """Budget is partially undeployed when a single BUY hits name_cap.

        This is intentional design: freed budget is not redistributed.
        The sum of BUY/ADD delta_pct will be < budget_frac in this scenario.
        """
        # Single BUY with budget_frac=0.20 and name_cap=0.12 → only 0.12 deployed
        result = size_book(
            [{"ticker": "AAPL", "action": "BUY", "conviction": 8}],
            {},
            budget_frac=0.20,
            cfg={"name_cap": 0.12},
        )
        buy_rows = [r for r in result if r["action"] == "BUY"]
        total_deployed = sum(r["delta_pct"] for r in buy_rows)
        # Deployed should be capped at name_cap (0.12), not full budget (0.20)
        assert total_deployed < 0.20
        assert abs(total_deployed - 0.12) < 1e-9


class TestSizeBookNoLeverage:
    """F5 — hard no-leverage guard: resulting gross must never exceed max_gross,
    even when the manual --cash-pct override requests more budget than free cash."""

    def test_large_budget_cannot_exceed_max_gross(self):
        """A 95%-invested book + an oversized budget must NOT breach gross=1.0."""
        # Held names sum to 0.95 (all HOLD → retained at current). Several fresh
        # high-cap BUYs with a 0.30 budget would push gross to ~1.07 without the
        # guard (final-review F5 reproduced this exactly).
        universe = [
            {"ticker": "HOLD1", "action": "HOLD", "conviction": 50, "beta": 1.0},
            {"ticker": "HOLD2", "action": "HOLD", "conviction": 50, "beta": 1.0},
            {"ticker": "NEW1", "action": "BUY", "conviction": 80, "beta": 1.0},
            {"ticker": "NEW2", "action": "BUY", "conviction": 80, "beta": 1.0},
            {"ticker": "NEW3", "action": "BUY", "conviction": 80, "beta": 1.0},
        ]
        current_weights = {"HOLD1": 0.50, "HOLD2": 0.45}
        result = size_book(universe, current_weights, budget_frac=0.30)
        gross = sum(r["target_pct"] for r in result)
        assert gross <= 1.0 + 1e-9, f"leverage breach: gross={gross:.4f} > 1.0"

    def test_max_gross_config_override_binds(self):
        """A tighter max_gross (0.80) must clamp deployment accordingly."""
        universe = [
            {"ticker": "HOLD1", "action": "HOLD", "conviction": 50, "beta": 1.0},
            {"ticker": "NEW1", "action": "BUY", "conviction": 80, "beta": 1.0},
            {"ticker": "NEW2", "action": "BUY", "conviction": 80, "beta": 1.0},
        ]
        current_weights = {"HOLD1": 0.70}
        result = size_book(universe, current_weights, budget_frac=0.50, cfg={"max_gross": 0.80})
        gross = sum(r["target_pct"] for r in result)
        assert gross <= 0.80 + 1e-9, f"gross {gross:.4f} exceeded max_gross 0.80"

    def test_guard_does_not_over_constrain_normal_case(self):
        """With ample headroom the guard is a no-op — budget deploys normally."""
        universe = [
            {"ticker": "HOLD1", "action": "HOLD", "conviction": 50, "beta": 1.0},
            {"ticker": "NEW1", "action": "BUY", "conviction": 80, "beta": 1.0},
        ]
        current_weights = {"HOLD1": 0.20}
        # 0.20 held + 0.10 budget = 0.30 gross, far below 1.0 → full deploy.
        result = size_book(universe, current_weights, budget_frac=0.10)
        new1 = next(r for r in result if r["ticker"] == "NEW1")
        # NEW1 target hits its ERC allocation (single BUY → full budget, capped
        # only by name_cap 0.12): 0.10 < 0.12 so it deploys the whole budget.
        assert abs(new1["target_pct"] - 0.10) < 1e-9


class TestSizeBookBudgetConcentration:
    """A large candidate set must not dilute the budget below min_position."""

    def test_large_deploy_set_funds_only_top_conviction(self):
        """20 BUYs, 6% budget, 1% min → only the top 6 by conviction fund at ~1%.

        Without concentration, ERC would spread 6% across all 20 (0.3% each),
        every name would fall below min_position, and NOTHING would deploy.
        """
        universe = [
            {"ticker": f"B{i:02d}", "action": "BUY", "conviction": float(i), "beta": 1.0}
            for i in range(20)
        ]
        result = size_book(universe, {}, budget_frac=0.06, cfg={"min_position": 0.01})
        funded = [r for r in result if r["action"] == "BUY"]
        # 0.06 / 0.01 = 6 fundable slots
        assert len(funded) == 6
        # they are the six highest convictions (19..14)
        assert sorted((r["conviction"] for r in funded), reverse=True) == [
            19.0,
            18.0,
            17.0,
            16.0,
            15.0,
            14.0,
        ]
        # each funded position clears min_position, and the whole budget deploys
        assert all(r["target_pct"] >= 0.01 - 1e-9 for r in funded)
        assert abs(sum(r["target_pct"] for r in funded) - 0.06) < 1e-6
