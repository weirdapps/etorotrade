"""
Tests for Committee Synthesis Engine — CIO v5 codified conviction scoring.

Tests cover all 8 public functions:
1. compute_sector_medians — sector-relative EXRET calculation
2. count_agent_votes — freshness-weighted bull/bear vote counting
3. determine_base_conviction — signal-aware base with floors/caps
4. compute_adjustments — bonus/penalty system with caps
5. apply_conviction_floors — prevents unreasonable suppression
6. determine_action — signal-aware action assignment
7. recalculate_trim_conviction — TRIM-specific confidence scoring
8. classify_hold_tier — HOLD sub-tier classification
9. synthesize_stock — full pipeline integration
10. build_concordance — end-to-end concordance matrix
11. compute_changes — previous-run comparison
"""

import pytest

from trade_modules.committee_synthesis import (
    ACTION_ORDER,
    AGENT_FRESHNESS,
    apply_conviction_floors,
    apply_opportunity_gate,
    build_concordance,
    classify_hold_tier,
    compute_adjustments,
    compute_changes,
    compute_sector_medians,
    count_agent_votes,
    detect_sector_gaps,
    determine_action,
    determine_base_conviction,
    recalculate_trim_conviction,
    synthesize_stock,
)


# ============================================================
# compute_sector_medians
# ============================================================


class TestComputeSectorMedians:
    def test_single_stock_per_sector(self):
        signals = {
            "AAPL": {"exret": 10},
            "XOM": {"exret": 20},
        }
        sectors = {"AAPL": "Technology", "XOM": "Energy"}
        medians, universe = compute_sector_medians(signals, sectors)
        assert medians["Technology"] == 10
        assert medians["Energy"] == 20
        assert universe in (10, 20)  # median of [10, 20]

    def test_multiple_stocks_same_sector(self):
        signals = {
            "AAPL": {"exret": 10},
            "MSFT": {"exret": 30},
            "GOOG": {"exret": 20},
        }
        sectors = {"AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology"}
        medians, universe = compute_sector_medians(signals, sectors)
        assert medians["Technology"] == 20  # median of [10, 20, 30]

    def test_even_count_uses_average(self):
        signals = {
            "A": {"exret": 10},
            "B": {"exret": 20},
        }
        sectors = {"A": "Tech", "B": "Tech"}
        medians, _ = compute_sector_medians(signals, sectors)
        assert medians["Tech"] == 15  # average of 10, 20

    def test_missing_exret_defaults_zero(self):
        signals = {"A": {}}
        sectors = {"A": "Tech"}
        medians, universe = compute_sector_medians(signals, sectors)
        assert medians["Tech"] == 0
        assert universe == 0

    def test_missing_sector_defaults_other(self):
        signals = {"A": {"exret": 15}}
        sectors = {}
        medians, _ = compute_sector_medians(signals, sectors)
        assert medians["Other"] == 15


# ============================================================
# count_agent_votes
# ============================================================


class TestCountAgentVotes:
    """Critical: neutral views must split evenly, not lean bullish."""

    def test_all_bullish(self):
        """When every agent is bullish, bull should dominate."""
        bull, bear = count_agent_votes(
            fund_score=85,         # bullish
            tech_signal="ENTER_NOW",  # bullish
            tech_momentum=40,
            macro_fit="FAVORABLE",    # bullish
            census_alignment="ALIGNED",  # bullish
            news_impact="HIGH_POSITIVE",  # bullish
            risk_warning=False,       # neutral (no warning)
            signal="B",
        )
        assert bull > bear
        assert bull / (bull + bear) > 0.7

    def test_all_bearish(self):
        bull, bear = count_agent_votes(
            fund_score=30,
            tech_signal="AVOID",
            tech_momentum=-50,
            macro_fit="UNFAVORABLE",
            census_alignment="DIVERGENT",
            news_impact="HIGH_NEGATIVE",
            risk_warning=True,
            signal="S",
        )
        assert bear > bull
        assert bear / (bull + bear) > 0.7

    def test_neutral_views_split_evenly(self):
        """CRITICAL FIX: neutral views must NOT inflate bull_pct."""
        bull, bear = count_agent_votes(
            fund_score=55,         # neutral (45-70 range)
            tech_signal="HOLD",    # neutral (fallthrough)
            tech_momentum=0,
            macro_fit="NEUTRAL",   # neutral
            census_alignment="NEUTRAL",  # neutral
            news_impact="NEUTRAL",  # neutral
            risk_warning=False,    # no warning = neutral
            signal="H",
        )
        total = bull + bear
        bull_pct = bull / total * 100
        # With all neutral views, should be ~50/50
        assert 45 <= bull_pct <= 55, (
            f"All-neutral views should be ~50% bull, got {bull_pct:.1f}%"
        )

    def test_fundamental_thresholds(self):
        """Fund score <45 = bearish, 45-70 = neutral, >=70 = bullish."""
        # Bearish
        b1, r1 = count_agent_votes(30, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        # Neutral
        b2, r2 = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        # Bullish
        b3, r3 = count_agent_votes(80, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        # Bearish fund should have more bear weight than neutral
        assert r1 > r2
        # Bullish fund should have more bull weight than neutral
        assert b3 > b2

    def test_risk_manager_sell_weight(self):
        """Risk manager gets 2x weight for SELL signals, 1.5x for BUY."""
        _, bear_sell = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", True, "S")
        _, bear_buy = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", True, "B")
        assert bear_sell > bear_buy

    def test_wait_for_pullback_leans_bull(self):
        """WAIT_FOR_PULLBACK is 0.6 bull / 0.4 bear — slight lean."""
        bull, bear = count_agent_votes(55, "WAIT_FOR_PULLBACK", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        # Tech contribution should be 0.6b/0.4r
        # Other agents neutral. Bull should slightly exceed bear.
        assert bull > bear

    def test_census_div_leans_bull(self):
        """CENSUS_DIV means PIs are bullish when signal isn't — PI view gets weight."""
        bull, bear = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "CENSUS_DIV", "NEUTRAL", False, "H")
        bull2, bear2 = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        assert bull > bull2  # Census DIV adds more bull weight


# ============================================================
# determine_base_conviction
# ============================================================


class TestDetermineBaseConviction:
    def test_buy_signal_floor(self):
        """BUY signal should never produce base below 55."""
        base = determine_base_conviction(
            bull_pct=30,  # low consensus
            signal="B",
            fund_score=50,
            excess_exret=0,
            bear_ratio=0.7,
        )
        assert base >= 55

    def test_hold_signal_cap(self):
        """HOLD signal should never produce base above 70."""
        base = determine_base_conviction(
            bull_pct=90,  # very high consensus
            signal="H",
            fund_score=90,
            excess_exret=30,
            bear_ratio=0.1,
        )
        assert base <= 70

    def test_sell_with_agent_disagreement(self):
        """When agents mostly disagree with SELL, base should be low (50)."""
        base = determine_base_conviction(
            bull_pct=60,  # agents are bullish but signal is SELL
            signal="S",
            fund_score=70,
            excess_exret=20,
            bear_ratio=0.4,
        )
        assert base == 50

    def test_sell_with_strong_bear_consensus(self):
        """When bears dominate for a SELL signal, base should be high."""
        base = determine_base_conviction(
            bull_pct=15,
            signal="S",
            fund_score=30,
            excess_exret=-10,
            bear_ratio=0.85,
        )
        assert base == 85

    def test_buy_quality_bonus_high_fund(self):
        """BUY + fund_score >= 80 → +10 bonus."""
        base_low = determine_base_conviction(50, "B", 60, 5, 0.5)
        base_high = determine_base_conviction(50, "B", 85, 5, 0.5)
        assert base_high > base_low
        assert base_high - base_low >= 10

    def test_buy_quality_bonus_moderate_fund(self):
        """BUY + fund_score 65-79 → +5 bonus."""
        base_low = determine_base_conviction(50, "B", 60, 5, 0.5)
        base_mid = determine_base_conviction(50, "B", 68, 5, 0.5)
        assert base_mid > base_low

    def test_buy_excess_exret_bonus(self):
        """BUY + excess_exret >= 12 → +5 bonus."""
        base_no = determine_base_conviction(50, "B", 60, 5, 0.5)
        base_ex = determine_base_conviction(50, "B", 60, 15, 0.5)
        assert base_ex == base_no + 5

    def test_hold_quality_bonus_NOT_applied(self):
        """HOLD signals should NOT get BUY-side quality bonus."""
        base = determine_base_conviction(50, "H", 90, 30, 0.5)
        # Should be capped at 70 regardless of fund_score
        assert base <= 70


# ============================================================
# compute_adjustments
# ============================================================


class TestComputeAdjustments:
    def _default_kwargs(self):
        return dict(
            signal="B", fund_score=70, tech_signal="HOLD", tech_momentum=0,
            rsi=50, macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            div_score=0, census_ts="stable", news_impact="NEUTRAL",
            risk_warning=False, buy_pct=60, excess_exret=10,
            beta=1.0, quality_trap=False, sector="Technology",
            sector_rankings={}, bull_count=3,
        )

    def test_agent_agreement_bonus_5(self):
        kw = self._default_kwargs()
        kw["bull_count"] = 5
        bonuses, _ = compute_adjustments(**kw)
        assert bonuses >= 10

    def test_agent_agreement_bonus_6(self):
        kw = self._default_kwargs()
        kw["bull_count"] = 6
        bonuses, _ = compute_adjustments(**kw)
        assert bonuses >= 15

    def test_consensus_warning_tiered(self):
        """Consensus penalty varies by excess EXRET."""
        kw = self._default_kwargs()
        kw["buy_pct"] = 95

        kw["excess_exret"] = 20
        _, p_high = compute_adjustments(**kw)

        kw["excess_exret"] = 5
        _, p_mid = compute_adjustments(**kw)

        kw["excess_exret"] = -5
        _, p_low = compute_adjustments(**kw)

        assert p_low > p_mid > p_high

    def test_census_alignment_bonus(self):
        kw = self._default_kwargs()
        kw["div_score"] = 0
        bonuses, _ = compute_adjustments(**kw)
        # div_score in [-20, 20] → +5
        assert bonuses >= 5

    def test_census_strong_alignment(self):
        kw = self._default_kwargs()
        kw["div_score"] = -30
        bonuses, _ = compute_adjustments(**kw)
        assert bonuses >= 8

    def test_high_beta_penalty(self):
        kw = self._default_kwargs()
        kw["beta"] = 2.5
        _, penalties = compute_adjustments(**kw)
        assert penalties >= 5

    def test_quality_trap_penalty(self):
        kw = self._default_kwargs()
        kw["quality_trap"] = True
        _, penalties = compute_adjustments(**kw)
        assert penalties >= 5

    def test_rsi_overbought_penalty(self):
        kw = self._default_kwargs()
        kw["rsi"] = 75
        _, penalties = compute_adjustments(**kw)
        assert penalties >= 5

    def test_macro_favorable_bonus(self):
        kw = self._default_kwargs()
        kw["macro_fit"] = "FAVORABLE"
        bonuses, _ = compute_adjustments(**kw)
        assert bonuses >= 5

    def test_macro_unfavorable_cyclical_penalty(self):
        kw = self._default_kwargs()
        kw["macro_fit"] = "UNFAVORABLE"
        kw["sector"] = "Financials"
        _, penalties = compute_adjustments(**kw)
        assert penalties >= 10

    def test_macro_unfavorable_noncyclical_penalty(self):
        kw = self._default_kwargs()
        kw["macro_fit"] = "UNFAVORABLE"
        kw["sector"] = "Technology"
        _, penalties = compute_adjustments(**kw)
        assert penalties >= 5

    def test_sell_tech_disagreement_penalty(self):
        """SELL signal + bullish tech → penalty (reduces sell conviction)."""
        kw = self._default_kwargs()
        kw["signal"] = "S"
        kw["tech_signal"] = "ENTER_NOW"
        _, penalties = compute_adjustments(**kw)
        assert penalties >= 5

    def test_bonus_cap_at_20(self):
        kw = self._default_kwargs()
        kw["bull_count"] = 6        # +15
        kw["div_score"] = -30       # +8
        kw["news_impact"] = "HIGH_POSITIVE"  # +5
        kw["macro_fit"] = "FAVORABLE"  # +5
        bonuses, _ = compute_adjustments(**kw)
        assert bonuses <= 20

    def test_penalty_cap_at_25(self):
        kw = self._default_kwargs()
        kw["buy_pct"] = 95
        kw["excess_exret"] = -10    # -15
        kw["rsi"] = 75              # -5
        kw["beta"] = 2.5            # -5
        kw["quality_trap"] = True   # -5
        kw["macro_fit"] = "UNFAVORABLE"  # -10
        _, penalties = compute_adjustments(**kw)
        assert penalties <= 25

    def test_census_distribution_penalty(self):
        kw = self._default_kwargs()
        kw["census_ts"] = "strong_distribution"
        _, penalties = compute_adjustments(**kw)
        assert penalties >= 5

    def test_census_accumulation_bonus(self):
        kw = self._default_kwargs()
        kw["census_ts"] = "strong_accumulation"
        bonuses, _ = compute_adjustments(**kw)
        assert bonuses >= 3

    def test_sector_rotation_leading(self):
        kw = self._default_kwargs()
        kw["sector"] = "Technology"
        kw["sector_rankings"] = {"XLK": {"rank": 1, "return_1m": 8.5}}
        bonuses, _ = compute_adjustments(**kw)
        assert bonuses >= 5

    def test_sector_rotation_lagging(self):
        kw = self._default_kwargs()
        kw["sector"] = "Technology"
        kw["sector_rankings"] = {"XLK": {"rank": 10, "return_1m": -5.0}}
        _, penalties = compute_adjustments(**kw)
        assert penalties >= 5


# ============================================================
# apply_conviction_floors
# ============================================================


class TestApplyConvictionFloors:
    def test_buy_unconditional_floor(self):
        """BUY signal stocks should never go below 40."""
        result = apply_conviction_floors(10, "B", 5, 20, 25, 2, 50, 50)
        assert result >= 40

    def test_buy_quality_floor_50(self):
        """BUY + excess_exret > 20 + improving PE → floor 50."""
        result = apply_conviction_floors(30, "B", 25, 15, 20, 3, 60, 60)
        assert result >= 50

    def test_buy_agent_agreement_floor(self):
        """BUY + 4+ agents agree → floor 50."""
        result = apply_conviction_floors(30, "B", 5, 20, 25, 4, 60, 60)
        assert result >= 50

    def test_buy_fund_quality_floor(self):
        """BUY + fund>=70 + buy_pct>=70 → floor 50."""
        result = apply_conviction_floors(30, "B", 5, 20, 25, 2, 75, 80)
        assert result >= 50

    def test_sell_no_floor(self):
        """SELL signals should NOT have floors applied."""
        result = apply_conviction_floors(10, "S", 5, 20, 25, 2, 50, 50)
        assert result == 10

    def test_hold_no_floor(self):
        """HOLD signals with low conviction stay low."""
        result = apply_conviction_floors(10, "H", 5, 20, 25, 2, 50, 50)
        assert result == 10

    def test_capped_at_100(self):
        result = apply_conviction_floors(110, "B", 30, 15, 20, 6, 90, 95)
        assert result == 100

    def test_floored_at_0(self):
        result = apply_conviction_floors(-10, "S", -5, 20, 25, 0, 30, 30)
        assert result == 0


# ============================================================
# determine_action
# ============================================================


class TestDetermineAction:
    # SELL signal actions
    def test_sell_high_conviction(self):
        assert determine_action(70, "S", "AVOID", True) == "SELL"

    def test_sell_low_conviction(self):
        assert determine_action(50, "S", "AVOID", True) == "REDUCE"

    def test_sell_threshold_60(self):
        assert determine_action(60, "S", "AVOID", True) == "SELL"
        assert determine_action(59, "S", "AVOID", True) == "REDUCE"

    # BUY signal actions
    def test_buy_high_conviction(self):
        assert determine_action(80, "B", "ENTER_NOW", False) == "BUY"

    def test_buy_moderate_conviction(self):
        assert determine_action(60, "B", "WAIT_FOR_PULLBACK", False) == "ADD"

    def test_buy_low_conviction(self):
        assert determine_action(45, "B", "HOLD", False) == "HOLD"

    def test_buy_threshold_75(self):
        assert determine_action(75, "B", "ENTER_NOW", False) == "BUY"
        assert determine_action(74, "B", "ENTER_NOW", False) == "ADD"

    def test_buy_threshold_55(self):
        assert determine_action(55, "B", "WAIT_FOR_PULLBACK", False) == "ADD"
        assert determine_action(54, "B", "WAIT_FOR_PULLBACK", False) == "HOLD"

    # HOLD signal actions
    def test_hold_high_conviction(self):
        assert determine_action(75, "H", "ENTER_NOW", False) == "ADD"

    def test_hold_moderate_conviction(self):
        assert determine_action(55, "H", "HOLD", False) == "HOLD"

    def test_hold_low_conviction(self):
        assert determine_action(40, "H", "HOLD", False) == "WEAK HOLD"

    def test_hold_very_low_conviction(self):
        assert determine_action(30, "H", "AVOID", True) == "TRIM"

    def test_hold_thresholds(self):
        assert determine_action(70, "H", "ENTER_NOW", False) == "ADD"
        assert determine_action(69, "H", "HOLD", False) == "HOLD"
        assert determine_action(50, "H", "HOLD", False) == "HOLD"
        assert determine_action(49, "H", "HOLD", False) == "WEAK HOLD"
        assert determine_action(35, "H", "HOLD", False) == "WEAK HOLD"
        assert determine_action(34, "H", "AVOID", True) == "TRIM"


# ============================================================
# recalculate_trim_conviction
# ============================================================


class TestRecalculateTrimConviction:
    def test_base_is_50(self):
        conv = recalculate_trim_conviction(
            "HOLD", "NEUTRAL", False, 1.0, 50, 60, "NEUTRAL"
        )
        assert conv == 50

    def test_tech_exit_adds_15(self):
        conv = recalculate_trim_conviction(
            "EXIT_SOON", "NEUTRAL", False, 1.0, 50, 60, "NEUTRAL"
        )
        assert conv == 65

    def test_tech_avoid_adds_15(self):
        conv = recalculate_trim_conviction(
            "AVOID", "NEUTRAL", False, 1.0, 50, 60, "NEUTRAL"
        )
        assert conv == 65

    def test_all_trim_factors(self):
        """Maximum trim conviction with all factors."""
        conv = recalculate_trim_conviction(
            "AVOID",           # +15
            "UNFAVORABLE",     # +10
            True,              # +10
            2.0,               # +5
            75,                # +5
            40,                # not >=80 so no -10
            "DIVERGENT",       # not ALIGNED so no -5
        )
        assert conv == 85  # capped at 85

    def test_cap_at_85(self):
        """Even with all factors, should not exceed 85."""
        conv = recalculate_trim_conviction(
            "AVOID", "UNFAVORABLE", True, 2.0, 80, 30, "DIVERGENT"
        )
        assert conv <= 85

    def test_strong_fundamentals_reduce(self):
        """High fund_score reduces trim conviction."""
        conv_low = recalculate_trim_conviction(
            "AVOID", "NEUTRAL", False, 1.0, 50, 60, "NEUTRAL"
        )
        conv_high = recalculate_trim_conviction(
            "AVOID", "NEUTRAL", False, 1.0, 50, 85, "NEUTRAL"
        )
        assert conv_high < conv_low

    def test_census_aligned_reduces(self):
        conv_aligned = recalculate_trim_conviction(
            "AVOID", "NEUTRAL", False, 1.0, 50, 60, "ALIGNED"
        )
        conv_neutral = recalculate_trim_conviction(
            "AVOID", "NEUTRAL", False, 1.0, 50, 60, "NEUTRAL"
        )
        assert conv_aligned < conv_neutral


# ============================================================
# classify_hold_tier
# ============================================================


class TestClassifyHoldTier:
    def test_strong_hold(self):
        assert classify_hold_tier(60) == "STRONG"

    def test_standard_hold(self):
        assert classify_hold_tier(45) == "STANDARD"

    def test_weak_hold(self):
        assert classify_hold_tier(35) == "WEAK"

    def test_boundary_55(self):
        assert classify_hold_tier(55) == "STRONG"

    def test_boundary_40(self):
        assert classify_hold_tier(40) == "STANDARD"

    def test_boundary_39(self):
        assert classify_hold_tier(39) == "WEAK"


# ============================================================
# synthesize_stock — integration tests
# ============================================================


class TestSynthesizeStock:
    def _make_sig_data(self, signal="B", exret=20, buy_pct=70, beta=1.0, pet=25, pef=20):
        return {"signal": signal, "exret": exret, "buy_pct": buy_pct,
                "beta": beta, "pet": pet, "pef": pef}

    def _make_fund_data(self, score=70, trap=False, pe_traj="improving"):
        return {"fundamental_score": score, "quality_trap_warning": trap,
                "pe_trajectory": pe_traj}

    def _make_tech_data(self, momentum=30, timing="ENTER_NOW", rsi=55, macd="BULLISH"):
        return {"momentum_score": momentum, "timing_signal": timing,
                "rsi": rsi, "macd_signal": macd}

    def test_strong_buy(self):
        """Stock with all bullish signals → BUY with high conviction."""
        result = synthesize_stock(
            ticker="NVDA",
            sig_data=self._make_sig_data("B", exret=40, buy_pct=95, beta=1.5),
            fund_data=self._make_fund_data(85),
            tech_data=self._make_tech_data(40, "ENTER_NOW", 60),
            macro_fit="FAVORABLE",
            census_alignment="ALIGNED",
            div_score=-5,
            census_ts_trend="strong_accumulation",
            news_impact="HIGH_POSITIVE",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=15,
            sector_rankings={"XLK": {"rank": 1, "return_1m": 8}},
            position_limit=3.5,
        )
        assert result["action"] == "BUY"
        assert result["conviction"] >= 70

    def test_clear_sell(self):
        """Stock with SELL signal + bearish agents → SELL or REDUCE."""
        result = synthesize_stock(
            ticker="BAD",
            sig_data=self._make_sig_data("S", exret=-5, buy_pct=20, beta=2.0),
            fund_data=self._make_fund_data(30, trap=True),
            tech_data=self._make_tech_data(-50, "AVOID", 25, "BEARISH"),
            macro_fit="UNFAVORABLE",
            census_alignment="DIVERGENT",
            div_score=40,
            census_ts_trend="strong_distribution",
            news_impact="HIGH_NEGATIVE",
            risk_warning=True,
            sector="Consumer Discretionary",
            sector_median_exret=10,
            sector_rankings={},
            position_limit=2.0,
        )
        assert result["action"] in ("SELL", "REDUCE")
        assert result["conviction"] >= 50

    def test_hold_signal_capped(self):
        """HOLD signal stock should NOT get ADD unless agents strongly agree."""
        result = synthesize_stock(
            ticker="AAPL",
            sig_data=self._make_sig_data("H", exret=10, buy_pct=60),
            fund_data=self._make_fund_data(55),
            tech_data=self._make_tech_data(10, "WAIT_FOR_PULLBACK", 55),
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=15,
            sector_rankings={},
            position_limit=5.0,
        )
        # HOLD signal caps base at 70, with neutral agents should be HOLD
        assert result["action"] in ("HOLD", "WEAK HOLD")

    def test_buy_signal_floor(self):
        """BUY signal stock should never drop below conviction 40."""
        result = synthesize_stock(
            ticker="WEAK_BUY",
            sig_data=self._make_sig_data("B", exret=5, buy_pct=55),
            fund_data=self._make_fund_data(45),
            tech_data=self._make_tech_data(-20, "AVOID", 75, "BEARISH"),
            macro_fit="UNFAVORABLE",
            census_alignment="DIVERGENT",
            div_score=30,
            census_ts_trend="distribution",
            news_impact="LOW_NEGATIVE",
            risk_warning=True,
            sector="Technology",
            sector_median_exret=15,
            sector_rankings={},
            position_limit=3.0,
        )
        assert result["conviction"] >= 40

    def test_trim_gets_recalculated(self):
        """When action is TRIM, conviction should reflect trim confidence."""
        result = synthesize_stock(
            ticker="TRIMME",
            sig_data=self._make_sig_data("H", exret=5, buy_pct=40, beta=2.0),
            fund_data=self._make_fund_data(40),
            tech_data=self._make_tech_data(-40, "AVOID", 75, "BEARISH"),
            macro_fit="UNFAVORABLE",
            census_alignment="DIVERGENT",
            div_score=30,
            census_ts_trend="distribution",
            news_impact="LOW_NEGATIVE",
            risk_warning=True,
            sector="Technology",
            sector_median_exret=15,
            sector_rankings={},
            position_limit=2.0,
        )
        if result["action"] in ("TRIM", "WEAK HOLD"):
            # Trim conviction should be reasonable (not near 0)
            assert result["conviction"] >= 40

    def test_pe_trajectory_fallback(self):
        """When fund data has stable pe_trajectory, derive from PET/PEF."""
        result = synthesize_stock(
            ticker="TEST",
            sig_data=self._make_sig_data("B", pet=30, pef=20),
            fund_data={"fundamental_score": 70, "pe_trajectory": "stable"},
            tech_data=self._make_tech_data(),
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=15,
            sector_rankings={},
            position_limit=5.0,
        )
        # PEF 20 < PET 30 × 0.8 = 24, so pe_traj should be "strong_improvement"
        assert result["pe_trajectory"] == "strong_improvement"

    def test_excess_exret_calculation(self):
        """Excess EXRET = stock EXRET - sector median."""
        result = synthesize_stock(
            ticker="TEST",
            sig_data=self._make_sig_data("B", exret=30),
            fund_data=self._make_fund_data(70),
            tech_data=self._make_tech_data(),
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=10,
            sector_rankings={},
            position_limit=5.0,
        )
        assert result["excess_exret"] == 20.0

    def test_output_keys(self):
        """Verify all required output keys are present."""
        result = synthesize_stock(
            ticker="TEST",
            sig_data=self._make_sig_data(),
            fund_data=self._make_fund_data(),
            tech_data=self._make_tech_data(),
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=15,
            sector_rankings={},
            position_limit=5.0,
        )
        required_keys = {
            "ticker", "signal", "sector", "fund_score", "fund_view",
            "pe_trajectory", "quality_trap", "tech_momentum", "tech_signal",
            "rsi", "macd", "macro_fit", "census", "div_score", "census_ts",
            "news_impact", "risk_warning", "exret", "excess_exret",
            "buy_pct", "beta", "bull_weight", "bear_weight", "bull_pct",
            "base", "bonuses", "penalties", "conviction", "action",
            "hold_tier", "max_pct",
        }
        assert required_keys.issubset(set(result.keys()))


# ============================================================
# build_concordance — end-to-end
# ============================================================


class TestBuildConcordance:
    @pytest.fixture
    def minimal_inputs(self):
        """Minimal valid inputs for build_concordance."""
        return dict(
            portfolio_signals={
                "AAPL": {"signal": "B", "exret": 15, "buy_pct": 70, "beta": 1.1, "pet": 30, "pef": 25},
                "XOM": {"signal": "H", "exret": 10, "buy_pct": 55, "beta": 0.8, "pet": 12, "pef": 10},
            },
            fund_report={
                "stocks": {
                    "AAPL": {"fundamental_score": 75, "quality_trap_warning": False, "pe_trajectory": "improving"},
                }
            },
            tech_report={
                "stocks": {
                    "AAPL": {"momentum_score": 20, "timing_signal": "WAIT_FOR_PULLBACK", "rsi": 55, "macd_signal": "BULLISH"},
                    "XOM": {"momentum_score": 10, "timing_signal": "HOLD", "rsi": 50, "macd_signal": "NEUTRAL"},
                }
            },
            macro_report={
                "portfolio_implications": {
                    "AAPL": {"macro_fit": "neutral"},
                    "XOM": {"macro_fit": "positive"},
                },
                "sector_rankings": {},
            },
            census_report={
                "divergences": {
                    "consensus_aligned": [
                        {"ticker": "AAPL", "divergence_score": 5},
                    ],
                    "signal_divergences": [],
                    "census_divergences": [],
                },
            },
            news_report={
                "portfolio_news": {
                    "AAPL": [{"impact": "LOW_POSITIVE", "headline": "Good news"}],
                },
            },
            risk_report={
                "consensus_warnings": [],
                "position_limits": {
                    "AAPL": {"max_pct": 4.0},
                    "XOM": {"max_pct": 5.0},
                },
            },
            sector_map={"AAPL": "Technology", "XOM": "Energy"},
        )

    def test_produces_sorted_concordance(self, minimal_inputs):
        result = build_concordance(**minimal_inputs)
        assert len(result) == 2
        # Should be sorted by action priority
        for i in range(len(result) - 1):
            a1 = ACTION_ORDER.get(result[i]["action"], 9)
            a2 = ACTION_ORDER.get(result[i + 1]["action"], 9)
            if a1 == a2:
                assert result[i]["conviction"] >= result[i + 1]["conviction"]
            else:
                assert a1 <= a2

    def test_fallback_fundamental_score(self, minimal_inputs):
        """XOM has no fund_report entry → should get fallback score."""
        result = build_concordance(**minimal_inputs)
        xom = next(r for r in result if r["ticker"] == "XOM")
        # Fallback score computed from signal data
        assert xom["fund_score"] > 0

    def test_macro_fit_parsing(self, minimal_inputs):
        """Macro fit strings should be normalized to FAVORABLE/UNFAVORABLE/NEUTRAL."""
        result = build_concordance(**minimal_inputs)
        xom = next(r for r in result if r["ticker"] == "XOM")
        assert xom["macro_fit"] == "FAVORABLE"  # "positive" → FAVORABLE
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["macro_fit"] == "NEUTRAL"

    def test_census_alignment_mapping(self, minimal_inputs):
        result = build_concordance(**minimal_inputs)
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["census"] == "ALIGNED"
        xom = next(r for r in result if r["ticker"] == "XOM")
        assert xom["census"] == "NEUTRAL"

    def test_news_impact_mapping(self, minimal_inputs):
        result = build_concordance(**minimal_inputs)
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["news_impact"] == "LOW_POSITIVE"
        xom = next(r for r in result if r["ticker"] == "XOM")
        assert xom["news_impact"] == "NEUTRAL"

    def test_position_limits(self, minimal_inputs):
        result = build_concordance(**minimal_inputs)
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["max_pct"] == 4.0

    def test_census_ts_map(self, minimal_inputs):
        minimal_inputs["census_ts_map"] = {"AAPL": "strong_accumulation"}
        result = build_concordance(**minimal_inputs)
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["census_ts"] == "strong_accumulation"


# ============================================================
# compute_changes
# ============================================================


class TestComputeChanges:
    def test_action_change_detected(self):
        current = [{"ticker": "AAPL", "action": "BUY", "conviction": 80}]
        previous = [{"ticker": "AAPL", "action": "HOLD", "conviction": 55}]
        changes = compute_changes(current, previous)
        assert len(changes) == 1
        assert changes[0]["type"] == "UPGRADE"
        assert changes[0]["prev_action"] == "HOLD"
        assert changes[0]["curr_action"] == "BUY"

    def test_conviction_change_detected(self):
        current = [{"ticker": "AAPL", "action": "HOLD", "conviction": 65}]
        previous = [{"ticker": "AAPL", "action": "HOLD", "conviction": 55}]
        changes = compute_changes(current, previous)
        assert len(changes) == 1
        assert changes[0]["delta"] == 10

    def test_small_conviction_change_ignored(self):
        """Changes <= 5 in conviction with same action should be ignored."""
        current = [{"ticker": "AAPL", "action": "HOLD", "conviction": 58}]
        previous = [{"ticker": "AAPL", "action": "HOLD", "conviction": 55}]
        changes = compute_changes(current, previous)
        assert len(changes) == 0

    def test_new_entry(self):
        current = [{"ticker": "NVDA", "action": "BUY", "conviction": 75}]
        previous = []
        changes = compute_changes(current, previous)
        assert len(changes) == 1
        assert changes[0]["type"] == "NEW"

    def test_downgrade(self):
        current = [{"ticker": "AAPL", "action": "SELL", "conviction": 70}]
        previous = [{"ticker": "AAPL", "action": "BUY", "conviction": 80}]
        changes = compute_changes(current, previous)
        assert changes[0]["type"] == "DOWNGRADE"

    def test_removed_stock_not_flagged(self):
        """Stocks in previous but not current are NOT flagged as changes."""
        current = []
        previous = [{"ticker": "AAPL", "action": "BUY", "conviction": 80}]
        changes = compute_changes(current, previous)
        assert len(changes) == 0


# ============================================================
# Regression tests for v3 live-run bugs
# ============================================================


class TestV3RegressionBugs:
    """Tests that reproduce the 5 critical bugs found during live committee run."""

    def test_aapl_hold_not_promoted_to_add(self):
        """
        AAPL bug: HOLD signal + fund=52.5 + EXRET=10.9% was getting ADD conv=85.
        Root cause: neutral views inflated bull_pct; quality bonus applied to HOLD.
        Fix: neutral splits evenly; HOLD cap at 70; quality bonus BUY-only.
        """
        result = synthesize_stock(
            ticker="AAPL",
            sig_data={"signal": "H", "exret": 10.9, "buy_pct": 60, "beta": 1.2, "pet": 35, "pef": 32},
            fund_data={"fundamental_score": 52.5, "quality_trap_warning": False, "pe_trajectory": "stable"},
            tech_data={"momentum_score": 10, "timing_signal": "WAIT_FOR_PULLBACK", "rsi": 55, "macd_signal": "BULLISH"},
            macro_fit="NEUTRAL",
            census_alignment="ALIGNED",
            div_score=-5,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=15,
            sector_rankings={},
            position_limit=5.0,
        )
        # Should NOT be ADD or BUY with these mediocre fundamentals
        assert result["action"] in ("HOLD", "WEAK HOLD"), (
            f"AAPL with HOLD signal + fund=52.5 should not get {result['action']} "
            f"(conv={result['conviction']})"
        )

    def test_nvda_buy_not_suppressed(self):
        """
        NVDA bug: Best fundamentals (fund=83.7, EXRET=48.4%) was getting ADD conv=65.
        Root cause: excess_exret cliff-edge (14.4 vs 15.0 threshold).
        Fix: threshold lowered to 12pp.
        """
        result = synthesize_stock(
            ticker="NVDA",
            sig_data={"signal": "B", "exret": 48.4, "buy_pct": 96, "beta": 2.4, "pet": 50, "pef": 30},
            fund_data={"fundamental_score": 83.7, "quality_trap_warning": False, "pe_trajectory": "improving"},
            tech_data={"momentum_score": -10, "timing_signal": "WAIT_FOR_PULLBACK", "rsi": 45, "macd_signal": "BEARISH"},
            macro_fit="NEUTRAL",
            census_alignment="ALIGNED",
            div_score=-10,
            census_ts_trend="stable",
            news_impact="LOW_POSITIVE",
            risk_warning=True,
            sector="Technology",
            sector_median_exret=20,
            sector_rankings={},
            position_limit=3.5,
        )
        # With fund=83.7 and BUY signal, should be at least ADD
        assert result["conviction"] >= 60, (
            f"NVDA with fund=83.7 + BUY signal should have conv>=60, got {result['conviction']}"
        )

    def test_btc_sell_with_bullish_agents(self):
        """
        BTC bug: SELL signal conv=73 when 62% of agents were bullish.
        Root cause: no penalty for SELL when tech/macro disagree.
        Fix: penalties when tech is bullish or macro favorable for SELL signals.
        """
        result = synthesize_stock(
            ticker="BTC-USD",
            sig_data={"signal": "S", "exret": -5, "buy_pct": 30, "beta": 1.5, "pet": 0, "pef": 0},
            fund_data={"fundamental_score": 40, "quality_trap_warning": False},
            tech_data={"momentum_score": 20, "timing_signal": "WAIT_FOR_PULLBACK", "rsi": 55, "macd_signal": "BULLISH"},
            macro_fit="FAVORABLE",
            census_alignment="CENSUS_DIV",
            div_score=-35,
            census_ts_trend="accumulation",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Crypto",
            sector_median_exret=0,
            sector_rankings={},
            position_limit=3.0,
        )
        # With mostly bullish agents, SELL conviction should be reduced
        if result["action"] == "SELL":
            assert result["conviction"] < 70, (
                f"BTC SELL with bullish agents should have reduced conviction, got {result['conviction']}"
            )

    def test_buy_signal_unconditional_floor_40(self):
        """
        Floor bug: BUY-signal stocks could drop to conv=15 with penalty stacking.
        Fix: unconditional floor of 40 for any BUY-signal stock.
        """
        result = synthesize_stock(
            ticker="FLOOR_TEST",
            sig_data={"signal": "B", "exret": 3, "buy_pct": 55, "beta": 2.5, "pet": 0, "pef": 0},
            fund_data={"fundamental_score": 40, "quality_trap_warning": True},
            tech_data={"momentum_score": -40, "timing_signal": "AVOID", "rsi": 80, "macd_signal": "BEARISH"},
            macro_fit="UNFAVORABLE",
            census_alignment="DIVERGENT",
            div_score=40,
            census_ts_trend="strong_distribution",
            news_impact="HIGH_NEGATIVE",
            risk_warning=True,
            sector="Technology",
            sector_median_exret=20,
            sector_rankings={},
            position_limit=2.0,
        )
        assert result["conviction"] >= 40, (
            f"BUY-signal stock conviction must be >=40, got {result['conviction']}"
        )


# ============================================================
# Constants and invariants
# ============================================================


# ============================================================
# apply_opportunity_gate (CIO C2)
# ============================================================


class TestApplyOpportunityGate:
    def _make_entry(self, conviction=70, sector="Energy", signal="B"):
        return {
            "ticker": "XOM",
            "conviction": conviction,
            "sector": sector,
            "signal": signal,
            "tech_signal": "WAIT_FOR_PULLBACK",
            "risk_warning": False,
            "action": "ADD",
        }

    def test_zero_confirmations_discount_15(self):
        """No non-scanner confirmations → -15 discount."""
        entry = self._make_entry(conviction=70)
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {}},
            tech_report={"stocks": {}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20},
        )
        assert result["conviction"] == 65  # 70 - 15 + 10 (sector gap)
        assert result["is_opportunity"] is True
        assert result["confirmations"] == 0

    def test_one_confirmation_discount_10(self):
        """One confirmation → -10 discount."""
        entry = self._make_entry(conviction=70)
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {"XOM": {"fundamental_score": 75}}},
            tech_report={"stocks": {}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20},
        )
        assert result["conviction"] == 70  # 70 - 10 + 10 (sector gap)
        assert result["confirmations"] == 1

    def test_two_confirmations_no_discount(self):
        """Two+ confirmations → no discount."""
        entry = self._make_entry(conviction=70)
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {"XOM": {"fundamental_score": 75}}},
            tech_report={"stocks": {"XOM": {"timing_signal": "ENTER_NOW"}}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20},
        )
        assert result["conviction"] == 75  # 70 + 0 + 10 (gap) = 80 → capped at 75
        assert result["confirmations"] == 2

    def test_sector_gap_bonus(self):
        """Opportunities filling missing sectors get +10."""
        entry = self._make_entry(conviction=60, sector="Energy")
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {}},
            tech_report={"stocks": {}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20},  # No Energy
        )
        assert result["fills_sector_gap"] is True
        # 60 - 15 (no confirm) + 10 (gap) = 55
        assert result["conviction"] == 55

    def test_no_sector_gap_bonus(self):
        """Opportunities in existing sectors don't get gap bonus."""
        entry = self._make_entry(conviction=60, sector="Technology")
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {}},
            tech_report={"stocks": {}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20},
        )
        assert result["fills_sector_gap"] is False
        # 60 - 15 (no confirm) = 45
        assert result["conviction"] == 45

    def test_hard_cap_75(self):
        """New opportunities cannot exceed 75 on first appearance."""
        entry = self._make_entry(conviction=90)
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {"XOM": {"fundamental_score": 85}}},
            tech_report={"stocks": {"XOM": {"timing_signal": "ENTER_NOW"}}},
            macro_fit="FAVORABLE",
            census_alignment="ALIGNED",
            portfolio_sectors={"Technology": 20},
        )
        assert result["conviction"] <= 75
        assert result["confirmations"] == 4

    def test_action_set_to_buy_new(self):
        """Opportunities with sufficient conviction get BUY NEW action."""
        entry = self._make_entry(conviction=70)
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {"XOM": {"fundamental_score": 75}}},
            tech_report={"stocks": {"XOM": {"timing_signal": "ENTER_NOW"}}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20},
        )
        assert result["action"] == "BUY NEW"

    def test_low_conviction_not_buy_new(self):
        """Opportunities with low conviction should not be BUY NEW."""
        entry = self._make_entry(conviction=40, signal="H")
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {}},
            tech_report={"stocks": {}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20, "Energy": 5},
        )
        # 40 - 15 = 25 → should be WEAK HOLD or TRIM
        assert result["action"] != "BUY NEW"


# ============================================================
# detect_sector_gaps
# ============================================================


class TestDetectSectorGaps:
    def test_finds_missing_leading_sector(self):
        portfolio_sectors = {"Technology": 20, "Financials": 5}
        sector_rankings = {
            "XLE": {"rank": 1, "return_1m": 12.5},
            "XLK": {"rank": 4, "return_1m": 5.0},
        }
        gaps = detect_sector_gaps(portfolio_sectors, sector_rankings)
        assert len(gaps) == 1
        assert gaps[0]["sector"] == "Energy"
        assert gaps[0]["urgency"] == "HIGH"

    def test_no_gaps_when_all_sectors_present(self):
        portfolio_sectors = {"Technology": 10, "Energy": 5, "Financials": 3}
        sector_rankings = {
            "XLE": {"rank": 1, "return_1m": 12.5},
            "XLK": {"rank": 2, "return_1m": 8.0},
        }
        gaps = detect_sector_gaps(portfolio_sectors, sector_rankings)
        assert len(gaps) == 0

    def test_lagging_sectors_not_flagged(self):
        portfolio_sectors = {"Technology": 20}
        sector_rankings = {
            "XLE": {"rank": 10, "return_1m": -5.0},  # Lagging
        }
        gaps = detect_sector_gaps(portfolio_sectors, sector_rankings)
        assert len(gaps) == 0

    def test_medium_urgency_for_rank_4_5(self):
        portfolio_sectors = {"Technology": 20}
        sector_rankings = {
            "XLE": {"rank": 5, "return_1m": 3.0},
        }
        gaps = detect_sector_gaps(portfolio_sectors, sector_rankings)
        assert len(gaps) == 1
        assert gaps[0]["urgency"] == "MEDIUM"

    def test_sorted_by_rank(self):
        portfolio_sectors = {}
        sector_rankings = {
            "XLV": {"rank": 3, "return_1m": 4.0},
            "XLE": {"rank": 1, "return_1m": 12.0},
            "XLU": {"rank": 2, "return_1m": 8.0},
        }
        gaps = detect_sector_gaps(portfolio_sectors, sector_rankings)
        assert gaps[0]["sector"] == "Energy"
        assert gaps[1]["sector"] == "Utilities"
        assert gaps[2]["sector"] == "Healthcare"


# ============================================================
# build_concordance with opportunities
# ============================================================


class TestBuildConcordanceWithOpportunities:
    @pytest.fixture
    def base_inputs(self):
        return dict(
            portfolio_signals={
                "AAPL": {"signal": "B", "exret": 15, "buy_pct": 70, "beta": 1.1, "pet": 30, "pef": 25},
            },
            fund_report={"stocks": {}},
            tech_report={"stocks": {}},
            macro_report={"portfolio_implications": {}, "sector_rankings": {}},
            census_report={"divergences": {"consensus_aligned": [], "signal_divergences": [], "census_divergences": []}},
            news_report={"portfolio_news": {}},
            risk_report={"consensus_warnings": [], "position_limits": {}},
            sector_map={"AAPL": "Technology"},
        )

    def test_opportunities_added_to_concordance(self, base_inputs):
        base_inputs["opportunity_signals"] = {
            "XOM": {"signal": "B", "exret": 25, "buy_pct": 85, "beta": 0.9, "pet": 12, "pef": 10},
        }
        base_inputs["opportunity_sector_map"] = {"XOM": "Energy"}
        result = build_concordance(**base_inputs)
        tickers = [r["ticker"] for r in result]
        assert "XOM" in tickers
        xom = next(r for r in result if r["ticker"] == "XOM")
        assert xom["is_opportunity"] is True

    def test_portfolio_stocks_not_duplicated(self, base_inputs):
        """If opportunity ticker is already in portfolio, skip it."""
        base_inputs["opportunity_signals"] = {
            "AAPL": {"signal": "B", "exret": 20, "buy_pct": 80, "beta": 1.0, "pet": 25, "pef": 20},
        }
        result = build_concordance(**base_inputs)
        aapl_count = sum(1 for r in result if r["ticker"] == "AAPL")
        assert aapl_count == 1

    def test_opportunity_conviction_capped(self, base_inputs):
        base_inputs["opportunity_signals"] = {
            "XOM": {"signal": "B", "exret": 40, "buy_pct": 95, "beta": 0.7, "pet": 12, "pef": 10},
        }
        base_inputs["opportunity_sector_map"] = {"XOM": "Energy"}
        result = build_concordance(**base_inputs)
        xom = next(r for r in result if r["ticker"] == "XOM")
        assert xom["conviction"] <= 75


# ============================================================
# Constants and invariants
# ============================================================


class TestConstants:
    def test_freshness_weights_valid(self):
        """All freshness weights should be between 0 and 1."""
        for agent, weight in AGENT_FRESHNESS.items():
            assert 0 < weight <= 1, f"{agent} has invalid freshness {weight}"

    def test_action_order_complete(self):
        """ACTION_ORDER should include all possible actions."""
        expected = {"SELL", "REDUCE", "TRIM", "WEAK HOLD", "BUY NEW", "BUY", "ADD", "HOLD", "STRONG HOLD"}
        assert expected == set(ACTION_ORDER.keys())

    def test_action_order_consistent(self):
        """SELL actions should sort before BUY actions."""
        assert ACTION_ORDER["SELL"] < ACTION_ORDER["BUY"]
        assert ACTION_ORDER["REDUCE"] < ACTION_ORDER["ADD"]
        assert ACTION_ORDER["TRIM"] < ACTION_ORDER["HOLD"]
        assert ACTION_ORDER["BUY NEW"] < ACTION_ORDER["BUY"]
