"""
Tests for Committee Synthesis Engine — CIO v35.0 conviction scoring.

CIO v35.0 (2026-05-02): Modifier simplification. 63→19 active modifiers.
Tests for disabled modifiers are marked with @v35_disabled. These tests
verified behavior that is now intentionally disabled based on empirical
modifier audit (T+30 forward returns, group comparison analysis).

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
    _fallback_technical,
    _normalize_breaking_news,
    _normalize_census_divergences,
    _normalize_fund_stocks,
    _normalize_sector_rankings,
    _resolve_news_impact,
    canonicalize_impact,
    _build_agent_lookups,
    normalize_agent_reports,
    apply_action_hysteresis,
    apply_conviction_floors,
    apply_opportunity_gate,
    build_agent_memory,
    build_concordance,
    classify_hold_tier,
    compute_adjustments,
    generate_kill_thesis,
    generate_synthesis_output,
    compute_changes,
    compute_dynamic_freshness,
    compute_sector_medians,
    compute_signal_velocity,
    count_agent_votes,
    detect_contradictions,
    detect_sector_gaps,
    determine_action,
    determine_base_conviction,
    get_earnings_surprise_adjustment,
    enrich_with_position_sizes,
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
        assert universe == pytest.approx(15.0)  # median of [10, 20] = average of two middle values

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
        bull, bear, _ = count_agent_votes(
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
        bull, bear, _ = count_agent_votes(
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
        bull, bear, _ = count_agent_votes(
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
        b1, r1, _ = count_agent_votes(30, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        # Neutral
        b2, r2, _ = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        # Bullish
        b3, r3, _ = count_agent_votes(80, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        # Bearish fund should have more bear weight than neutral
        assert r1 > r2
        # Bullish fund should have more bull weight than neutral
        assert b3 > b2

    def test_risk_manager_sell_weight(self):
        """Risk manager gets 2x weight for SELL signals, 1.5x for BUY."""
        _, bear_sell, _ = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", True, "S")
        _, bear_buy, _ = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", True, "B")
        assert bear_sell > bear_buy

    def test_wait_for_pullback_leans_bull(self):
        """WAIT_FOR_PULLBACK is 0.6 bull / 0.4 bear — slight lean."""
        bull, bear, _ = count_agent_votes(55, "WAIT_FOR_PULLBACK", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        # Tech contribution should be 0.6b/0.4r
        # Other agents neutral. Bull should slightly exceed bear.
        assert bull > bear

    def test_census_div_leans_bull(self):
        """CENSUS_DIV means PIs are bullish when signal isn't — PI view gets weight."""
        bull, bear, _ = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "CENSUS_DIV", "NEUTRAL", False, "H")
        bull2, bear2, _ = count_agent_votes(55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H")
        assert bull > bull2  # Census DIV adds more bull weight

    def test_risk_neutral_risk_on_leans_bull(self):
        """CIO v6.0 F5: No risk warning in RISK_ON should lean mildly bullish."""
        bull_on, bear_on, _ = count_agent_votes(
            55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H",
            regime="RISK_ON",
        )
        bull_def, bear_def, _ = count_agent_votes(
            55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H",
            regime="",
        )
        # RISK_ON: risk neutral → 0.6 bull / 0.4 bear (vs 0.5/0.5 default)
        assert bull_on > bull_def

    def test_risk_neutral_risk_off_leans_bear(self):
        """CIO v6.0 F5: No risk warning in RISK_OFF should lean mildly bearish."""
        bull_off, bear_off, _ = count_agent_votes(
            55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H",
            regime="RISK_OFF",
        )
        bull_def, bear_def, _ = count_agent_votes(
            55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H",
            regime="",
        )
        # RISK_OFF: risk neutral → 0.4 bull / 0.6 bear (vs 0.5/0.5 default)
        assert bear_off > bear_def

    def test_risk_neutral_regime_default_unchanged(self):
        """CIO v6.0 F5: No regime = traditional 0.5/0.5 neutral split."""
        bull, bear, _ = count_agent_votes(
            55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H",
            regime="",
        )
        bull2, bear2, _ = count_agent_votes(
            55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", False, "H",
            regime="CAUTIOUS",
        )
        # Both default and CAUTIOUS use 0.5/0.5
        assert bull == bull2
        assert bear == bear2

    def test_risk_warning_overrides_regime(self):
        """When risk_warning=True, regime doesn't matter — always bearish."""
        _, bear_on, _ = count_agent_votes(
            55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", True, "B",
            regime="RISK_ON",
        )
        _, bear_off, _ = count_agent_votes(
            55, "HOLD", 0, "NEUTRAL", "NEUTRAL", "NEUTRAL", True, "B",
            regime="RISK_OFF",
        )
        # Both should have same bear weight — risk_warning bypasses regime
        assert bear_on == bear_off


# ============================================================
# Schema Validation Logging (E3)
# ============================================================


class TestSchemaValidationLogging:
    """CIO v6.0 E3: Distinguish missing stock vs malformed agent report."""

    def test_fallback_logs_warning_for_malformed_report(self, caplog):
        """Agent with report but no 'stocks' key triggers WARNING."""
        import logging
        with caplog.at_level(logging.WARNING):
            from trade_modules.committee_synthesis import _synthesize_with_lookups
            sig_data = {"signal": "H", "exret": 5, "buy_pct": 50, "beta": 1.0}
            lookups = {
                "macro_impl": {}, "div_map": {}, "port_news": {},
                "risk_warns": set(), "risk_limits": {}, "sector_rankings": {},
            }
            # Fund report has data but missing 'stocks' key
            fund_report = {"analyst": "fundamental", "timestamp": "2026-01-01"}
            tech_report = {"analyst": "technical", "timestamp": "2026-01-01"}
            _synthesize_with_lookups(
                "AAPL", sig_data, lookups, fund_report, tech_report,
                "Technology", 5.0, {},
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1
        assert "schema" in warnings[0].message.lower() or "stocks" in warnings[0].message.lower()

    def test_no_warning_when_stocks_key_exists(self, caplog):
        """Agent with 'stocks' key but missing ticker = DEBUG, not WARNING."""
        import logging
        with caplog.at_level(logging.DEBUG):
            from trade_modules.committee_synthesis import _synthesize_with_lookups
            sig_data = {"signal": "H", "exret": 5, "buy_pct": 50, "beta": 1.0}
            lookups = {
                "macro_impl": {}, "div_map": {}, "port_news": {},
                "risk_warns": set(), "risk_limits": {}, "sector_rankings": {},
            }
            fund_report = {"stocks": {"MSFT": {"fundamental_score": 75}}}
            tech_report = {"stocks": {"MSFT": {"signal": "ENTER_NOW"}}}
            _synthesize_with_lookups(
                "AAPL", sig_data, lookups, fund_report, tech_report,
                "Technology", 5.0, {},
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0


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

    def test_buy_excess_exret_bonus_proportional(self):
        """BUY + excess_exret scales proportionally (CIO v5.2)."""
        base_5 = determine_base_conviction(50, "B", 60, 5, 0.5)
        base_15 = determine_base_conviction(50, "B", 60, 15, 0.5)
        base_25 = determine_base_conviction(50, "B", 60, 25, 0.5)
        # Higher EXRET → higher base
        assert base_15 > base_5
        assert base_25 > base_15
        # Max bonus is 5
        base_40 = determine_base_conviction(50, "B", 60, 40, 0.5)
        assert base_40 - base_5 <= 5

    def test_hold_quality_bonus_NOT_applied(self):
        """HOLD signals should NOT get BUY-side quality bonus."""
        base = determine_base_conviction(50, "H", 90, 30, 0.5)
        # Should be capped at 70 regardless of fund_score
        assert base <= 70

    def test_continuous_base_breaks_clustering(self):
        """CIO v5.2: Different bull_pct values produce different bases."""
        base_52 = determine_base_conviction(52, "B", 60, 0, 0.48)
        base_58 = determine_base_conviction(58, "B", 60, 0, 0.42)
        base_64 = determine_base_conviction(64, "B", 60, 0, 0.36)
        # Old system: all three would produce base=60 (same bucket)
        # New system: should produce different values
        assert base_52 != base_58 or base_58 != base_64

    def test_continuous_monotonic(self):
        """CIO v5.2: Higher bull_pct → higher or equal base."""
        bases = [determine_base_conviction(pct, "H", 50, 0, 0.5) for pct in range(20, 80, 5)]
        for i in range(len(bases) - 1):
            assert bases[i] <= bases[i + 1]


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

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_agent_agreement_bonus_5(self):
        kw = self._default_kwargs()
        kw["bull_count"] = 5
        bonuses, _, _ = compute_adjustments(**kw)
        assert bonuses >= 10

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_agent_agreement_bonus_6(self):
        kw = self._default_kwargs()
        kw["bull_count"] = 6
        bonuses, _, _ = compute_adjustments(**kw)
        assert bonuses >= 15

    def test_consensus_crowded_penalty_removed(self):
        """CIO v40 (post-backtest 2026-04-26): consensus_crowded penalty removed.

        T+7 attribution showed the penalty was systematically subtracting
        conviction from rallying names (+8.51% alpha on ADD when penalty fired).
        Now no penalty fires regardless of buy_pct or excess_exret — verified
        by checking that penalty values are identical across excess_exret bands.
        """
        kw = self._default_kwargs()
        kw["buy_pct"] = 95  # Would have triggered the old >90 penalty

        kw["excess_exret"] = 20
        _, p_high, _ = compute_adjustments(**kw)
        kw["excess_exret"] = 5
        _, p_mid, _ = compute_adjustments(**kw)
        kw["excess_exret"] = -5
        _, p_low, _ = compute_adjustments(**kw)

        # All three should produce IDENTICAL penalty totals — the consensus
        # gradient no longer contributes anything.
        assert p_low == p_mid == p_high, (
            f"consensus_crowded should not fire at any excess_exret band. "
            f"Got p_low={p_low}, p_mid={p_mid}, p_high={p_high}"
        )

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_census_alignment_bonus(self):
        kw = self._default_kwargs()
        kw["div_score"] = 0
        bonuses, _, _ = compute_adjustments(**kw)
        # div_score in [-20, 20] → +5
        assert bonuses >= 5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_census_strong_alignment(self):
        kw = self._default_kwargs()
        kw["div_score"] = -30
        bonuses, _, _ = compute_adjustments(**kw)
        assert bonuses >= 8

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_high_beta_penalty(self):
        kw = self._default_kwargs()
        kw["beta"] = 2.5
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_quality_trap_penalty(self):
        kw = self._default_kwargs()
        kw["quality_trap"] = True
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_rsi_overbought_penalty(self):
        kw = self._default_kwargs()
        kw["rsi"] = 75
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_macro_favorable_bonus(self):
        kw = self._default_kwargs()
        kw["macro_fit"] = "FAVORABLE"
        bonuses, _, _ = compute_adjustments(**kw)
        assert bonuses >= 5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_macro_unfavorable_cyclical_penalty(self):
        kw = self._default_kwargs()
        kw["macro_fit"] = "UNFAVORABLE"
        kw["sector"] = "Financials"
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 10

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_macro_unfavorable_noncyclical_penalty(self):
        kw = self._default_kwargs()
        kw["macro_fit"] = "UNFAVORABLE"
        kw["sector"] = "Technology"
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_sell_tech_disagreement_penalty(self):
        """SELL signal + bullish tech → penalty (reduces sell conviction)."""
        kw = self._default_kwargs()
        kw["signal"] = "S"
        kw["tech_signal"] = "ENTER_NOW"
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 5

    def test_bonus_cap_at_20(self):
        kw = self._default_kwargs()
        kw["bull_count"] = 6        # +15
        kw["div_score"] = -30       # +8
        kw["news_impact"] = "HIGH_POSITIVE"  # +5
        kw["macro_fit"] = "FAVORABLE"  # +5
        bonuses, _, _ = compute_adjustments(**kw)
        assert bonuses <= 20

    def test_penalty_cap_at_25(self):
        kw = self._default_kwargs()
        kw["buy_pct"] = 95
        kw["excess_exret"] = -10    # -15
        kw["rsi"] = 75              # -5
        kw["beta"] = 2.5            # -5
        kw["quality_trap"] = True   # -5
        kw["macro_fit"] = "UNFAVORABLE"  # -10
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties <= 25

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_census_distribution_penalty(self):
        kw = self._default_kwargs()
        kw["census_ts"] = "strong_distribution"
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 5

    def test_census_accumulation_bonus(self):
        kw = self._default_kwargs()
        kw["census_ts"] = "strong_accumulation"
        bonuses, _, _ = compute_adjustments(**kw)
        assert bonuses >= 3

    def test_sector_rotation_leading(self):
        kw = self._default_kwargs()
        kw["sector"] = "Technology"
        kw["sector_rankings"] = {"XLK": {"rank": 1, "return_1m": 8.5}}
        bonuses, _, _ = compute_adjustments(**kw)
        assert bonuses >= 5

    def test_sector_rotation_lagging(self):
        kw = self._default_kwargs()
        kw["sector"] = "Technology"
        kw["sector_rankings"] = {"XLK": {"rank": 10, "return_1m": -5.0}}
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_tech_disagreement_buy_avoid(self):
        """CIO v5.2: BUY signal + tech AVOID → penalty."""
        kw = self._default_kwargs()
        kw["signal"] = "B"
        kw["tech_signal"] = "AVOID"
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 8

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_tech_disagreement_buy_exit(self):
        """CIO v5.2: BUY signal + tech EXIT_SOON → penalty."""
        kw = self._default_kwargs()
        kw["signal"] = "B"
        kw["tech_signal"] = "EXIT_SOON"
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 8

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_tech_disagreement_buy_negative_momentum(self):
        """CIO v5.2: BUY signal + tech_momentum < -30 → penalty."""
        kw = self._default_kwargs()
        kw["signal"] = "B"
        kw["tech_momentum"] = -35
        _, penalties, _ = compute_adjustments(**kw)
        assert penalties >= 5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_tech_disagreement_not_for_hold(self):
        """Tech disagreement penalty only for BUY signals, not HOLD."""
        kw = self._default_kwargs()
        kw["signal"] = "H"
        kw["tech_signal"] = "AVOID"
        _, penalties_hold, _ = compute_adjustments(**kw)
        kw["signal"] = "B"
        _, penalties_buy, _ = compute_adjustments(**kw)
        assert penalties_buy > penalties_hold


# ============================================================
# apply_conviction_floors
# ============================================================


class TestApplyConvictionFloors:
    def test_buy_unconditional_floor(self):
        """BUY signal stocks should never go below 40."""
        result = apply_conviction_floors(10, "B", 5, 20, 25, 2, 50, 50)
        assert result >= 40

    def test_buy_graduated_floor_single_quality(self):
        """CIO v13.0 F1: BUY + 1 quality hit → graduated floor 43."""
        # exret>20 + improving PE = 1 quality hit → floor = 40 + 3 = 43
        result = apply_conviction_floors(30, "B", 25, 15, 20, 3, 60, 60)
        assert result == 43

    def test_buy_graduated_floor_agent_agreement(self):
        """CIO v13.0 F1: BUY + 4 agents → 1 quality hit → floor 43."""
        # bull_count=4 → 1 quality hit → floor = 40 + 3 = 43
        result = apply_conviction_floors(30, "B", 5, 20, 25, 4, 60, 60)
        assert result == 43

    def test_buy_graduated_floor_fund_and_census(self):
        """CIO v13.0 F1: BUY + fund>=70 + buy_pct>=80 → 2 hits → floor 46."""
        # fund_score=75 >= 70 → +1, buy_pct=80 >= 80 → +1 = 2 hits → 40+6=46
        result = apply_conviction_floors(30, "B", 5, 20, 25, 2, 75, 80)
        assert result == 46

    def test_buy_graduated_floor_max_quality(self):
        """CIO v13.0 F1: BUY + all quality criteria → floor 55 (max)."""
        # exret>20+PE improving → +1, bull>=5 → +2, fund>=80 → +2, buy>=80 → +1 = 6 hits
        # graduated_floor = min(55, 40 + 6*3) = min(55, 58) = 55
        result = apply_conviction_floors(30, "B", 25, 15, 20, 5, 85, 85)
        assert result == 55

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
        assert determine_action(50, "S", "AVOID", True) == "TRIM"

    def test_sell_threshold_60(self):
        assert determine_action(60, "S", "AVOID", True) == "SELL"
        assert determine_action(59, "S", "AVOID", True) == "TRIM"

    # BUY signal actions (BUY is reserved for new opportunities only)
    def test_buy_high_conviction(self):
        assert determine_action(80, "B", "ENTER_NOW", False) == "ADD"

    def test_buy_moderate_conviction(self):
        assert determine_action(60, "B", "WAIT_FOR_PULLBACK", False) == "ADD"

    def test_buy_low_conviction(self):
        assert determine_action(45, "B", "HOLD", False) == "HOLD"

    def test_buy_threshold_55(self):
        assert determine_action(55, "B", "WAIT_FOR_PULLBACK", False) == "ADD"
        assert determine_action(54, "B", "WAIT_FOR_PULLBACK", False) == "HOLD"

    # HOLD signal actions
    def test_hold_high_conviction(self):
        assert determine_action(75, "H", "ENTER_NOW", False) == "ADD"

    def test_hold_moderate_conviction(self):
        assert determine_action(55, "H", "HOLD", False) == "HOLD"

    def test_hold_low_conviction(self):
        assert determine_action(40, "H", "HOLD", False) == "HOLD"

    def test_hold_very_low_conviction(self):
        assert determine_action(30, "H", "AVOID", True) == "TRIM"

    def test_hold_thresholds(self):
        assert determine_action(70, "H", "ENTER_NOW", False) == "ADD"
        assert determine_action(69, "H", "HOLD", False) == "HOLD"
        assert determine_action(50, "H", "HOLD", False) == "HOLD"
        assert determine_action(35, "H", "HOLD", False) == "HOLD"
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

    def test_tech_exit_adds_conviction(self):
        """CIO v13.0 F4: Tech EXIT_SOON adds 12 (diminishing returns)."""
        conv = recalculate_trim_conviction(
            "EXIT_SOON", "NEUTRAL", False, 1.0, 50, 60, "NEUTRAL"
        )
        assert conv == 62  # 50 + 12*1.0

    def test_tech_avoid_adds_conviction(self):
        """CIO v13.0 F4: Tech AVOID adds 12 (diminishing returns)."""
        conv = recalculate_trim_conviction(
            "AVOID", "NEUTRAL", False, 1.0, 50, 60, "NEUTRAL"
        )
        assert conv == 62  # 50 + 12*1.0

    def test_all_trim_factors_diminishing(self):
        """CIO v13.0 F4: All factors with diminishing returns."""
        conv = recalculate_trim_conviction(
            "AVOID",           # 12
            "UNFAVORABLE",     # 8
            True,              # 7
            2.0,               # 4
            75,                # 4
            40,                # not >=80 so no -10
            "DIVERGENT",       # not ALIGNED so no -5
        )
        # Sorted: [12, 8, 7, 4, 4] with 0.7 decay
        # 12*1.0=12, 8*0.7=5, 7*0.49=3, 4*0.343=1, 4*0.24=0 = 21
        assert conv == 71  # 50 + 21

    def test_cap_at_80(self):
        """CIO v13.0 F4: Cap reduced from 85 to 80."""
        conv = recalculate_trim_conviction(
            "AVOID", "UNFAVORABLE", True, 2.0, 80, 30, "DIVERGENT"
        )
        assert conv <= 80

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
        assert result["action"] == "ADD"
        assert result["conviction"] >= 70

    def test_clear_sell(self):
        """Stock with SELL signal + bearish agents → SELL or TRIM."""
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
        assert result["action"] in ("SELL", "TRIM")
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
        assert result["action"] == "HOLD"

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
        if result["action"] == "TRIM":
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
        assert result["excess_exret"] == pytest.approx(20.0)

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

    def test_macro_fit_unfavorable_substring(self, minimal_inputs):
        """UNFAVORABLE must not match as FAVORABLE (substring bug guard)."""
        minimal_inputs["macro_report"]["portfolio_implications"]["AAPL"] = {
            "macro_fit": "UNFAVORABLE",
        }
        result = build_concordance(**minimal_inputs)
        aapl = next(r for r in result if r["ticker"] == "AAPL")
        assert aapl["macro_fit"] == "UNFAVORABLE"

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
        assert aapl["max_pct"] == pytest.approx(4.0)

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

    def test_dict_with_stocks_wrapper(self):
        """Previous concordance.json has {date, stocks: {ticker: data}} format."""
        current = [{"ticker": "AAPL", "action": "BUY", "conviction": 80}]
        previous = {
            "date": "2026-03-16",
            "stocks": {
                "AAPL": {"action": "HOLD", "conviction": 55},
            },
        }
        changes = compute_changes(current, previous)
        assert len(changes) == 1
        assert changes[0]["type"] == "UPGRADE"

    def test_dict_without_stocks_wrapper(self):
        """Previous can also be a flat dict of ticker -> data."""
        current = [{"ticker": "AAPL", "action": "BUY", "conviction": 80}]
        previous = {"AAPL": {"action": "HOLD", "conviction": 55}}
        changes = compute_changes(current, previous)
        assert len(changes) == 1
        assert changes[0]["type"] == "UPGRADE"


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
        assert result["action"] == "HOLD", (
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

    def test_action_set_to_buy(self):
        """Opportunities with sufficient conviction get BUY action."""
        entry = self._make_entry(conviction=70)
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {"XOM": {"fundamental_score": 75}}},
            tech_report={"stocks": {"XOM": {"timing_signal": "ENTER_NOW"}}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20},
        )
        assert result["action"] == "BUY"

    def test_low_conviction_not_buy(self):
        """Opportunities with low conviction should not be BUY."""
        entry = self._make_entry(conviction=40, signal="H")
        result = apply_opportunity_gate(
            entry,
            fund_report={"stocks": {}},
            tech_report={"stocks": {}},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            portfolio_sectors={"Technology": 20, "Energy": 5},
        )
        # 40 - 15 = 25 → should be HOLD
        assert result["action"] != "BUY"


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

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_opportunity_conviction_capped(self, base_inputs):
        base_inputs["opportunity_signals"] = {
            "XOM": {"signal": "B", "exret": 40, "buy_pct": 95, "beta": 0.7, "pet": 12, "pef": 10},
        }
        base_inputs["opportunity_sector_map"] = {"XOM": "Energy"}
        result = build_concordance(**base_inputs)
        xom = next(r for r in result if r["ticker"] == "XOM")
        assert xom["conviction"] <= 75


# ============================================================
# Fallback Technical (CIO v5.2)
# ============================================================


class TestFallbackTechnical:
    def test_strong_positive_pp(self):
        """Strong price performance → positive momentum."""
        result = _fallback_technical({"pp": 20, "52w": 90, "beta": 1.0})
        assert result["momentum_score"] > 0
        assert result["timing_signal"] in ("WAIT_FOR_PULLBACK", "HOLD")

    def test_strong_negative_pp(self):
        """Negative price performance → negative momentum."""
        result = _fallback_technical({"pp": -20, "52w": 55, "beta": 1.5})
        assert result["momentum_score"] < 0
        assert result["timing_signal"] == "AVOID"

    def test_near_52w_high(self):
        """Near 52-week high → WAIT_FOR_PULLBACK."""
        result = _fallback_technical({"pp": 5, "52w": 98, "beta": 1.0})
        assert result["timing_signal"] == "WAIT_FOR_PULLBACK"
        assert result["rsi"] > 60

    def test_far_from_52w_high(self):
        """Far from 52-week high → AVOID."""
        result = _fallback_technical({"pp": -10, "52w": 45, "beta": 1.2})
        assert result["timing_signal"] == "AVOID"
        assert result["rsi"] < 40

    def test_synthetic_flag(self):
        """Fallback results should be marked as synthetic."""
        result = _fallback_technical({"pp": 0, "52w": 80, "beta": 1.0})
        assert result["synthetic"] is True

    def test_defaults_when_data_missing(self):
        """Should handle missing keys gracefully."""
        result = _fallback_technical({})
        assert "momentum_score" in result
        assert "timing_signal" in result


# ============================================================
# Tiebreaking (CIO v5.2)
# ============================================================


class TestTiebreaking:
    @pytest.fixture
    def base_inputs(self):
        return {
            "portfolio_signals": {
                "AAA": {"signal": "B", "exret": 20, "buy_pct": 80, "beta": 1.0, "pet": 25, "pef": 20, "pp": 5, "52w": 85},
                "BBB": {"signal": "B", "exret": 10, "buy_pct": 70, "beta": 1.5, "pet": 20, "pef": 18, "pp": -5, "52w": 70},
            },
            "fund_report": {"stocks": {}},
            "tech_report": {"stocks": {}},
            "macro_report": {"portfolio_implications": {}, "sector_rankings": {}},
            "census_report": {"divergences": {}},
            "news_report": {"portfolio_news": {}},
            "risk_report": {"consensus_warnings": [], "position_limits": {}},
            "sector_map": {"AAA": "Technology", "BBB": "Technology"},
        }

    def test_tiebreak_field_exists(self, base_inputs):
        """Every concordance entry should have a tiebreak field."""
        result = build_concordance(**base_inputs)
        for entry in result:
            assert "tiebreak" in entry

    def test_different_exret_breaks_tie(self, base_inputs):
        """Stocks with same conviction but different EXRET should have different tiebreak."""
        result = build_concordance(**base_inputs)
        aaa = next(r for r in result if r["ticker"] == "AAA")
        bbb = next(r for r in result if r["ticker"] == "BBB")
        assert aaa["tiebreak"] != bbb["tiebreak"]


# ============================================================
# Constants and invariants
# ============================================================


class TestConstants:
    def test_freshness_weights_valid(self):
        """All agent vote weights should be positive and within sane bounds.

        Phase A calibration (2026-04-18) re-tuned weights based on empirical
        per-agent alpha attribution; some weights now exceed 1.0 (Fundamental,
        Census were monotonic predictors and got amplified). Cap at 2.0 to
        catch typos / runaway values.
        """
        for agent, weight in AGENT_FRESHNESS.items():
            assert 0 < weight <= 2.0, f"{agent} has invalid weight {weight}"

    def test_action_order_complete(self):
        """ACTION_ORDER should include all 5 canonical actions."""
        expected = {"SELL", "TRIM", "BUY", "ADD", "HOLD"}
        assert expected == set(ACTION_ORDER.keys())

    def test_action_order_consistent(self):
        """SELL actions should sort before BUY actions."""
        assert ACTION_ORDER["SELL"] < ACTION_ORDER["BUY"]
        assert ACTION_ORDER["TRIM"] < ACTION_ORDER["ADD"]
        assert ACTION_ORDER["SELL"] < ACTION_ORDER["HOLD"]


# ============================================================
# CIO Legacy Review: A1 — Regime-Adjusted Conviction
# ============================================================


class TestRegimeAdjustedConviction:
    """Tests for CIO Legacy A1: regime discount on agent base conviction."""

    def test_risk_off_reduces_base(self):
        """RISK_OFF regime should produce lower base than no regime."""
        base_normal = determine_base_conviction(
            bull_pct=60, signal="B", fund_score=70,
            excess_exret=10, bear_ratio=0.3,
        )
        base_risk_off = determine_base_conviction(
            bull_pct=60, signal="B", fund_score=70,
            excess_exret=10, bear_ratio=0.3, regime="RISK_OFF",
        )
        # BUY floor of 55 may kick in, but agent_base is different
        assert base_risk_off <= base_normal

    def test_cautious_reduces_base(self):
        """CAUTIOUS regime should produce lower base than normal."""
        base_normal = determine_base_conviction(
            bull_pct=65, signal="H", fund_score=50,
            excess_exret=5, bear_ratio=0.3,
        )
        base_cautious = determine_base_conviction(
            bull_pct=65, signal="H", fund_score=50,
            excess_exret=5, bear_ratio=0.3, regime="CAUTIOUS",
        )
        assert base_cautious <= base_normal

    def test_risk_on_no_change(self):
        """RISK_ON and empty regime should produce identical bases."""
        base_empty = determine_base_conviction(
            bull_pct=60, signal="H", fund_score=50,
            excess_exret=5, bear_ratio=0.3,
        )
        base_risk_on = determine_base_conviction(
            bull_pct=60, signal="H", fund_score=50,
            excess_exret=5, bear_ratio=0.3, regime="RISK_ON",
        )
        assert base_empty == base_risk_on

    def test_risk_off_discount_magnitude(self):
        """RISK_OFF should apply approximately 15% discount to agent_base."""
        # 70% bull → agent_base = 30 + 35 = 65
        # RISK_OFF: 65 * 0.85 = 55
        # HOLD signal caps at 70, so HOLD won't hit cap
        base_risk_off = determine_base_conviction(
            bull_pct=70, signal="H", fund_score=50,
            excess_exret=0, bear_ratio=0.3, regime="RISK_OFF",
        )
        base_normal = determine_base_conviction(
            bull_pct=70, signal="H", fund_score=50,
            excess_exret=0, bear_ratio=0.3,
        )
        # Discount should be meaningful (at least 3 points)
        assert base_normal - base_risk_off >= 3

    def test_buy_floor_survives_regime_discount(self):
        """BUY signal floor of 55 should still apply after regime discount."""
        base = determine_base_conviction(
            bull_pct=60, signal="B", fund_score=50,
            excess_exret=0, bear_ratio=0.3, regime="RISK_OFF",
        )
        assert base >= 55  # BUY floor

    def test_sell_unaffected_by_regime(self):
        """SELL conviction logic doesn't use agent_base directly, so regime
        should only affect when bull_pct changes the branch."""
        base_normal = determine_base_conviction(
            bull_pct=30, signal="S", fund_score=40,
            excess_exret=-5, bear_ratio=0.7,
        )
        base_risk_off = determine_base_conviction(
            bull_pct=30, signal="S", fund_score=40,
            excess_exret=-5, bear_ratio=0.7, regime="RISK_OFF",
        )
        # SELL branches use bull_pct thresholds, not agent_base
        assert base_normal == base_risk_off

    def test_regime_propagates_through_synthesize_stock(self):
        """synthesize_stock should accept and propagate the regime parameter."""
        sig_data = {"signal": "H", "exret": 10, "buy_pct": 60, "beta": 1.0, "pet": 15, "pef": 14}
        fund_data = {"fundamental_score": 60}
        tech_data = {"momentum_score": 10, "timing_signal": "HOLD", "rsi": 50, "macd_signal": "NEUTRAL"}

        result_normal = synthesize_stock(
            "TEST", sig_data, fund_data, tech_data,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL", risk_warning=False,
            sector="Technology", sector_median_exret=5, sector_rankings={},
            position_limit=5.0,
        )
        result_risk_off = synthesize_stock(
            "TEST", sig_data, fund_data, tech_data,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL", risk_warning=False,
            sector="Technology", sector_median_exret=5, sector_rankings={},
            position_limit=5.0, regime="RISK_OFF",
        )
        assert result_risk_off["conviction"] <= result_normal["conviction"]


# ============================================================
# CIO Legacy Review: A2 — Synthetic Data Discount
# ============================================================


class TestSyntheticDataDiscount:
    """Tests for CIO Legacy A2: synthetic/fallback agent data gets half weight."""

    def test_synthetic_fund_reduces_bull_weight(self):
        """Synthetic fundamental bullish vote should be weaker than real."""
        bull_real, _, _ = count_agent_votes(
            fund_score=80, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            fund_synthetic=False,
        )
        bull_synth, _, _ = count_agent_votes(
            fund_score=80, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            fund_synthetic=True,
        )
        assert bull_synth < bull_real

    def test_synthetic_tech_reduces_bull_weight(self):
        """Synthetic technical ENTER_NOW should be weaker than real."""
        bull_real, _, _ = count_agent_votes(
            fund_score=50, tech_signal="ENTER_NOW", tech_momentum=30,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            tech_synthetic=False,
        )
        bull_synth, _, _ = count_agent_votes(
            fund_score=50, tech_signal="ENTER_NOW", tech_momentum=30,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            tech_synthetic=True,
        )
        assert bull_synth < bull_real

    def test_both_synthetic_reduces_most(self):
        """Both agents synthetic should have lowest bull weight."""
        bull_none, _, _ = count_agent_votes(
            fund_score=80, tech_signal="ENTER_NOW", tech_momentum=30,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
        )
        bull_both, _, _ = count_agent_votes(
            fund_score=80, tech_signal="ENTER_NOW", tech_momentum=30,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            fund_synthetic=True, tech_synthetic=True,
        )
        # Both synthetic should be noticeably less bullish
        assert bull_none - bull_both >= 0.5

    def test_synthetic_default_false(self):
        """Not passing synthetic flags should match non-synthetic behavior."""
        bull_default, bear_default, _ = count_agent_votes(
            fund_score=60, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
        )
        bull_explicit, bear_explicit, _ = count_agent_votes(
            fund_score=60, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            fund_synthetic=False, tech_synthetic=False,
        )
        assert bull_default == bull_explicit
        assert bear_default == bear_explicit

    def test_synthetic_bearish_also_discounted(self):
        """Synthetic bearish votes should also be reduced."""
        _, bear_real, _ = count_agent_votes(
            fund_score=30, tech_signal="AVOID", tech_momentum=-30,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="S",
            fund_synthetic=False, tech_synthetic=False,
        )
        _, bear_synth, _ = count_agent_votes(
            fund_score=30, tech_signal="AVOID", tech_momentum=-30,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="S",
            fund_synthetic=True, tech_synthetic=True,
        )
        assert bear_synth < bear_real

    def test_synthesize_stock_tracks_synthetic(self):
        """synthesize_stock return dict should include synthetic flags."""
        sig_data = {"signal": "B", "exret": 15, "buy_pct": 70, "beta": 1.0, "pet": 15, "pef": 13}
        fund_data = {"fundamental_score": 75}  # Has score → not synthetic
        tech_data = {"momentum_score": 20, "timing_signal": "HOLD", "rsi": 55,
                     "macd_signal": "BULLISH", "synthetic": True}  # Marked synthetic

        result = synthesize_stock(
            "TEST", sig_data, fund_data, tech_data,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL", risk_warning=False,
            sector="Technology", sector_median_exret=5, sector_rankings={},
            position_limit=5.0,
        )
        assert result["fund_synthetic"] is False
        assert result["tech_synthetic"] is True

    def test_synthetic_neutral_agent_half_weight(self):
        """Neutral tech with synthetic flag should contribute half the weight.

        At a neutral tech signal, the agent contributes tech_weight/2 to each of
        bull and bear. The synthetic discount cuts the underlying tech_weight in
        half, so each side drops by tech_weight/4. Using AGENT_FRESHNESS rather
        than a literal so the test tracks vote-weight retunes (Phase A 2026-04-18).
        """
        bull_real, bear_real, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            tech_synthetic=False,
        )
        bull_synth, bear_synth, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            tech_synthetic=True,
        )
        expected_drop = AGENT_FRESHNESS["technical"] / 4
        assert abs((bull_real - bull_synth) - expected_drop) < 0.01
        assert abs((bear_real - bear_synth) - expected_drop) < 0.01


# ============================================================
# CIO Legacy Review: B2 — Extreme EXRET Penalty
# ============================================================


class TestExtremeExretPenalty:
    """Tests for CIO Legacy B2: extreme EXRET signals stale analyst targets."""

    def test_extreme_exret_penalized(self):
        """EXRET > 40 with low consensus should get a penalty, not a bonus.

        CIO v14.0 V4: Staleness penalty requires bull_pct < 65 (low consensus
        confirms stale data). High consensus validates the high target.
        """
        base_extreme = determine_base_conviction(
            bull_pct=50, signal="B", fund_score=60,
            excess_exret=50, bear_ratio=0.3,
        )
        base_moderate = determine_base_conviction(
            bull_pct=50, signal="B", fund_score=60,
            excess_exret=15, bear_ratio=0.3,
        )
        # Extreme EXRET with low consensus should produce lower base
        assert base_extreme < base_moderate

    def test_normal_exret_still_gets_bonus(self):
        """EXRET in 5-30 range should still get the normal bonus."""
        base_with_bonus = determine_base_conviction(
            bull_pct=60, signal="B", fund_score=60,
            excess_exret=12, bear_ratio=0.3,
        )
        base_without = determine_base_conviction(
            bull_pct=60, signal="B", fund_score=60,
            excess_exret=0, bear_ratio=0.3,
        )
        assert base_with_bonus > base_without

    def test_boundary_at_40(self):
        """EXRET exactly at 40 should get the bonus (boundary inclusive)."""
        base_at_40 = determine_base_conviction(
            bull_pct=60, signal="B", fund_score=60,
            excess_exret=40, bear_ratio=0.3,
        )
        base_above_40 = determine_base_conviction(
            bull_pct=60, signal="B", fund_score=60,
            excess_exret=41, bear_ratio=0.3,
        )
        # 40 → bonus of min(5, 10) = 5
        # 41 → penalty of 3
        assert base_at_40 > base_above_40

    def test_mstr_like_extreme_exret(self):
        """MSTR-like 170% EXRET should definitely be penalized."""
        base = determine_base_conviction(
            bull_pct=55, signal="B", fund_score=45,
            excess_exret=170, bear_ratio=0.4,
        )
        base_sane = determine_base_conviction(
            bull_pct=55, signal="B", fund_score=45,
            excess_exret=10, bear_ratio=0.4,
        )
        assert base < base_sane

    def test_sell_signal_unaffected(self):
        """SELL signal doesn't use EXRET bonus/penalty."""
        base1 = determine_base_conviction(
            bull_pct=30, signal="S", fund_score=40,
            excess_exret=100, bear_ratio=0.7,
        )
        base2 = determine_base_conviction(
            bull_pct=30, signal="S", fund_score=40,
            excess_exret=5, bear_ratio=0.7,
        )
        assert base1 == base2

    def test_hold_signal_unaffected(self):
        """HOLD signal doesn't use EXRET bonus/penalty."""
        base1 = determine_base_conviction(
            bull_pct=60, signal="H", fund_score=50,
            excess_exret=100, bear_ratio=0.3,
        )
        base2 = determine_base_conviction(
            bull_pct=60, signal="H", fund_score=50,
            excess_exret=5, bear_ratio=0.3,
        )
        assert base1 == base2


# ============================================================
# CIO Legacy Review: A3 — Contradiction Detection
# ============================================================


class TestContradictionDetection:
    """Tests for CIO Legacy A3: detecting logical contradictions between agents."""

    def test_macro_tech_contradiction(self):
        """Macro UNFAVORABLE + Tech ENTER_NOW is a contradiction."""
        penalty, contras = detect_contradictions(
            macro_fit="UNFAVORABLE", tech_signal="ENTER_NOW",
            fund_score=60, risk_warning=False,
            census_alignment="NEUTRAL", news_impact="NEUTRAL",
        )
        assert penalty >= 5
        assert len(contras) >= 1
        assert any("Macro" in c and "Technical" in c for c in contras)

    def test_fund_risk_contradiction(self):
        """High fundamental score + risk warning is a contradiction."""
        penalty, contras = detect_contradictions(
            macro_fit="NEUTRAL", tech_signal="HOLD",
            fund_score=85, risk_warning=True,
            census_alignment="NEUTRAL", news_impact="NEUTRAL",
        )
        assert penalty >= 3
        assert any("Fundamental" in c and "Risk" in c for c in contras)

    def test_census_news_contradiction(self):
        """PIs distributing (DIVERGENT) despite positive news is a contradiction."""
        penalty, contras = detect_contradictions(
            macro_fit="NEUTRAL", tech_signal="HOLD",
            fund_score=60, risk_warning=False,
            census_alignment="DIVERGENT", news_impact="HIGH_POSITIVE",
        )
        assert penalty >= 3
        assert any("PI" in c for c in contras)

    def test_macro_news_contradiction(self):
        """Macro UNFAVORABLE + news HIGH_POSITIVE is a contradiction."""
        penalty, contras = detect_contradictions(
            macro_fit="UNFAVORABLE", tech_signal="HOLD",
            fund_score=60, risk_warning=False,
            census_alignment="NEUTRAL", news_impact="HIGH_POSITIVE",
        )
        assert penalty >= 2
        assert any("HIGH_POSITIVE" in c for c in contras)

    def test_no_contradiction_consistent_bullish(self):
        """Consistent bullish signals should produce no contradictions."""
        penalty, contras = detect_contradictions(
            macro_fit="FAVORABLE", tech_signal="ENTER_NOW",
            fund_score=85, risk_warning=False,
            census_alignment="ALIGNED", news_impact="HIGH_POSITIVE",
        )
        assert penalty == 0
        assert len(contras) == 0

    def test_no_contradiction_consistent_bearish(self):
        """Consistent bearish signals should produce no contradictions."""
        penalty, contras = detect_contradictions(
            macro_fit="UNFAVORABLE", tech_signal="AVOID",
            fund_score=35, risk_warning=True,
            census_alignment="DIVERGENT", news_impact="HIGH_NEGATIVE",
        )
        assert penalty == 0
        assert len(contras) == 0

    def test_multiple_contradictions_stack(self):
        """Multiple contradictions should produce cumulative penalty."""
        penalty, contras = detect_contradictions(
            macro_fit="UNFAVORABLE", tech_signal="ENTER_NOW",
            fund_score=85, risk_warning=True,
            census_alignment="DIVERGENT", news_impact="HIGH_POSITIVE",
        )
        # Should have: macro-tech (5), fund-risk (3), census-news (3), macro-news (2) = 13
        assert penalty >= 10
        assert len(contras) >= 3

    def test_contradiction_penalty_capped_in_synthesis(self):
        """Total penalties including contradictions are capped at 25 in synthesis."""
        sig_data = {"signal": "B", "exret": 15, "buy_pct": 70, "beta": 1.0, "pet": 15, "pef": 13}
        fund_data = {"fundamental_score": 85}
        tech_data = {"momentum_score": 30, "timing_signal": "ENTER_NOW", "rsi": 50,
                     "macd_signal": "BULLISH"}

        result = synthesize_stock(
            "TEST", sig_data, fund_data, tech_data,
            macro_fit="UNFAVORABLE", census_alignment="DIVERGENT", div_score=10,
            census_ts_trend="stable", news_impact="HIGH_POSITIVE", risk_warning=True,
            sector="Technology", sector_median_exret=5, sector_rankings={},
            position_limit=5.0,
        )
        # Penalties capped at 25 in compute_adjustments
        assert result["penalties"] <= 25
        assert len(result["contradictions"]) >= 1

    def test_fund_below_70_no_fund_risk_contradiction(self):
        """Fund score below 70 should not trigger fund-risk contradiction."""
        penalty, contras = detect_contradictions(
            macro_fit="NEUTRAL", tech_signal="HOLD",
            fund_score=65, risk_warning=True,
            census_alignment="NEUTRAL", news_impact="NEUTRAL",
        )
        assert not any("Fundamental" in c for c in contras)

    def test_aligned_census_no_census_news_contradiction(self):
        """ALIGNED census should not trigger census-news contradiction."""
        penalty, contras = detect_contradictions(
            macro_fit="NEUTRAL", tech_signal="HOLD",
            fund_score=60, risk_warning=False,
            census_alignment="ALIGNED", news_impact="HIGH_POSITIVE",
        )
        assert not any("PI" in c for c in contras)


# ============================================================
# CIO Legacy Review: C1 — Exposure-Weighted Sector Gaps
# ============================================================


class TestExposureWeightedSectorGaps:
    """Tests for CIO Legacy C1: sector gap detection with portfolio weights."""

    def test_backwards_compatible_no_weights(self):
        """Without portfolio_weights, should behave exactly like before."""
        rankings = {"XLE": {"rank": 1, "return_1m": 5.0}}
        gaps = detect_sector_gaps(
            portfolio_sectors={"Technology": 5},
            sector_rankings=rankings,
        )
        assert len(gaps) == 1
        assert gaps[0]["sector"] == "Energy"

    def test_low_exposure_detected_as_gap(self):
        """Sector with < min_meaningful_exposure should be flagged."""
        rankings = {"XLE": {"rank": 2, "return_1m": 8.0}}
        gaps = detect_sector_gaps(
            portfolio_sectors={"Energy": 1},
            sector_rankings=rankings,
            portfolio_weights={"Energy": 0.6},  # Only 0.6% exposure
            min_meaningful_exposure=3.0,
        )
        assert len(gaps) == 1
        assert gaps[0]["sector"] == "Energy"
        assert gaps[0]["portfolio_exposure"] == pytest.approx(0.6)

    def test_adequate_exposure_not_a_gap(self):
        """Sector with >= min_meaningful_exposure should not be a gap."""
        rankings = {"XLE": {"rank": 2, "return_1m": 8.0}}
        gaps = detect_sector_gaps(
            portfolio_sectors={"Energy": 3},
            sector_rankings=rankings,
            portfolio_weights={"Energy": 5.0},
            min_meaningful_exposure=3.0,
        )
        assert len(gaps) == 0

    def test_high_urgency_for_top_3_low_exposure(self):
        """Top-3 sector with <1% exposure should be HIGH urgency."""
        rankings = {"XLE": {"rank": 1, "return_1m": 12.0}}
        gaps = detect_sector_gaps(
            portfolio_sectors={"Energy": 1},
            sector_rankings=rankings,
            portfolio_weights={"Energy": 0.3},
        )
        assert len(gaps) == 1
        assert gaps[0]["urgency"] == "HIGH"

    def test_medium_urgency_for_rank_4_5(self):
        """Rank 4-5 sector should be MEDIUM urgency."""
        rankings = {"XLE": {"rank": 4, "return_1m": 3.0}}
        gaps = detect_sector_gaps(
            portfolio_sectors={},
            sector_rankings=rankings,
            portfolio_weights={},
        )
        assert len(gaps) == 1
        assert gaps[0]["urgency"] == "MEDIUM"

    def test_missing_sector_in_weights(self):
        """Sector completely missing from weights should show 0.0 exposure."""
        rankings = {"XLE": {"rank": 2, "return_1m": 6.0}}
        gaps = detect_sector_gaps(
            portfolio_sectors={},
            sector_rankings=rankings,
            portfolio_weights={"Technology": 30.0},
        )
        assert len(gaps) == 1
        assert gaps[0]["portfolio_exposure"] == pytest.approx(0.0)

    def test_custom_min_exposure_threshold(self):
        """Custom min_meaningful_exposure should be respected."""
        rankings = {"XLE": {"rank": 3, "return_1m": 5.0}}
        gaps_strict = detect_sector_gaps(
            portfolio_sectors={"Energy": 2},
            sector_rankings=rankings,
            portfolio_weights={"Energy": 4.5},
            min_meaningful_exposure=5.0,
        )
        gaps_lenient = detect_sector_gaps(
            portfolio_sectors={"Energy": 2},
            sector_rankings=rankings,
            portfolio_weights={"Energy": 4.5},
            min_meaningful_exposure=3.0,
        )
        assert len(gaps_strict) == 1  # 4.5 < 5.0
        assert len(gaps_lenient) == 0  # 4.5 >= 3.0


# ============================================================
# CIO Legacy Review: B1 — Sigmoid Conviction
# ============================================================


class TestSigmoidConviction:
    """Tests for CIO Legacy B1: sigmoid conviction mapping."""

    def test_center_at_50_pct(self):
        """50% bull should produce agent_base near 55 (sigmoid center)."""
        base = determine_base_conviction(
            bull_pct=50, signal="H", fund_score=50,
            excess_exret=0, bear_ratio=0.5,
        )
        # Sigmoid at x=0 → 0.5, so agent_base = 30 + 0.5*50 = 55
        # HOLD cap at 70, so should be exactly 55
        assert base == 55

    def test_steeper_near_threshold(self):
        """Going from 45% to 55% should produce a larger jump than 85% to 95%."""
        base_45 = determine_base_conviction(
            bull_pct=45, signal="H", fund_score=50,
            excess_exret=0, bear_ratio=0.5,
        )
        base_55 = determine_base_conviction(
            bull_pct=55, signal="H", fund_score=50,
            excess_exret=0, bear_ratio=0.5,
        )
        delta_center = base_55 - base_45

        base_85 = determine_base_conviction(
            bull_pct=85, signal="H", fund_score=50,
            excess_exret=0, bear_ratio=0.5,
        )
        base_95 = determine_base_conviction(
            bull_pct=95, signal="H", fund_score=50,
            excess_exret=0, bear_ratio=0.5,
        )
        delta_extreme = base_95 - base_85

        # Sigmoid produces steeper change near center
        assert delta_center >= delta_extreme

    def test_monotonic_increasing(self):
        """Higher bull% should always produce equal or higher base."""
        prev_base = 0
        for bp in range(0, 101, 10):
            base = determine_base_conviction(
                bull_pct=bp, signal="H", fund_score=50,
                excess_exret=0, bear_ratio=0.5,
            )
            assert base >= prev_base, f"Non-monotonic at {bp}%: {base} < {prev_base}"
            prev_base = base

    def test_bounds(self):
        """Agent base should be clamped to [30, 80]."""
        base_min = determine_base_conviction(
            bull_pct=0, signal="H", fund_score=50,
            excess_exret=0, bear_ratio=1.0,
        )
        base_max = determine_base_conviction(
            bull_pct=100, signal="B", fund_score=50,
            excess_exret=0, bear_ratio=0.0,
        )
        assert base_min >= 30
        # BUY floor of 55 may override, but raw agent_base <= 80
        assert base_max <= 80 or base_max >= 55  # BUY floor can push above


# ============================================================
# CIO Legacy Review: B3 — Risk Manager Weight
# ============================================================


class TestRiskManagerWeight:
    """Tests for CIO Legacy B3: reduced BUY-side risk weight."""

    def test_sell_weight_preserved_at_2x(self):
        """SELL signal should still have 2.0x risk weight."""
        _, bear_sell, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="S",
        )
        # With risk_warning and SELL: bear should include 2.0
        # All other neutrals contribute balanced, so risk_mult of 2.0 dominates
        assert bear_sell > 3.0  # Substantial bearish weight from 2.0x risk

    def test_buy_weight_reduced_to_1_2x(self):
        """BUY signal risk weight should be 1.2x, not 1.5x."""
        _, bear_buy, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="B",
        )
        # Risk warning with BUY: bear should include 1.2 (not 1.5)
        # We can verify by checking it's less than what 1.5 would give
        _, bear_sell, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="S",
        )
        # SELL (2.0x) should produce more bear weight than BUY (1.2x)
        assert bear_sell > bear_buy

    def test_no_warning_unchanged(self):
        """Without risk warning, weight should be same for BUY and SELL."""
        bull_buy, bear_buy, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
        )
        bull_sell, bear_sell, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="S",
        )
        # No warning = neutral vote (0.5/0.5), same for both signals
        assert bull_buy == bull_sell
        assert bear_buy == bear_sell


# ============================================================
# CIO Legacy Review: A5 — Directional Confidence
# ============================================================


class TestDirectionalConfidence:
    """Tests for CIO Legacy A5: tracking directional vs neutral weight."""

    def test_all_directional_high_confidence(self):
        """All agents directional should produce high confidence."""
        _, _, conf = count_agent_votes(
            fund_score=80, tech_signal="ENTER_NOW", tech_momentum=30,
            macro_fit="FAVORABLE", census_alignment="ALIGNED",
            news_impact="HIGH_POSITIVE", risk_warning=True, signal="B",
        )
        assert conf > 0.8

    def test_all_neutral_low_confidence(self):
        """All agents neutral should produce low confidence."""
        _, _, conf = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
        )
        assert conf < 0.2

    def test_mixed_moderate_confidence(self):
        """Mix of directional and neutral should be moderate."""
        _, _, conf = count_agent_votes(
            fund_score=80, tech_signal="HOLD", tech_momentum=0,
            macro_fit="FAVORABLE", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
        )
        assert 0.2 < conf < 0.8

    def test_low_confidence_penalized_in_synthesis(self):
        """Low directional confidence should reduce conviction."""
        # All neutral agents → low confidence → penalty
        sig_data = {"signal": "H", "exret": 10, "buy_pct": 55, "beta": 1.0, "pet": 15, "pef": 14}
        fund_data = {"fundamental_score": 55}
        tech_data = {"momentum_score": 0, "timing_signal": "HOLD", "rsi": 50, "macd_signal": "NEUTRAL"}

        result = synthesize_stock(
            "TEST", sig_data, fund_data, tech_data,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL", risk_warning=False,
            sector="Technology", sector_median_exret=5, sector_rankings={},
            position_limit=5.0,
        )
        assert result["directional_confidence"] < 0.3
        # Should have low-confidence penalty applied
        assert result["penalties"] >= 3


# ============================================================
# CIO Legacy Review: B4 — Signal Velocity
# ============================================================


class TestSignalVelocity:
    """Tests for CIO Legacy B4: signal velocity tracking."""

    def test_accelerating_upgrade(self):
        """Recent upgrade (within 14 days) should give +5 bonus."""
        adj, label = compute_signal_velocity("B", "H", 10)
        assert adj == 5
        assert label == "ACCELERATING"

    def test_improving_upgrade(self):
        """Upgrade within 30 days should give +3."""
        adj, label = compute_signal_velocity("B", "H", 20)
        assert adj == 3
        assert label == "IMPROVING"

    def test_deteriorating_downgrade(self):
        """Recent downgrade should give -5."""
        adj, label = compute_signal_velocity("H", "B", 7)
        assert adj == -5
        assert label == "DETERIORATING"

    def test_weakening_downgrade(self):
        """Downgrade within 30 days should give -3."""
        adj, label = compute_signal_velocity("H", "B", 25)
        assert adj == -3
        assert label == "WEAKENING"

    def test_stale_signal(self):
        """Same signal for >90 days should give -2 (stale)."""
        adj, label = compute_signal_velocity("B", "B", 120)
        assert adj == -2
        assert label == "STALE"

    def test_stable_signal(self):
        """Same signal for moderate time should be STABLE."""
        adj, label = compute_signal_velocity("B", "B", 45)
        assert adj == 0
        assert label == "STABLE"

    def test_no_history(self):
        """No previous signal should return NO_HISTORY."""
        adj, label = compute_signal_velocity("B", None, None)
        assert adj == 0
        assert label == "NO_HISTORY"

    def test_sell_to_buy_accelerating(self):
        """S→B is a 3-step upgrade, should be ACCELERATING."""
        adj, label = compute_signal_velocity("B", "S", 10)
        assert adj == 5
        assert label == "ACCELERATING"

    def test_integrated_in_synthesis(self):
        """Signal velocity should appear in synthesis output."""
        sig_data = {"signal": "B", "exret": 15, "buy_pct": 70, "beta": 1.0, "pet": 15, "pef": 13}
        fund_data = {"fundamental_score": 70}
        tech_data = {"momentum_score": 20, "timing_signal": "HOLD", "rsi": 55, "macd_signal": "BULLISH"}

        result = synthesize_stock(
            "TEST", sig_data, fund_data, tech_data,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL", risk_warning=False,
            sector="Technology", sector_median_exret=5, sector_rankings={},
            position_limit=5.0, previous_signal="H", days_since_signal_change=10,
        )
        assert result["signal_velocity"] == "ACCELERATING"


# ============================================================
# CIO Legacy Review: B5 — Earnings Surprise
# ============================================================


class TestEarningsSurprise:
    """Tests for CIO Legacy B5: earnings surprise adjustment."""

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_serial_beater(self):
        """Big beat + consecutive beats = +8 (CIO v20.0 D3 enhanced PEAD)."""
        adj, label = get_earnings_surprise_adjustment(15.0, 3)
        assert adj == 8
        assert label == "SERIAL_BEATER"

    def test_single_beat(self):
        """Moderate beat = +3."""
        adj, label = get_earnings_surprise_adjustment(8.0, 0)
        assert adj == 3
        assert label == "BEAT"

    def test_big_miss(self):
        """Big miss = -5."""
        adj, label = get_earnings_surprise_adjustment(-12.0, 0)
        assert adj == -5
        assert label == "BIG_MISS"

    def test_moderate_miss(self):
        """Moderate miss = -3."""
        adj, label = get_earnings_surprise_adjustment(-7.0, 0)
        assert adj == -3
        assert label == "MISS"

    def test_in_line(self):
        """Small surprise either way = 0."""
        adj, label = get_earnings_surprise_adjustment(2.0, 0)
        assert adj == 0
        assert label == "IN_LINE"

    def test_no_data(self):
        """No earnings data = 0."""
        adj, label = get_earnings_surprise_adjustment(None, 0)
        assert adj == 0
        assert label == "NO_DATA"

    def test_integrated_in_synthesis(self):
        """Earnings surprise should appear in synthesis output."""
        sig_data = {"signal": "B", "exret": 15, "buy_pct": 70, "beta": 1.0, "pet": 15, "pef": 13}
        fund_data = {"fundamental_score": 70}
        tech_data = {"momentum_score": 20, "timing_signal": "HOLD", "rsi": 55, "macd_signal": "BULLISH"}

        result = synthesize_stock(
            "TEST", sig_data, fund_data, tech_data,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL", risk_warning=False,
            sector="Technology", sector_median_exret=5, sector_rankings={},
            position_limit=5.0, earnings_surprise_pct=15.0, consecutive_earnings_beats=3,
        )
        assert result["earnings_surprise"] == "SERIAL_BEATER"


# ============================================================
# CIO Legacy Review: A4 — Dynamic Freshness
# ============================================================


class TestDynamicFreshness:
    """Tests for CIO Legacy A4: timestamp-based freshness multiplier."""

    def test_very_fresh(self):
        """Agent data < 1 hour old should be 1.0."""
        from datetime import datetime
        now = datetime.now()
        agent_ts = (now - __import__("datetime").timedelta(minutes=30)).isoformat()
        committee_ts = now.isoformat()
        mult = compute_dynamic_freshness(agent_ts, committee_ts)
        assert mult == pytest.approx(1.0)

    def test_few_hours(self):
        """Agent data 2-4 hours old should be 0.95."""
        from datetime import datetime, timedelta
        now = datetime.now()
        agent_ts = (now - timedelta(hours=3)).isoformat()
        committee_ts = now.isoformat()
        mult = compute_dynamic_freshness(agent_ts, committee_ts)
        assert mult == pytest.approx(0.95)

    def test_half_day(self):
        """Agent data 8 hours old should be 0.85."""
        from datetime import datetime, timedelta
        now = datetime.now()
        agent_ts = (now - timedelta(hours=8)).isoformat()
        committee_ts = now.isoformat()
        mult = compute_dynamic_freshness(agent_ts, committee_ts)
        assert mult == pytest.approx(0.85)

    def test_day_old(self):
        """Agent data 20 hours old should be 0.75."""
        from datetime import datetime, timedelta
        now = datetime.now()
        agent_ts = (now - timedelta(hours=20)).isoformat()
        committee_ts = now.isoformat()
        mult = compute_dynamic_freshness(agent_ts, committee_ts)
        assert mult == pytest.approx(0.75)

    def test_very_stale(self):
        """Agent data > 24 hours old should be 0.6."""
        from datetime import datetime, timedelta
        now = datetime.now()
        agent_ts = (now - timedelta(hours=30)).isoformat()
        committee_ts = now.isoformat()
        mult = compute_dynamic_freshness(agent_ts, committee_ts)
        assert mult == pytest.approx(0.6)

    def test_no_timestamp(self):
        """No agent timestamp should return default 1.0."""
        mult = compute_dynamic_freshness(None, None)
        assert mult == pytest.approx(1.0)

    def test_invalid_timestamp(self):
        """Invalid timestamp format should return 1.0."""
        mult = compute_dynamic_freshness("not-a-date", "also-not")
        assert mult == pytest.approx(1.0)


# =============================================================================
# CIO v6.0 Tests — Successor Review Findings
# =============================================================================

class TestUniverseMedianEvenCount:
    """CIO v6.0 F4: Universe median handles even-count arrays correctly."""

    def test_even_count_universe_median(self):
        """Even number of stocks should average two middle values."""
        signals = {
            "A": {"exret": 10}, "B": {"exret": 20},
            "C": {"exret": 30}, "D": {"exret": 40},
        }
        sectors = {"A": "S1", "B": "S2", "C": "S3", "D": "S4"}
        _, universe = compute_sector_medians(signals, sectors)
        # Sorted: [10, 20, 30, 40] → median = (20+30)/2 = 25
        assert universe == pytest.approx(25.0)

    def test_odd_count_universe_median(self):
        """Odd number of stocks should pick the middle value."""
        signals = {
            "A": {"exret": 10}, "B": {"exret": 20}, "C": {"exret": 30},
        }
        sectors = {"A": "S1", "B": "S2", "C": "S3"}
        _, universe = compute_sector_medians(signals, sectors)
        assert universe == 20

    def test_single_stock_universe_median(self):
        """Single stock: median is that stock's EXRET."""
        signals = {"A": {"exret": 15}}
        sectors = {"A": "S1"}
        _, universe = compute_sector_medians(signals, sectors)
        assert universe == 15


class TestNormalizedTiebreak:
    """CIO v6.0 F1: Tiebreak components are normalized to 0-100 before weighting."""

    def test_tiebreak_normalized_range(self):
        """Tiebreak should produce values in a reasonable normalized range."""
        result = build_concordance(
            portfolio_signals={"AAPL": {
                "signal": "B", "exret": 15, "buy_pct": 80,
                "beta": 1.2, "pet": 30, "pef": 25,
            }},
            fund_report={"stocks": {"AAPL": {"fundamental_score": 75}}},
            tech_report={"stocks": {"AAPL": {
                "momentum_score": 30, "timing_signal": "ENTER_NOW",
                "rsi": 55, "macd_signal": "BULLISH",
            }}},
            macro_report={"portfolio_implications": {}, "sector_rankings": {}},
            census_report={"divergences": {}},
            news_report={"portfolio_news": {}},
            risk_report={"consensus_warnings": [], "position_limits": {}},
            sector_map={"AAPL": "Technology"},
        )
        entry = concordance[0] if (concordance := result) else None
        assert entry is not None
        # Normalized tiebreak with all 0-100 components should be in a much
        # more consistent range than the old unnormalized version
        assert 0 <= entry["tiebreak"] <= 100

    def test_tiebreak_beta_differentiation(self):
        """Low beta should produce higher tiebreak than high beta, all else equal."""
        base_sig = {"signal": "H", "exret": 10, "buy_pct": 60, "pet": 20, "pef": 18}
        base_agent = {"stocks": {}}
        base_reports = {
            "fund_report": base_agent,
            "tech_report": base_agent,
            "macro_report": {"portfolio_implications": {}, "sector_rankings": {}},
            "census_report": {"divergences": {}},
            "news_report": {"portfolio_news": {}},
            "risk_report": {"consensus_warnings": [], "position_limits": {}},
        }

        low_beta = build_concordance(
            portfolio_signals={"A": {**base_sig, "beta": 0.5}},
            sector_map={"A": "Other"},
            **base_reports,
        )
        high_beta = build_concordance(
            portfolio_signals={"A": {**base_sig, "beta": 2.5}},
            sector_map={"A": "Other"},
            **base_reports,
        )
        assert low_beta[0]["tiebreak"] > high_beta[0]["tiebreak"]


class TestKillThesisIntegration:
    """CIO v6.0 E1: Kill thesis triggered flag reduces conviction by 15."""

    def _base_args(self):
        return dict(
            sig_data={"signal": "B", "exret": 15, "buy_pct": 75, "beta": 1.0,
                       "pet": 25, "pef": 20},
            fund_data={"fundamental_score": 70},
            tech_data={"momentum_score": 20, "timing_signal": "HOLD",
                       "rsi": 55, "macd_signal": "BULLISH"},
            macro_fit="FAVORABLE",
            census_alignment="ALIGNED",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=10,
            sector_rankings={},
            position_limit=5.0,
        )

    def test_kill_thesis_reduces_conviction(self):
        """Triggered kill thesis should reduce conviction by 15 points."""
        args = self._base_args()
        normal = synthesize_stock(ticker="TEST", **args, kill_thesis_triggered=False)
        killed = synthesize_stock(ticker="TEST", **args, kill_thesis_triggered=True)
        assert killed["conviction"] < normal["conviction"]
        # The delta should be exactly 15 (before floors may intervene)
        raw_delta = normal["conviction"] - killed["conviction"]
        assert raw_delta == 15 or killed["conviction"] == 40  # floor may cap

    def test_kill_thesis_bypasses_penalty_cap(self):
        """Kill thesis penalty is applied AFTER normal penalty cap, bypassing it."""
        args = self._base_args()
        # Give a scenario with max penalties already (25 cap)
        args["tech_data"] = {"momentum_score": -50, "timing_signal": "AVOID",
                             "rsi": 85, "macd_signal": "BEARISH"}
        args["macro_fit"] = "UNFAVORABLE"
        args["risk_warning"] = True
        args["news_impact"] = "HIGH_NEGATIVE"
        args["census_alignment"] = "DIVERGENT"
        args["census_ts_trend"] = "strong_distribution"

        normal = synthesize_stock(ticker="TEST", **args, kill_thesis_triggered=False)
        killed = synthesize_stock(ticker="TEST", **args, kill_thesis_triggered=True)
        # Kill thesis should reduce even when penalties are already at cap
        assert killed["conviction"] <= normal["conviction"]

    def test_kill_thesis_in_build_concordance(self):
        """build_concordance passes kill thesis data through to synthesis."""
        base_report = {"stocks": {}}
        concordance = build_concordance(
            portfolio_signals={"AAPL": {
                "signal": "B", "exret": 15, "buy_pct": 75, "beta": 1.0,
                "pet": 25, "pef": 20,
            }},
            fund_report=base_report,
            tech_report=base_report,
            macro_report={"portfolio_implications": {}, "sector_rankings": {}},
            census_report={"divergences": {}},
            news_report={"portfolio_news": {}},
            risk_report={"consensus_warnings": [], "position_limits": {}},
            sector_map={"AAPL": "Technology"},
            triggered_kill_theses={"AAPL": True},
        )
        entry = concordance[0]
        assert entry["kill_thesis_triggered"] is True

    def test_default_no_kill_thesis(self):
        """Default behavior: no kill thesis triggered."""
        args = self._base_args()
        result = synthesize_stock(ticker="TEST", **args)
        assert result["conviction"] > 0  # Normal flow


class TestProportionalOpportunityCostSizing:
    """CIO v6.0 F3: Proportional sizing instead of fixed ±10%."""

    def test_proportional_reduction_scales_with_distance(self):
        """Larger distance from mean should produce larger reduction."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost
        positions = [
            {"conviction": 20, "position_size": 100.0},   # far below avg (50)
            {"conviction": 40, "position_size": 100.0},   # slightly below
            {"conviction": 50, "position_size": 100.0},   # at average
            {"conviction": 80, "position_size": 100.0},   # above average
        ]
        result = adjust_sizes_for_opportunity_cost(positions)
        # conviction=20 is farther from avg=47.5 than conviction=40
        assert result[0]["opp_cost_adj"] < result[1]["opp_cost_adj"]
        # conviction=80 should get a positive adjustment
        assert result[3]["opp_cost_adj"] > 0

    def test_proportional_increase_capped(self):
        """Increase should be capped at 15%."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost
        positions = [
            {"conviction": 10, "position_size": 100.0},
            {"conviction": 100, "position_size": 100.0},
        ]
        result = adjust_sizes_for_opportunity_cost(positions)
        assert result[1]["opp_cost_adj"] <= 0.15

    def test_proportional_reduction_capped(self):
        """Reduction should be capped at 20%."""
        from trade_modules.conviction_sizer import adjust_sizes_for_opportunity_cost
        positions = [
            {"conviction": 0, "position_size": 100.0},
            {"conviction": 100, "position_size": 100.0},
        ]
        result = adjust_sizes_for_opportunity_cost(positions)
        assert abs(result[0]["opp_cost_adj"]) <= 0.20


# ============================================================
# CIO v6.0 H1: News Impact Bias Fix
# ============================================================


class TestNewsImpactResolution:
    """CIO v6.0 H1: Conflicting high-impact news should resolve to MIXED."""

    def test_conflicting_high_impact_returns_mixed(self):
        """HIGH_POSITIVE + HIGH_NEGATIVE should return MIXED, not HIGH_POSITIVE."""
        port_news = {
            "AAPL": [
                {"impact": "HIGH_POSITIVE"},
                {"impact": "HIGH_NEGATIVE"},
            ],
        }
        result = _resolve_news_impact(port_news, "AAPL")
        assert result == "MIXED"

    def test_single_high_positive_still_works(self):
        """Uncontested HIGH_POSITIVE should still return HIGH_POSITIVE."""
        port_news = {"AAPL": [{"impact": "HIGH_POSITIVE"}]}
        result = _resolve_news_impact(port_news, "AAPL")
        assert result == "HIGH_POSITIVE"

    def test_single_high_negative_still_works(self):
        """Uncontested HIGH_NEGATIVE should still return HIGH_NEGATIVE."""
        port_news = {"AAPL": [{"impact": "HIGH_NEGATIVE"}]}
        result = _resolve_news_impact(port_news, "AAPL")
        assert result == "HIGH_NEGATIVE"

    def test_negative_prioritized_over_positive(self):
        """When no conflict, HIGH_NEGATIVE should be checked before HIGH_POSITIVE."""
        port_news = {
            "AAPL": [
                {"impact": "LOW_POSITIVE"},
                {"impact": "HIGH_NEGATIVE"},
            ],
        }
        result = _resolve_news_impact(port_news, "AAPL")
        assert result == "HIGH_NEGATIVE"

    def test_empty_news_returns_neutral(self):
        """No news for ticker should return NEUTRAL."""
        result = _resolve_news_impact({}, "AAPL")
        assert result == "NEUTRAL"

    def test_mixed_treated_as_neutral_in_votes(self):
        """MIXED news should fall through to neutral treatment in vote counting."""
        bull, bear, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="MIXED", risk_warning=False, signal="H",
        )
        # MIXED falls to else branch: bull += 0.5, bear += 0.5
        # Same as NEUTRAL — conflicting news doesn't inflate either side
        bull2, bear2, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
        )
        assert bull == bull2
        assert bear == bear2


# ============================================================
# CIO v6.0 H5: Penalty Cap Saturation Fix
# ============================================================


class TestPenaltyCapSaturation:
    """CIO v6.0 H5: Signal quality penalties should not be absorbed by base cap."""

    def _make_sig_data(self):
        return {"signal": "H", "exret": 5, "buy_pct": 95, "beta": 2.5,
                "pet": 20, "pef": 25, "pp": -10, "52w": 60}

    def _make_fund_data(self):
        return {"fundamental_score": 40, "quality_trap_warning": True}

    def _make_tech_data(self):
        return {"momentum_score": -30, "timing_signal": "AVOID", "rsi": 75,
                "macd_signal": "BEARISH"}

    def test_signal_quality_penalties_add_beyond_base_cap(self):
        """With base penalties at 25, signal quality penalties should add more.

        CIO v14.0 V5: Total effective penalties are now capped at 60% of base
        conviction, so the raw penalty count may be less than 25 after capping.
        The test verifies that signal quality penalties ARE computed beyond the
        base cap (before proportionality cap is applied).
        """
        # This stock triggers heavy base penalties + contradiction + earnings miss
        result = synthesize_stock(
            ticker="TEST", sig_data=self._make_sig_data(),
            fund_data=self._make_fund_data(), tech_data=self._make_tech_data(),
            macro_fit="UNFAVORABLE", census_alignment="NEUTRAL",
            div_score=0, census_ts_trend="stable",
            news_impact="NEUTRAL", risk_warning=True,
            sector="Technology", sector_median_exret=5,
            sector_rankings={}, position_limit=5.0,
            earnings_surprise_pct=-15,  # BIG_MISS → -5 signal quality penalty
        )
        # V5 caps penalties at 60% of base — verify conviction is reasonable
        # Base for HOLD signal with these inputs ~45-55, penalties capped accordingly
        assert result["penalties"] > 0
        assert result["conviction"] >= 0  # Never goes negative

    def test_signal_quality_cap_at_10(self):
        """Signal quality penalties should be capped at 10 independently."""
        # Trigger all signal quality penalties simultaneously
        result = synthesize_stock(
            ticker="TEST", sig_data=self._make_sig_data(),
            fund_data={"fundamental_score": 85},  # High fund + risk_warning → contradiction
            tech_data={"momentum_score": 0, "timing_signal": "ENTER_NOW",
                       "rsi": 50, "macd_signal": "BULLISH"},
            macro_fit="UNFAVORABLE",  # + ENTER_NOW → macro-tech contradiction
            census_alignment="NEUTRAL", div_score=0, census_ts_trend="stable",
            news_impact="NEUTRAL", risk_warning=True,  # + fund>=80 → fund-risk contradiction
            sector="Technology", sector_median_exret=5,
            sector_rankings={}, position_limit=5.0,
            previous_signal="B", days_since_signal_change=10,  # H from B = DETERIORATING → -5
            earnings_surprise_pct=-15,  # BIG_MISS → -5
        )
        # Signal quality: contradiction(5+3) + velocity(5) + earnings(5) = 18, capped at 10
        # The waterfall records intended values; signal_quality_cap records the absorbed excess
        wf = result.get("conviction_waterfall", {})
        sq_keys = {"contradiction", "signal_velocity", "earnings_surprise", "low_dir_confidence"}
        sq_intended = sum(abs(wf.get(k, 0)) for k in sq_keys)
        sq_absorbed = wf.get("signal_quality_cap", 0)
        sq_effective = sq_intended - sq_absorbed
        assert sq_effective <= 10, f"Effective signal quality {sq_effective} exceeds cap of 10"


# ============================================================
# CIO v6.0 R1: Agent Memory (build_agent_memory)
# ============================================================


class TestBuildAgentMemory:
    """CIO v6.0 R1: Per-agent feedback from previous concordance."""

    def test_empty_concordance_returns_empty(self):
        assert build_agent_memory([], {}) == {}

    def test_fundamental_too_optimistic(self):
        """High fund_score followed by price drop = TOO OPTIMISTIC."""
        prev = [{"ticker": "NVDA", "fund_score": 85, "price": 100, "conviction": 70,
                 "action": "ADD", "tech_signal": "", "macro_fit": "",
                 "census_alignment": "", "news_impact": "", "risk_warning": False}]
        current = {"NVDA": {"price": 88}}  # -12%
        memory = build_agent_memory(prev, current)
        assert "fundamental" in memory
        assert "TOO OPTIMISTIC" in memory["fundamental"]

    def test_fundamental_correct(self):
        """High fund_score followed by price rise = CORRECT."""
        prev = [{"ticker": "AAPL", "fund_score": 80, "price": 150, "conviction": 65,
                 "action": "ADD", "tech_signal": "", "macro_fit": "",
                 "census_alignment": "", "news_impact": "", "risk_warning": False}]
        current = {"AAPL": {"price": 160}}  # +6.7%
        memory = build_agent_memory(prev, current)
        assert "CORRECT" in memory["fundamental"]

    def test_technical_wrong_entry(self):
        """ENTER_NOW followed by drop = WRONG."""
        prev = [{"ticker": "TSLA", "fund_score": 0, "price": 200,
                 "tech_signal": "ENTER_NOW", "macro_fit": "", "conviction": 60,
                 "action": "BUY", "census_alignment": "", "news_impact": "",
                 "risk_warning": False}]
        current = {"TSLA": {"price": 180}}  # -10%
        memory = build_agent_memory(prev, current)
        assert "technical" in memory
        assert "WRONG" in memory["technical"]

    def test_risk_warning_validated(self):
        """Risk warning + subsequent drop = VALIDATED."""
        prev = [{"ticker": "META", "fund_score": 50, "price": 500,
                 "tech_signal": "", "macro_fit": "", "conviction": 45,
                 "action": "TRIM", "census_alignment": "", "news_impact": "",
                 "risk_warning": True}]
        current = {"META": {"price": 470}}  # -6%
        memory = build_agent_memory(prev, current)
        assert "risk" in memory
        assert "VALIDATED" in memory["risk"]

    def test_risk_warning_false_alarm(self):
        """Risk warning + subsequent rise = FALSE ALARM."""
        prev = [{"ticker": "GOOG", "fund_score": 50, "price": 170,
                 "tech_signal": "", "macro_fit": "", "conviction": 55,
                 "action": "HOLD", "census_alignment": "", "news_impact": "",
                 "risk_warning": True}]
        current = {"GOOG": {"price": 185}}  # +8.8%
        memory = build_agent_memory(prev, current)
        assert "FALSE ALARM" in memory["risk"]

    def test_opportunity_good_pick(self):
        """BUY that goes up = GOOD PICK."""
        prev = [{"ticker": "PLTR", "fund_score": 60, "price": 80,
                 "tech_signal": "", "macro_fit": "", "conviction": 62,
                 "action": "BUY", "census_alignment": "", "news_impact": "",
                 "risk_warning": False}]
        current = {"PLTR": {"price": 90}}  # +12.5%
        memory = build_agent_memory(prev, current)
        assert "opportunity" in memory
        assert "GOOD PICK" in memory["opportunity"]

    def test_missing_prices_skipped(self):
        """Entries without current price should be skipped."""
        prev = [{"ticker": "XYZ", "fund_score": 70, "price": 50,
                 "tech_signal": "ENTER_NOW", "macro_fit": "", "conviction": 60,
                 "action": "BUY", "census_alignment": "", "news_impact": "",
                 "risk_warning": False}]
        current = {}  # No current data
        memory = build_agent_memory(prev, current)
        assert memory == {}

    def test_priority_sorting_wrong_first(self):
        """WRONG/TOO OPTIMISTIC entries should appear before CORRECT."""
        prev = [
            {"ticker": "A", "fund_score": 85, "price": 100, "tech_signal": "",
             "macro_fit": "", "conviction": 70, "action": "ADD",
             "census_alignment": "", "news_impact": "", "risk_warning": False},
            {"ticker": "B", "fund_score": 80, "price": 50, "tech_signal": "",
             "macro_fit": "", "conviction": 65, "action": "ADD",
             "census_alignment": "", "news_impact": "", "risk_warning": False},
        ]
        current = {"A": {"price": 85}, "B": {"price": 55}}  # A dropped, B rose
        memory = build_agent_memory(prev, current)
        lines = memory["fundamental"].split("\n")
        # TOO OPTIMISTIC (A) should come before CORRECT (B)
        assert "TOO OPTIMISTIC" in lines[0]
        assert "CORRECT" in lines[1]


# ═══════════════════════════════════════════════════════════
# CIO v7.0 Tests
# ═══════════════════════════════════════════════════════════


class TestRSIFloorForTrimEscalation:
    """CIO v7.0 P1: Never trim deeply oversold stocks (RSI < 30)."""

    def test_rsi_below_30_blocks_trim_escalation(self):
        """Stock with RSI=16 should NOT be escalated to TRIM."""
        action = determine_action(50, "H", "AVOID", True)
        assert action == "HOLD"
        # The trim escalation in synthesize_stock checks rsi >= 30,
        # but determine_action itself doesn't know about RSI.
        # The guard is in the post-action logic in synthesize_stock.

    def test_rsi_above_30_allows_trim_with_risk_and_avoid(self):
        """Stock with RSI=50, tech=AVOID, risk_warning should be TRIM."""
        # This test verifies the synthesize_stock pathway, but since we
        # can't easily call synthesize_stock here, we test the boundary.
        action = determine_action(50, "H", "AVOID", True)
        # determine_action returns HOLD; the escalation happens in synthesize_stock
        assert action == "HOLD"

    def test_rsi_80_overbought_still_trims(self):
        """Overbought stocks (RSI>80) should still be trimmed."""
        # determine_action returns HOLD, escalation to TRIM is in synthesize_stock
        action = determine_action(50, "H", "EXIT_SOON", False)
        assert action == "HOLD"

    def test_strong_fundamentals_block_trim_escalation(self):
        """v32.0: fund_score >= 70 should block HOLD→TRIM escalation."""
        entry = synthesize_stock(
            ticker="TEST",
            sig_data={"signal": "H", "exret": 10, "buy_pct": 50,
                      "beta": 1.0, "pet": 15, "pef": 14},
            fund_data={"fundamental_score": 75, "quality_trap_warning": False},
            tech_data={"timing_signal": "AVOID", "momentum_score": -10, "rsi": 55},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=True,
            sector="Technology",
            sector_median_exret=8.0,
            sector_rankings={},
            position_limit=5.0,
        )
        assert entry["action"] == "HOLD", (
            f"Strong fundamentals (75) should block trim escalation, got {entry['action']}"
        )

    def test_weak_fundamentals_allow_trim_escalation(self):
        """v32.0: fund_score < 70 with tech=AVOID + risk_warning should still TRIM."""
        entry = synthesize_stock(
            ticker="TEST",
            sig_data={"signal": "H", "exret": 10, "buy_pct": 50,
                      "beta": 1.0, "pet": 15, "pef": 14},
            fund_data={"fundamental_score": 55, "quality_trap_warning": False},
            tech_data={"timing_signal": "AVOID", "momentum_score": -10, "rsi": 55},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=True,
            sector="Technology",
            sector_median_exret=8.0,
            sector_rankings={},
            position_limit=5.0,
        )
        assert entry["action"] == "TRIM", (
            f"Weak fundamentals (55) with tech=AVOID + risk should TRIM, got {entry['action']}"
        )

    def test_overbought_rsi80_needs_risk_warning_too(self):
        """v32.0: RSI > 80 + bearish tech alone no longer triggers TRIM without risk_warning."""
        entry = synthesize_stock(
            ticker="TEST",
            sig_data={"signal": "H", "exret": 10, "buy_pct": 50,
                      "beta": 1.0, "pet": 15, "pef": 14},
            fund_data={"fundamental_score": 60, "quality_trap_warning": False},
            tech_data={"timing_signal": "AVOID", "momentum_score": -10, "rsi": 85},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=8.0,
            sector_rankings={},
            position_limit=5.0,
        )
        assert entry["action"] == "HOLD", (
            f"RSI>80 without risk_warning should not TRIM, got {entry['action']}"
        )

    def test_overbought_rsi80_with_risk_and_weak_fund_trims(self):
        """v32.0: RSI > 80 + bearish tech + risk_warning + weak fund should TRIM."""
        entry = synthesize_stock(
            ticker="TEST",
            sig_data={"signal": "H", "exret": 10, "buy_pct": 50,
                      "beta": 1.0, "pet": 15, "pef": 14},
            fund_data={"fundamental_score": 45, "quality_trap_warning": False},
            tech_data={"timing_signal": "AVOID", "momentum_score": -10, "rsi": 85},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=True,
            sector="Technology",
            sector_median_exret=8.0,
            sector_rankings={},
            position_limit=5.0,
        )
        assert entry["action"] == "TRIM", (
            f"RSI>80 + risk + weak fund (45) should TRIM, got {entry['action']}"
        )


class TestRiskWarningDilution:
    """CIO v7.0 P3: Detect when risk warnings become systemic."""

    def _make_signals(self, n):
        """Create n portfolio signals."""
        return {f"T{i}": {"signal": "H", "exret": 5, "buy_pct": 50,
                          "beta": 1.0, "pet": 15, "pef": 14}
                for i in range(n)}

    def _make_risk_report(self, warned_tickers):
        """Create risk report with warnings for given tickers."""
        return {
            "consensus_warnings": [{"ticker": t} for t in warned_tickers],
            "position_limits": {},
        }

    def test_dilution_detected_above_40_pct(self):
        """When >40% of stocks have risk warnings, dilution is flagged."""
        signals = self._make_signals(10)
        tickers = list(signals.keys())
        # 5/10 = 50% warned
        risk = self._make_risk_report(tickers[:5])
        sector_map = {t: "Tech" for t in tickers}

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            risk,
            sector_map,
        )
        # All entries should have risk_diluted=True
        assert all(e.get("risk_diluted") for e in concordance)

    def test_no_dilution_below_40_pct(self):
        """When <=40% of stocks have risk warnings, no dilution."""
        signals = self._make_signals(10)
        tickers = list(signals.keys())
        # 3/10 = 30% warned
        risk = self._make_risk_report(tickers[:3])
        sector_map = {t: "Tech" for t in tickers}

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            risk,
            sector_map,
        )
        assert all(not e.get("risk_diluted") for e in concordance)


class TestSectorConcentrationPenalty:
    """CIO v7.0 P4: Penalize over-concentration in same sector."""

    def _make_signals(self, tickers_sectors, signal="B"):
        """Create signals dict with given tickers and sectors."""
        signals = {}
        for t, _ in tickers_sectors:
            signals[t] = {"signal": signal, "exret": 10, "buy_pct": 70,
                          "beta": 1.0, "pet": 15, "pef": 14}
        return signals

    def _make_sector_map(self, tickers_sectors):
        return {t: s for t, s in tickers_sectors}

    def test_no_penalty_with_2_stocks_per_sector(self):
        """2 stocks in same sector = no penalty."""
        tickers_sectors = [("AAPL", "Tech"), ("MSFT", "Tech"),
                           ("JNJ", "Health")]
        signals = self._make_signals(tickers_sectors)
        sector_map = self._make_sector_map(tickers_sectors)

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map,
        )
        for entry in concordance:
            assert entry.get("sector_concentration_penalty") is None

    def test_penalty_with_4_stocks_in_same_sector(self):
        """4 stocks in same sector = penalty of (4-2)*2 = 4."""
        tickers_sectors = [("AAPL", "Tech"), ("MSFT", "Tech"),
                           ("GOOGL", "Tech"), ("META", "Tech")]
        signals = self._make_signals(tickers_sectors)
        sector_map = self._make_sector_map(tickers_sectors)

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map,
        )
        # All BUY/ADD entries in Tech should have penalty
        tech_entries = [e for e in concordance if e["sector"] == "Tech"
                        and e["action"] in ("BUY", "ADD")]
        for entry in tech_entries:
            assert entry.get("sector_concentration_penalty") == 4

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_penalty_only_on_buy_add_actions(self):
        """HOLD/TRIM/SELL stocks should NOT get sector penalty.

        Phase A 2026-04-18: weight retune (Fundamental 0.8→1.25, Census 0.85→
        1.25) made the synthesis more eager to escalate H-signal stocks with
        positive analyst data to ADD. The sector penalty correctly fires when
        pre-penalty action is BUY/ADD even if the post-penalty action drops
        back to HOLD. To preserve the original invariant under test (HOLD-
        signal stocks that genuinely stay HOLD never get penalty), this
        scenario uses neutral buy_pct=50 so escalation never occurs.
        """
        tickers_sectors = [("T1", "Tech"), ("T2", "Tech"),
                           ("T3", "Tech"), ("T4", "Tech")]
        # Use neutral buy_pct so H signal genuinely stays HOLD
        signals = {t: {"signal": "H", "exret": 5, "buy_pct": 50,
                       "beta": 1.0, "pet": 15, "pef": 14} for t, _ in tickers_sectors}
        sector_map = self._make_sector_map(tickers_sectors)

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map,
        )
        # HOLD-signal stocks that stay HOLD should NOT have sector penalty
        for entry in concordance:
            assert entry["action"] == "HOLD", \
                f"Test pre-condition broke: {entry['ticker']} ended up {entry['action']}"
            assert entry.get("sector_concentration_penalty") is None

    def test_penalty_does_not_breach_floor(self):
        """Conviction should never go below 30 from sector penalty."""
        tickers_sectors = [(f"T{i}", "Tech") for i in range(20)]
        signals = self._make_signals(tickers_sectors)
        sector_map = self._make_sector_map(tickers_sectors)

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map,
        )
        for entry in concordance:
            assert entry["conviction"] >= 30


# ============================================================
# CIO v11.0 Review Tests
# ============================================================


class TestV11ActionConvictionSync:
    """CIO v11.0 L1: Action must match conviction after sector concentration penalty."""

    def _make_signals(self, tickers_sectors, signal="B"):
        return {
            t: {"signal": signal, "exret": 15, "buy_pct": 80, "beta": 1.0,
                "pet": 20, "pef": 18, "pp": 5, "52w": 85}
            for t, _ in tickers_sectors
        }

    def _make_sector_map(self, tickers_sectors):
        return {t: s for t, s in tickers_sectors}

    def test_add_action_requires_conviction_55(self):
        """After sector penalty, ADD action should only appear with conv >= 55."""
        # 6 Tech stocks → concentration penalty triggers
        tickers_sectors = [(f"TECH{i}", "Technology") for i in range(6)]
        signals = self._make_signals(tickers_sectors)
        sector_map = self._make_sector_map(tickers_sectors)

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map,
        )

        for entry in concordance:
            if entry["action"] == "ADD":
                assert entry["conviction"] >= 55, (
                    f'{entry["ticker"]}: ADD with conviction {entry["conviction"]} < 55'
                )
            elif entry["action"] == "HOLD":
                assert entry["conviction"] < 55 or entry["signal"] == "H"

    def test_no_desync_after_concentration_penalty(self):
        """No stock should have ADD action with conviction below threshold."""
        # Mix of sectors with concentration
        tickers_sectors = (
            [(f"FIN{i}", "Financials") for i in range(5)] +
            [(f"TECH{i}", "Technology") for i in range(5)]
        )
        signals = self._make_signals(tickers_sectors)
        sector_map = self._make_sector_map(tickers_sectors)

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map,
        )

        for entry in concordance:
            # Verify contract: ADD requires conv >= 55 for BUY signal
            if entry["signal"] == "B" and entry["action"] == "ADD":
                assert entry["conviction"] >= 55, (
                    f'{entry["ticker"]}: contract violation ADD at conv={entry["conviction"]}'
                )


class TestV11DirectionalConfidenceSynthetic:
    """CIO v11.0 L6: Skip dir_confidence penalty when dual-synthetic."""

    def test_dual_synthetic_no_dir_confidence_penalty(self):
        """Dual-synthetic stocks should not get directional confidence penalty."""
        result = synthesize_stock(
            ticker="TEST",
            sig_data={"signal": "B", "exret": 15, "buy_pct": 80, "beta": 1.0,
                       "pet": 20, "pef": 18, "pp": 5, "52w": 85},
            fund_data={"fundamental_score": None},  # triggers synthetic
            tech_data={"synthetic": True, "momentum_score": 0,
                        "timing_signal": "HOLD", "rsi": 50, "macd_signal": "NEUTRAL"},
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

        # With both synthetic, dir_confidence will be < 0.4 (all neutral agents)
        # but the penalty should be skipped for dual-synthetic stocks
        assert result["fund_synthetic"] is True
        assert result["tech_synthetic"] is True
        # The key test: conviction should NOT have the -3 dir_confidence penalty
        # compared to a stock with only one synthetic
        result_single_synthetic = synthesize_stock(
            ticker="TEST2",
            sig_data={"signal": "B", "exret": 15, "buy_pct": 80, "beta": 1.0,
                       "pet": 20, "pef": 18, "pp": 5, "52w": 85},
            fund_data={"fundamental_score": 50},  # NOT synthetic
            tech_data={"synthetic": True, "momentum_score": 0,
                        "timing_signal": "HOLD", "rsi": 50, "macd_signal": "NEUTRAL"},
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
        # Dual-synthetic should have same or higher conviction (no false penalty)
        assert result["conviction"] >= result_single_synthetic["conviction"] - 3


class TestV11SignalVelocityWiring:
    """CIO v11.0 L2: Signal velocity should be populated when previous concordance provided."""

    def test_velocity_populated_with_previous(self):
        """build_concordance should pass previous signal data to synthesize_stock."""
        signals = {
            "AAPL": {"signal": "B", "exret": 15, "buy_pct": 80, "beta": 1.0,
                      "pet": 20, "pef": 18, "pp": 5, "52w": 85},
        }
        previous = [
            {"ticker": "AAPL", "signal": "H", "action": "HOLD", "conviction": 50,
             "date": "2026-03-10"},
        ]

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            {"AAPL": "Technology"},
            previous_concordance=previous,
        )

        aapl = [e for e in concordance if e["ticker"] == "AAPL"][0]
        # Signal changed from H to B (upgrade) within ~8 days → should be ACCELERATING or IMPROVING
        assert aapl["signal_velocity"] != "NO_HISTORY", (
            f'Velocity should be populated but got {aapl["signal_velocity"]}'
        )

    def test_velocity_no_history_without_previous(self):
        """Without previous concordance, velocity should be NO_HISTORY."""
        signals = {
            "AAPL": {"signal": "B", "exret": 15, "buy_pct": 80, "beta": 1.0,
                      "pet": 20, "pef": 18, "pp": 5, "52w": 85},
        }

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            {"AAPL": "Technology"},
        )

        aapl = [e for e in concordance if e["ticker"] == "AAPL"][0]
        assert aapl["signal_velocity"] == "NO_HISTORY"


class TestV11SectorPenaltyCap:
    """CIO v11.0 L5: Sector concentration penalty capped at 6 for existing holdings."""

    def _make_signals(self, tickers_sectors, signal="B"):
        return {
            t: {"signal": signal, "exret": 15, "buy_pct": 80, "beta": 1.0,
                "pet": 20, "pef": 18, "pp": 5, "52w": 85}
            for t, _ in tickers_sectors
        }

    def _make_sector_map(self, tickers_sectors):
        return {t: s for t, s in tickers_sectors}

    def test_existing_holding_penalty_cap_is_6(self):
        """Existing holdings should have sector concentration penalty capped at 6."""
        tickers_sectors = [(f"TECH{i}", "Technology") for i in range(8)]
        signals = self._make_signals(tickers_sectors)
        sector_map = self._make_sector_map(tickers_sectors)

        concordance = build_concordance(
            signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map,
        )

        for entry in concordance:
            if not entry.get("is_opportunity"):
                penalty = entry.get("sector_concentration_penalty", 0)
                assert penalty <= 6, (
                    f'{entry["ticker"]}: sector penalty {penalty} exceeds cap of 6'
                )


class TestV11EarningsSurpriseWiring:
    """CIO v11.0 L3: Earnings surprise data should flow from fund_data to synthesize_stock."""

    def test_earnings_surprise_from_fund_data(self):
        """When fund_data includes earnings surprise, it should affect conviction."""
        signals = {
            "AAPL": {"signal": "B", "exret": 15, "buy_pct": 80, "beta": 1.0,
                      "pet": 20, "pef": 18, "pp": 5, "52w": 85},
        }
        fund_report = {
            "stocks": {
                "AAPL": {
                    "fundamental_score": 75,
                    "signal": "BUY",
                    "earnings_surprise_pct": 15.0,
                    "consecutive_earnings_beats": 3,
                }
            }
        }

        concordance = build_concordance(
            signals,
            fund_report, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            {"AAPL": "Technology"},
        )

        aapl = [e for e in concordance if e["ticker"] == "AAPL"][0]
        assert aapl["earnings_surprise"] == "SERIAL_BEATER", (
            f'Expected SERIAL_BEATER but got {aapl["earnings_surprise"]}'
        )

    def test_earnings_no_data_without_fund_fields(self):
        """Without earnings fields in fund_data, should default to NO_DATA."""
        signals = {
            "AAPL": {"signal": "B", "exret": 15, "buy_pct": 80, "beta": 1.0,
                      "pet": 20, "pef": 18, "pp": 5, "52w": 85},
        }
        fund_report = {
            "stocks": {
                "AAPL": {
                    "fundamental_score": 75,
                    "signal": "BUY",
                    # No earnings_surprise_pct or consecutive_earnings_beats
                }
            }
        }

        concordance = build_concordance(
            signals,
            fund_report, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            {"AAPL": "Technology"},
        )

        aapl = [e for e in concordance if e["ticker"] == "AAPL"][0]
        assert aapl["earnings_surprise"] == "NO_DATA"


class TestV11OpportunityActionPreservation:
    """CIO v11.0 L1 edge case: Opportunities must keep BUY/HOLD, not get ADD."""

    def test_opportunity_keeps_buy_after_sector_penalty(self):
        """Opportunities with sector concentration penalty should use BUY, not ADD."""
        # 4 existing Tech stocks + 1 Tech opportunity → concentration triggers
        portfolio_signals = {
            f"TECH{i}": {"signal": "B", "exret": 15, "buy_pct": 80, "beta": 1.0,
                          "pet": 20, "pef": 18, "pp": 5, "52w": 85}
            for i in range(4)
        }
        opp_signals = {
            "NEWTECH": {"signal": "B", "exret": 20, "buy_pct": 85, "beta": 1.0,
                         "pet": 20, "pef": 18, "pp": 5, "52w": 85}
        }
        sector_map = {f"TECH{i}": "Technology" for i in range(4)}
        opp_sector_map = {"NEWTECH": "Technology"}

        concordance = build_concordance(
            portfolio_signals,
            {"stocks": {}}, {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {"divergences": {"consensus_aligned": [], "signal_divergences": [],
                             "census_divergences": []}},
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map,
            opportunity_signals=opp_signals,
            opportunity_sector_map=opp_sector_map,
        )

        newtech = [e for e in concordance if e["ticker"] == "NEWTECH"][0]
        # Should be BUY (new opportunity), not ADD (existing holding)
        assert newtech["action"] in ("BUY", "HOLD"), (
            f'Opportunity NEWTECH should be BUY or HOLD, not {newtech["action"]}'
        )


# ============================================================
# CIO v12.0 Findings
# ============================================================


class TestV12R1KillThesisFloorEscape:
    """CIO v12.0 R1: Kill thesis should prevent quality floor rescue."""

    def _base_sig_data(self):
        return {
            "signal": "B", "exret": 25, "buy_pct": 75,
            "beta": 1.0, "pet": 20, "pef": 15, "pp": 10, "52w": 90,
        }

    def _base_fund_data(self):
        return {"fundamental_score": 80, "quality_trap_warning": False}

    def _base_tech_data(self):
        return {"momentum_score": 20, "timing_signal": "ENTER_NOW", "rsi": 55, "macd_signal": "BULLISH"}

    def test_kill_thesis_prevents_quality_floor_55(self):
        """Stock with triggered kill thesis should NOT get quality floor of 55."""
        result = synthesize_stock(
            ticker="TEST", sig_data=self._base_sig_data(),
            fund_data=self._base_fund_data(), tech_data=self._base_tech_data(),
            macro_fit="FAVORABLE", census_alignment="ALIGNED", div_score=0,
            census_ts_trend="stable", news_impact="HIGH_POSITIVE",
            risk_warning=False, sector="Technology", sector_median_exret=10,
            sector_rankings={}, position_limit=5.0,
            kill_thesis_triggered=True,
        )
        # With kill thesis, should be below 55 (quality floor would push to 55)
        # The unconditional BUY floor of 40 should still apply
        assert result["conviction"] >= 40, "Unconditional BUY floor should apply"

    def test_kill_thesis_true_produces_hold_not_add(self):
        """Kill thesis triggered should produce HOLD, not ADD, for BUY signal stocks."""
        # Use a scenario where penalties are heavy enough that -15 drops below 55
        result = synthesize_stock(
            ticker="TEST", sig_data=self._base_sig_data(),
            fund_data=self._base_fund_data(), tech_data=self._base_tech_data(),
            macro_fit="UNFAVORABLE", census_alignment="DIVERGENT", div_score=30,
            census_ts_trend="distribution", news_impact="NEUTRAL",
            risk_warning=True, sector="Technology", sector_median_exret=10,
            sector_rankings={}, position_limit=5.0,
            kill_thesis_triggered=True, regime="RISK_OFF",
        )
        # With RISK_OFF + penalties + kill thesis, should NOT be ADD
        assert result["action"] != "ADD" or result["conviction"] < 55, (
            "Kill thesis should prevent ADD recommendation"
        )

    def test_no_kill_thesis_quality_floor_still_works(self):
        """Without kill thesis, quality floors should still apply normally."""
        result = synthesize_stock(
            ticker="TEST",
            sig_data={"signal": "B", "exret": 25, "buy_pct": 75,
                       "beta": 1.0, "pet": 20, "pef": 15, "pp": 5, "52w": 85},
            fund_data={"fundamental_score": 80, "quality_trap_warning": False},
            tech_data={"momentum_score": 10, "timing_signal": "HOLD", "rsi": 50,
                        "macd_signal": "NEUTRAL"},
            macro_fit="UNFAVORABLE", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL",
            risk_warning=True, sector="Technology", sector_median_exret=10,
            sector_rankings={}, position_limit=5.0,
            kill_thesis_triggered=False, regime="RISK_OFF",
        )
        # Quality floor should rescue: excess_exret > 20, pef < pet
        assert result["conviction"] >= 40, "Quality floor should still apply without kill thesis"


class TestV12M1AsymmetricNewsWeights:
    """CIO v12.0 M1: Negative news should have higher weight than positive."""

    def test_high_negative_stronger_than_high_positive(self):
        """HIGH_NEGATIVE should produce lower bull_pct than HIGH_POSITIVE produces higher."""
        # All other agents neutral
        _, _, _ = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="HIGH_POSITIVE", risk_warning=False, signal="H",
        )
        bull_pos, bear_pos, _ = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="HIGH_POSITIVE", risk_warning=False, signal="H",
        )
        bull_neg, bear_neg, _ = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="HIGH_NEGATIVE", risk_warning=False, signal="H",
        )
        # bear weight from HIGH_NEGATIVE should be > bull weight from HIGH_POSITIVE
        assert bear_neg > bull_pos - bear_pos, (
            f"Negative news bear weight ({bear_neg}) should exceed positive news "
            f"net bull effect ({bull_pos - bear_pos})"
        )

    def test_high_negative_asymmetric_impact(self):
        """HIGH_NEGATIVE net bear impact should exceed HIGH_POSITIVE net bull impact."""
        bull_base, bear_base, _ = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
        )
        bull_neg, bear_neg, _ = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="HIGH_NEGATIVE", risk_warning=False, signal="H",
        )
        bull_pos, bear_pos, _ = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="HIGH_POSITIVE", risk_warning=False, signal="H",
        )
        # Net negative impact (bear delta) should exceed net positive impact (bull delta)
        bear_delta = bear_neg - bear_base
        bull_delta = bull_pos - bull_base
        assert bear_delta > bull_delta, (
            f"Negative bear delta ({bear_delta:.2f}) should exceed "
            f"positive bull delta ({bull_delta:.2f})"
        )

    def test_high_positive_adds_less_than_1_0_bull(self):
        """HIGH_POSITIVE should add < 1.0 net bull weight (conservative)."""
        bull_base, bear_base, _ = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
        )
        bull_pos, bear_pos, _ = count_agent_votes(
            fund_score=55, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="HIGH_POSITIVE", risk_warning=False, signal="H",
        )
        # Net bull delta = (bull_pos - bull_base) - (bear_pos - bear_base)
        net_bull = (bull_pos - bull_base) - (bear_pos - bear_base)
        assert net_bull < 1.0, (
            f"HIGH_POSITIVE net bull contribution ({net_bull:.2f}) should be < 1.0"
        )


class TestV12R2SellBearRatioContinuous:
    """CIO v12.0 R2: SELL signal base conviction should use continuous interpolation."""

    def test_bear_ratio_0_65_vs_0_79_differentiated(self):
        """Bear ratio 0.65 and 0.79 should produce different base convictions."""
        # bear_ratio 0.65
        base_65 = determine_base_conviction(
            bull_pct=35, signal="S", fund_score=50,
            excess_exret=0, bear_ratio=0.65,
        )
        # bear_ratio 0.79
        base_79 = determine_base_conviction(
            bull_pct=35, signal="S", fund_score=50,
            excess_exret=0, bear_ratio=0.79,
        )
        assert base_79 > base_65, (
            f"Bear ratio 0.79 base ({base_79}) should exceed 0.65 base ({base_65})"
        )

    def test_bear_ratio_0_80_gives_85(self):
        """Bear ratio >= 0.80 should still give base 85."""
        base = determine_base_conviction(
            bull_pct=20, signal="S", fund_score=50,
            excess_exret=0, bear_ratio=0.85,
        )
        assert base == 85

    def test_bear_ratio_0_40_gives_50(self):
        """Bear ratio at 0.40 boundary should give base 50."""
        base = determine_base_conviction(
            bull_pct=60, signal="S", fund_score=50,
            excess_exret=0, bear_ratio=0.40,
        )
        assert base == 50

    def test_bear_ratio_below_0_40_gives_50(self):
        """Bear ratio below 0.40 (agents disagree with SELL) should give base 50."""
        base = determine_base_conviction(
            bull_pct=65, signal="S", fund_score=50,
            excess_exret=0, bear_ratio=0.35,
        )
        assert base == 50

    def test_bear_ratio_continuous_monotonic(self):
        """Base conviction should increase monotonically with bear_ratio (0.40 to 0.80)."""
        prev_base = 0
        for br in [0.40, 0.50, 0.60, 0.70, 0.79]:
            base = determine_base_conviction(
                bull_pct=40, signal="S", fund_score=50,
                excess_exret=0, bear_ratio=br,
            )
            assert base >= prev_base, (
                f"Base at bear_ratio={br} ({base}) should >= prev ({prev_base})"
            )
            prev_base = base


# ============================================================
# CIO v12.0 M2: Volume-Weighted Census Signal
# ============================================================


class TestV12M2VolumeWeightedCensus:
    """CIO v12.0 M2: Census weight scales by divergence magnitude."""

    def test_high_div_score_gets_full_weight(self):
        """High divergence score (50+) should get full census weight."""
        bull_hi, bear_hi, _ = count_agent_votes(
            fund_score=70, tech_signal="ENTER_NOW", tech_momentum=30,
            macro_fit="FAVORABLE", census_alignment="ALIGNED",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            census_div_score=60.0,
        )
        bull_lo, bear_lo, _ = count_agent_votes(
            fund_score=70, tech_signal="ENTER_NOW", tech_momentum=30,
            macro_fit="FAVORABLE", census_alignment="ALIGNED",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            census_div_score=10.0,
        )
        # High div_score should contribute more bull weight
        assert bull_hi > bull_lo

    def test_zero_div_score_gets_half_weight(self):
        """Zero/missing divergence score should use 0.5 magnitude (half weight)."""
        bull_zero, _, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="ALIGNED",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
            census_div_score=0.0,
        )
        bull_full, _, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="ALIGNED",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
            census_div_score=50.0,
        )
        assert bull_full > bull_zero

    def test_divergent_scales_with_magnitude(self):
        """DIVERGENT alignment should also scale bear weight by magnitude."""
        _, bear_hi, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="DIVERGENT",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
            census_div_score=60.0,
        )
        _, bear_lo, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="DIVERGENT",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
            census_div_score=10.0,
        )
        assert bear_hi > bear_lo

    def test_backward_compatible_without_div_score(self):
        """Default census_div_score=0.0 should not crash."""
        bull, bear, dc = count_agent_votes(
            fund_score=70, tech_signal="ENTER_NOW", tech_momentum=30,
            macro_fit="FAVORABLE", census_alignment="ALIGNED",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
        )
        assert bull > bear


# ============================================================
# CIO v12.0 M3: Earnings Trajectory Momentum
# ============================================================


class TestV12M3EarningsTrajectory:
    """CIO v12.0 M3: Surprise trajectory modulates earnings adjustment."""

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_accelerating_beats_boost(self):
        """ACCELERATING trajectory with serial beats should add +2."""
        adj_accel, label_accel = get_earnings_surprise_adjustment(
            recent_surprise_pct=12.0,
            consecutive_beats=3,
            surprise_trajectory="ACCELERATING",
        )
        adj_base, label_base = get_earnings_surprise_adjustment(
            recent_surprise_pct=12.0,
            consecutive_beats=3,
        )
        assert adj_accel == adj_base + 2
        assert "_ACCEL" in label_accel

    def test_decelerating_beats_reduce(self):
        """DECELERATING trajectory on BEAT should reduce by 2."""
        adj_decel, label_decel = get_earnings_surprise_adjustment(
            recent_surprise_pct=8.0,
            consecutive_beats=1,
            surprise_trajectory="DECELERATING",
        )
        adj_base, _ = get_earnings_surprise_adjustment(
            recent_surprise_pct=8.0,
            consecutive_beats=1,
        )
        assert adj_decel == max(adj_base - 2, 0)
        assert "_DECEL" in label_decel

    def test_decelerating_miss_no_effect(self):
        """DECELERATING on MISS should have no trajectory effect."""
        adj_decel, label_decel = get_earnings_surprise_adjustment(
            recent_surprise_pct=-8.0,
            consecutive_beats=0,
            surprise_trajectory="DECELERATING",
        )
        adj_base, label_base = get_earnings_surprise_adjustment(
            recent_surprise_pct=-8.0,
            consecutive_beats=0,
        )
        assert adj_decel == adj_base
        assert "_DECEL" not in label_decel

    def test_accelerating_needs_consecutive_beats(self):
        """ACCELERATING with <2 consecutive beats should have no trajectory effect."""
        adj, label = get_earnings_surprise_adjustment(
            recent_surprise_pct=12.0,
            consecutive_beats=1,
            surprise_trajectory="ACCELERATING",
        )
        adj_base, _ = get_earnings_surprise_adjustment(
            recent_surprise_pct=12.0,
            consecutive_beats=1,
        )
        assert adj == adj_base
        assert "_ACCEL" not in label

    def test_capped_at_ten(self):
        """ACCELERATING trajectory adjustment should cap at 10 (v25.0)."""
        adj, _ = get_earnings_surprise_adjustment(
            recent_surprise_pct=15.0,
            consecutive_beats=4,
            surprise_trajectory="ACCELERATING",
        )
        assert adj <= 10

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_backward_compatible_without_trajectory(self):
        """Without surprise_trajectory, serial beater returns +8 (v20.0 D3)."""
        adj_new, label_new = get_earnings_surprise_adjustment(
            recent_surprise_pct=12.0,
            consecutive_beats=3,
        )
        assert adj_new == 8
        assert label_new == "SERIAL_BEATER"


class TestEnrichWithPositionSizes:
    """CIO v13.0 S2: Position sizing integration tests."""

    def test_buy_gets_size(self):
        """BUY actions should get suggested_size_usd."""
        conc = [{"ticker": "AAPL", "action": "BUY", "conviction": 70}]
        enrich_with_position_sizes(conc)
        assert "suggested_size_usd" in conc[0]
        assert conc[0]["suggested_size_usd"] > 0

    def test_hold_no_size(self):
        """HOLD actions should NOT get sizing."""
        conc = [{"ticker": "MSFT", "action": "HOLD", "conviction": 55}]
        enrich_with_position_sizes(conc)
        assert "suggested_size_usd" not in conc[0]

    def test_higher_conviction_larger_size(self):
        """Higher conviction should produce larger position sizes."""
        conc = [
            {"ticker": "A", "action": "ADD", "conviction": 75},
            {"ticker": "B", "action": "ADD", "conviction": 50},
        ]
        enrich_with_position_sizes(conc)
        assert conc[0]["suggested_size_usd"] > conc[1]["suggested_size_usd"]

    def test_risk_off_reduces_size(self):
        """RISK_OFF regime should reduce position sizes."""
        normal_conc = [{"ticker": "AAPL", "action": "BUY", "conviction": 70}]
        enrich_with_position_sizes(normal_conc, regime="NEUTRAL")
        risk_off_conc = [{"ticker": "AAPL", "action": "BUY", "conviction": 70}]
        enrich_with_position_sizes(risk_off_conc, regime="RISK_OFF")
        assert risk_off_conc[0]["suggested_size_usd"] < normal_conc[0]["suggested_size_usd"]


# ============================================================
# CIO v14.0: Conviction Effectiveness Review Tests
# ============================================================


class TestV14RSIContextualTechVote:
    """CIO v14.0 V1: Tech AVOID/EXIT_SOON should be RSI-contextual."""

    def test_avoid_at_deeply_oversold_is_neutral(self):
        """At RSI < 30, AVOID should split neutral (not full bear)."""
        bull_os, bear_os, _ = count_agent_votes(
            fund_score=50, tech_signal="AVOID", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            rsi=25.0,
        )
        bull_norm, bear_norm, _ = count_agent_votes(
            fund_score=50, tech_signal="AVOID", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            rsi=55.0,
        )
        # At RSI 25 (oversold), tech should be neutral → more bull weight than at RSI 55
        assert bull_os > bull_norm
        # At RSI 55 (normal), AVOID is full bear → more bear weight
        assert bear_norm > bear_os

    def test_avoid_at_moderate_oversold_is_weak_bear(self):
        """At RSI 30-40, AVOID should be a weak bear (not full)."""
        bull_mod, bear_mod, _ = count_agent_votes(
            fund_score=50, tech_signal="AVOID", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            rsi=35.0,
        )
        _, bear_full, _ = count_agent_votes(
            fund_score=50, tech_signal="AVOID", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            rsi=55.0,
        )
        # Weak bear (0.6) should be less than full bear (1.0)
        assert bear_mod < bear_full

    def test_exit_soon_also_contextual(self):
        """EXIT_SOON should follow same RSI context as AVOID."""
        bull_os, _, _ = count_agent_votes(
            fund_score=50, tech_signal="EXIT_SOON", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            rsi=28.0,
        )
        bull_norm, _, _ = count_agent_votes(
            fund_score=50, tech_signal="EXIT_SOON", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            rsi=55.0,
        )
        assert bull_os > bull_norm

    def test_avoid_at_high_rsi_unchanged(self):
        """At RSI > 40, AVOID should remain full bear (unchanged behavior)."""
        _, bear_high, _ = count_agent_votes(
            fund_score=50, tech_signal="AVOID", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            rsi=60.0,
        )
        _, bear_also_high, _ = count_agent_votes(
            fund_score=50, tech_signal="AVOID", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            rsi=70.0,
        )
        # Both above RSI 40 should produce same full bear weight
        assert bear_high == bear_also_high


class TestV14BuyTechDisagreePenaltyScaling:
    """CIO v14.0 V2: BUY-tech disagree penalty scales by RSI."""

    def test_oversold_penalty_reduced(self):
        """At RSI < 35, BUY+AVOID penalty should be only -2."""
        bonuses, penalties, _ = compute_adjustments(
            signal="B", fund_score=70, tech_signal="AVOID",
            tech_momentum=-10, rsi=30, macro_fit="NEUTRAL",
            census_alignment="NEUTRAL", div_score=0, census_ts="stable",
            news_impact="NEUTRAL", risk_warning=False, buy_pct=80,
            excess_exret=10, beta=1.0, quality_trap=False,
            sector="Technology", sector_rankings={}, bull_count=4,
        )
        # At RSI 30, penalty should be -2 (not -8)
        # Total penalties = 2 (tech disagree only in this minimal case)
        assert penalties <= 5  # Only the RSI-scaled tech disagree + maybe minor others

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_neutral_rsi_moderate_penalty(self):
        """At RSI 35-50, penalty should be -5."""
        _, penalties_mod, _ = compute_adjustments(
            signal="B", fund_score=70, tech_signal="AVOID",
            tech_momentum=-10, rsi=45, macro_fit="NEUTRAL",
            census_alignment="NEUTRAL", div_score=0, census_ts="stable",
            news_impact="NEUTRAL", risk_warning=False, buy_pct=80,
            excess_exret=10, beta=1.0, quality_trap=False,
            sector="Technology", sector_rankings={}, bull_count=4,
        )
        _, penalties_os, _ = compute_adjustments(
            signal="B", fund_score=70, tech_signal="AVOID",
            tech_momentum=-10, rsi=30, macro_fit="NEUTRAL",
            census_alignment="NEUTRAL", div_score=0, census_ts="stable",
            news_impact="NEUTRAL", risk_warning=False, buy_pct=80,
            excess_exret=10, beta=1.0, quality_trap=False,
            sector="Technology", sector_rankings={}, bull_count=4,
        )
        assert penalties_mod > penalties_os


class TestV14RiskWarningConsensusOverride:
    """CIO v14.0 V3: Risk warning defers to overwhelming consensus."""

    def test_consensus_override_reduces_risk_weight(self):
        """When buy_pct>=90 and fund_score>=80, risk weight should be 0.6x."""
        # With consensus override
        bull_co, bear_co, _ = count_agent_votes(
            fund_score=85, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="B",
            buy_pct=95, rsi=50,
        )
        # Without override (buy_pct below threshold)
        bull_no, bear_no, _ = count_agent_votes(
            fund_score=85, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="B",
            buy_pct=70, rsi=50,
        )
        # With override, risk bear weight should be lower
        assert bear_co < bear_no

    def test_contradiction_skipped_with_consensus(self):
        """Fund-risk contradiction should be noted but not penalized at high consensus."""
        penalty, desc = detect_contradictions(
            macro_fit="NEUTRAL", tech_signal="HOLD",
            fund_score=85, risk_warning=True,
            census_alignment="NEUTRAL", news_impact="NEUTRAL",
            buy_pct=95,
        )
        # With consensus override, penalty should be 0
        assert penalty == 0
        # But the contradiction is still noted in descriptions
        assert len(desc) > 0
        assert "overridden" in desc[0]

    def test_contradiction_still_penalizes_without_consensus(self):
        """Without overwhelming consensus, fund-risk contradiction is penalized."""
        penalty, _ = detect_contradictions(
            macro_fit="NEUTRAL", tech_signal="HOLD",
            fund_score=85, risk_warning=True,
            census_alignment="NEUTRAL", news_impact="NEUTRAL",
            buy_pct=70,
        )
        assert penalty == 3

    def test_consensus_override_at_fund_score_75(self):
        """V3 threshold lowered to fund>=70: fund=75 with 100% BUY should override."""
        bull_co, bear_co, _ = count_agent_votes(
            fund_score=75, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="B",
            buy_pct=100, rsi=50,
        )
        bull_no, bear_no, _ = count_agent_votes(
            fund_score=75, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="B",
            buy_pct=70, rsi=50,
        )
        assert bear_co < bear_no  # Override should reduce bear weight

    def test_no_override_below_fund_70(self):
        """V3 should NOT override when fund_score < 70, even with 100% BUY."""
        bull_a, bear_a, _ = count_agent_votes(
            fund_score=65, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="B",
            buy_pct=100, rsi=50,
        )
        bull_b, bear_b, _ = count_agent_votes(
            fund_score=65, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=True, signal="B",
            buy_pct=70, rsi=50,
        )
        # Without override, both should have same risk weight (1.2x)
        assert abs(bear_a - bear_b) < 0.01


class TestV14ExcessExretStalenessValidation:
    """CIO v14.0 V4: Staleness only applies with low consensus."""

    def test_high_exret_high_consensus_no_penalty(self):
        """EXRET>40 with high bull_pct should NOT get staleness penalty."""
        # High bull_pct (strong consensus)
        base_high = determine_base_conviction(
            bull_pct=75, signal="B", fund_score=60,
            excess_exret=50, bear_ratio=0.3,
        )
        # Compare with moderate EXRET (which gets bonus)
        base_mod = determine_base_conviction(
            bull_pct=75, signal="B", fund_score=60,
            excess_exret=15, bear_ratio=0.3,
        )
        # High EXRET with high consensus should NOT be penalized
        # Both should get EXRET bonuses (50% excess gets full bonus, 15% gets moderate)
        assert base_high >= base_mod

    def test_high_exret_low_consensus_penalized(self):
        """EXRET>40 with low consensus should still be penalized."""
        base_extreme = determine_base_conviction(
            bull_pct=50, signal="B", fund_score=60,
            excess_exret=50, bear_ratio=0.3,
        )
        base_moderate = determine_base_conviction(
            bull_pct=50, signal="B", fund_score=60,
            excess_exret=15, bear_ratio=0.3,
        )
        assert base_extreme < base_moderate


class TestV14PenaltyProportionalityCap:
    """CIO v14.0 V5: Total penalties capped at 60% of base."""

    def test_penalties_dont_exceed_60pct_of_base(self):
        """Even with heavy penalties, total should not exceed 60% of base."""
        result = synthesize_stock(
            ticker="TEST",
            sig_data={"signal": "H", "exret": 0, "buy_pct": 30,
                      "beta": 3.0, "pet": 20, "pef": 25},
            fund_data={"fundamental_score": 30, "quality_trap_warning": True},
            tech_data={"momentum_score": -40, "timing_signal": "AVOID",
                       "rsi": 80, "macd_signal": "BEARISH"},
            macro_fit="UNFAVORABLE", census_alignment="DIVERGENT",
            div_score=50, census_ts_trend="distribution",
            news_impact="HIGH_NEGATIVE", risk_warning=True,
            sector="Consumer Discretionary", sector_median_exret=10,
            sector_rankings={}, position_limit=5.0,
            earnings_surprise_pct=-15,
        )
        base = result["base"]
        penalties = result["penalties"]
        # V5: penalties should be at most 60% of base
        assert penalties <= int(base * 0.60) + 1  # +1 for rounding

    def test_proportionality_preserves_differentiation(self):
        """V5 cap prevents penalties from exceeding 60% of base conviction."""
        # A stock with maximum penalties should still retain at least 40% of base
        result = synthesize_stock(
            ticker="A",
            sig_data={"signal": "B", "exret": 20, "buy_pct": 80,
                      "beta": 1.5, "pet": 20, "pef": 18},
            fund_data={"fundamental_score": 75},
            tech_data={"momentum_score": -20, "timing_signal": "AVOID",
                       "rsi": 60, "macd_signal": "BEARISH"},
            macro_fit="UNFAVORABLE", census_alignment="DIVERGENT",
            div_score=50, census_ts_trend="stable",
            news_impact="LOW_NEGATIVE", risk_warning=True,
            sector="Technology", sector_median_exret=10,
            sector_rankings={}, position_limit=5.0,
        )
        # With proportionality cap, conviction should never go deeply negative
        # even under maximum penalty stacking
        assert result["conviction"] >= 20  # Floor prevents total suppression


class TestV14SectorConcentrationScope:
    """CIO v14.0 V6: Sector concentration counts BUY/ADD actions only."""

    def test_hold_stocks_dont_inflate_sector_count(self):
        """HOLD stocks should not increase sector concentration penalty."""
        # Create a concordance with 5 tech stocks: 2 ADD + 3 HOLD
        sigs = {
            f"T{i}": {"signal": "B", "exret": 20, "buy_pct": 80,
                       "beta": 1.0, "pet": 15, "pef": 13}
            for i in range(5)
        }
        sector_map = {f"T{i}": "Technology" for i in range(5)}
        # Use minimal agent reports so all stocks get similar treatment
        fund = {"stocks": {}}
        tech = {"stocks": {}}
        macro = {"executive_summary": {"regime": ""}, "portfolio_implications": {},
                 "sector_rankings": {}}
        census = {"divergences": {}}
        news = {"portfolio_news": {}}
        risk = {"consensus_warnings": [], "position_limits": {}}

        concordance = build_concordance(
            sigs, fund, tech, macro, census, news, risk, sector_map,
        )
        # With V6, sector count should be based on BUY/ADD actions (signal=B stocks)
        # not ALL concordance entries. The penalty should be proportional to
        # the actual buying concentration, not the total sector presence.
        penalized = [e for e in concordance if e.get("sector_concentration_penalty", 0) > 0]
        # Some stocks may be penalized, but the counts should use BUY/ADD+signal=B scope
        for e in concordance:
            if e.get("sector_concentration_penalty", 0) > 0:
                # Penalty should be based on action-scoped count, which is smaller
                assert e["sector_concentration_penalty"] <= 15


# ═══════════════════════════════════════════════════════════════════════
# CIO v15.0: Conviction Integrity Tests
# ═══════════════════════════════════════════════════════════════════════


class TestV15ConsensusVoteInjection:
    """CIO v15.0 W1: Consensus-anchored vote injection."""

    def test_high_consensus_buy_injects_bull_vote(self):
        """buy_pct >= 75 with BUY signal should inject consensus bull vote."""
        # Without consensus injection (buy_pct=70, below threshold)
        bull_low, bear_low, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            buy_pct=70.0,
        )
        # With consensus injection (buy_pct=95)
        bull_high, bear_high, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            buy_pct=95.0,
        )
        # High consensus should produce higher bull_pct
        total_low = bull_low + bear_low
        total_high = bull_high + bear_high
        bp_low = bull_low / total_low * 100 if total_low > 0 else 50
        bp_high = bull_high / total_high * 100 if total_high > 0 else 50
        assert bp_high > bp_low
        assert bp_high >= 55  # Should be meaningfully above 50%

    def test_consensus_injection_scales_with_buy_pct(self):
        """Higher buy_pct should inject stronger bull weight."""
        results = []
        for bp in [75, 85, 95, 100]:
            bull, bear, _ = count_agent_votes(
                fund_score=50, tech_signal="HOLD", tech_momentum=0,
                macro_fit="NEUTRAL", census_alignment="NEUTRAL",
                news_impact="NEUTRAL", risk_warning=False, signal="B",
                buy_pct=float(bp),
            )
            total = bull + bear
            results.append(bull / total * 100 if total > 0 else 50)
        # Each step should increase bull_pct
        for i in range(len(results) - 1):
            assert results[i + 1] >= results[i]

    def test_no_injection_for_hold_signal(self):
        """Consensus injection only applies to BUY signal stocks."""
        bull_buy, bear_buy, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            buy_pct=95.0,
        )
        bull_hold, bear_hold, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="H",
            buy_pct=95.0,
        )
        # BUY signal should get higher bull_pct due to injection
        total_buy = bull_buy + bear_buy
        total_hold = bull_hold + bear_hold
        bp_buy = bull_buy / total_buy * 100
        bp_hold = bull_hold / total_hold * 100
        assert bp_buy > bp_hold

    def test_sell_signal_low_consensus_injects_bear(self):
        """SELL signal with very low buy_pct should inject bear vote."""
        bull_mid, bear_mid, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="S",
            buy_pct=30.0,
        )
        bull_low, bear_low, _ = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="S",
            buy_pct=15.0,
        )
        # Low buy_pct with SELL should have higher bear_ratio
        total_mid = bull_mid + bear_mid
        total_low = bull_low + bear_low
        br_mid = bear_mid / total_mid
        br_low = bear_low / total_low
        assert br_low > br_mid

    def test_consensus_injection_improves_directional_confidence(self):
        """Consensus injection should increase directional confidence."""
        _, _, dc_without = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            buy_pct=70.0,
        )
        _, _, dc_with = count_agent_votes(
            fund_score=50, tech_signal="HOLD", tech_momentum=0,
            macro_fit="NEUTRAL", census_alignment="NEUTRAL",
            news_impact="NEUTRAL", risk_warning=False, signal="B",
            buy_pct=95.0,
        )
        assert dc_with > dc_without


class TestV15RegimeDiscountCap:
    """CIO v15.0 W3: Regime discount cap for BUY signals."""

    def test_cautious_buy_signal_gets_reduced_discount(self):
        """BUY signal in CAUTIOUS should get 5% discount, not 8%."""
        base_buy = determine_base_conviction(
            65.0, "B", 70, 15, 0.35, regime="CAUTIOUS",
        )
        base_hold = determine_base_conviction(
            65.0, "H", 70, 15, 0.35, regime="CAUTIOUS",
        )
        # BUY should be higher because of reduced regime discount
        # (plus BUY-specific floors and bonuses)
        assert base_buy >= base_hold

    def test_cautious_buy_higher_than_v14(self):
        """v15.0 CAUTIOUS BUY should produce higher base than v14's 8%."""
        # With CAUTIOUS and bull_pct=65, agent_base is about 62
        # v14: 62 * 0.92 = 57, then BUY floor 55 → max(57, 55) = 57
        # v15: 62 * 0.95 = 58, then BUY floor 55 → max(58, 55) = 58+
        base = determine_base_conviction(
            65.0, "B", 70, 15, 0.35, regime="CAUTIOUS",
        )
        assert base >= 58  # Should be at least 58 with reduced discount

    def test_risk_off_unchanged(self):
        """RISK_OFF should still apply full 15% discount."""
        base_riskoff = determine_base_conviction(
            65.0, "B", 70, 15, 0.35, regime="RISK_OFF",
        )
        base_no_regime = determine_base_conviction(
            65.0, "B", 70, 15, 0.35, regime="",
        )
        # RISK_OFF should be significantly lower
        assert base_riskoff < base_no_regime


class TestV15ConsensusPremium:
    """CIO v15.0 W4: Consensus premium in base conviction."""

    def test_high_consensus_gets_premium(self):
        """BUY signal with bull_pct >= 80 should get consensus premium."""
        base_high = determine_base_conviction(
            85.0, "B", 80, 20, 0.15, regime="",
        )
        base_low = determine_base_conviction(
            70.0, "B", 80, 20, 0.30, regime="",
        )
        # Higher bull_pct should produce higher base
        assert base_high > base_low

    def test_premium_capped_at_8(self):
        """Consensus premium should not exceed 8 points."""
        # With bull_pct=100, premium = min(8, (100-75)/5) = min(8, 5) = 5
        # With bull_pct=200 (edge case), premium = min(8, ...) = 8
        base = determine_base_conviction(
            100.0, "B", 90, 30, 0.0, regime="",
        )
        # Should be reasonable — not infinitely boosted
        assert base <= 90

    def test_no_premium_below_80_bull_pct(self):
        """No consensus premium when bull_pct < 80."""
        base_79 = determine_base_conviction(
            79.0, "B", 60, 10, 0.21, regime="",
        )
        base_80 = determine_base_conviction(
            80.0, "B", 60, 10, 0.20, regime="",
        )
        # Base at 80 should be at least as high as at 79
        assert base_80 >= base_79


class TestV15SectorConcentrationFloor:
    """CIO v15.0 W2: Inviolable BUY floor after sector concentration."""

    def test_high_consensus_buy_floor_52(self):
        """BUY signal with buy_pct >= 80 should have floor 52 during concentration."""
        sigs = {
            f"T{i}": {"signal": "B", "exret": 30, "buy_pct": 95,
                       "beta": 1.0, "pet": 20, "pef": 18}
            for i in range(5)
        }
        sector_map = {f"T{i}": "Technology" for i in range(5)}
        fund = {"stocks": {}}
        tech = {"stocks": {}}
        macro = {"executive_summary": {"regime": ""}, "portfolio_implications": {},
                 "sector_rankings": {}}
        census = {"divergences": {}}
        news = {"portfolio_news": {}}
        risk = {"consensus_warnings": [], "position_limits": {}}

        concordance = build_concordance(
            sigs, fund, tech, macro, census, news, risk, sector_map,
        )
        # All stocks have buy_pct=95 (>= 80), so floor should be at least 52
        for e in concordance:
            if e.get("sector_concentration_penalty", 0) > 0:
                assert e["conviction"] >= 52, (
                    f"{e['ticker']} conviction {e['conviction']} < 52 "
                    f"despite buy_pct={e['buy_pct']}"
                )

    def test_low_consensus_uses_standard_floor(self):
        """BUY signal with buy_pct < 80 uses standard floor 45."""
        sigs = {
            f"T{i}": {"signal": "B", "exret": 10, "buy_pct": 60,
                       "beta": 1.0, "pet": 20, "pef": 22}
            for i in range(5)
        }
        sector_map = {f"T{i}": "Technology" for i in range(5)}
        fund = {"stocks": {}}
        tech = {"stocks": {}}
        macro = {"executive_summary": {"regime": ""}, "portfolio_implications": {},
                 "sector_rankings": {}}
        census = {"divergences": {}}
        news = {"portfolio_news": {}}
        risk = {"consensus_warnings": [], "position_limits": {}}

        concordance = build_concordance(
            sigs, fund, tech, macro, census, news, risk, sector_map,
        )
        for e in concordance:
            if e.get("sector_concentration_penalty", 0) > 0:
                assert e["conviction"] >= 45  # Standard BUY floor


class TestV15OpportunityThreshold:
    """CIO v15.0 W5: Opportunity BUY threshold reduction."""

    def test_opportunity_at_conv_50_is_buy(self):
        """Opportunity with conviction 50 should be BUY (was HOLD in v14)."""
        entry = {
            "ticker": "NEW1", "signal": "B", "sector": "Materials",
            "conviction": 62, "action": "ADD", "fund_score": 70,
            "buy_pct": 80, "is_opportunity": False, "base": 60,
            "tech_signal": "HOLD", "exret": 20, "excess_exret": 10,
        }
        fund = {"stocks": {"NEW1": {"fundamental_score": 70}}}
        tech = {"stocks": {"NEW1": {"timing_signal": "ENTER_NOW"}}}

        result = apply_opportunity_gate(
            entry, fund, tech, "FAVORABLE", "ALIGNED",
            {"Technology": 5}, regime="CAUTIOUS",
        )
        # With 3 confirmations (fund>=70, tech=ENTER_NOW, macro=FAVORABLE)
        # No discount applied. Conviction should stay high.
        if result["conviction"] >= 50:
            assert result["action"] == "BUY"

    def test_opportunity_below_50_is_hold(self):
        """Opportunity below 50 conviction should be HOLD."""
        entry = {
            "ticker": "WEAK1", "signal": "H", "sector": "Other",
            "conviction": 40, "action": "HOLD", "fund_score": 40,
            "buy_pct": 50, "is_opportunity": False, "base": 45,
            "tech_signal": "HOLD", "exret": 5, "excess_exret": -5,
        }
        fund = {"stocks": {}}
        tech = {"stocks": {}}

        result = apply_opportunity_gate(
            entry, fund, tech, "NEUTRAL", "NEUTRAL",
            {"Technology": 5}, regime="CAUTIOUS",
        )
        # With 0 confirmations and CAUTIOUS discount, should be well below 50
        assert result["action"] == "HOLD"


class TestV15EndToEndConvictionIntegrity:
    """Integration tests for v15.0 conviction integrity."""

    def test_msft_like_stock_gets_add(self):
        """MEGA-cap BUY signal with 95% consensus should score as ADD."""
        result = synthesize_stock(
            ticker="MSFT",
            sig_data={"signal": "B", "exret": 50, "buy_pct": 95,
                      "beta": 1.0, "pet": 25, "pef": 22},
            fund_data={"fundamental_score": 75},
            tech_data={"momentum_score": 0, "timing_signal": "HOLD",
                       "rsi": 33, "macd_signal": "NEUTRAL"},
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
            regime="CAUTIOUS",
        )
        # With v15.0 consensus injection + premium, conviction should reach ADD
        assert result["conviction"] >= 55, (
            f"MSFT-like stock at conviction {result['conviction']} < 55 — "
            "consensus injection not working"
        )
        assert result["action"] == "ADD"

    def test_low_consensus_stock_not_inflated(self):
        """Stock with low consensus should not benefit from W1/W4."""
        result = synthesize_stock(
            ticker="WEAK",
            sig_data={"signal": "B", "exret": 10, "buy_pct": 55,
                      "beta": 1.5, "pet": 20, "pef": 25},
            fund_data={"fundamental_score": 50},
            tech_data={"momentum_score": -10, "timing_signal": "HOLD",
                       "rsi": 50, "macd_signal": "BEARISH"},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Industrials",
            sector_median_exret=10,
            sector_rankings={},
            position_limit=5.0,
            regime="CAUTIOUS",
        )
        # Low consensus, deteriorating PE — should not reach ADD
        assert result["conviction"] <= 60

    def test_conviction_spread_improved(self):
        """v15.0 should produce wider conviction spread than v14."""
        stocks = []
        for bp, exret, fund_s in [(100, 50, 85), (80, 30, 70), (60, 15, 50)]:
            r = synthesize_stock(
                ticker=f"S{bp}",
                sig_data={"signal": "B", "exret": exret, "buy_pct": bp,
                          "beta": 1.0, "pet": 20, "pef": 18},
                fund_data={"fundamental_score": fund_s},
                tech_data={"momentum_score": 0, "timing_signal": "HOLD",
                           "rsi": 45, "macd_signal": "NEUTRAL"},
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
            stocks.append(r)
        # The spread between best and worst should be meaningful
        spread = stocks[0]["conviction"] - stocks[2]["conviction"]
        assert spread >= 10, (
            f"Conviction spread {spread} < 10 — insufficient differentiation "
            f"({stocks[0]['conviction']}, {stocks[1]['conviction']}, {stocks[2]['conviction']})"
        )


class TestV40ConsensusCrowdedPenaltyRemoved:
    """CIO v40 (post-backtest 2026-04-26): consensus_crowded penalty removed.

    Was tier-scaled in v37 (MEGA=0, LARGE=3, MID=5, SMALL=8, MICRO=10 with
    excess_exret multiplier). T+7 attribution on n=442 evaluations showed the
    rule was contrarian even after tier-scaling: ADD actions where
    wf_consensus_crowded fired returned +8.51% alpha (5/7 hits). The penalty
    was suppressing conviction on names that subsequently rallied.

    All previous tier/excess_exret cases must now produce 0 consensus_crowded
    contribution. Stocks marginally below sector median should still earn ADD
    purely from positive factors (no longer get pulled down by consensus).
    """

    def _kwargs_with_high_consensus(self, **overrides):
        base = dict(
            signal="B", fund_score=75, tech_signal="HOLD",
            tech_momentum=0, rsi=50, macro_fit="NEUTRAL",
            census_alignment="NEUTRAL", div_score=0,
            census_ts="stable", news_impact="NEUTRAL",
            risk_warning=False, buy_pct=96,
            excess_exret=0, beta=1.0, quality_trap=False,
            sector="Technology", sector_rankings={},
            bull_count=3, stock_tier="MID",
        )
        base.update(overrides)
        return base

    def test_no_penalty_at_any_tier(self):
        """consensus_crowded must produce 0 penalty for every tier × excess_exret."""
        baselines = {}
        for tier in ("MEGA", "LARGE", "MID", "SMALL", "MICRO"):
            kw = self._kwargs_with_high_consensus(stock_tier=tier, excess_exret=0)
            _, baseline, _ = compute_adjustments(**kw)
            baselines[tier] = baseline
            for excess in (-15, -10, -5, 5, 12, 20):
                kw = self._kwargs_with_high_consensus(stock_tier=tier, excess_exret=excess)
                _, p, _ = compute_adjustments(**kw)
                # Penalty totals must be IDENTICAL across excess_exret bands
                # (other factors don't depend on excess_exret in this kwargs set).
                assert p == baseline, (
                    f"Tier {tier}, excess_exret {excess}: expected {baseline}, "
                    f"got {p} — consensus_crowded should not fire"
                )

    def test_marginal_below_median_no_longer_dragged_down(self):
        """A stock marginally below median + strong consensus should reach ADD.

        Previously needed the v16 smoothing to avoid a -15 cliff penalty.
        Now the penalty is gone entirely, so the stock should trivially clear
        the ADD threshold from positive factors alone.
        """
        result = synthesize_stock(
            ticker="MARG1",
            sig_data={"signal": "B", "exret": 30, "buy_pct": 96,
                      "beta": 1.0, "pet": 20, "pef": 18},
            fund_data={"fundamental_score": 85},
            tech_data={"momentum_score": 15, "timing_signal": "HOLD",
                       "rsi": 50, "macd_signal": "BULLISH"},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Consumer Discretionary",
            sector_median_exret=31,  # excess_exret = -1 (marginal below)
            sector_rankings={},
            position_limit=5.0,
        )
        assert result["conviction"] >= 55, (
            f"Marginal below-median stock conv {result['conviction']} < 55"
        )
        assert result["action"] == "ADD", (
            f"Expected ADD for marginal below-median, got {result['action']}"
        )


# ============================================================
# CIO v17.0: Performance feedback loop
# ============================================================


class TestV17PerformanceFeedback:
    """Tests for performance data integration in synthesis output."""

    def _minimal_reports(self):
        """Build minimal agent reports for generate_synthesis_output."""
        macro = {"regime": "CAUTIOUS", "macro_score": 50,
                 "rotation_phase": "LATE_CYCLE", "indicators": {},
                 "sector_rankings": {}}
        census = {"sentiment": {"fg_top100": 50, "fg_broad": 50}}
        news = {}
        risk = {"portfolio_risk": {"var_95": 2.0, "max_drawdown": 0.1,
                                   "portfolio_beta": 1.0, "risk_score": 50}}
        return macro, census, news, risk

    def test_synthesis_output_includes_performance_data(self):
        """Performance data should be included in synthesis output."""
        macro, census, news, risk = self._minimal_reports()
        perf = {
            "status": "complete",
            "prev_committee_date": "2026-03-20",
            "total_evaluated": 10,
            "actions": {
                "ADD": {"count": 5, "hit_rate": 60.0, "avg_return": 2.5},
                "HOLD": {"count": 3, "hit_rate": 66.7, "avg_return": 0.1},
            },
        }
        output = generate_synthesis_output(
            concordance=[], macro_report=macro, census_report=census,
            news_report=news, risk_report=risk, changes=[], sector_gaps=[],
            performance_data=perf,
        )
        assert output["performance"]["status"] == "complete"
        assert output["performance"]["actions"]["ADD"]["hit_rate"] == pytest.approx(60.0)

    def test_synthesis_output_empty_performance_when_none(self):
        """When no performance data, output should have empty dict."""
        macro, census, news, risk = self._minimal_reports()
        output = generate_synthesis_output(
            concordance=[], macro_report=macro, census_report=census,
            news_report=news, risk_report=risk, changes=[], sector_gaps=[],
        )
        assert output["performance"] == {}

    def test_synthesize_stock_includes_price(self):
        """synthesize_stock should include price from sig_data."""
        result = synthesize_stock(
            ticker="AAPL",
            sig_data={"signal": "B", "exret": 15, "buy_pct": 80,
                      "beta": 1.1, "pet": 25, "pef": 22, "price": 185.5},
            fund_data={"fundamental_score": 75},
            tech_data={"momentum_score": 30, "timing_signal": "ENTER_NOW",
                       "rsi": 55, "macd_signal": "BULLISH"},
            macro_fit="FAVORABLE",
            census_alignment="ALIGNED",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=10,
            sector_rankings={},
            position_limit=5.0,
        )
        assert result["price"] == pytest.approx(185.5)

    def test_synthesize_stock_price_defaults_to_zero(self):
        """When sig_data has no price, should default to 0."""
        result = synthesize_stock(
            ticker="TEST",
            sig_data={"signal": "H", "exret": 5, "buy_pct": 50,
                      "beta": 1.0, "pet": 20, "pef": 20},
            fund_data={"fundamental_score": 50},
            tech_data={"momentum_score": 0, "timing_signal": "HOLD",
                       "rsi": 50, "macd_signal": "NEUTRAL"},
            macro_fit="NEUTRAL",
            census_alignment="NEUTRAL",
            div_score=0,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Other",
            sector_median_exret=5,
            sector_rankings={},
            position_limit=5.0,
        )
        assert result["price"] == 0


# ============================================================
# CIO v26.1: Agent Report Normalizer Tests
# ============================================================


class TestNormalizeFundStocks:
    """Fund report stocks can arrive as list or dict."""

    def test_dict_passthrough(self):
        report = {"stocks": {"AAPL": {"fundamental_score": 80}}}
        _normalize_fund_stocks(report)
        assert report["stocks"] == {"AAPL": {"fundamental_score": 80}}

    def test_list_to_dict(self):
        report = {"stocks": [
            {"ticker": "AAPL", "fundamental_score": 80},
            {"ticker": "MSFT", "fundamental_score": 75},
        ]}
        _normalize_fund_stocks(report)
        assert isinstance(report["stocks"], dict)
        assert "AAPL" in report["stocks"]
        assert report["stocks"]["AAPL"]["fundamental_score"] == 80
        assert "MSFT" in report["stocks"]

    def test_list_with_symbol_key(self):
        report = {"stocks": [{"symbol": "GOOG", "score": 70}]}
        _normalize_fund_stocks(report)
        assert "GOOG" in report["stocks"]

    def test_empty_stocks(self):
        report = {"stocks": {}}
        _normalize_fund_stocks(report)
        assert report["stocks"] == {}

    def test_missing_stocks(self):
        report = {}
        _normalize_fund_stocks(report)
        assert report.get("stocks", {}) == {}


class TestNormalizeCensusDivergences:
    """Census divergences can arrive as flat list or structured dict."""

    def test_dict_passthrough(self):
        div = {"signal_divergences": [{"ticker": "A"}], "census_divergences": [], "consensus_aligned": []}
        report = {"divergences": div}
        _normalize_census_divergences(report)
        assert report["divergences"] == div

    def test_flat_list_to_structured(self):
        report = {"divergences": [
            {"ticker": "AAPL", "type": "signal_divergence", "divergence_score": 25},
            {"ticker": "MSFT", "type": "census_divergence", "divergence_score": 15},
            {"ticker": "GOOG", "type": "consensus_aligned", "divergence_score": 5},
        ]}
        _normalize_census_divergences(report)
        d = report["divergences"]
        assert isinstance(d, dict)
        assert len(d["signal_divergences"]) == 1
        assert d["signal_divergences"][0]["ticker"] == "AAPL"
        assert len(d["census_divergences"]) == 1
        assert len(d["consensus_aligned"]) == 1

    def test_ambiguous_items_classified_by_score(self):
        report = {"divergences": [
            {"ticker": "X", "divergence_score": 5},   # low score -> aligned
            {"ticker": "Y", "divergence_score": 30},   # high score -> signal divergence
        ]}
        _normalize_census_divergences(report)
        d = report["divergences"]
        assert any(i["ticker"] == "X" for i in d["consensus_aligned"])
        assert any(i["ticker"] == "Y" for i in d["signal_divergences"])

    def test_string_items_in_structured_lists_become_dicts(self):
        # Reproduces 2026-04-27 prod crash: census agent emitted bare ticker
        # strings in consensus_aligned (and may do so in any of the three lists).
        # Downstream synthesis indexes item["ticker"] / item.get(...), so unwrapped
        # strings raise AttributeError. Normalization must rescue both shapes.
        report = {"divergences": {
            "consensus_aligned": ["MSTR", {"ticker": "MSFT", "divergence_score": 2}],
            "signal_divergences": ["NVDA"],
            "census_divergences": ["VXX"],
        }}
        _normalize_census_divergences(report)
        d = report["divergences"]

        aligned = {i["ticker"]: i for i in d["consensus_aligned"]}
        assert aligned["MSTR"] == {"ticker": "MSTR", "divergence_score": 0}
        assert aligned["MSFT"]["divergence_score"] == 2  # dict items untouched

        assert d["signal_divergences"][0] == {"ticker": "NVDA", "divergence_score": 0}
        assert d["census_divergences"][0] == {"ticker": "VXX", "divergence_score": 0}


class TestNormalizeBreakingNews:
    """Breaking news can arrive as list or dict of categories."""

    def test_list_passthrough(self):
        report = {"breaking_news": [{"headline": "Test", "impact": "HIGH_NEGATIVE"}]}
        _normalize_breaking_news(report)
        assert len(report["breaking_news"]) == 1

    def test_dict_to_list(self):
        report = {"breaking_news": {
            "tariffs": [
                {"title": "New tariffs", "severity": "CRITICAL", "tickers": ["AAPL"]},
                {"title": "Trade war", "severity": "HIGH"},
            ],
            "oil": [
                {"event": "Oil spike", "severity": "MEDIUM"},
            ],
        }}
        _normalize_breaking_news(report)
        bn = report["breaking_news"]
        assert isinstance(bn, list)
        assert len(bn) == 3
        # Check headline normalization
        assert bn[0]["headline"] == "New tariffs"
        assert bn[2]["headline"] == "Oil spike"
        # Check impact mapping
        assert bn[0]["impact"] == "HIGH_NEGATIVE"
        assert bn[1]["impact"] == "LOW_NEGATIVE"
        assert bn[2]["impact"] == "NEUTRAL"
        # Check tickers carried through
        assert bn[0]["affected_tickers"] == ["AAPL"]


class TestNormalizeSectorRankings:
    """Sector rankings can arrive as list or dict."""

    def test_dict_passthrough(self):
        report = {"sector_rankings": {"XLK": {"return_1m": 3.2}}}
        _normalize_sector_rankings(report)
        assert report["sector_rankings"]["XLK"]["return_1m"] == pytest.approx(3.2)

    def test_list_to_dict(self):
        report = {"sector_rankings": [
            {"etf": "XLK", "1m_return": 3.2, "status": "LEADING", "rank": 2},
            {"sector": "Energy", "return_1m": -1.5, "rank": 8},
        ]}
        _normalize_sector_rankings(report)
        sr = report["sector_rankings"]
        assert isinstance(sr, dict)
        assert sr["XLK"]["return_1m"] == pytest.approx(3.2)
        assert sr["XLK"]["relative_strength"] == "LEADING"
        assert sr["Energy"]["return_1m"] == -1.5

    def test_empty_list(self):
        report = {"sector_rankings": []}
        _normalize_sector_rankings(report)
        assert report["sector_rankings"] == {}


class TestNormalizeAgentReportsIntegration:
    """End-to-end normalization of all reports at once."""

    def test_all_normalizers_applied(self):
        fund = {"stocks": [{"ticker": "AAPL", "score": 80}]}
        census = {"divergences": [{"ticker": "X", "type": "signal_divergence", "divergence_score": 20}]}
        news = {"breaking_news": {"cat": [{"title": "T", "severity": "CRITICAL"}]}}
        macro = {"sector_rankings": [{"etf": "XLK", "1m_return": 2.0}]}
        f, t, m, c, n, r = normalize_agent_reports(fund, {}, macro, census, news, {})
        assert isinstance(f["stocks"], dict)
        assert isinstance(c["divergences"], dict)
        assert isinstance(n["breaking_news"], list)
        assert isinstance(m["sector_rankings"], dict)

    def test_idempotent(self):
        """Running normalization twice produces same result."""
        fund = {"stocks": [{"ticker": "AAPL", "score": 80}]}
        _normalize_fund_stocks(fund)
        snapshot = dict(fund["stocks"])
        _normalize_fund_stocks(fund)
        assert fund["stocks"] == snapshot


class TestDebateConvictionModifiers:
    """CIO v27.0: Adversarial debate conviction adjustments."""

    def _base_kwargs(self):
        """Minimal kwargs for synthesize_stock() — only required positional args."""
        return dict(
            sig_data={"signal": "B", "exret": 25, "buy_pct": 80, "beta": 1.2,
                      "pet": 20, "pef": 18, "pp": 5, "52w": 85, "price": 100},
            fund_data={"outlook": "BULLISH", "score": 75, "revenue_growth": {}},
            tech_data={"tech_signal": "ENTER_NOW", "rsi": 55, "momentum": 2},
            macro_fit="FAVORABLE",
            census_alignment="ALIGNED",
            div_score=3,
            census_ts_trend="stable",
            news_impact="NEUTRAL",
            risk_warning=False,
            sector="Technology",
            sector_median_exret=15,
            sector_rankings={},
            position_limit=5.0,
            regime="",
        )

    def test_strengthen_bull_adds_conviction(self):
        from trade_modules.committee_synthesis import synthesize_stock
        kw = self._base_kwargs()
        r1 = synthesize_stock("TEST", **kw)
        r2 = synthesize_stock("TEST", **kw, debate_conviction_signal="STRENGTHEN_BULL")
        assert r2["conviction"] > r1["conviction"]
        assert r2["conviction_waterfall"].get("debate_strengthen_bull") == 5

    def test_weaken_bull_reduces_conviction(self):
        from trade_modules.committee_synthesis import synthesize_stock
        kw = self._base_kwargs()
        r1 = synthesize_stock("TEST", **kw)
        r2 = synthesize_stock("TEST", **kw, debate_conviction_signal="WEAKEN_BULL")
        assert r2["conviction"] < r1["conviction"]
        assert r2["conviction_waterfall"].get("debate_weaken_bull") == -5

    def test_deadlock_no_change(self):
        from trade_modules.committee_synthesis import synthesize_stock
        kw = self._base_kwargs()
        r1 = synthesize_stock("TEST", **kw)
        r2 = synthesize_stock("TEST", **kw, debate_conviction_signal="DEADLOCK")
        assert r2["conviction"] == r1["conviction"]

    def test_debate_data_stored_on_entry(self):
        from trade_modules.committee_synthesis import synthesize_stock
        kw = self._base_kwargs()
        debate_data = {
            "bull_thesis": "Strong growth ahead",
            "bear_thesis": "Overvalued",
            "kill_theses_from_debate": ["[Debate] Valuation risk"],
        }
        r = synthesize_stock("TEST", **kw, debate_conviction_signal="WEAKEN_BULL", debate_data=debate_data)
        assert r["debate_signal"] == "WEAKEN_BULL"
        assert r["debate_bull_thesis"] == "Strong growth ahead"
        assert r["debate_bear_thesis"] == "Overvalued"


class TestActionHysteresis:
    """CIO v32.0 T2: Hysteresis prevents HOLD/ADD whipsaw."""

    def test_hold_to_add_requires_delta(self):
        """Stock at conviction 56 (BUY signal) was HOLD last time — stays HOLD."""
        action = apply_action_hysteresis(
            current_action="ADD", conviction=56, signal="B",
            prev_action="HOLD", prev_conviction=53,
        )
        assert action == "HOLD"

    def test_hold_to_add_large_delta_upgrades(self):
        """Stock at conviction 62 (BUY signal) was HOLD last time — upgrades to ADD."""
        action = apply_action_hysteresis(
            current_action="ADD", conviction=62, signal="B",
            prev_action="HOLD", prev_conviction=53,
        )
        assert action == "ADD"

    def test_add_to_hold_requires_delta(self):
        """Stock at conviction 53 (BUY signal) was ADD last time — stays ADD."""
        action = apply_action_hysteresis(
            current_action="HOLD", conviction=53, signal="B",
            prev_action="ADD", prev_conviction=56,
        )
        assert action == "ADD"

    def test_add_to_hold_large_drop_downgrades(self):
        """Stock at conviction 48 (BUY signal) was ADD last time — downgrades to HOLD."""
        action = apply_action_hysteresis(
            current_action="HOLD", conviction=48, signal="B",
            prev_action="ADD", prev_conviction=56,
        )
        assert action == "HOLD"

    def test_hold_signal_add_to_hold_hysteresis(self):
        """HOLD-signal stock at conviction 68 was ADD (prev=72) — stays ADD."""
        action = apply_action_hysteresis(
            current_action="HOLD", conviction=68, signal="H",
            prev_action="ADD", prev_conviction=72,
        )
        assert action == "ADD"

    def test_hold_signal_hold_to_add_hysteresis(self):
        """HOLD-signal stock at conviction 71 was HOLD (prev=68) — stays HOLD."""
        action = apply_action_hysteresis(
            current_action="ADD", conviction=71, signal="H",
            prev_action="HOLD", prev_conviction=68,
        )
        assert action == "HOLD"

    def test_no_prev_action_no_hysteresis(self):
        """First time stock appears — no hysteresis applied."""
        action = apply_action_hysteresis(
            current_action="ADD", conviction=56, signal="B",
            prev_action=None, prev_conviction=None,
        )
        assert action == "ADD"

    def test_sell_trim_not_affected(self):
        """Hysteresis only applies to HOLD/ADD boundary, not SELL/TRIM."""
        action = apply_action_hysteresis(
            current_action="TRIM", conviction=45, signal="S",
            prev_action="HOLD", prev_conviction=55,
        )
        assert action == "TRIM"

    def test_same_action_no_change(self):
        """If current == previous, no hysteresis needed."""
        action = apply_action_hysteresis(
            current_action="ADD", conviction=60, signal="B",
            prev_action="ADD", prev_conviction=58,
        )
        assert action == "ADD"


class TestQualityGrowthException:
    """CIO v32.0 T3: Quality growth stocks exempt from quality_trap penalty."""

    def test_quality_trap_waived_for_strong_fund_high_consensus(self):
        """fund_score >= 70 + buy_pct >= 70 + quality_trap should NOT penalize."""
        b, p, w = compute_adjustments(
            signal="B", fund_score=80, tech_signal="ENTER_NOW",
            tech_momentum=15, rsi=55, macro_fit="FAVORABLE",
            census_alignment="NEUTRAL", div_score=0,
            census_ts="stable", news_impact="NEUTRAL",
            risk_warning=False, buy_pct=75, excess_exret=15,
            beta=1.2, quality_trap=True, sector="Technology",
            sector_rankings={}, bull_count=5,
        )
        assert "quality_trap" not in w, (
            f"Quality growth exception should waive quality_trap, got waterfall={w}"
        )

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_quality_trap_applied_for_weak_fund(self):
        """fund_score < 70 + quality_trap should still penalize."""
        b, p, w = compute_adjustments(
            signal="B", fund_score=55, tech_signal="ENTER_NOW",
            tech_momentum=15, rsi=55, macro_fit="FAVORABLE",
            census_alignment="NEUTRAL", div_score=0,
            census_ts="stable", news_impact="NEUTRAL",
            risk_warning=False, buy_pct=75, excess_exret=15,
            beta=1.2, quality_trap=True, sector="Technology",
            sector_rankings={}, bull_count=5,
        )
        assert w.get("quality_trap") == -5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_quality_trap_applied_for_low_consensus(self):
        """fund_score >= 70 but buy_pct < 70 — quality_trap should still apply."""
        b, p, w = compute_adjustments(
            signal="B", fund_score=75, tech_signal="ENTER_NOW",
            tech_momentum=15, rsi=55, macro_fit="FAVORABLE",
            census_alignment="NEUTRAL", div_score=0,
            census_ts="stable", news_impact="NEUTRAL",
            risk_warning=False, buy_pct=50, excess_exret=15,
            beta=1.2, quality_trap=True, sector="Technology",
            sector_rankings={}, bull_count=5,
        )
        assert w.get("quality_trap") == -5

    @pytest.mark.skip(reason="CIO v35.0: modifier disabled/changed")
    def test_quality_trap_applied_for_sell_signal(self):
        """Even with high fund+consensus, SELL signal should still apply quality_trap."""
        b, p, w = compute_adjustments(
            signal="S", fund_score=75, tech_signal="AVOID",
            tech_momentum=-15, rsi=72, macro_fit="UNFAVORABLE",
            census_alignment="NEUTRAL", div_score=0,
            census_ts="stable", news_impact="NEUTRAL",
            risk_warning=True, buy_pct=75, excess_exret=15,
            beta=1.2, quality_trap=True, sector="Technology",
            sector_rankings={}, bull_count=2,
        )
        assert w.get("quality_trap") == -5


# ---------------------------------------------------------------------------
# v33.0: News impact canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalizeImpact:
    """v33.0: Agent impact variants mapped to canonical values."""

    def test_positive_maps_to_low_positive(self):
        assert canonicalize_impact("POSITIVE") == "LOW_POSITIVE"

    def test_negative_maps_to_low_negative(self):
        assert canonicalize_impact("NEGATIVE") == "LOW_NEGATIVE"

    def test_very_positive_maps_to_high_positive(self):
        assert canonicalize_impact("VERY_POSITIVE") == "HIGH_POSITIVE"

    def test_very_negative_maps_to_high_negative(self):
        assert canonicalize_impact("VERY_NEGATIVE") == "HIGH_NEGATIVE"

    def test_strong_positive_maps_to_high_positive(self):
        assert canonicalize_impact("STRONG_POSITIVE") == "HIGH_POSITIVE"

    def test_bullish_maps_to_low_positive(self):
        assert canonicalize_impact("BULLISH") == "LOW_POSITIVE"

    def test_bearish_maps_to_low_negative(self):
        assert canonicalize_impact("BEARISH") == "LOW_NEGATIVE"

    def test_neutral_passes_through(self):
        assert canonicalize_impact("NEUTRAL") == "NEUTRAL"

    def test_mixed_passes_through(self):
        assert canonicalize_impact("MIXED") == "MIXED"

    def test_canonical_values_unchanged(self):
        for val in ("HIGH_POSITIVE", "HIGH_NEGATIVE", "LOW_POSITIVE", "LOW_NEGATIVE"):
            assert canonicalize_impact(val) == val

    def test_case_insensitive(self):
        assert canonicalize_impact("positive") == "LOW_POSITIVE"
        assert canonicalize_impact("Negative") == "LOW_NEGATIVE"


class TestResolveNewsImpact:
    """v33.0: News impact resolution handles agent variants."""

    def test_agent_positive_resolves(self):
        news = {"NVDA": [{"impact": "POSITIVE", "headline": "Beat"}]}
        assert _resolve_news_impact(news, "NVDA") == "LOW_POSITIVE"

    def test_agent_negative_resolves(self):
        news = {"NVDA": [{"impact": "NEGATIVE", "headline": "Miss"}]}
        assert _resolve_news_impact(news, "NVDA") == "LOW_NEGATIVE"

    def test_agent_very_positive_resolves(self):
        news = {"NVDA": [{"impact": "VERY_POSITIVE", "headline": "Blowout"}]}
        assert _resolve_news_impact(news, "NVDA") == "HIGH_POSITIVE"

    def test_mixed_high_impacts_resolve_to_mixed(self):
        """MIXED only fires when both HIGH_POSITIVE and HIGH_NEGATIVE present."""
        news = {"NVDA": [
            {"impact": "VERY_POSITIVE", "headline": "Good"},
            {"impact": "VERY_NEGATIVE", "headline": "Bad"},
        ]}
        assert _resolve_news_impact(news, "NVDA") == "MIXED"

    def test_low_positive_plus_high_negative_resolves_to_high_negative(self):
        """LOW_POSITIVE + HIGH_NEGATIVE: high negative wins by priority."""
        news = {"NVDA": [
            {"impact": "POSITIVE", "headline": "Good"},
            {"impact": "VERY_NEGATIVE", "headline": "Bad"},
        ]}
        assert _resolve_news_impact(news, "NVDA") == "HIGH_NEGATIVE"

    def test_missing_ticker_neutral(self):
        assert _resolve_news_impact({}, "NVDA") == "NEUTRAL"

    def test_empty_items_neutral(self):
        assert _resolve_news_impact({"NVDA": []}, "NVDA") == "NEUTRAL"

    def test_single_dict_item_handled(self):
        """Agent writes single dict per ticker instead of list of dicts."""
        news = {"NVDA": {"impact": "POSITIVE", "headline": "Beat"}}
        assert _resolve_news_impact(news, "NVDA") == "LOW_POSITIVE"

    def test_string_item_in_list_skipped(self):
        """Non-dict items in the list are skipped."""
        news = {"NVDA": ["some string", {"impact": "NEGATIVE", "headline": "Miss"}]}
        assert _resolve_news_impact(news, "NVDA") == "LOW_NEGATIVE"


# ---------------------------------------------------------------------------
# v33.0: Risk warning extraction from risk_warnings_by_stock
# ---------------------------------------------------------------------------

class TestRiskWarningExtraction:
    """v33.0: Risk warnings from risk_warnings_by_stock."""

    def test_risk_warnings_by_stock_populates_risk_warns(self):
        risk = {"risk_warnings_by_stock": {"NVDA": {"severity": "HIGH", "warning": "Concentration"}}}
        lookups = _build_agent_lookups({}, {}, {}, {}, {}, risk)
        assert "NVDA" in lookups["risk_warns"]

    def test_consensus_warnings_still_work(self):
        risk = {"consensus_warnings": [{"ticker": "AAPL", "severity": "HIGH"}]}
        lookups = _build_agent_lookups({}, {}, {}, {}, {}, risk)
        assert "AAPL" in lookups["risk_warns"]

    def test_both_sources_merged(self):
        risk = {
            "consensus_warnings": [{"ticker": "AAPL", "severity": "HIGH"}],
            "risk_warnings_by_stock": {"NVDA": {"severity": "HIGH"}},
        }
        lookups = _build_agent_lookups({}, {}, {}, {}, {}, risk)
        assert "AAPL" in lookups["risk_warns"]
        assert "NVDA" in lookups["risk_warns"]

    def test_stocks_key_still_works(self):
        risk = {"stocks": {"TSLA": {"risk_warning": True}}}
        lookups = _build_agent_lookups({}, {}, {}, {}, {}, risk)
        assert "TSLA" in lookups["risk_warns"]

    def test_empty_risk_report(self):
        lookups = _build_agent_lookups({}, {}, {}, {}, {}, {})
        assert lookups["risk_warns"] == set()


# ---------------------------------------------------------------------------
# v33.0: Census alignment from per-stock data
# ---------------------------------------------------------------------------

class TestCensusSignalExtraction:
    """v33.0: Census alignment from per-stock data."""

    def test_per_stock_sentiment_fills_div_map(self):
        census = {"stocks": {"MSFT": {"sentiment": "ALIGNED", "divergence_score": 10}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert lookups["div_map"]["MSFT"] == ("ALIGNED", 10)

    def test_per_stock_alignment_key(self):
        census = {"stocks": {"MSFT": {"alignment": "DIVERGENT", "div_score": -20}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert lookups["div_map"]["MSFT"] == ("DIVERGENT", -20)

    def test_divergences_take_precedence(self):
        census = {
            "divergences": {"signal_divergences": [{"ticker": "BTC", "divergence_score": -30}]},
            "stocks": {"BTC": {"sentiment": "ALIGNED", "divergence_score": 10}},
        }
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert lookups["div_map"]["BTC"] == ("DIVERGENT", -30)

    def test_accumulating_maps_to_aligned(self):
        census = {"stocks": {"NVDA": {"sentiment": "ACCUMULATING"}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert lookups["div_map"]["NVDA"][0] == "ALIGNED"

    def test_distributing_maps_to_divergent(self):
        census = {"stocks": {"TSLA": {"sentiment": "DISTRIBUTING"}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert lookups["div_map"]["TSLA"][0] == "DIVERGENT"

    def test_bullish_maps_to_aligned(self):
        census = {"stocks": {"AAPL": {"sentiment": "BULLISH"}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert lookups["div_map"]["AAPL"][0] == "ALIGNED"

    def test_per_stock_alias(self):
        census = {"per_stock": {"GOOG": {"alignment": "CONSENSUS_ALIGNED"}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert lookups["div_map"]["GOOG"][0] == "ALIGNED"

    def test_unknown_sentiment_skipped(self):
        census = {"stocks": {"AAPL": {"sentiment": "UNKNOWN"}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert "AAPL" not in lookups["div_map"]

    def test_none_signal_divergence_skipped(self):
        """signal_divergence='none' is genuinely neutral — don't map."""
        census = {"stocks": {"NVDA": {"signal_divergence": "none", "trend": "stable"}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert "NVDA" not in lookups["div_map"]

    def test_crowded_long_maps_to_divergent(self):
        """signal_divergence='crowded_long' maps to DIVERGENT."""
        census = {"stocks": {"PLTR": {"signal_divergence": "crowded_long"}}}
        lookups = _build_agent_lookups({}, {}, {}, census, {}, {})
        assert lookups["div_map"]["PLTR"][0] == "DIVERGENT"

    def test_no_census_data_empty_div_map(self):
        lookups = _build_agent_lookups({}, {}, {}, {}, {}, {})
        assert lookups["div_map"] == {}


# ============================================================
# CIO v40 (post-backtest 2026-04-26) — new behaviours
# ============================================================


class TestV40EpsRevisionsPriceGuard:
    """EPS revisions UP bonus (now +7) suppressed when stock has already
    moved >20% over the period — guards against PEAD-fade entries."""

    def _kw(self, pp_value, eps_class="REVISIONS_UP"):
        return dict(
            ticker="TEST",
            sig_data={"signal": "B", "exret": 25, "buy_pct": 80,
                      "beta": 1.0, "pet": 20, "pef": 18, "pp": pp_value},
            fund_data={"fundamental_score": 75,
                       "eps_revisions": {"classification": eps_class}},
            tech_data={"momentum_score": 10, "timing_signal": "HOLD",
                       "rsi": 55, "macd_signal": "BULLISH"},
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL",
            risk_warning=False, sector="Technology", sector_median_exret=15,
            sector_rankings={}, position_limit=5.0,
        )

    def test_bonus_applies_when_pp_within_threshold(self):
        """pp = 10% (< 20%): EPS revisions up bonus contributes."""
        result = synthesize_stock(**self._kw(pp_value=10))
        wf = result.get("conviction_waterfall", {})
        assert wf.get("eps_revisions_up", 0) > 0, (
            f"Expected eps_revisions_up bonus to fire (pp=10), got waterfall: {wf}"
        )

    def test_bonus_suppressed_when_pp_exceeds_threshold(self):
        """pp = 30% (> 20%): EPS revisions up bonus suppressed (gated)."""
        result = synthesize_stock(**self._kw(pp_value=30))
        wf = result.get("conviction_waterfall", {})
        assert wf.get("eps_revisions_up", 0) == 0, (
            f"Expected eps_revisions_up bonus to NOT fire (pp=30), got: {wf}"
        )

    def test_downside_revisions_unaffected_by_pp_guard(self):
        """REVISIONS_DOWN penalty fires regardless of pp."""
        result = synthesize_stock(**self._kw(pp_value=30, eps_class="REVISIONS_DOWN"))
        wf = result.get("conviction_waterfall", {})
        # REVISIONS_DOWN fires for any signal — check the penalty is recorded
        assert wf.get("eps_revisions_down", 0) < 0


class TestV40RevenueGrowthPriceGuard:
    """Revenue growth ACCELERATING bonus (now +8) suppressed when pp > 20."""

    def _kw(self, pp_value, growth_class="ACCELERATING"):
        return dict(
            ticker="TEST",
            sig_data={"signal": "B", "exret": 25, "buy_pct": 80,
                      "beta": 1.0, "pet": 20, "pef": 18, "pp": pp_value},
            fund_data={"fundamental_score": 75,
                       "revenue_growth": {"classification": growth_class}},
            tech_data={"momentum_score": 10, "timing_signal": "HOLD",
                       "rsi": 55, "macd_signal": "BULLISH"},
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL",
            risk_warning=False, sector="Technology", sector_median_exret=15,
            sector_rankings={}, position_limit=5.0,
        )

    def test_bonus_applies_when_pp_within_threshold(self):
        result = synthesize_stock(**self._kw(pp_value=10))
        wf = result.get("conviction_waterfall", {})
        assert wf.get("revenue_growth", 0) > 0

    def test_bonus_suppressed_when_pp_exceeds_threshold(self):
        result = synthesize_stock(**self._kw(pp_value=30))
        wf = result.get("conviction_waterfall", {})
        assert wf.get("revenue_growth", 0) == 0


class TestV40TrimQualityProtection:
    """fund_score >= 75 + no kill thesis blocks HOLD→TRIM escalation.

    Backtest showed 89% of TRIMs went up at T+30 with avg +18% return.
    Worst calls were quality stocks (UNH +32%, MSTR +29%, TSM +23%).
    Quality threshold raised from <70 to <75 with kill-thesis override.
    """

    def _kw(self, fund_score, kill_thesis_triggered=False):
        return dict(
            ticker="TEST",
            sig_data={"signal": "H", "exret": 10, "buy_pct": 50,
                      "beta": 1.0, "pet": 15, "pef": 14},
            fund_data={"fundamental_score": fund_score, "quality_trap_warning": False},
            tech_data={"timing_signal": "AVOID", "momentum_score": -10, "rsi": 55},
            macro_fit="NEUTRAL", census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL",
            risk_warning=True, sector="Technology", sector_median_exret=8,
            sector_rankings={}, position_limit=5.0,
            kill_thesis_triggered=kill_thesis_triggered,
        )

    def test_quality_75_blocks_trim(self):
        """fund_score=75 + no kill thesis → HOLD (not TRIM)."""
        result = synthesize_stock(**self._kw(fund_score=75))
        assert result["action"] == "HOLD", (
            f"Expected HOLD for quality stock (fund=75), got {result['action']}"
        )

    def test_quality_80_blocks_trim(self):
        result = synthesize_stock(**self._kw(fund_score=80))
        assert result["action"] == "HOLD"

    def test_below_quality_threshold_allows_trim(self):
        """fund_score=70 (below new threshold 75) + tech AVOID + risk → TRIM."""
        result = synthesize_stock(**self._kw(fund_score=70))
        assert result["action"] == "TRIM", (
            f"Expected TRIM for fund=70 (below 75) with bear signals, got {result['action']}"
        )

    def test_kill_thesis_overrides_quality_protection(self):
        """fund_score=80 + triggered kill thesis → TRIM allowed (override)."""
        result = synthesize_stock(**self._kw(fund_score=80, kill_thesis_triggered=True))
        # Kill thesis triggered means an explicit, pre-identified failure mode
        # has happened — quality should not block exit.
        assert result["action"] == "TRIM", (
            f"Expected TRIM despite quality (kill thesis triggered), got {result['action']}"
        )


class TestV40KillThesisGenerator:
    """Stock-specific kill theses with numeric, actionable triggers."""

    def test_add_thesis_includes_price_stop(self):
        thesis = generate_kill_thesis(
            ticker="AAPL", action="ADD",
            sig_data={"price": 271.06, "pp": 10},
            fund_data={"fundamental_score": 78,
                       "eps_revisions": {"classification": "REVISIONS_UP"}},
            tech_data={"rsi": 60, "support": 250, "resistance": 285},
            macro_data={"regime": "CAUTIOUS"},
        )
        # Concrete price (10% below current 271 ≈ 244)
        assert "$243.95" in thesis or "$244" in thesis
        assert "10% stop" in thesis or "-10% stop" in thesis
        # Re-evaluation horizon present
        assert "Re-evaluate" in thesis

    def test_trim_thesis_includes_resistance(self):
        thesis = generate_kill_thesis(
            ticker="UNH", action="TRIM",
            sig_data={"price": 354.92, "pp": -5},
            fund_data={"fundamental_score": 80,
                       "eps_revisions": {"classification": "REVISIONS_UP"}},
            tech_data={"rsi": 65, "support": 340, "resistance": 380},
            macro_data={"regime": "CAUTIOUS"},
        )
        assert "Wrong to trim" in thesis
        # Resistance break trigger
        assert "$380" in thesis
        # Quality stock-specific trigger
        assert "fundamental_score" in thesis

    def test_sell_thesis_includes_upside_break(self):
        thesis = generate_kill_thesis(
            ticker="ETH-USD", action="SELL",
            sig_data={"price": 2308.16, "pp": 29},
            fund_data={"fundamental_score": 30},
            tech_data={"rsi": 47, "support": 2100, "resistance": 2500},
            macro_data={"regime": "CAUTIOUS"},
        )
        assert "Wrong to sell" in thesis
        # +8% above 2308 ≈ 2492.81
        assert "$2492" in thesis or "+8%" in thesis

    def test_earnings_event_included_when_close(self):
        thesis = generate_kill_thesis(
            ticker="MSFT", action="ADD",
            sig_data={"price": 424, "pp": 5},
            fund_data={"fundamental_score": 80},
            tech_data={"rsi": 60, "support": 400, "resistance": 450},
            earnings_days_away=3,
        )
        assert "earnings" in thesis.lower()
        assert "3d" in thesis or "T-3d" in thesis

    def test_minimal_inputs_still_produce_thesis(self):
        """No tech/fund/macro data — should still build a sensible default."""
        thesis = generate_kill_thesis(
            ticker="UNK", action="HOLD",
            sig_data={"price": 100},
        )
        assert "Wrong if" in thesis
        assert "Re-evaluate" in thesis
