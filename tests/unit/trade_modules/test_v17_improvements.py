"""Tests for CIO v17 improvement implementations.

Coverage: each finding from the v17 review gets at least one focused
test that exercises the public API and verifies the documented behavior.
"""

import json
import math
from pathlib import Path

import pytest

from trade_modules.agent_sign_calibrator import (
    AGENT_POLARITY,
    calibrate_agent_signs,
    load_applied_signs,
    persist_calibration,
)
from trade_modules.committee_backtester import (
    DEFAULT_SPREAD_BPS_BY_TIER,
    _information_ratio,
    _pearson_r,
    _round_trip_cost_pct,
    _spearman_rho,
)
from trade_modules.committee_scorecard import (
    backfill_action_log_from_concordance,
    log_committee_actions,
)
from trade_modules.committee_synthesis import (
    apply_trim_regime_gate,
    determine_action,
)
from trade_modules.conviction_sizer import (
    apply_small_cap_cap,
    get_small_cap_cap,
    kelly_fraction,
    quarter_kelly_size_pct,
)
from trade_modules.conviction_thresholds import (
    LEGACY_THRESHOLDS,
    compute_rolling_thresholds,
    get_action_thresholds,
    load_thresholds,
    persist_thresholds,
)
from trade_modules.factor_attribution import (
    HORIZONS,
    PRIMARY_HORIZON_DAYS,
    generate_attribution_summary,
)
from trade_modules.waterfall_categories import (
    CATEGORY_ORDER,
    categorize_waterfall,
    render_category_summary,
)


# ── H4.a — Spearman / Pearson / Information Ratio ──────────────────────


class TestRankCorrelations:
    def test_spearman_perfect_positive(self):
        # Strict monotonic increasing ⇒ ρ = +1
        rho = _spearman_rho([1, 2, 3, 4], [10, 20, 30, 40])
        assert rho == pytest.approx(1.0)

    def test_spearman_perfect_negative(self):
        rho = _spearman_rho([1, 2, 3, 4], [40, 30, 20, 10])
        assert rho == pytest.approx(-1.0)

    def test_spearman_zero_correlation(self):
        # Permuted ⇒ near-zero
        rho = _spearman_rho([1, 2, 3, 4], [3, 1, 4, 2])
        assert -0.6 <= (rho or 0) <= 0.6

    def test_spearman_too_few_points(self):
        assert _spearman_rho([1, 2], [3, 4]) is None

    def test_pearson_basic(self):
        r = _pearson_r([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r == pytest.approx(1.0)

    def test_information_ratio_t30(self):
        alphas = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean=3, std≈1.41
        ir = _information_ratio(alphas, 30)
        # mean/std ≈ 2.12, annualized × sqrt(12.17) ≈ 7.4
        assert ir is not None and 6.5 <= ir <= 8.5

    def test_information_ratio_zero_vol(self):
        # No dispersion ⇒ undefined
        assert _information_ratio([2.0, 2.0, 2.0], 30) is None


# ── L2 — Net-of-cost backtests ─────────────────────────────────────────


class TestRoundTripCost:
    def test_default_mid_30d(self):
        c = _round_trip_cost_pct(tier="MID", holding_days=30, financing_apr=0.0)
        # MID spread 18bp × 2 = 0.36%
        assert c == pytest.approx(0.36, abs=0.01)

    def test_crypto_higher(self):
        c = _round_trip_cost_pct(is_crypto=True, holding_days=30)
        assert c >= 1.5

    def test_financing_added(self):
        c0 = _round_trip_cost_pct(tier="LARGE", holding_days=30, financing_apr=0.0)
        c1 = _round_trip_cost_pct(tier="LARGE", holding_days=30, financing_apr=0.06)
        assert c1 > c0

    def test_unknown_tier_default(self):
        # Unknown tier falls back to MID-equivalent (18 bps).
        c = _round_trip_cost_pct(tier="UNKNOWN", holding_days=30)
        assert c == pytest.approx(0.36, abs=0.01)


# ── M2 — TRIM regime gating ─────────────────────────────────────────────


class TestTrimRegimeGate:
    def test_non_trim_passthrough(self):
        action, label = apply_trim_regime_gate("ADD", regime="RISK_ON")
        assert action == "ADD"
        assert label is None

    def test_trim_demoted_in_risk_on_improving(self):
        action, label = apply_trim_regime_gate(
            "TRIM", regime="RISK_ON", regime_momentum="IMPROVING",
            tech_signal="HOLD", rsi=50, position_pct=2.0,
        )
        assert action == "HOLD"
        assert label == "trim_regime_gate"

    def test_kill_thesis_passthrough(self):
        action, _ = apply_trim_regime_gate(
            "TRIM", regime="RISK_ON", regime_momentum="IMPROVING",
            tech_signal="HOLD", rsi=50, position_pct=2.0,
            kill_thesis_triggered=True,
        )
        assert action == "TRIM"

    def test_high_rsi_passthrough(self):
        action, _ = apply_trim_regime_gate(
            "TRIM", regime="RISK_ON", regime_momentum="IMPROVING",
            tech_signal="HOLD", rsi=82, position_pct=2.0,
        )
        assert action == "TRIM"

    def test_oversize_passthrough(self):
        action, _ = apply_trim_regime_gate(
            "TRIM", regime="RISK_ON", regime_momentum="IMPROVING",
            tech_signal="HOLD", rsi=50, position_pct=7.0,
        )
        assert action == "TRIM"

    def test_tech_exit_passthrough(self):
        action, _ = apply_trim_regime_gate(
            "TRIM", regime="RISK_ON", regime_momentum="IMPROVING",
            tech_signal="EXIT_SOON", rsi=50, position_pct=2.0,
        )
        assert action == "TRIM"

    def test_risk_off_no_demotion(self):
        action, _ = apply_trim_regime_gate(
            "TRIM", regime="RISK_OFF", regime_momentum="STABLE",
            tech_signal="HOLD", rsi=50, position_pct=2.0,
        )
        assert action == "TRIM"


# ── R2 — Fundamental composite bonus (verified via determine_action's
#     parent — we just confirm the cap waterfall key exists when stacked)


class TestFundamentalComposite:
    def test_synthesize_stock_caps_fundamental_at_eight(self):
        from trade_modules.committee_synthesis import synthesize_stock

        # Stock with all 4 fundamental positives → raw +13 → capped to +8.
        # Use minimal but legal inputs to exercise compute_adjustments.
        sig_data = {
            "signal": "B", "exret": 25, "buy_pct": 80, "beta": 1.0,
            "pet": 20, "pef": 16, "price": 100, "num_targets": 8,
            "am": 0.5, "analyst_type": "A", "short_interest": 1,
        }
        fund_data = {
            "fundamental_score": 75,
            "quality_trap_warning": False,
            "pe_trajectory": "improving",
            "piotroski": {"f_score": 8},
            "revenue_growth": {"classification": "ACCELERATING"},
            "eps_revisions": {"classification": "REVISIONS_UP"},
            "fcf_quality": {"classification": "STRONG"},
            "cap_tier": "LARGE",
        }
        tech_data = {
            "momentum_score": 30, "timing_signal": "ENTER_NOW",
            "rsi": 60, "macd_signal": "BULLISH",
        }
        out = synthesize_stock(
            ticker="TEST", sig_data=sig_data, fund_data=fund_data,
            tech_data=tech_data, macro_fit="NEUTRAL",
            census_alignment="NEUTRAL", div_score=0,
            census_ts_trend="stable", news_impact="NEUTRAL",
            risk_warning=False, sector="Technology",
            sector_median_exret=10.0, sector_rankings={},
            position_limit=5.0,
        )
        wf = out.get("conviction_waterfall", {}) or {}
        # Fundamental sub-keys still present, but their sum is ≤8.
        sub_sum = sum(
            wf.get(k, 0) for k in (
                "piotroski_quality", "revenue_growth",
                "eps_revisions_up", "fcf_quality_strong",
            )
        )
        assert sub_sum <= 8, f"fund composite uncapped (sum={sub_sum})"
        # The cap key fires when raw > 8.
        assert wf.get("fund_composite_cap", 0) > 0


# ── R3 — Kelly sizer ───────────────────────────────────────────────────


class TestKellyFraction:
    def test_kelly_zero_when_negative_edge(self):
        assert kelly_fraction(-1.0, 5.0, risk_free_pct=0.0) == 0.0

    def test_kelly_zero_when_no_vol(self):
        assert kelly_fraction(5.0, 0.0) == 0.0

    def test_kelly_basic(self):
        # μ=10%, σ=20% → f* = 0.10/0.04 = 2.5
        f = kelly_fraction(10.0, 20.0, risk_free_pct=0.0)
        assert f == pytest.approx(2.5, abs=0.01)

    def test_quarter_kelly_caps_at_5pct(self):
        out = quarter_kelly_size_pct(80, atr_pct_daily=1.5)
        assert out["size_pct"] <= 5.0
        assert isinstance(out["kelly_f_full"], float)
        assert out["horizon_days"] == 30

    def test_quarter_kelly_low_conviction(self):
        out = quarter_kelly_size_pct(40, atr_pct_daily=1.5)
        # Conv <45 → expected return 0 → size 0
        assert out["size_pct"] == 0.0


# ── R4 — Census band tightening (verified via output keys) ─────────────


class TestCensusBand:
    def test_tight_band_full_bonus(self):
        # Direct test: invoke compute_adjustments with div_score in [-10,10].
        from trade_modules.committee_synthesis import compute_adjustments

        bonuses, penalties, wf = compute_adjustments(
            signal="B", fund_score=70, tech_signal="ENTER_NOW",
            tech_momentum=20, rsi=55, macro_fit="NEUTRAL",
            census_alignment="ALIGNED", div_score=5,
            census_ts="stable", news_impact="NEUTRAL", risk_warning=False,
            buy_pct=70, excess_exret=0, beta=1.0,
            quality_trap=False, sector="Technology",
            sector_rankings={}, bull_count=4,
        )
        assert wf.get("census_alignment") == 3  # CIO v35.0: reduced from 5 to 3

    def test_loose_band_partial_bonus(self):
        from trade_modules.committee_synthesis import compute_adjustments

        bonuses, penalties, wf = compute_adjustments(
            signal="B", fund_score=70, tech_signal="ENTER_NOW",
            tech_momentum=20, rsi=55, macro_fit="NEUTRAL",
            census_alignment="ALIGNED", div_score=15,
            census_ts="stable", news_impact="NEUTRAL", risk_warning=False,
            buy_pct=70, excess_exret=0, beta=1.0,
            quality_trap=False, sector="Technology",
            sector_rankings={}, bull_count=4,
        )
        # In [10, 20] → +2, not +5
        assert wf.get("census_alignment") == 2

    def test_strong_contrarian_eight(self):
        from trade_modules.committee_synthesis import compute_adjustments

        bonuses, penalties, wf = compute_adjustments(
            signal="B", fund_score=70, tech_signal="ENTER_NOW",
            tech_momentum=20, rsi=55, macro_fit="NEUTRAL",
            census_alignment="DIVERGENT", div_score=-30,
            census_ts="stable", news_impact="NEUTRAL", risk_warning=False,
            buy_pct=70, excess_exret=0, beta=1.0,
            quality_trap=False, sector="Technology",
            sector_rankings={}, bull_count=4,
        )
        assert wf.get("census_alignment") == 5  # CIO v35.0: reduced from 8 to 5


# ── H4.b — Rolling percentile thresholds ───────────────────────────────


class TestConvictionThresholds:
    def test_legacy_used_when_no_state(self):
        thr = get_action_thresholds("B", rolling=None)
        assert thr["add_pct"] == LEGACY_THRESHOLDS["B"]["add_pct"]
        assert thr["source"] == "legacy"

    def test_rolling_overrides_when_present(self):
        rolling = {"B": {"add_pct": 62, "trim_pct": 0,
                          "n": 50, "source": "rolling"}}
        thr = get_action_thresholds("B", rolling=rolling)
        assert thr["add_pct"] == 62
        assert thr["source"] == "rolling"

    def test_compute_rolling_with_data(self):
        # 22 BUY-signal observations spanning 50-80 conviction.
        history = [
            {"date": "2026-04-01", "concordance": [
                {"signal": "B", "conviction": 50 + i} for i in range(22)
            ]},
        ]
        thr = compute_rolling_thresholds(history, lookback_snapshots=8, min_per_signal=10)
        assert thr["B"]["source"] == "rolling"
        assert "add_pct" in thr["B"]
        # 50th percentile of 50-71 ≈ 60
        assert 58 <= thr["B"]["add_pct"] <= 64

    def test_compute_rolling_below_min_uses_legacy(self):
        history = [{"date": "2026-04-01", "concordance": [
            {"signal": "B", "conviction": 50}
        ]}]
        thr = compute_rolling_thresholds(history, min_per_signal=10)
        assert thr["B"]["source"] == "legacy"

    def test_persist_and_load(self, tmp_path):
        path = tmp_path / "thresholds.json"
        sample = {"B": {"add_pct": 60.0, "source": "rolling"}}
        persist_thresholds(sample, path=path)
        loaded = load_thresholds(path=path)
        assert loaded is not None
        assert loaded["B"]["add_pct"] == 60.0

    def test_load_stale_returns_none(self, tmp_path):
        # Write a file with an old timestamp.
        path = tmp_path / "thresholds.json"
        path.write_text(json.dumps({
            "generated_at": "2020-01-01T00:00:00",
            "thresholds": {"B": {"add_pct": 60.0}},
        }))
        assert load_thresholds(path=path, max_age_days=7) is None

    def test_determine_action_uses_rolling(self):
        rolling = {"B": {"add_pct": 65, "source": "rolling"}}
        # Conviction 60: legacy says ADD (≥55), rolling says HOLD (<65).
        action = determine_action(60, "B", "ENTER_NOW", False, thresholds=rolling)
        assert action == "HOLD"


# ── N1 — Portfolio-size-aware small-cap cap ─────────────────────────────


class TestSmallCapCap:
    def test_cap_for_small_book(self):
        assert get_small_cap_cap(50_000) == 25.0

    def test_cap_for_medium_book(self):
        assert get_small_cap_cap(250_000) == 15.0

    def test_cap_for_large_book(self):
        assert get_small_cap_cap(1_000_000) == 8.0

    def test_cap_for_unknown(self):
        assert get_small_cap_cap(None) == 25.0

    def test_apply_scales_excess(self):
        positions = [
            {"ticker": "X", "cap_tier": "SMALL", "position_size": 10.0},
            {"ticker": "Y", "cap_tier": "MICRO", "position_size": 10.0},
            {"ticker": "Z", "cap_tier": "MEGA", "position_size": 5.0},
        ]
        # Large book: cap 8%, no current exposure, total small/micro = 20 → scale = 8/20 = 0.4
        out = apply_small_cap_cap(positions, current_small_cap_pct=0.0,
                                  portfolio_value_eur=1_000_000)
        small = next(p for p in out if p["ticker"] == "X")
        mega = next(p for p in out if p["ticker"] == "Z")
        assert small["small_cap_cap_applied"] is True
        assert small["position_size"] == pytest.approx(4.0)
        assert mega.get("small_cap_cap_applied") is None  # MEGA untouched

    def test_apply_no_scaling_when_within_cap(self):
        positions = [{"ticker": "X", "cap_tier": "SMALL", "position_size": 5.0}]
        out = apply_small_cap_cap(positions, current_small_cap_pct=0.0,
                                  portfolio_value_eur=1_000_000)
        assert out[0]["small_cap_cap_applied"] is False
        assert out[0]["position_size"] == 5.0


# ── L1 — Waterfall categorisation ──────────────────────────────────────


class TestWaterfallCategories:
    def test_categorize_basic(self):
        wf = {"piotroski_quality": 3, "rsi_overbought": -8, "kill_thesis": -15}
        cats = categorize_waterfall(wf)
        assert "Quality" in cats and cats["Quality"]["total"] == 3
        assert "Momentum" in cats and cats["Momentum"]["total"] == -8
        assert "Risk" in cats and cats["Risk"]["total"] == -15

    def test_category_order(self):
        wf = {"kill_thesis": -15, "piotroski_quality": 3, "macro_sector": -5}
        cats = categorize_waterfall(wf)
        keys = list(cats.keys())
        # Quality must come before Macro must come before Risk.
        assert keys.index("Quality") < keys.index("Macro") < keys.index("Risk")

    def test_unknown_falls_to_other(self):
        wf = {"foo_unknown": 7}
        cats = categorize_waterfall(wf)
        assert "Other" in cats
        assert cats["Other"]["total"] == 7

    def test_render_text(self):
        wf = {"piotroski_quality": 3, "kill_thesis": -15}
        s = render_category_summary(wf, text_only=True)
        assert "Quality +3" in s
        assert "Risk -15" in s

    def test_render_html(self):
        wf = {"piotroski_quality": 3}
        s = render_category_summary(wf)
        assert "Quality" in s
        assert "<div" in s
        assert "+3" in s

    def test_underscore_keys_skipped(self):
        wf = {"_internal": 99, "piotroski_quality": 3}
        cats = categorize_waterfall(wf)
        assert sum(b["total"] for b in cats.values()) == 3

    def test_string_values_are_coerced(self):
        # Regression: values can arrive as strings after JSON round-trip
        # via concordance.json. Must not crash; must coerce or skip.
        wf = {"piotroski_quality": "3", "rsi_overbought": "-8",
              "garbage": "not_a_number"}
        cats = categorize_waterfall(wf)
        assert cats["Quality"]["total"] == 3
        assert cats["Momentum"]["total"] == -8
        # "garbage" key should be silently skipped, not crash.
        assert "garbage" not in (cats.get("Other", {}).get("modifiers") or {})

    def test_render_handles_string_values(self):
        wf = {"piotroski_quality": "3", "kill_thesis": "-15"}
        # Should not raise even though values are strings.
        s = render_category_summary(wf, text_only=True)
        assert "Quality +3" in s
        assert "Risk -15" in s


# ── H1 — Agent sign calibrator (shadow mode) ────────────────────────────


class TestAgentSignCalibrator:
    def test_no_history_returns_no_history(self, tmp_path):
        out = calibrate_agent_signs(
            forward_returns={}, history_dir=tmp_path,
            horizon="T+30", lookback_days=60,
        )
        assert out["status"] == "no_history"

    def test_inverted_macro_flagged(self, tmp_path):
        # Build a tiny synthetic history where macro_fit=FAVORABLE
        # consistently has α<0 across two views.
        from datetime import datetime, timedelta
        today = datetime.now()
        for i in range(3):
            d = (today - timedelta(days=i * 7)).strftime("%Y-%m-%d")
            (tmp_path / f"concordance-{d}.json").write_text(json.dumps({
                "date": d,
                "concordance": [
                    {"ticker": f"T{j}", "macro_fit": "FAVORABLE",
                     "tech_signal": "ENTER_NOW", "fund_view": "BUY",
                     "census": "ALIGNED", "news_impact": "NEUTRAL"}
                    for j in range(8)
                ],
            }))
        # Forward returns: macro=FAVORABLE → all -2% alpha.
        forward = {}
        for i in range(3):
            d = (today - timedelta(days=i * 7)).strftime("%Y-%m-%d")
            for j in range(8):
                forward[f"T{j}:{d}"] = {"T+30_alpha": -2.0}
        out = calibrate_agent_signs(
            forward_returns=forward, history_dir=tmp_path,
            horizon="T+30", lookback_days=60,
        )
        assert out["status"] == "ok"
        macro = out["agents"].get("macro", {})
        # FAVORABLE has polarity +1; if all alphas <0, P(α>0)=0 < 0.45 → INVERTED
        # but only if the count crosses min_evidence (10).
        assert macro["evidence_total"] == 24
        # FAVORABLE view should be flagged INVERTED.
        assert macro["views"]["FAVORABLE"]["verdict"] == "INVERTED"

    def test_persist_shadow_mode_does_not_apply(self, tmp_path):
        cal = {
            "status": "ok",
            "agents": {
                "macro": {"verdict": "INVERTED", "recommended_sign": -1,
                          "evidence_total": 50, "flip_signal_count": 2},
            },
        }
        path = tmp_path / "calibration.json"
        persist_calibration(cal, path=path, enabled=False)
        # Even with consecutive=2 inverted, shadow mode should NOT apply.
        applied = load_applied_signs(path=path)
        assert applied == {}  # SHADOW mode → no signs applied

    def test_persist_auto_mode_applies_after_two(self, tmp_path):
        cal = {
            "status": "ok",
            "agents": {
                "macro": {"verdict": "INVERTED", "recommended_sign": -1,
                          "evidence_total": 50, "flip_signal_count": 2},
            },
        }
        path = tmp_path / "calibration.json"
        # First run: consecutive_inverted = 1 → not yet applied
        persist_calibration(cal, path=path, enabled=True)
        # Second run: consecutive_inverted = 2 → applied
        persist_calibration(cal, path=path, enabled=True)
        applied = load_applied_signs(path=path)
        assert applied.get("macro") == -1


# ── H2 — log_committee_actions waterfall enrichment + backfill ──────────


class TestActionLogWaterfall:
    def test_log_enriches_from_concordance(self, tmp_path):
        log = tmp_path / "actions.jsonl"
        actions = [{"ticker": "AAA", "action": "ADD", "conviction": 60,
                    "signal": "B", "price": 100}]
        concordance = [{"ticker": "AAA",
                        "conviction_waterfall": {"piotroski_quality": 3}}]
        log_committee_actions(
            "2026-04-20", actions, log_path=log, concordance=concordance,
        )
        line = log.read_text().strip()
        entry = json.loads(line)
        assert entry["conviction_waterfall"] == {"piotroski_quality": 3}

    def test_log_assertion_fires_when_threshold_breached(self, tmp_path):
        log = tmp_path / "actions.jsonl"
        actions = [
            {"ticker": "AAA", "action": "ADD", "conviction": 60,
             "signal": "B", "price": 100},
            {"ticker": "BBB", "action": "ADD", "conviction": 60,
             "signal": "B", "price": 100},
        ]
        # No concordance, no waterfall → 100% missing → above 10% threshold.
        with pytest.raises(AssertionError):
            log_committee_actions(
                "2026-04-20", actions, log_path=log,
                require_waterfall=True,
            )

    def test_backfill_from_concordance_archive(self, tmp_path):
        history = tmp_path / "history"
        history.mkdir()
        log = tmp_path / "action_log.jsonl"
        # One archived concordance with two stocks, both with waterfalls.
        (history / "concordance-2026-04-19.json").write_text(json.dumps([
            {"ticker": "AAA", "action": "ADD", "conviction": 60,
             "signal": "B", "price": 100,
             "conviction_waterfall": {"piotroski_quality": 3}},
            {"ticker": "BBB", "action": "HOLD", "conviction": 50,
             "signal": "H", "price": 200,
             "conviction_waterfall": {"census_alignment": 5}},
        ]))
        n = backfill_action_log_from_concordance(
            history_dir=history, log_path=log, overwrite=True,
        )
        assert n == 2
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert {r["ticker"] for r in rows} == {"AAA", "BBB"}
        assert all(r.get("conviction_waterfall") for r in rows)


# ── H7 — Factor attribution at T+30 ────────────────────────────────────


class TestFactorAttributionT30:
    def test_horizons_include_t30(self):
        assert 30 in HORIZONS
        assert PRIMARY_HORIZON_DAYS == 30

    def test_attribution_summary_prefers_t30(self):
        attribution = {
            "factors": {
                "demo": {
                    "fires_total": 50,
                    "fires_pct": 50.0,
                    "description": "test",
                    "category": "test",
                    "by_action": {
                        "ADD": {
                            "T+7": {"evaluated": 30, "hits": 20,
                                    "hit_rate": 66.7, "avg_alpha": 1.5,
                                    "avg_return": 1.5},
                            "T+30": {"evaluated": 25, "hits": 18,
                                     "hit_rate": 72.0, "avg_alpha": 3.0,
                                     "avg_return": 3.5},
                        },
                    },
                },
            },
        }
        summary = generate_attribution_summary(attribution)
        assert summary
        first = summary[0]
        assert first["primary_horizon"] == "T+30"
        assert first["primary_hit_rate"] == 72.0
        assert first["primary_avg_alpha"] == 3.0

    def test_attribution_summary_fallback_when_no_t30(self):
        attribution = {
            "factors": {
                "demo": {
                    "fires_total": 5,
                    "fires_pct": 5.0,
                    "description": "test",
                    "category": "test",
                    "by_action": {
                        "ADD": {
                            "T+7": {"evaluated": 5, "hits": 4,
                                    "hit_rate": 80.0, "avg_alpha": 2.0,
                                    "avg_return": 2.0},
                        },
                    },
                },
            },
        }
        summary = generate_attribution_summary(attribution)
        assert summary
        # No T+30 → primary fields stay None, signal still derived from best.
        assert summary[0]["primary_horizon"] is None
        assert summary[0]["best_hit_rate"] == 80.0


# ── Sprint S1.3 — Weekly backtest helpers (refresh_*) ───────────────────


class TestWeeklyBacktestHelpers:
    def test_refresh_rolling_thresholds_runs(self, tmp_path, monkeypatch):
        # Simulate a tiny history dir.
        hist = tmp_path / "history"
        hist.mkdir()
        (hist / "concordance-2026-04-15.json").write_text(json.dumps({
            "concordance": [
                {"signal": "B", "conviction": 50 + i, "ticker": f"T{i}"}
                for i in range(25)
            ],
        }))
        # Patch the default threshold path so the test doesn't pollute the
        # user's real cache.
        from trade_modules import conviction_thresholds as ct
        monkeypatch.setattr(ct, "DEFAULT_THRESHOLDS_PATH",
                            tmp_path / "thresholds.json")
        # Patch CommitteeBacktester to use our tmp dir.
        from trade_modules.committee_backtester import CommitteeBacktester
        monkeypatch.setattr(
            CommitteeBacktester, "__init__",
            lambda self, log_dir=None: setattr(self, "log_dir", tmp_path)
            or setattr(self, "history", []) or setattr(self, "forward_returns", {})
            or None,
        )

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from run_weekly_backtest import refresh_rolling_thresholds  # type: ignore
        thresholds = refresh_rolling_thresholds()
        # Either rolling (sufficient evidence) or empty (no history) — both legal.
        assert isinstance(thresholds, dict)

    def test_refresh_agent_sign_calibrator_shadow(self, tmp_path, monkeypatch):
        from trade_modules import agent_sign_calibrator as asc
        monkeypatch.setattr(asc, "DEFAULT_CALIBRATOR_PATH",
                            tmp_path / "calibration.json")
        from trade_modules.committee_backtester import CommitteeBacktester
        monkeypatch.setattr(
            CommitteeBacktester, "__init__",
            lambda self, log_dir=None: setattr(self, "log_dir", tmp_path)
            or setattr(self, "history", []) or setattr(self, "forward_returns", {})
            or None,
        )

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from run_weekly_backtest import refresh_agent_sign_calibrator  # type: ignore
        cal = refresh_agent_sign_calibrator(shadow_mode=True)
        # No history → returns {} per docstring.
        assert cal == {}


# ── Sprint S2.1 — HTML panel for sign-calibrator (smoke render) ─────────


class TestSignCalibratorHTMLPanel:
    def test_panel_renders_when_evidence_present(self):
        from trade_modules.committee_html import generate_report_html

        sign_cal_block = {
            "mode": "SHADOW",
            "horizon": "T+30",
            "agents": {
                "macro": {"verdict": "INVERTED", "evidence_total": 50,
                          "consecutive_inverted": 2, "applied": False},
                "fundamental": {"verdict": "OK", "evidence_total": 50,
                                "consecutive_inverted": 0, "applied": False},
            },
        }
        synth = {
            "version": "v17", "regime": "RISK_ON",
            "macro_score": 25, "rotation_phase": "EARLY_CYCLE",
            "verdict": "OK", "narrative": "Test",
            "concordance": [],
            "agent_sign_calibration": sign_cal_block,
        }
        html = generate_report_html(
            synth=synth, fund={}, tech={}, macro={}, census={}, news={},
            opps={}, risk={}, date_str="2026-04-20", mode="full",
        )
        assert "Agent Sign Calibrator" in html
        assert "INVERTED" in html
        assert "macro" in html.lower()

    def test_panel_hidden_when_no_evidence(self):
        from trade_modules.committee_html import generate_report_html
        synth = {
            "version": "v17", "regime": "RISK_ON",
            "macro_score": 25, "rotation_phase": "EARLY_CYCLE",
            "verdict": "OK", "narrative": "Test",
            "concordance": [],
            "agent_sign_calibration": {
                "mode": "SHADOW",
                "agents": {
                    "macro": {"verdict": "INSUFFICIENT_DATA",
                              "evidence_total": 0,
                              "consecutive_inverted": 0, "applied": False},
                },
            },
        }
        html = generate_report_html(
            synth=synth, fund={}, tech={}, macro={}, census={}, news={},
            opps={}, risk={}, date_str="2026-04-20", mode="full",
        )
        # Panel should NOT render when nothing has any evidence.
        assert "Agent Sign Calibrator" not in html

    def test_panel_hidden_in_daily_mode(self):
        from trade_modules.committee_html import generate_report_html
        synth = {
            "version": "v17", "regime": "RISK_ON",
            "macro_score": 25, "rotation_phase": "EARLY_CYCLE",
            "verdict": "OK", "narrative": "Test",
            "concordance": [],
            "agent_sign_calibration": {
                "mode": "SHADOW",
                "agents": {
                    "macro": {"verdict": "INVERTED", "evidence_total": 50,
                              "consecutive_inverted": 2, "applied": False},
                },
            },
        }
        html = generate_report_html(
            synth=synth, fund={}, tech={}, macro={}, census={}, news={},
            opps={}, risk={}, date_str="2026-04-20", mode="daily",
        )
        # Daily/digest mode skips the panel to keep email lightweight.
        assert "Agent Sign Calibrator" not in html

    def test_render_survives_string_waterfall_values(self):
        # Regression: when concordance comes back from JSON round-trip,
        # waterfall values can be strings. The renderer must not crash on
        # `abs(str)` or `int + str`.
        from trade_modules.committee_html import generate_report_html
        synth = {
            "version": "v17", "regime": "RISK_ON",
            "macro_score": 25, "rotation_phase": "EARLY_CYCLE",
            "verdict": "OK", "narrative": "Test",
            "concordance": [
                {
                    "ticker": "TEST", "action": "ADD", "conviction": 60,
                    "signal": "B", "fund_view": "BUY", "tech_signal": "ENTER_NOW",
                    "macro_fit": "FAVORABLE", "census": "ALIGNED",
                    "news_impact": "NEUTRAL", "rsi": 55, "sector": "Technology",
                    "exret": 20.0, "buy_pct": 80, "beta": 1.0,
                    "conviction_waterfall": {
                        "piotroski_quality": "3",   # string!
                        "rsi_overbought": "-8",      # string!
                        "kill_thesis": "-15",        # string!
                        "garbage_modifier": "n/a",   # non-numeric
                    },
                },
            ],
        }
        # Must not raise.
        html = generate_report_html(
            synth=synth, fund={}, tech={}, macro={}, census={}, news={},
            opps={}, risk={}, date_str="2026-04-20", mode="full",
        )
        assert "TEST" in html


# ── Sprint S1.3 — agent_sign_calibrator legacy-format normalization ─────


class TestSignCalibratorLegacyNormalization:
    def test_handles_legacy_stocks_dict_format(self, tmp_path):
        # Some old archives store concordance as {"stocks": {ticker: row}}.
        # The calibrator must normalize to a list of dicts before iterating;
        # otherwise it crashes with "str has no attribute 'get'".
        from datetime import datetime, timedelta
        d = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        (tmp_path / f"concordance-{d}.json").write_text(json.dumps({
            "date": d,
            "stocks": {
                "AAA": {"macro_fit": "FAVORABLE", "tech_signal": "ENTER_NOW",
                        "fund_view": "BUY", "census": "ALIGNED",
                        "news_impact": "POSITIVE"},
                "BBB": {"macro_fit": "UNFAVORABLE", "tech_signal": "AVOID",
                        "fund_view": "SELL", "census": "DIVERGENT",
                        "news_impact": "NEGATIVE"},
            },
        }))
        forward = {
            f"AAA:{d}": {"T+30_alpha": +1.5},
            f"BBB:{d}": {"T+30_alpha": -2.0},
        }
        out = calibrate_agent_signs(
            forward_returns=forward, history_dir=tmp_path,
            horizon="T+30", lookback_days=60,
        )
        # Critical: must NOT raise. Status should be 'ok' (history loaded).
        assert out["status"] == "ok"
        assert "agents" in out
