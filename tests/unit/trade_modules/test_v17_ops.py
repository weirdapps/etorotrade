"""Tests for CIO v17 operational improvements (ops #1-#8).

Coverage:
- price_cache: cache freshness, fetch+persist roundtrip, refresh logic
- kill_thesis_auditor: classification, pattern grouping, no-data paths
- conviction_cells: cell aggregation, Spearman, confidence multiplier
- debate_scorecard: signal aggregation, control comparison, verdict
- post_mortem: drawdown detection, lesson generation, dedup
- bayesian_conviction: prior/posterior math, likelihood building, shadow
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trade_modules.bayesian_conviction import (
    PRIOR_HITS, PRIOR_MISSES,
    _conviction_to_prior, _logit, _prior_to_conviction, _sigmoid,
    bayesian_posterior, compute_likelihoods, persist_likelihoods,
    load_likelihoods, shadow_score_concordance,
)
from trade_modules.conviction_cells import (
    cell_confidence_multiplier, compute_cells, persist_cells, load_cells,
    _cap_tier, _consensus_band, _conv_band,
)
from trade_modules.debate_scorecard import (
    compute_debate_scorecard, persist_scorecard, load_scorecard,
)
from trade_modules.kill_thesis_auditor import (
    audit_triggered_theses, FALSE_RISE_PCT, TRUE_DROP_PCT,
)
from trade_modules.post_mortem import (
    DRAWDOWN_TRIGGER_PCT,
    append_lessons, detect_post_mortems, load_recent_lessons,
    summarise_for_committee,
)
from trade_modules.price_cache import (
    DEFAULT_CACHE_DIR, _cache_path, cache_stats,
    fetch_and_cache, freshness_status, load_prices, refresh_if_stale,
    write_health_report,
)


# ── #1 Weekly Backtest brittleness — verified via unit tests on
#     build_report's behavior with empty signal_summary ────────────────


class TestWeeklyBacktestSoftFail:
    def test_build_report_returns_zero_when_signal_summary_empty(self):
        # We don't import the script directly because it touches sys.path
        # at module load. Instead, test the contract: headline must
        # always carry buy_count_t7 (int 0 not None) when no data.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from run_weekly_backtest import build_report  # type: ignore

        report = build_report(
            signal_summary={},
            committee_result={"status": "no_data"},
            scorecard={},
            calibration={},
        )
        assert "headline" in report
        h = report["headline"]
        assert h["buy_count_t7"] == 0
        assert h["sell_count_t7"] == 0
        # The new status field flags the "no data" condition.
        assert h["signal_backtest_status"] == "no_data"

    def test_build_report_passes_through_when_signal_summary_present(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from run_weekly_backtest import build_report  # type: ignore

        report = build_report(
            signal_summary={"t7_B": {"count": 50, "hit_rate": 60.0}},
            committee_result={},
            scorecard={},
            calibration={},
        )
        h = report["headline"]
        assert h["buy_count_t7"] == 50
        assert h["buy_hit_rate_t7"] == 60.0
        assert h["signal_backtest_status"] == "ok"


# ── #8 Price Cache ──────────────────────────────────────────────────────


class TestPriceCache:
    def test_freshness_status_missing(self, tmp_path):
        assert freshness_status("AAPL", tmp_path) == "missing"

    def test_freshness_fresh_after_write(self, tmp_path):
        # Build a tiny parquet with today's bar.
        df = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime([datetime.now() - timedelta(days=1),
                                   datetime.now()]),
        )
        df.to_parquet(_cache_path("AAPL", tmp_path))
        assert freshness_status("AAPL", tmp_path) == "fresh"

    def test_freshness_very_stale(self, tmp_path):
        df = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime([datetime.now() - timedelta(days=14)]),
        )
        df.to_parquet(_cache_path("AAPL", tmp_path))
        assert freshness_status("AAPL", tmp_path) == "very_stale"

    def test_load_prices_returns_dict(self, tmp_path):
        df = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime([datetime.now() - timedelta(days=1),
                                   datetime.now()]),
        )
        df.to_parquet(_cache_path("AAPL", tmp_path))
        out = load_prices(["AAPL", "MISSING"], cache_dir=tmp_path)
        assert "AAPL" in out
        assert "MISSING" not in out
        assert len(out["AAPL"]) == 2

    def test_cache_stats(self, tmp_path):
        # Add one fresh entry
        df = pd.DataFrame({"Close": [100.0]},
                          index=pd.to_datetime([datetime.now()]))
        df.to_parquet(_cache_path("AAPL", tmp_path))
        # Add one stale
        df2 = pd.DataFrame({"Close": [100.0]},
                           index=pd.to_datetime([datetime.now() - timedelta(days=4)]))
        df2.to_parquet(_cache_path("MSFT", tmp_path))
        stats = cache_stats(tmp_path)
        assert stats["total"] == 2
        assert stats["fresh"] == 1
        assert stats["stale"] == 1

    def test_write_health_report(self, tmp_path):
        out = write_health_report(tmp_path)
        assert out.exists()
        data = json.load(open(out))
        assert "stats" in data
        assert "generated_at" in data


# ── #2 Kill Thesis Auditor ──────────────────────────────────────────────


class TestKillThesisAuditor:
    def test_no_file_returns_status(self, tmp_path):
        result = audit_triggered_theses(
            kill_theses_path=tmp_path / "missing.json",
            output_path=tmp_path / "out.json",
        )
        assert result.get("status") == "no_kill_theses_file"

    def test_no_triggered_returns_status(self, tmp_path):
        kt_path = tmp_path / "kt.json"
        kt_path.write_text(json.dumps({"triggered_theses": []}))
        result = audit_triggered_theses(
            kill_theses_path=kt_path,
            output_path=tmp_path / "out.json",
            use_price_cache=False,
        )
        assert result.get("status") == "no_triggered_theses"

    def test_classifies_true_positive(self, tmp_path, monkeypatch):
        kt_path = tmp_path / "kt.json"
        # Trigger 14 days ago at price 100; current price 80 → -20% drop = TRUE
        rec_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        kt_path.write_text(json.dumps({
            "triggered_theses": [{
                "ticker": "TEST",
                "thesis": "Will fail if X",
                "trigger_date": rec_date,
                "committee_date": rec_date,
            }],
        }))

        # Patch price cache to return synthetic data
        from trade_modules import price_cache as pc

        def _fake_load(tickers, cache_dir=None, allow_stale=True):
            df = pd.DataFrame(
                {"Close": [100.0, 80.0]},
                index=pd.to_datetime([rec_date, datetime.now().strftime("%Y-%m-%d")]),
            )
            return {"TEST": df}

        monkeypatch.setattr(pc, "load_prices", _fake_load)

        result = audit_triggered_theses(
            kill_theses_path=kt_path,
            output_path=tmp_path / "out.json",
            use_price_cache=True,
        )
        assert result["total_audited"] == 1
        assert result["summary"]["true_positive_count"] == 1


# ── #4 Per-Cell Conviction Calibration ─────────────────────────────────


class TestConvictionCells:
    def test_cap_tier_thresholds(self):
        assert _cap_tier(800) == "MEGA"
        assert _cap_tier(200) == "LARGE"
        assert _cap_tier(50) == "MID"
        assert _cap_tier(5) == "SMALL"
        assert _cap_tier(1) == "MICRO"
        assert _cap_tier(None) == "UNKNOWN"

    def test_consensus_band(self):
        assert _consensus_band(95) == "EXTREME"
        assert _consensus_band(80) == "HIGH"
        assert _consensus_band(65) == "MODERATE"
        assert _consensus_band(40) == "LOW"
        assert _consensus_band(None) == "UNKNOWN"

    def test_compute_cells_with_strong_signal(self):
        # Build a small history where conviction monotonically predicts
        # alpha within a single cell (B|MEGA|RISK_ON|HIGH).
        history = [
            {
                "date": "2026-04-01",
                "regime": "RISK_ON",
                "concordance": [
                    {"ticker": f"T{i}", "signal": "B", "conviction": 50 + i,
                     "market_cap_str": "600B", "buy_pct": 80}
                    for i in range(10)
                ],
            },
        ]
        forward = {f"T{i}:2026-04-01": {"T+30_alpha": float(i)} for i in range(10)}
        cells = compute_cells(history, forward, horizon="T+30", min_cell_n=3)
        # There should be one cell key
        cell_keys = list(cells["cells"].keys())
        assert len(cell_keys) == 1
        sp = cells["cells"][cell_keys[0]]["spearman"]
        assert sp == pytest.approx(1.0)
        assert cells["cells"][cell_keys[0]]["n"] == 10
        assert cells["cells"][cell_keys[0]]["verdict"] == "EVIDENCE"

    def test_cell_confidence_multiplier_high_ic(self):
        cells_data = {
            "cells": {
                "B|MEGA|RISK_ON|HIGH": {
                    "n": 20, "spearman": 0.45, "verdict": "EVIDENCE",
                },
            },
        }
        m = cell_confidence_multiplier("B", 600, "RISK_ON", 80, cells_data)
        assert m == 1.2

    def test_cell_confidence_multiplier_low_ic(self):
        cells_data = {
            "cells": {
                "B|MEGA|RISK_ON|HIGH": {
                    "n": 20, "spearman": 0.05, "verdict": "EVIDENCE",
                },
            },
        }
        m = cell_confidence_multiplier("B", 600, "RISK_ON", 80, cells_data)
        assert m == 0.7

    def test_cell_confidence_multiplier_no_data(self):
        m = cell_confidence_multiplier("B", 600, "RISK_ON", 80, None)
        assert m == 1.0

    def test_persist_and_load(self, tmp_path):
        cells = {"horizon": "T+30", "cells": {}, "high_ic_cells": []}
        path = tmp_path / "cells.json"
        persist_cells(cells, path=path)
        loaded = load_cells(path=path)
        assert loaded == cells


# ── #5 Debate Scorecard ─────────────────────────────────────────────────


class TestDebateScorecard:
    def test_no_signals_returns_insufficient(self):
        sc = compute_debate_scorecard(history=[], forward_returns={})
        assert sc["control_n"] == 0
        assert "INSUFFICIENT_EVIDENCE" in sc["verdict"]

    def test_aggregates_strengthen_bull(self):
        history = [{
            "date": "2026-04-01",
            "concordance": [
                {"ticker": "T1", "signal": "B", "conviction": 65,
                 "conviction_waterfall": {"debate_strengthen_bull": 5}},
                {"ticker": "T2", "signal": "B", "conviction": 60,
                 "conviction_waterfall": {"debate_strengthen_bull": 3}},
                # control
                {"ticker": "T3", "signal": "B", "conviction": 60,
                 "conviction_waterfall": {}},
            ],
        }]
        forward = {
            "T1:2026-04-01": {"T+30_alpha": 5.0},
            "T2:2026-04-01": {"T+30_alpha": 3.0},
            "T3:2026-04-01": {"T+30_alpha": 1.0},
        }
        sc = compute_debate_scorecard(history, forward)
        assert sc["control_n"] == 1
        assert "STRENGTHEN_BULL" in sc["signals"]
        assert sc["signals"]["STRENGTHEN_BULL"]["count"] == 2
        # Excess alpha = 4.0 - 1.0 = 3.0pp
        assert sc["signals"]["STRENGTHEN_BULL"]["excess_alpha_vs_control"] == 3.0

    def test_persist_load(self, tmp_path):
        sc = {"verdict": "test"}
        path = tmp_path / "ds.json"
        persist_scorecard(sc, path=path)
        assert load_scorecard(path=path) == sc


# ── #6 Post-Mortem ──────────────────────────────────────────────────────


class TestPostMortem:
    def test_no_history_returns_empty(self, tmp_path):
        # tmp_path is a fresh dir with no concordance files.
        result = detect_post_mortems(history_dir=tmp_path, price_data={})
        assert result == []

    def test_detect_drawdown(self, tmp_path):
        # ADD recommendation 14 days ago that dropped 15%.
        rec_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        (tmp_path / f"concordance-{rec_date}.json").write_text(json.dumps({
            "concordance": [{
                "ticker": "FAIL", "action": "ADD", "conviction": 70,
                "signal": "B", "price": 100.0, "fund_view": "BUY",
                "tech_signal": "ENTER_NOW", "macro_fit": "FAVORABLE",
                "census": "ALIGNED", "news_impact": "NEUTRAL",
                "kill_thesis": "Fails if X",
                "conviction_waterfall": {"piotroski_quality": 3, "high_beta": -5},
            }],
        }))
        # Provide price data showing -15% drop on day 7
        drop_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        price_data = {"FAIL": {rec_date: 100.0, drop_date: 85.0}}
        pms = detect_post_mortems(history_dir=tmp_path, price_data=price_data)
        assert len(pms) == 1
        pm = pms[0]
        assert pm.ticker == "FAIL"
        assert pm.drawdown_pct == -15.0
        assert pm.days_to_drawdown == 7
        assert "Fundamental" in " ".join(pm.endorsing_agents)

    def test_no_drawdown_no_post_mortem(self, tmp_path):
        rec_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        (tmp_path / f"concordance-{rec_date}.json").write_text(json.dumps({
            "concordance": [{
                "ticker": "OK", "action": "ADD", "conviction": 70,
                "signal": "B", "price": 100.0, "fund_view": "BUY",
                "tech_signal": "ENTER_NOW",
            }],
        }))
        # Position rose 5% — no drawdown
        recent_date = datetime.now().strftime("%Y-%m-%d")
        price_data = {"OK": {rec_date: 100.0, recent_date: 105.0}}
        pms = detect_post_mortems(history_dir=tmp_path, price_data=price_data)
        assert pms == []

    def test_append_lessons_dedup(self, tmp_path):
        from trade_modules.post_mortem import PostMortem
        pm = PostMortem(
            ticker="X", recommendation_date="2026-04-01", action="ADD",
            conviction=70, entry_price=100.0, drawdown_date="2026-04-08",
            drawdown_pct=-15.0, days_to_drawdown=7, lesson="t",
        )
        path = tmp_path / "lessons.jsonl"
        # First write: 1 row
        assert append_lessons([pm], path=path) == 1
        # Second write of same row: dedup — 0 new
        assert append_lessons([pm], path=path) == 0

    def test_summarise_for_committee(self):
        lessons = [{
            "ticker": "X", "recommendation_date": "2026-04-01",
            "conviction": 70, "drawdown_pct": -15.0, "days_to_drawdown": 7,
            "lesson": "Failed because Y", "dissenting_agents": ["Risk (WARN)"],
        }]
        s = summarise_for_committee(lessons)
        assert "Post-Mortem Library" in s
        assert "X" in s


# ── #7 Bayesian Conviction ──────────────────────────────────────────────


class TestBayesianConviction:
    def test_sigmoid_roundtrip(self):
        for c in (0, 25, 50, 75, 100):
            p = _conviction_to_prior(c)
            c2 = _prior_to_conviction(p)
            assert abs(c - c2) <= 1  # rounding tolerance

    def test_conviction_50_maps_to_prob_05(self):
        assert _conviction_to_prior(50) == pytest.approx(0.5)

    def test_conviction_100_maps_to_prob_095(self):
        assert _conviction_to_prior(100) == pytest.approx(0.95, abs=0.01)

    def test_conviction_0_maps_to_prob_005(self):
        assert _conviction_to_prior(0) == pytest.approx(0.05, abs=0.01)

    def test_compute_likelihoods_with_clear_signal(self):
        # Fundamental BUY perfectly predicts positive alpha
        history = [{
            "date": "2026-04-01",
            "concordance": [
                {"ticker": "T1", "fund_view": "BUY", "tech_signal": "ENTER_NOW",
                 "macro_fit": "FAVORABLE", "census": "ALIGNED",
                 "news_impact": "POSITIVE"},
                {"ticker": "T2", "fund_view": "BUY", "tech_signal": "ENTER_NOW",
                 "macro_fit": "FAVORABLE", "census": "ALIGNED",
                 "news_impact": "POSITIVE"},
                {"ticker": "T3", "fund_view": "SELL", "tech_signal": "AVOID",
                 "macro_fit": "UNFAVORABLE", "census": "DIVERGENT",
                 "news_impact": "NEGATIVE"},
            ],
        }]
        forward = {
            "T1:2026-04-01": {"T+30_alpha": 5.0},
            "T2:2026-04-01": {"T+30_alpha": 3.0},
            "T3:2026-04-01": {"T+30_alpha": -2.0},
        }
        lik = compute_likelihoods(history, forward)
        assert lik["evidence_total"] == 3
        # P(α>0 | fund=BUY) with 2 hits, 0 misses + Beta(2,2) shrinkage
        # = (2 + 2) / (2 + 2 + 2) = 0.667
        fund_buy_p = lik["agents"]["fundamental"]["BUY"]["p_alpha_pos"]
        assert fund_buy_p == pytest.approx(0.667, abs=0.005)

    def test_bayesian_posterior_reflects_priors(self):
        # No likelihoods → posterior = prior conviction
        stock = {"conviction": 60, "fund_view": "BUY"}
        post = bayesian_posterior(stock, likelihoods={})
        # No agent contributions → posterior = prior
        assert post["conviction_posterior"] == post["conviction_prior"]
        assert post["delta"] == 0

    def test_bayesian_posterior_updates_with_strong_likelihood(self):
        # A strong agent likelihood should move conviction
        stock = {"conviction": 50, "fund_view": "BUY"}
        likelihoods = {
            "agents": {
                "fundamental": {"BUY": {"p_alpha_pos": 0.85}},
            },
        }
        post = bayesian_posterior(stock, likelihoods=likelihoods)
        # Prior 50 → 0.5; Bayes update with P=0.85 → posterior > 50
        assert post["conviction_posterior"] > post["conviction_prior"]
        assert post["delta"] > 0
        assert "fundamental" in post["agent_contributions"]

    def test_persist_load_likelihoods(self, tmp_path):
        lik = {"horizon": "T+30", "agents": {}}
        path = tmp_path / "lik.json"
        persist_likelihoods(lik, path=path)
        assert load_likelihoods(path=path) == lik

    def test_shadow_score_concordance(self):
        concordance = [
            {"ticker": "X", "conviction": 50, "fund_view": "BUY",
             "signal": "B", "action": "ADD"},
        ]
        likelihoods = {
            "agents": {"fundamental": {"BUY": {"p_alpha_pos": 0.85}}},
        }
        out = shadow_score_concordance(concordance, likelihoods)
        assert out["n_stocks"] == 1
        assert out["rows"][0]["ticker"] == "X"
        assert "summary" in out
