"""
Weekly Backtest Runner

Consolidates all backtesting operations into a single entry point:
1. Signal-level backtest (BacktestEngine + ThresholdAnalyzer)
2. Committee conviction backtest (CommitteeBacktester)
3. Committee scorecard (performance tracking + modifier attribution)
4. Modifier self-calibration
5. CIO v17 H4.b — refresh rolling-percentile action thresholds for next /committee
6. CIO v17 H1  — refresh agent-sign calibrator in shadow mode

Produces a consolidated backtest_report.json for downstream consumption
(morning briefing, /backtest command).

Usage:
    python scripts/run_weekly_backtest.py [--ci]

    --ci: CI mode (action_log.jsonl is now in-repo, scorecard runs in CI)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent / "yahoofinance" / "output"
REPORT_PATH = OUTPUT_DIR / "backtest_report.json"


def run_signal_backtest():
    """Run signal-level backtest with threshold analysis."""
    print("\n" + "=" * 60)
    print("  PHASE 1: Signal-Level Backtest")
    print("=" * 60)

    from trade_modules.backtest_engine import BacktestEngine, ThresholdAnalyzer

    engine = BacktestEngine()
    engine.run()

    # Run threshold analysis on results
    results_path = engine.output_dir / "backtest_results.csv"
    summary = {}
    if results_path.exists():
        import pandas as pd

        results_df = pd.read_csv(results_path)
        if not results_df.empty:
            analyzer = ThresholdAnalyzer()
            analyzer.run(results_df)

            # Extract key metrics for report
            summary_path = engine.output_dir / "backtest_summary.csv"
            if summary_path.exists():
                sdf = pd.read_csv(summary_path)
                for _, row in sdf.iterrows():
                    if row.get("group_type") == "by_signal":
                        key = f"{row['horizon']}_{row['group']}"
                        summary[key] = {
                            "count": int(row["count"]),
                            "hit_rate": float(row["hit_rate"]),
                            "hit_rate_ci_lo": float(row["hit_rate_ci_lo"])
                            if pd.notna(row.get("hit_rate_ci_lo"))
                            else None,
                            "hit_rate_ci_hi": float(row["hit_rate_ci_hi"])
                            if pd.notna(row.get("hit_rate_ci_hi"))
                            else None,
                            "alpha_hit_rate": float(row["alpha_hit_rate"])
                            if pd.notna(row.get("alpha_hit_rate"))
                            else None,
                            "mean_return": float(row["mean_return"])
                            if pd.notna(row.get("mean_return"))
                            else None,
                            "avg_alpha": float(row["avg_alpha"])
                            if pd.notna(row.get("avg_alpha"))
                            else None,
                            "proven_signal": bool(row["proven_signal"])
                            if pd.notna(row.get("proven_signal"))
                            else None,
                        }

    return summary


def run_committee_backtest():
    """Run committee conviction backtest."""
    print("\n" + "=" * 60)
    print("  PHASE 2: Committee Conviction Backtest")
    print("=" * 60)

    from trade_modules.committee_backtester import run_backtest

    result = run_backtest(horizon="T+30")
    status = result.get("status", "unknown")
    print(f"\nCommittee backtest status: {status}")

    if status == "complete":
        for label, key in [("T+7", "performance_7d"), ("T+30", "performance_30d")]:
            perf = result.get(key, {})
            actions = perf.get("actions", {})
            if actions:
                print(f"\n  {label} Performance:")
                for action, data in actions.items():
                    print(
                        f"    {action}: {data.get('count', 0)} recs, "
                        f"{data.get('hit_rate', 0):.1f}% hit rate, "
                        f"{data.get('avg_return', 0):.2f}% avg return"
                    )

    return result


def run_scorecard(ci_mode=False):
    """Run committee scorecard and modifier calibration."""
    print("\n" + "=" * 60)
    print("  PHASE 3: Committee Scorecard + Calibration")
    print("=" * 60)

    from trade_modules.committee_scorecard import (
        COMMITTEE_LOG_PATH,
        calibrate_modifiers,
        generate_committee_scorecard,
    )

    if not COMMITTEE_LOG_PATH.exists():
        print(f"\n  Action log not found at {COMMITTEE_LOG_PATH}")
        print("  No committee actions logged yet. Run /committee first.")
        return {}, {}

    scorecard = generate_committee_scorecard(months_back=3)
    buy_recs = scorecard.get("buy_recommendations", {})
    sell_recs = scorecard.get("sell_recommendations", {})
    hold_recs = scorecard.get("hold_recommendations", {})
    print(
        f"\nScorecard: {buy_recs.get('total', 0)} BUY, "
        f"{sell_recs.get('total', 0)} SELL, "
        f"{hold_recs.get('total', 0)} HOLD tracked"
    )
    if buy_recs.get("hit_rate_30d") is not None:
        print(f"  BUY alpha hit rate (T+30): {buy_recs['hit_rate_30d']:.1f}% (vs SPY)")
    if sell_recs.get("validated_30d") is not None:
        print(f"  SELL validated (T+30): {sell_recs['validated_30d']:.1f}% (vs SPY)")

    calibration = calibrate_modifiers(months_back=3)
    if calibration.get("sufficient_data"):
        mods = calibration.get("modifiers", {})
        effective = sum(1 for m in mods.values() if m.get("recommendation") == "KEEP")
        ineffective = sum(1 for m in mods.values() if m.get("recommendation") == "REMOVE")
        adjust = sum(1 for m in mods.values() if m.get("recommendation") == "ADJUST")
        print(f"\nCalibration: {effective} effective, {adjust} adjust, {ineffective} ineffective")

        # Save calibration report for committee HTML to consume
        cal_path = Path.home() / ".weirdapps-trading" / "committee" / "calibration_report.json"
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "w") as f:
            json.dump(calibration, f, indent=2, default=str)
        print(f"  Calibration report saved to {cal_path}")

    return scorecard, calibration


def refresh_rolling_thresholds():
    """
    CIO v17 H4.b: Compute and persist rolling-percentile action thresholds.

    The next /committee run will pick these up via load_thresholds() and
    use them in determine_action(). On insufficient evidence (<20 stocks
    per signal class), legacy fixed cuts remain in effect — see
    conviction_thresholds.LEGACY_THRESHOLDS.
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: Rolling-Percentile Action Thresholds (CIO v17 H4.b)")
    print("=" * 60)

    from trade_modules.committee_backtester import CommitteeBacktester
    from trade_modules.conviction_thresholds import (
        compute_rolling_thresholds,
        persist_thresholds,
    )

    bt = CommitteeBacktester()
    bt.load_history()
    if not bt.history:
        print("  No history — skipping threshold refresh.")
        return {}

    thresholds = compute_rolling_thresholds(
        bt.history,
        lookback_snapshots=8,
        min_per_signal=20,
    )
    persist_thresholds(thresholds)

    for sig in ("B", "H", "S", "I"):
        block = thresholds.get(sig, {})
        source = block.get("source", "n/a")
        if source == "rolling":
            print(
                f"  Signal {sig}: source=rolling n={block.get('n')} "
                f"add_pct={block.get('add_pct')} trim_pct={block.get('trim_pct')} "
                f"sell_pct={block.get('sell_pct')}"
            )
        else:
            print(
                f"  Signal {sig}: source=legacy n={block.get('n', 0)} "
                "(insufficient evidence — keeping fixed cuts)"
            )
    return thresholds


def refresh_agent_sign_calibrator(shadow_mode: bool = True):
    """
    CIO v17 H1: Refresh per-agent sign-flip calibration.

    Default is **shadow mode** for the first 8 weeks: compute, log, and
    persist proposed flips with `applied=false`. Promote to AUTO only after
    we've seen consistent INVERTED verdicts across the lookback window.
    """
    print("\n" + "=" * 60)
    print(
        "  PHASE 5: Agent Sign Calibrator (CIO v17 H1, "
        + ("SHADOW" if shadow_mode else "AUTO")
        + ")"
    )
    print("=" * 60)

    from trade_modules.agent_sign_calibrator import (
        calibrate_agent_signs,
        persist_calibration,
    )
    from trade_modules.committee_backtester import CommitteeBacktester

    try:
        from trade_modules.price_service import PriceService

        svc = PriceService()
    except Exception:
        svc = None

    bt = CommitteeBacktester()
    bt.load_history()
    if not bt.history:
        print("  No history — skipping sign-calibrator refresh.")
        return {}

    # Need forward returns for the calibrator to evaluate P(α30>0|view).
    if svc is not None:
        bt.compute_forward_returns(price_service=svc, horizons=(7, 14, 30))
    else:
        from trade_modules.committee_backtester import yfinance_price_fetcher

        bt.compute_forward_returns(price_fetcher=yfinance_price_fetcher, horizons=(7, 14, 30))

    calibration = calibrate_agent_signs(
        forward_returns=bt.forward_returns,
        horizon="T+30",
        lookback_days=60,
    )
    persist_calibration(calibration, enabled=not shadow_mode)

    if calibration.get("status") == "ok":
        for agent, data in (calibration.get("agents") or {}).items():
            verdict = data.get("verdict", "?")
            consec = data.get("consecutive_inverted", 0)
            applied = data.get("applied", False)
            n = data.get("evidence_total", 0)
            tag = "APPLIED -1" if applied else "shadow"
            print(
                f"  {agent:<12s} n={n:<4d} consecutive_inverted={consec} "
                f"verdict={verdict:<18s} ({tag})"
            )
    return calibration


def refresh_conviction_cells():
    """
    CIO v17 op #4: Compute Spearman ρ(conv, α30) per
    signal × tier × regime × consensus cell.

    Output is consumed by the synthesis (next /committee run) as a
    confidence multiplier — high-IC cells boost, low-IC cells dampen.
    """
    print("\n" + "=" * 60)
    print("  PHASE 6: Per-Cell Conviction Calibration (CIO v17 op #4)")
    print("=" * 60)

    from trade_modules.committee_backtester import (
        CommitteeBacktester,
        yfinance_price_fetcher,
    )
    from trade_modules.conviction_cells import (
        compute_cells,
        persist_cells,
    )

    bt = CommitteeBacktester()
    bt.load_history()
    if not bt.history:
        print("  No history — skipping cell refresh.")
        return {}

    # Use price cache when available; fall back to yfinance.
    try:
        from trade_modules.price_service import PriceService

        bt.compute_forward_returns(price_service=PriceService(), horizons=(7, 14, 30))
    except Exception:
        bt.compute_forward_returns(price_fetcher=yfinance_price_fetcher, horizons=(7, 14, 30))

    cells = compute_cells(bt.history, bt.forward_returns, horizon="T+30")
    persist_cells(cells)
    print(
        f"  total_observations={cells.get('total_observations')}, "
        f"aggregate ρ={cells.get('aggregate_spearman')}, "
        f"high_ic_cells={len(cells.get('high_ic_cells', []))}, "
        f"low_ic_cells={len(cells.get('low_ic_cells', []))}"
    )
    print(f"  → {cells.get('recommendation', '')}")
    return cells


def refresh_debate_scorecard():
    """CIO v17 op #5: aggregate adversarial-debate signal effectiveness."""
    print("\n" + "=" * 60)
    print("  PHASE 7: Adversarial Debate Scorecard (CIO v17 op #5)")
    print("=" * 60)

    from trade_modules.committee_backtester import (
        CommitteeBacktester,
        yfinance_price_fetcher,
    )
    from trade_modules.debate_scorecard import (
        compute_debate_scorecard,
        persist_scorecard,
    )

    bt = CommitteeBacktester()
    bt.load_history()
    if not bt.history:
        print("  No history — skipping debate scorecard.")
        return {}
    try:
        from trade_modules.price_service import PriceService

        bt.compute_forward_returns(price_service=PriceService(), horizons=(7, 14, 30))
    except Exception:
        bt.compute_forward_returns(price_fetcher=yfinance_price_fetcher, horizons=(7, 14, 30))

    sc = compute_debate_scorecard(bt.history, bt.forward_returns)
    persist_scorecard(sc)
    print(
        f"  control_n={sc.get('control_n')}, "
        f"signal_classes={list((sc.get('signals') or {}).keys())}"
    )
    print(f"  → {sc.get('verdict', '')}")
    return sc


def refresh_bayesian_likelihoods():
    """CIO v17 op #7: build per-agent likelihoods for Bayesian update."""
    print("\n" + "=" * 60)
    print("  PHASE 8: Bayesian Conviction Likelihoods (CIO v17 op #7)")
    print("=" * 60)

    from trade_modules.bayesian_conviction import (
        compute_likelihoods,
        persist_likelihoods,
    )
    from trade_modules.committee_backtester import (
        CommitteeBacktester,
        yfinance_price_fetcher,
    )

    bt = CommitteeBacktester()
    bt.load_history()
    if not bt.history:
        print("  No history — skipping Bayesian refresh.")
        return {}
    try:
        from trade_modules.price_service import PriceService

        bt.compute_forward_returns(price_service=PriceService(), horizons=(7, 14, 30))
    except Exception:
        bt.compute_forward_returns(price_fetcher=yfinance_price_fetcher, horizons=(7, 14, 30))

    lik = compute_likelihoods(bt.history, bt.forward_returns)
    persist_likelihoods(lik)
    n_views = sum(len(v) for v in (lik.get("agents") or {}).values())
    print(
        f"  evidence_total={lik.get('evidence_total')}, "
        f"agents={len(lik.get('agents') or {})}, view_buckets={n_views}"
    )
    return lik


def refresh_post_mortems():
    """CIO v17 op #6: detect ADD/BUY drawdowns ≥10% within 30d."""
    print("\n" + "=" * 60)
    print("  PHASE 9: Post-Mortem Detection (CIO v17 op #6)")
    print("=" * 60)

    from trade_modules.post_mortem import (
        append_lessons,
        detect_post_mortems,
    )

    pms = detect_post_mortems()
    n = append_lessons(pms)
    if pms:
        print(f"  detected={len(pms)}, new_appended={n}")
        for pm in pms[:3]:
            print(
                f"  * {pm.ticker} ({pm.recommendation_date}): "
                f"{pm.drawdown_pct}% in {pm.days_to_drawdown}d — {pm.lesson[:80]}"
            )
    else:
        print("  No new post-mortem-worthy positions found.")
    return n


def refresh_kill_thesis_audit():
    """CIO v17 op #2: classify triggered kill theses as TRUE/FALSE positive."""
    print("\n" + "=" * 60)
    print("  PHASE 10: Kill Thesis Audit (CIO v17 op #2)")
    print("=" * 60)

    from trade_modules.kill_thesis_auditor import audit_triggered_theses

    audit = audit_triggered_theses()
    if audit.get("status") in ("no_kill_theses_file", "no_triggered_theses"):
        print(f"  {audit.get('status')}")
        return audit
    summary = audit.get("summary", {})
    print(f"  total_audited={audit.get('total_audited')}")
    print(
        f"  TRUE_POSITIVE={summary.get('true_positive_count')}, "
        f"FALSE_POSITIVE={summary.get('false_positive_count')}, "
        f"INCONCLUSIVE={summary.get('inconclusive_count')}, "
        f"UNVERIFIED={summary.get('unverified_count')}"
    )
    print(f"  TRUE-positive rate: {summary.get('true_positive_rate')}")
    return audit


def build_report(signal_summary, committee_result, scorecard, calibration):
    """Build consolidated backtest report JSON.

    CIO v17 op: Phase 1 (signal-level backtest) can legitimately produce
    no rows in CI when yfinance is rate-limited or upstream data is
    stale. The report MUST always emit `headline.buy_count_t7` (and the
    other headline keys) so the workflow validation never crashes on
    `assert h.get('buy_count_t7') is not None`. Use 0 as the empty
    sentinel and surface `signal_backtest_status: 'no_data'` so the
    downstream consumer can distinguish "no signals today" from "20
    signals today".
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "signal_backtest": {
            "status": "complete" if signal_summary else "no_data",
            "metrics": signal_summary,
        },
        "committee_backtest": {
            "status": committee_result.get("status", "unknown"),
            "history_entries": committee_result.get("history_entries", 0),
            "performance_7d": committee_result.get("performance_7d", {}),
            "performance_30d": committee_result.get("performance_30d", {}),
        },
        "scorecard": {
            "buy_total": scorecard.get("buy_recommendations", {}).get("total", 0),
            "buy_alpha_hit_rate_7d": scorecard.get("buy_recommendations", {}).get("hit_rate_7d"),
            "buy_alpha_hit_rate_30d": scorecard.get("buy_recommendations", {}).get("hit_rate_30d"),
            "buy_avg_alpha_7d": scorecard.get("buy_recommendations", {}).get("avg_alpha_7d"),
            "buy_avg_alpha_30d": scorecard.get("buy_recommendations", {}).get("avg_alpha_30d"),
            "conviction_predictive": scorecard.get("conviction_calibration", {}).get(
                "conviction_predictive"
            ),
        },
        "calibration": {
            "sufficient_data": calibration.get("sufficient_data", False),
            "modifiers_evaluated": len(calibration.get("modifiers", {})),
        },
    }

    # Add key signal metrics at top level for easy access
    buy_t7 = signal_summary.get("t7_B", {})
    buy_t30 = signal_summary.get("t30_B", {})
    sell_t7 = signal_summary.get("t7_S", {})
    sell_t30 = signal_summary.get("t30_S", {})

    # CIO v17 op: empty-data sentinel (0 not None) so workflow validators
    # never crash on `assert h.get('buy_count_t7') is not None`. The
    # status field below tells consumers to distinguish a real "no
    # signals today" from a Phase-1 data outage.
    no_signal_data = not signal_summary

    def _h(d, k, default=0):
        v = d.get(k)
        return default if v is None else v

    report["headline"] = {
        "buy_hit_rate_t7": _h(buy_t7, "hit_rate"),
        "buy_alpha_hit_rate_t7": _h(buy_t7, "alpha_hit_rate"),
        "buy_hit_rate_t30": _h(buy_t30, "hit_rate"),
        "buy_alpha_hit_rate_t30": _h(buy_t30, "alpha_hit_rate"),
        "buy_avg_alpha_t7": _h(buy_t7, "avg_alpha"),
        "buy_avg_alpha_t30": _h(buy_t30, "avg_alpha"),
        "buy_count_t7": _h(buy_t7, "count"),
        "sell_hit_rate_t7": _h(sell_t7, "hit_rate"),
        "sell_alpha_hit_rate_t7": _h(sell_t7, "alpha_hit_rate"),
        "sell_hit_rate_t30": _h(sell_t30, "hit_rate"),
        "sell_alpha_hit_rate_t30": _h(sell_t30, "alpha_hit_rate"),
        "sell_avg_alpha_t7": _h(sell_t7, "avg_alpha"),
        "sell_avg_alpha_t30": _h(sell_t30, "avg_alpha"),
        "sell_count_t7": _h(sell_t7, "count"),
        "buy_hit_rate_ci_t7": [
            _h(buy_t7, "hit_rate_ci_lo", None),
            _h(buy_t7, "hit_rate_ci_hi", None),
        ],
        "buy_proven_signal_t7": buy_t7.get("proven_signal", False),
        "signal_backtest_status": "no_data" if no_signal_data else "ok",
    }

    return report


def main():
    ci_mode = "--ci" in sys.argv

    print("\n" + "#" * 60)
    print("  WEEKLY BACKTEST PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("#" * 60)

    # Phase 1: Signal backtest
    signal_summary = {}
    try:
        signal_summary = run_signal_backtest()
    except Exception as e:
        print(f"\n  Phase 1 FAILED: {e}")

    # Phase 2: Committee backtest
    committee_result = {}
    try:
        committee_result = run_committee_backtest()
    except Exception as e:
        print(f"\n  Phase 2 FAILED: {e}")

    # Phase 3: Scorecard + calibration
    scorecard, calibration = {}, {}
    try:
        scorecard, calibration = run_scorecard(ci_mode)
    except Exception as e:
        print(f"\n  Phase 3 FAILED: {e}")

    # Phase 4: Rolling-percentile thresholds (CIO v17 H4.b)
    rolling_thresholds = {}
    try:
        rolling_thresholds = refresh_rolling_thresholds()
    except Exception as e:
        print(f"\n  Phase 4 FAILED: {e}")

    # Phase 5: Agent sign calibrator (CIO v17 H1, shadow by default)
    sign_cal = {}
    try:
        sign_cal = refresh_agent_sign_calibrator(shadow_mode=True)
    except Exception as e:
        print(f"\n  Phase 5 FAILED: {e}")

    # Phase 6: Per-cell conviction calibration (CIO v17 op #4)
    cells = {}
    try:
        cells = refresh_conviction_cells()
    except Exception as e:
        print(f"\n  Phase 6 FAILED: {e}")

    # Phase 7: Adversarial debate scorecard (CIO v17 op #5)
    debate_sc = {}
    try:
        debate_sc = refresh_debate_scorecard()
    except Exception as e:
        print(f"\n  Phase 7 FAILED: {e}")

    # Phase 8: Bayesian likelihoods (CIO v17 op #7) — shadow only.
    bayes_likelihoods = {}
    try:
        bayes_likelihoods = refresh_bayesian_likelihoods()
    except Exception as e:
        print(f"\n  Phase 8 FAILED: {e}")

    # Phase 9: Post-mortem detection (CIO v17 op #6) — appends new
    # ADD/BUY positions that breached -10% within 30d to lessons log.
    pm_count = 0
    try:
        pm_count = refresh_post_mortems()
    except Exception as e:
        print(f"\n  Phase 9 FAILED: {e}")

    # Phase 10: Kill thesis audit (CIO v17 op #2) — classify triggered
    # theses as TRUE/FALSE positive based on 30-day forward return.
    audit = {}
    try:
        audit = refresh_kill_thesis_audit()
    except Exception as e:
        print(f"\n  Phase 10 FAILED: {e}")

    # Phase 11: Parameter effectiveness analysis (CIO v35.0)
    # Correlates ALL archived parameters with forward returns
    param_effectiveness = {}
    try:
        print("\n" + "=" * 60)
        print("  PHASE 11: Parameter Effectiveness (CIO v35.0)")
        print("=" * 60)
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "etoro-portfolio" / "src"))
        from etoro_portfolio.modifier_audit import run_modifier_audit

        param_effectiveness = run_modifier_audit(horizon_days=30, min_observations=10)
        keep = param_effectiveness.get("keep_count", 0)
        cut = param_effectiveness.get("cut_count", 0)
        inv = param_effectiveness.get("investigate_count", 0)
        conv_r = param_effectiveness.get("conviction_correlation", 0)
        print(f"  Modifiers: {keep} keep, {cut} cut, {inv} investigate")
        print(f"  Conviction→returns correlation: r={conv_r:.4f}")
    except Exception as e:
        print(f"\n  Phase 11 FAILED: {e}")

    # Build consolidated report
    report = build_report(signal_summary, committee_result, scorecard, calibration)
    # CIO v17 wiring: surface H1 + H4.b + ops 4/5/6/7 state in the report.
    report["rolling_thresholds"] = rolling_thresholds
    report["agent_sign_calibration"] = {
        "status": sign_cal.get("status", "no_data") if sign_cal else "no_data",
        "horizon": sign_cal.get("horizon") if sign_cal else None,
        "agents": {
            a: {
                "verdict": d.get("verdict"),
                "consecutive_inverted": d.get("consecutive_inverted"),
                "applied": d.get("applied", False),
                "evidence_total": d.get("evidence_total"),
            }
            for a, d in (sign_cal.get("agents") or {}).items()
        }
        if sign_cal
        else {},
    }
    report["conviction_cells"] = {
        "total_observations": cells.get("total_observations") if cells else 0,
        "aggregate_spearman": cells.get("aggregate_spearman") if cells else None,
        "high_ic_cell_count": len(cells.get("high_ic_cells", [])) if cells else 0,
        "low_ic_cell_count": len(cells.get("low_ic_cells", [])) if cells else 0,
        "recommendation": cells.get("recommendation") if cells else None,
    }
    report["debate_scorecard_summary"] = {
        "control_n": debate_sc.get("control_n") if debate_sc else 0,
        "verdict": debate_sc.get("verdict") if debate_sc else None,
    }
    report["bayesian_likelihoods_summary"] = {
        "evidence_total": bayes_likelihoods.get("evidence_total") if bayes_likelihoods else 0,
        "n_agents": len((bayes_likelihoods or {}).get("agents") or {}),
    }
    report["post_mortems_appended"] = pm_count
    report["kill_thesis_audit"] = {
        "summary": (audit or {}).get("summary", {}),
        "status": (audit or {}).get("status", "ok"),
    }
    report["parameter_effectiveness"] = {
        "keep_count": param_effectiveness.get("keep_count", 0),
        "cut_count": param_effectiveness.get("cut_count", 0),
        "investigate_count": param_effectiveness.get("investigate_count", 0),
        "conviction_correlation": param_effectiveness.get("conviction_correlation"),
    }

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nConsolidated report saved to: {REPORT_PATH}")

    # Print headline metrics
    headline = report.get("headline", {})
    print("\n" + "=" * 60)
    print("  HEADLINE METRICS")
    print("=" * 60)
    for key, val in headline.items():
        if val is not None:
            print(f"  {key}: {val}")
    print()

    # Validate output
    report_size = REPORT_PATH.stat().st_size
    if report_size < 500:
        print(f"\nWARNING: Report file is suspiciously small ({report_size} bytes)")
        print("This likely indicates a pipeline failure. Check logs above.")
        sys.exit(1)

    # Validate key fields are populated
    missing_fields = []
    headline = report.get("headline", {})
    for key in ["buy_hit_rate_t7", "buy_count_t7", "sell_hit_rate_t7"]:
        if headline.get(key) is None:
            missing_fields.append(key)

    if missing_fields:
        print(f"\nWARNING: Missing headline fields: {missing_fields}")


if __name__ == "__main__":
    main()
