"""
Weekly Backtest Runner

Consolidates all backtesting operations into a single entry point:
1. Signal-level backtest (BacktestEngine + ThresholdAnalyzer)
2. Committee conviction backtest (CommitteeBacktester)
3. Committee scorecard (performance tracking + modifier attribution)
4. Modifier self-calibration

Produces a consolidated backtest_report.json for downstream consumption
(morning briefing, /backtest command).

Usage:
    python scripts/run_weekly_backtest.py [--ci]

    --ci: Skip committee scorecard (requires local action_log.jsonl)
"""

import json
import sys
import os
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
                            "mean_return": float(row["mean_return"]) if pd.notna(row.get("mean_return")) else None,
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
        perf = result.get("performance_30d", {})
        actions = perf.get("actions", {})
        for action, data in actions.items():
            print(f"  {action}: {data.get('count', 0)} recs, "
                  f"{data.get('hit_rate', 0):.1f}% hit rate, "
                  f"{data.get('avg_return', 0):.2f}% avg return")

    return result


def run_scorecard(ci_mode=False):
    """Run committee scorecard and modifier calibration."""
    if ci_mode:
        print("\n  PHASE 3: Skipped (--ci mode, no local action_log)")
        return {}, {}

    print("\n" + "=" * 60)
    print("  PHASE 3: Committee Scorecard + Calibration")
    print("=" * 60)

    from trade_modules.committee_scorecard import (
        generate_committee_scorecard,
        calibrate_modifiers,
    )

    scorecard = generate_committee_scorecard(months_back=3)
    buy_recs = scorecard.get("buy_recommendations", {})
    print(f"\nScorecard: {buy_recs.get('total', 0)} BUY recommendations tracked")
    if buy_recs.get("hit_rate_30d") is not None:
        print(f"  BUY hit rate (T+30): {buy_recs['hit_rate_30d']:.1f}%")

    calibration = calibrate_modifiers(months_back=3)
    if calibration.get("sufficient_data"):
        mods = calibration.get("modifiers", {})
        effective = sum(1 for m in mods.values() if m.get("recommendation") == "KEEP")
        ineffective = sum(1 for m in mods.values() if m.get("recommendation") == "REMOVE")
        print(f"\nCalibration: {effective} effective, {ineffective} ineffective modifiers")

    return scorecard, calibration


def build_report(signal_summary, committee_result, scorecard, calibration):
    """Build consolidated backtest report JSON."""
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
        },
        "scorecard": {
            "buy_total": scorecard.get("buy_recommendations", {}).get("total", 0),
            "buy_hit_rate_7d": scorecard.get("buy_recommendations", {}).get("hit_rate_7d"),
            "buy_hit_rate_30d": scorecard.get("buy_recommendations", {}).get("hit_rate_30d"),
            "conviction_predictive": scorecard.get("conviction_calibration", {}).get("conviction_predictive"),
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

    report["headline"] = {
        "buy_hit_rate_t7": buy_t7.get("hit_rate"),
        "buy_hit_rate_t30": buy_t30.get("hit_rate"),
        "buy_count_t7": buy_t7.get("count"),
        "sell_hit_rate_t7": sell_t7.get("hit_rate"),
        "sell_hit_rate_t30": sell_t30.get("hit_rate"),
        "sell_count_t7": sell_t7.get("count"),
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

    # Build consolidated report
    report = build_report(signal_summary, committee_result, scorecard, calibration)

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


if __name__ == "__main__":
    main()
