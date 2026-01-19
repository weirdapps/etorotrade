#!/usr/bin/env python3
"""
Framework Analysis Script

Runs comprehensive analysis of the trading framework including:
1. Signal validation against historical performance
2. Metric effectiveness analysis
3. Data quality assessment
4. Automated improvement suggestions

Usage:
    python scripts/analyze_framework.py [--validate] [--suggestions] [--full]
    python scripts/analyze_framework.py --scheduled --output /path/to/report.txt
    python scripts/analyze_framework.py --json --output /path/to/summary.json

Options:
    --validate      Run signal validation only
    --suggestions   Generate improvement suggestions only
    --full          Run complete analysis (default)
    --scheduled     Run in scheduled mode (for cron jobs)
    --json          Output summary as JSON for programmatic consumption
    --output FILE   Write output to file instead of stdout
    --min-days N    Minimum days for signal maturity (default: 30)

Cron Example (run daily at 6 AM):
    0 6 * * * /usr/bin/python3 /path/to/scripts/analyze_framework.py --scheduled --output /path/to/logs/validation_$(date +\\%Y\\%m\\%d).txt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_validation(min_days: int = 30) -> None:
    """Run signal validation analysis."""
    print("=" * 60)
    print("SIGNAL VALIDATION ANALYSIS")
    print("=" * 60)
    print()

    from trade_modules.signal_validator import SignalValidator

    validator = SignalValidator()
    signals = validator.load_signals()

    print(f"Total historical signals: {len(signals)}")
    print()

    # Run validation
    report = validator.generate_report(min_days=min_days)
    print(report)


def run_suggestions() -> None:
    """Generate improvement suggestions."""
    print("=" * 60)
    print("IMPROVEMENT SUGGESTIONS GENERATION")
    print("=" * 60)
    print()

    from trade_modules.improvement_analyzer import ImprovementAnalyzer

    analyzer = ImprovementAnalyzer()
    doc = analyzer.generate_suggestions_document()

    # Save to file
    output_path = analyzer.save_document(doc)
    print(f"Suggestions document saved to: {output_path}")
    print()

    # Print summary
    print("SUMMARY")
    print("-" * 40)
    print(doc.summary)
    print()

    print("KEY INSIGHTS")
    print("-" * 40)
    for i, insight in enumerate(doc.key_insights, 1):
        print(f"{i}. {insight}")
    print()

    print("HIGH PRIORITY SUGGESTIONS")
    print("-" * 40)
    high_priority = [s for s in doc.suggestions if s.priority == "HIGH"]
    for s in high_priority:
        print(f"- {s.title}")
        print(f"  Action: {s.action}")
        print()


def run_full_analysis() -> None:
    """Run complete framework analysis."""
    print()
    print("*" * 60)
    print("TRADING FRAMEWORK COMPREHENSIVE ANALYSIS")
    print(f"Generated: {datetime.now().isoformat()}")
    print("*" * 60)
    print()

    # Run all analyses
    run_validation()
    print()
    run_suggestions()

    print()
    print("*" * 60)
    print("ANALYSIS COMPLETE")
    print("*" * 60)
    print()
    print("Generated files:")
    print("  - docs/IMPROVEMENT_SUGGESTIONS.md")
    print("  - docs/FRAMEWORK_CRITIQUE.md")
    print()
    print("Next steps:")
    print("  1. Review HIGH priority suggestions")
    print("  2. Update config.yaml thresholds if needed")
    print("  3. Run 'python trade.py -o p' to verify changes")
    print("  4. Re-run this analysis after market data update")


def run_scheduled_validation(min_days: int = 30, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run scheduled validation suitable for cron jobs.

    Returns a summary dict and optionally writes to file.
    """
    from trade_modules.signal_validator import SignalValidator
    from trade_modules.signal_tracker import get_signal_summary

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "errors": [],
        "summary": {},
    }

    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("SCHEDULED SIGNAL VALIDATION REPORT")
    output_lines.append(f"Generated: {datetime.now().isoformat()}")
    output_lines.append("=" * 60)
    output_lines.append("")

    try:
        # Get signal log stats
        signal_stats = get_signal_summary()
        output_lines.append("SIGNAL LOG STATUS")
        output_lines.append("-" * 40)
        output_lines.append(f"Total signals logged: {signal_stats.get('total', 0)}")

        by_signal = signal_stats.get("by_signal", {})
        for sig, count in sorted(by_signal.items()):
            label = {"B": "BUY", "S": "SELL", "H": "HOLD", "I": "INCONCLUSIVE"}.get(sig, sig)
            output_lines.append(f"  {label}: {count}")

        date_range = signal_stats.get("date_range", {})
        if date_range:
            output_lines.append(f"Date range: {date_range.get('first', 'N/A')} to {date_range.get('last', 'N/A')}")
        output_lines.append("")

        results["summary"]["signal_stats"] = signal_stats

        # Run validation
        validator = SignalValidator()
        signals = validator.load_signals()

        output_lines.append("SIGNAL VALIDATION")
        output_lines.append("-" * 40)
        output_lines.append(f"Signals to validate: {len(signals)}")
        output_lines.append(f"Minimum maturity: {min_days} days")
        output_lines.append("")

        if len(signals) > 0:
            report = validator.generate_report(min_days=min_days)
            output_lines.append(report)

            # Get summary for JSON
            validation_results = validator.validate_signals_batch(signals, min_days=min_days)
            if validation_results:
                summary = validator.generate_summary(validation_results)
                results["summary"]["validation"] = {
                    "total_signals": summary.total_signals,
                    "validated_signals": summary.validated_signals,
                    "hit_rate": summary.hit_rate,
                    "avg_return": summary.avg_return,
                    "median_return": summary.median_return,
                    "excess_vs_benchmark": summary.excess_vs_benchmark,
                    "by_signal_type": summary.by_signal_type,
                    "by_tier": summary.by_tier,
                    "by_region": summary.by_region,
                    "improvement_suggestions": summary.improvement_suggestions,
                }
        else:
            output_lines.append("No signals available for validation.")
            output_lines.append("Signals need at least 30 days to mature for meaningful validation.")

    except Exception as e:
        results["status"] = "error"
        results["errors"].append(str(e))
        output_lines.append(f"ERROR: {e}")

    # Write output
    output_text = "\n".join(output_lines)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(output_text)
        print(f"Report written to: {output_file}")
    else:
        print(output_text)

    return results


def run_json_export(min_days: int = 30, output_file: Optional[str] = None) -> None:
    """
    Export validation summary as JSON for programmatic consumption.
    """
    results = run_scheduled_validation(min_days=min_days)

    json_output = json.dumps(results, indent=2, default=str)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(json_output)
        print(f"JSON written to: {output_file}")
    else:
        print(json_output)


def main():
    parser = argparse.ArgumentParser(
        description="Trading Framework Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full interactive analysis
  python scripts/analyze_framework.py --full

  # Scheduled validation for cron (writes to file)
  python scripts/analyze_framework.py --scheduled --output /var/log/trading/validation.txt

  # JSON export for programmatic consumption
  python scripts/analyze_framework.py --json --output /var/log/trading/summary.json

  # Validate signals that are at least 60 days old
  python scripts/analyze_framework.py --validate --min-days 60

Cron Setup:
  # Daily validation at 6 AM
  0 6 * * * cd /path/to/etorotrade && /usr/bin/python3 scripts/analyze_framework.py --scheduled --output logs/validation_$(date +\\%Y\\%m\\%d).txt
        """
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run signal validation only"
    )
    parser.add_argument(
        "--suggestions",
        action="store_true",
        help="Generate improvement suggestions only"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete analysis (default)"
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Run in scheduled mode (for cron jobs), outputs clean report"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output summary as JSON for programmatic consumption"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write output to file instead of stdout"
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=30,
        help="Minimum days for signal maturity in validation (default: 30)"
    )

    args = parser.parse_args()

    if args.json:
        run_json_export(min_days=args.min_days, output_file=args.output)
    elif args.scheduled:
        run_scheduled_validation(min_days=args.min_days, output_file=args.output)
    elif args.validate:
        run_validation(args.min_days)
    elif args.suggestions:
        run_suggestions()
    else:
        run_full_analysis()


if __name__ == "__main__":
    main()
