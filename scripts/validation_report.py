"""
validation_report.py — S0 Validation Baseline Runner

Loads backtest_results.csv, optionally attaches regime labels and action_log,
calls harness.evaluate(), and writes MD + JSON reports to ~/Downloads/.

Usage:
    python3 -m scripts.validation_report
    # or:
    python3 scripts/validation_report.py

All I/O is inside main().  # pragma: no cover for all I/O + CLI code.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers (no I/O)
# ---------------------------------------------------------------------------


def _load_csv(path: str | Path) -> list[dict]:  # pragma: no cover
    """Load a CSV as list of dicts; cast horizon to int, floats where possible."""
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Cast horizon to int for the harness
            if "horizon" in row:
                try:
                    row["horizon"] = int(row["horizon"])
                except (ValueError, TypeError):
                    pass
            # Cast numeric columns to float (empty string → None)
            for col in ("alpha", "net_alpha", "future_price", "price_at_signal"):
                if col in row:
                    val = row[col].strip() if isinstance(row[col], str) else row[col]
                    if val == "" or val is None:
                        row[col] = None
                    else:
                        try:
                            row[col] = float(val)
                        except (ValueError, TypeError):
                            row[col] = None
            rows.append(row)
    return rows


def _load_jsonl(path: str | Path) -> list[dict]:  # pragma: no cover
    """Load a .jsonl file; return empty list if file does not exist."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _athens_timestamp() -> str:  # pragma: no cover
    """Return Athens local time as YYYYMMDDHHMM string."""
    result = subprocess.run(
        ["bash", "-c", "TZ='Europe/Athens' date '+%Y%m%d%H%M'"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fmt(value: float | None, decimals: int = 3, pct: bool = False) -> str:
    """Format a float or return 'N/A'."""
    if value is None:
        return "N/A"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Report generation (pure, no I/O — easier to test)
# ---------------------------------------------------------------------------


def build_md_report(result: dict, data_date: str) -> str:  # pragma: no cover
    """Render the VerdictReport dict as Markdown."""
    lines: list[str] = []

    overall = result["overall"]
    status = "PASS" if overall["passed"] else "FAIL"

    lines.append(f"# S0 Validation Report — {data_date}")
    lines.append("")
    lines.append(f"**Overall verdict: {status}**")
    lines.append("")

    lines.append("## DSR assumptions")
    lines.append(f"- n_trials = {result['n_trials']} (signal engine heavy parameterisation)")
    lines.append("- var_sr = 0.5 (prior variance of Sharpe under null)")
    lines.append("")

    lines.append("## Overall gate")
    lines.append(f"- DSR: {_fmt(overall['dsr'])}")
    lines.append(f"- PBO: {_fmt(overall['pbo'])}")
    if overall["reasons"]:
        lines.append("- Reasons for FAIL:")
        for r in overall["reasons"]:
            lines.append(f"  - {r}")
    else:
        lines.append("- No gate failures")
    lines.append("")

    # Honest framing
    if not overall["passed"]:
        lines.append(
            "> **Honest framing**: A FAIL verdict is expected for the baseline run. "
            "The existing signal engine was tuned on ~60 observations in a single bull regime "
            "with ~220 parameters — a textbook overfitting setup. "
            "The purpose of this harness is to quantify that overfitting and establish a "
            "performance floor before S1–S5 rebuilding begins."
        )
        lines.append("")

    # Per-family table
    lines.append("## Per-family results")
    lines.append("")
    lines.append("| Family | n | Sharpe | DSR | PBO | OOS_hit | OOS_alpha | gross_alpha |")
    lines.append("|--------|---|--------|-----|-----|---------|-----------|-------------|")

    families = result.get("families", {})
    # Sort by n descending
    sorted_fams = sorted(
        families.items(),
        key=lambda kv: kv[1].get("n", 0) if not kv[1].get("insufficient_data") else 0,
        reverse=True,
    )
    for fam, stats in sorted_fams:
        if stats.get("insufficient_data"):
            lines.append(
                f"| {fam} | {stats.get('n', '?')} | — | — | — | — | — | — | *(insufficient data)* |"
            )
        else:
            lines.append(
                f"| {fam} | {stats.get('n', '?')} "
                f"| {_fmt(stats.get('sharpe'))} "
                f"| {_fmt(stats.get('dsr'))} "
                f"| {_fmt(stats.get('pbo'))} "
                f"| {_fmt(stats.get('oos_hit'), pct=True)} "
                f"| {_fmt(stats.get('oos_alpha'))} "
                f"| {_fmt(stats.get('alpha_gross'))} |"
            )
    lines.append("")

    # IC half-life per family
    lines.append("## IC half-life per family")
    lines.append("")
    has_ic = False
    for fam, stats in sorted_fams:
        if stats.get("insufficient_data"):
            continue
        ic = stats.get("ic_decay", {})
        hl = ic.get("half_life_days")
        note = ic.get("note", "")
        if hl is not None:
            lines.append(f"- **{fam}**: half-life = {hl:.1f} days (IC0={_fmt(ic.get('ic0'))})")
            has_ic = True
        elif note:
            lines.append(f"- **{fam}**: N/A — {note}")
    if not has_ic:
        lines.append("*(IC decay not computable — insufficient horizon overlap in data)*")
    lines.append("")

    # Regime stratified
    regime_stratified = result.get("regime_stratified", {})
    if regime_stratified:
        lines.append("## Regime-stratified performance")
        lines.append("")
        lines.append("| Regime | n | Hit rate | Avg alpha |")
        lines.append("|--------|---|----------|-----------|")
        for regime, stats in sorted(regime_stratified.items()):
            lines.append(
                f"| {regime} | {stats['n']} | {stats['hit']:.1%} | {stats['avg_alpha']:.3f} |"
            )
        lines.append("")
    else:
        lines.append("## Regime-stratified performance")
        lines.append("")
        lines.append(
            "*(Regime data unavailable — yfinance fetch skipped or no regime key in rows)*"
        )
        lines.append("")

    # Survivorship
    surv = result.get("survivorship", {})
    lines.append("## Survivorship bias check")
    lines.append("")
    lines.append(f"- Total rows: {surv.get('total_rows', 'N/A')}")
    lines.append(f"- Rows missing future_price: {surv.get('no_forward_price', 'N/A')}")
    lines.append(f"- Pct dropped: {_fmt(surv.get('pct_dropped'), decimals=1)}%")
    lines.append("")

    # Turnover
    turnover = result.get("turnover")
    lines.append("## Turnover")
    lines.append("")
    if turnover is None:
        # turnover is None only when action_records was not passed at all
        lines.append("*(No action_log passed to harness — turnover not computed)*")
    elif "note" in turnover:
        # action_log was found but computation was skipped or failed
        lines.append(f"*(Turnover skipped: {turnover['note']})*")
        if "n_records" in turnover:
            lines.append(f"- Action records found: {turnover['n_records']}")
    else:
        lines.append(
            f"- Annualised turnover: {_fmt(turnover.get('turnover_annual_pct'), decimals=1)}%"
        )
        lines.append(
            f"- Annualised drag: {_fmt(turnover.get('annualized_drag_bps'), decimals=1)} bps"
        )
        lines.append(f"- N trades in window: {turnover.get('n_trades', 'N/A')}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """Entry point — all I/O lives here."""
    from trade_modules.validation.harness import evaluate
    from trade_modules.validation.regime_join import attach_regime, fetch_regime_inputs, label_dates

    repo_root = Path(__file__).parent.parent

    # Load backtest results
    csv_path = repo_root / "yahoofinance" / "output" / "backtest_results.csv"
    print(f"Loading {csv_path} ...")
    results_rows = _load_csv(csv_path)
    print(f"  Loaded {len(results_rows)} rows")

    # Load action_log (fail-open)
    action_log_path = repo_root / "data" / "committee" / "action_log.jsonl"
    action_records = _load_jsonl(action_log_path)
    if action_records:
        print(f"  Loaded {len(action_records)} action records from {action_log_path}")
    else:
        print("  No action_log found — turnover will be None")

    # Attach regime labels (fail-open)
    try:
        # Collect date range
        dates = sorted(
            {r.get("signal_date", "")[:10] for r in results_rows if r.get("signal_date")}
        )
        if dates:
            start = dates[0]
            end = dates[-1]
            print(f"  Fetching regime inputs for {start} to {end} ...")
            date_index, vix, vix3m, spy = fetch_regime_inputs(start, end)
            date_to_regime = label_dates(dates, vix, vix3m, spy, date_index)
            results_rows = attach_regime(results_rows, date_to_regime)
            regime_count = len({r.get("regime") for r in results_rows if r.get("regime")})
            print(f"  Regime labels attached: {regime_count} distinct regimes")
        else:
            print("  No signal_dates found — skipping regime attachment")
    except Exception as e:
        print(f"  Regime fetch failed (fail-open): {e}")

    # Evaluate
    print("Running harness.evaluate() ...")
    result = evaluate(results_rows, action_records if action_records else None)

    # Print verdict to stdout
    overall = result["overall"]
    status = "PASS" if overall["passed"] else "FAIL"
    print(f"\n=== VERDICT: {status} ===")
    print(f"  DSR:  {_fmt(overall['dsr'])}")
    print(f"  PBO:  {_fmt(overall['pbo'])}")
    if overall["reasons"]:
        print("  Reasons:")
        for r in overall["reasons"]:
            print(f"    - {r}")
    surv = result["survivorship"]
    print(
        f"\nSurvivorship: {surv['total_rows']} total rows, {surv['no_forward_price']} missing future_price ({_fmt(surv['pct_dropped'], decimals=1)}% dropped)"
    )

    print("\nPer-family summary:")
    families = result.get("families", {})
    sorted_fams = sorted(
        families.items(),
        key=lambda kv: kv[1].get("n", 0) if not kv[1].get("insufficient_data") else 0,
        reverse=True,
    )
    for fam, stats in sorted_fams[:10]:
        if stats.get("insufficient_data"):
            print(f"  {fam}: INSUFFICIENT DATA (n={stats.get('n', '?')})")
        else:
            print(
                f"  {fam}: n={stats.get('n', '?')} | Sharpe={_fmt(stats.get('sharpe'))} "
                f"| DSR={_fmt(stats.get('dsr'))} | PBO={_fmt(stats.get('pbo'))}"
            )

    # Write output files
    ts = _athens_timestamp()
    downloads = Path.home() / "Downloads"
    md_path = downloads / f"{ts}_validation_report.md"
    json_path = downloads / f"{ts}_validation_report.json"

    # Determine data date
    backtest_dates = sorted(
        r.get("signal_date", "")[:10] for r in results_rows if r.get("signal_date")
    )
    data_date = f"{backtest_dates[0]} → {backtest_dates[-1]}" if backtest_dates else "unknown"

    md_text = build_md_report(result, data_date)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"\nReport written to: {md_path}")

    # JSON: make result serialisable
    def _serialise(obj):
        if isinstance(obj, (float, int, bool, str, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        return str(obj)

    json_path.write_text(json.dumps(_serialise(result), indent=2), encoding="utf-8")
    print(f"JSON written to:   {json_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
