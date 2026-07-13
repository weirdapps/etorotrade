"""Trading Model v3 — IC-join report (Phase 5E).

Reads the forward-IC log at $V3_IC_LOG (or ~/.weirdapps-trading/v3_ic_log.csv),
computes cross-sectional Spearman IC for each horizon in (5, 10, 21, 63) log-date
steps, prints a summary, and saves a markdown report to:

    ~/Downloads/<UTCstamp>_v3_ic_report.md

Gracefully handles insufficient history for any horizon.

Run:   .venv/bin/python scripts/v3_ic_report.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.v3_ic_logger import DEFAULT_LOG, ic_from_log  # noqa: E402

HORIZONS = (5, 10, 21, 63)

_HORIZON_LABELS = {
    5: " ~1w",
    10: " ~2w",
    21: " ~1m",
    63: " ~3m",
}


def _format_val(v: float | None, fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if v is not None else "n/a"


def _build_markdown(results: dict, log_path: str, n_rows: int, n_dates: int) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# v3 Forward IC Report",
        "",
        f"Generated: {now_str}",
        f"Log: `{log_path}`  ({n_rows:,} rows, {n_dates} distinct dates)",
        "",
        "## Cross-Sectional IC by Horizon",
        "",
        "| Horizon | Label | mean IC | IC std | t-stat | Hit rate | n dates |",
        "|--------:|:------|--------:|-------:|-------:|---------:|--------:|",
    ]

    for h in HORIZONS:
        r = results.get(h, {})
        label = _HORIZON_LABELS.get(h, "")
        if r.get("insufficient_history"):
            avail = r.get("n_dates_available", "?")
            need = r.get("n_dates_needed", h + 1)
            lines.append(f"| {h:>7} | {label} | *insufficient ({avail}/{need} dates)* | | | | |")
        else:
            mean_ic = _format_val(r.get("mean_ic"))
            ic_std = _format_val(r.get("ic_std"))
            t_stat = _format_val(r.get("t_stat"))
            hit = _format_val(r.get("hit_rate"))
            n_d = r.get("n_dates", 0)
            lines.append(
                f"| {h:>7} | {label} | {mean_ic} | {ic_std} | {t_stat} | {hit} | {n_d:>7} |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- **Horizon** is measured in log-date steps (one step = one daily logger run).",
        "- IC is the mean Spearman rank correlation of conviction vs forward return across dates.",
        "- A date contributes to IC only when >= 3 clean name-pairs are present.",
        "- IC weighting in `compute_scores` activates once >63 dates are logged.",
        "",
        "## Raw JSON",
        "",
        "```json",
    ]

    # Serialise results (drop ic_by_date to keep the report concise).
    slim = {}
    for h, r in results.items():
        r2 = {k: v for k, v in r.items() if k != "ic_by_date"}
        slim[str(h)] = r2
    lines.append(json.dumps(slim, indent=2, default=str))
    lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> None:
    log_path = os.environ.get("V3_IC_LOG", DEFAULT_LOG)
    expanded = os.path.expanduser(log_path)

    if not os.path.exists(expanded):
        print(f"log not found: {expanded}")
        print("Run v3_ic_logger.py at least once first.")
        sys.exit(0)

    try:
        log_df = pd.read_csv(expanded, dtype={"date": str})
    except Exception as exc:  # noqa: BLE001
        print(f"error reading log {expanded}: {exc}", file=sys.stderr)
        sys.exit(1)

    n_rows = len(log_df)
    dates = sorted(log_df["date"].astype(str).unique())
    n_dates = len(dates)
    print(f"log: {n_rows:,} rows, {n_dates} distinct dates  ({expanded})")
    print(f"date range: {dates[0] if dates else 'n/a'} .. {dates[-1] if dates else 'n/a'}")
    print()

    results = ic_from_log(log_df, horizons=HORIZONS)

    # Print summary table.
    header = f"{'H':>5}  {'Label':5}  {'mean IC':>8}  {'IC std':>8}  {'t-stat':>7}  {'Hit%':>6}  {'n':>4}"
    print(header)
    print("-" * len(header))
    for h in HORIZONS:
        r = results[h]
        label = _HORIZON_LABELS.get(h, "")
        if r.get("insufficient_history"):
            avail = r.get("n_dates_available", "?")
            need = r.get("n_dates_needed", h + 1)
            print(f"{h:>5}  {label:5}  insufficient history ({avail}/{need} dates)")
        else:
            mean_ic = r.get("mean_ic")
            ic_std = r.get("ic_std")
            t_stat = r.get("t_stat")
            hit = r.get("hit_rate")
            n_d = r.get("n_dates", 0)
            print(
                f"{h:>5}  {label:5}"
                f"  {_format_val(mean_ic):>8}"
                f"  {_format_val(ic_std):>8}"
                f"  {_format_val(t_stat):>7}"
                f"  {_format_val(hit, '.2%') if hit is not None else 'n/a':>6}"
                f"  {n_d:>4}"
            )

    # Write markdown report.
    md = _build_markdown(results, expanded, n_rows, n_dates)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_ic_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
