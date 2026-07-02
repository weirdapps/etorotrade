"""
universe_filter_report.py — S1 Universe Filter Validation & Live Effect Report

Two goals:
  A) LIVE effect: run filter_universe on today's etoro.csv → show counts
     (eligible / price_only / excluded), per-gate drop counts, dropped-name list.
  B) Referee-backed validation on the ONE dim present in history (tier/region):
     compare_tier_region_filters() runs harness.evaluate on (i) full set,
     (ii) excl-micro/small, (iii) each region alone → ranks by OOS alpha@T+30.
     Honest caveat: analyst/earnings/trend gates are NOT in historical rows
     → forward-gated only, no OOS claim.

Usage:
    python3 -m scripts.universe_filter_report

All I/O is inside main(). # pragma: no cover applied to all I/O + CLI code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pure helpers — no I/O
# ---------------------------------------------------------------------------


def summarize_filter_result(result: dict) -> dict:
    """Compute high-level summary counts from a filter_universe result.

    Returns a dict with keys:
        eligible, price_only, excluded, fundamental_total, eligible_pct
    """
    summary = result["summary"]
    fundamental_total = summary["fundamental"]
    eligible = summary["eligible"]
    pct = (eligible / fundamental_total * 100.0) if fundamental_total > 0 else 0.0
    return {
        "eligible": eligible,
        "price_only": summary["price_only"],
        "excluded": summary["excluded"],
        "fundamental_total": fundamental_total,
        "eligible_pct": round(pct, 1),
    }


def aggregate_gate_counts(result: dict) -> dict:
    """Return the per-gate failure counts from a filter_universe result."""
    return dict(result["summary"]["gate_failures"])


def aggregate_dropped_names(result: dict) -> list[dict]:
    """Return list of {ticker, reason} for every fundamental name that FAILED.

    Price-only assets (crypto, ETFs) are routed — they are NOT in this list.
    Excluded assets (no usable price) ARE in this list.
    """
    dropped: list[dict] = []
    # Fundamental failures: reasons dict, non-empty reasons list = failed
    reasons = result["reasons"]
    for tkr, rsns in reasons.items():
        if rsns:  # non-empty = failed at least one gate
            dropped.append({"ticker": tkr, "reason": "; ".join(rsns)})
    # Excluded (no price): always show
    excluded = result["excluded"]
    for tkr, reason in excluded.items():
        dropped.append({"ticker": tkr, "reason": reason})
    return dropped


def compare_tier_region_filters(
    results_rows: list[dict],
    harness_evaluate,
) -> list[dict]:
    """Compare universe quality across tier/region subsets via the referee.

    Subsets compared:
      - "full" — all rows
      - "excl_micro_small" — rows where tier not in {micro, small}
      - "region_us", "region_eu", "region_hk" — region-only slices

    Each subset is evaluated with harness_evaluate (same signature as
    trade_modules.validation.harness.evaluate).  Ranked by:
      1. OOS alpha at T+30 (from the first material family with OOS computed),
         or overall DSR if OOS not computed, or -999 sentinel.

    Args:
        results_rows: list of dicts from backtest_results.csv.
        harness_evaluate: callable with signature (rows: list[dict]) -> dict.
            Must return a dict with an "overall" key.

    Returns:
        Sorted list of dicts (best first), each with:
            name, subset_desc, n_rows, score, rank, passed, verdict
    """
    subsets: list[tuple[str, str, list[dict]]] = [
        ("full", "All rows (no tier/region filter)", results_rows),
        (
            "excl_micro_small",
            "Excl. MICRO+SMALL tiers (large/mega/mid only)",
            [r for r in results_rows if str(r.get("tier", "")).lower() not in ("micro", "small")],
        ),
        (
            "region_us",
            "US region only",
            [r for r in results_rows if str(r.get("region", "")).lower() == "us"],
        ),
        (
            "region_eu",
            "EU region only",
            [r for r in results_rows if str(r.get("region", "")).lower() == "eu"],
        ),
        (
            "region_hk",
            "HK region only",
            [r for r in results_rows if str(r.get("region", "")).lower() == "hk"],
        ),
    ]

    entries: list[dict] = []
    for name, desc, rows in subsets:
        verdict = harness_evaluate(rows)
        overall = verdict.get("overall", {})
        passed = overall.get("passed", False)

        # Extract score: prefer OOS alpha from any material family at T+30,
        # fallback to overall DSR, fallback to sentinel.
        score = _extract_score(verdict)

        entries.append(
            {
                "name": name,
                "subset_desc": desc,
                "n_rows": len(rows),
                "score": score,
                "passed": passed,
                "verdict": overall,
            }
        )

    # Sort descending by score (higher = better)
    entries.sort(key=lambda e: e["score"], reverse=True)
    for i, e in enumerate(entries, 1):
        e["rank"] = i

    return entries


def _extract_score(verdict: dict) -> float:
    """Extract a scalar quality score from a harness verdict dict.

    Priority:
      1. First material family's OOS alpha (T+30) when computed.
      2. Overall pooled DSR.
      3. Direct _subset_score (injected by fake harness in tests).
      4. Sentinel -999.0 (no data / not computed).
    """
    # Injected by fake harness in tests
    if "_subset_score" in verdict:
        raw = verdict["_subset_score"]
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass

    # OOS alpha from material families
    families = verdict.get("families", {})
    for _fam, fstats in families.items():
        if fstats.get("insufficient_data"):
            continue
        oos = fstats.get("oos", {})
        if oos.get("computed"):
            oa = oos.get("oos_alpha")
            if oa is not None:
                try:
                    return float(oa)
                except (TypeError, ValueError):
                    pass

    # Fallback: pooled DSR
    overall = verdict.get("overall", {})
    dsr = overall.get("dsr")
    if dsr is not None:
        try:
            return float(dsr)
        except (TypeError, ValueError):
            pass

    return -999.0


def _format_report(
    live_summary: dict,
    gate_counts: dict,
    dropped_names: list[dict],
    price_only_sample: list[str],
    tier_region_rankings: list[dict],
    data_date: str,
    n_total: int,
    forward_gated_pass_rates: dict,
    results_rows: int = 0,
) -> str:
    """Format the markdown report."""
    lines = [
        "# S1 Universe Filter — Validation & Live Effect Report",
        "",
        f"**Data date:** {data_date}  **Total universe:** {n_total:,} instruments",
        "",
        "---",
        "",
        "## A) Live Filter Effect (today's universe)",
        "",
        "### Routing Summary",
        "",
        "| Category | Count | % of total |",
        "|---|---|---|",
        f"| Fundamental-eligible (all gates pass) | {live_summary['eligible']:,} | "
        f"{live_summary['eligible'] / n_total * 100:.1f}% |",
        f"| Fundamental-assessed (all fundamental rows) | {live_summary['fundamental_total']:,} | "
        f"{live_summary['fundamental_total'] / n_total * 100:.1f}% |",
        f"| Price-only (crypto/ETF/leveraged) | {live_summary['price_only']:,} | "
        f"{live_summary['price_only'] / n_total * 100:.1f}% |",
        f"| Excluded (no usable price) | {live_summary['excluded']:,} | "
        f"{live_summary['excluded'] / n_total * 100:.1f}% |",
        f"| Eligible % of fundamental assessed | {live_summary['eligible_pct']}% | — |",
        "",
        "### Per-Gate Drop Counts (fundamental rows only)",
        "",
        "| Gate | Names dropped |",
        "|---|---|",
        f"| Cap floor <$2B (min_cap) | {gate_counts['min_cap']:,} |",
        f"| Analyst coverage <5 (min_analysts) | {gate_counts['min_analysts']:,} |",
        f"| Negative/missing earnings (positive_earnings) | {gate_counts['positive_earnings']:,} |",
        f"| Melting 52W <30 (trend) | {gate_counts['trend']:,} |",
        "",
        "> **Note on forward-gated gates:** analyst, earnings, and trend data are",
        "> point-in-time and NOT preserved in `backtest_results.csv` historical rows.",
        "> These gates are applied for correctness/prudence and accrue forward evidence,",
        "> but are not presented as historically validated.",
        "",
        "### Forward-Gated Pass Rates (live, no OOS claim)",
        "",
        "| Gate | Pass rate (today) |",
        "|---|---|",
        f"| Min analysts ≥5 | {forward_gated_pass_rates.get('analysts_pass_pct', 'N/A')} |",
        f"| Positive earnings (PET/PEF/FCF) | {forward_gated_pass_rates.get('earnings_pass_pct', 'N/A')} |",
        f"| Not melting (52W ≥30) | {forward_gated_pass_rates.get('trend_pass_pct', 'N/A')} |",
        "",
    ]

    # Dropped names (top 30 by first gate fail)
    lines += [
        "### Dropped Names (fundamental, sample — up to 30)",
        "",
        "| Ticker | Reason |",
        "|---|---|",
    ]
    for d in dropped_names[:30]:
        lines.append(f"| {d['ticker']} | {d['reason']} |")
    if len(dropped_names) > 30:
        lines.append(f"| … | (+{len(dropped_names) - 30} more) |")
    lines.append("")

    # Price-only sample
    lines += [
        "### Price-Only Routed (sample — up to 20)",
        "",
        ", ".join(price_only_sample[:20]),
        "",
        "---",
        "",
        "## B) Referee-Backed Validation — Tier/Region Filter Ranking",
        "",
        "> **Honest caveat:** only `tier` and `region` are present in",
        "> `backtest_results.csv`. OOS evidence covers ONLY these dimensions.",
        "> Analyst/earnings/trend gates are forward-gated (see § A above).",
        f"> History: {results_rows:,} rows across {len(tier_region_rankings)} subsets.",
        "",
        "| Rank | Subset | N rows | Score (OOS α or DSR) | PASS |",
        "|---|---|---|---|---|",
    ]
    for e in tier_region_rankings:
        score_str = f"{e['score']:.4f}" if e["score"] != -999.0 else "N/A (no data)"
        passed_str = "✓" if e["passed"] else "✗"
        lines.append(
            f"| {e['rank']} | {e['subset_desc']} | {e['n_rows']:,} | {score_str} | {passed_str} |"
        )

    lines += [
        "",
        "> **Score definition:** OOS alpha at T+30 (first material family with OOS computed);",
        "> fallback to pooled DSR; fallback to −999 (insufficient data).",
        "> Short-history caveat: all S0 DSR/OOS estimates carry single-regime risk",
        "> (~5 months of data, all bull market). Rankings are directional only.",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "- The live filter produces a **clean, routed universe** with explicit reasons",
        "  for every pass/fail decision.",
        "- The tier/region ranking shows which subsets the S0 referee scores best",
        "  on historical alpha/OOS — but with the short-history caveat.",
        "- Analyst/earnings/trend gates accrue evidence going forward.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main (pragma: no cover — all I/O)
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """Load data, run filter, run referee comparison, write report."""
    import csv
    import subprocess
    from pathlib import Path

    import pandas as pd

    from trade_modules.universe.filter import filter_universe
    from trade_modules.validation.harness import evaluate as harness_evaluate

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    repo_root = Path(__file__).parent.parent
    etoro_csv = repo_root / "yahoofinance" / "output" / "etoro.csv"
    backtest_csv = repo_root / "yahoofinance" / "output" / "backtest_results.csv"

    # ------------------------------------------------------------------
    # A) Live filter on etoro.csv
    # ------------------------------------------------------------------
    print(f"Loading {etoro_csv} …")
    df = pd.read_csv(etoro_csv)
    data_date_raw = (
        subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=short", str(etoro_csv)],
            capture_output=True,
            text=True,
            cwd=repo_root,
        ).stdout.strip()
        or "unknown"
    )

    result = filter_universe(df)
    live_summary = summarize_filter_result(result)
    gate_counts = aggregate_gate_counts(result)
    dropped_names = aggregate_dropped_names(result)
    price_only_sample = result["price_only"]

    # Forward-gated pass rates (live, point-in-time)
    fundamental_total = live_summary["fundamental_total"]

    def _pct(n: int) -> str:
        if fundamental_total == 0:
            return "N/A"
        return f"{n:,} / {fundamental_total:,} ({n / fundamental_total * 100:.1f}%)"

    analysts_pass = fundamental_total - gate_counts["min_analysts"]
    earnings_pass = fundamental_total - gate_counts["positive_earnings"]
    trend_pass = fundamental_total - gate_counts["trend"]
    forward_gated_pass_rates = {
        "analysts_pass_pct": _pct(analysts_pass),
        "earnings_pass_pct": _pct(earnings_pass),
        "trend_pass_pct": _pct(trend_pass),
    }

    # ------------------------------------------------------------------
    # B) Tier/region referee comparison
    # ------------------------------------------------------------------
    print(f"Loading {backtest_csv} …")
    results_rows: list[dict] = []
    with open(backtest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["horizon"] = int(float(row["horizon"]))
            except (ValueError, KeyError):
                pass
            for key in ("alpha", "net_alpha", "future_price", "stock_return", "spy_return"):
                try:
                    row[key] = float(row[key]) if row.get(key) else None
                except (ValueError, TypeError):
                    row[key] = None
            results_rows.append(row)

    def _wrapped_evaluate(rows: list[dict]) -> dict:
        """Wrap harness.evaluate with standard kwargs."""
        if not rows:
            return {"overall": {"passed": False, "dsr": None, "reasons": ["no rows"]}}
        return harness_evaluate(rows, family_key="signal", min_obs=30)

    print("Running referee comparison across tier/region subsets …")
    rankings = compare_tier_region_filters(results_rows, _wrapped_evaluate)

    # ------------------------------------------------------------------
    # Format + write report
    # ------------------------------------------------------------------
    report_md = _format_report(
        live_summary=live_summary,
        gate_counts=gate_counts,
        dropped_names=dropped_names,
        price_only_sample=price_only_sample,
        tier_region_rankings=rankings,
        data_date=data_date_raw,
        n_total=len(df),
        forward_gated_pass_rates=forward_gated_pass_rates,
        results_rows=len(results_rows),
    )

    ts = subprocess.run(
        ["date", "+%Y%m%d%H%M"],
        capture_output=True,
        text=True,
        env={"TZ": "Europe/Athens", "PATH": "/bin:/usr/bin"},
    ).stdout.strip()
    out_path = Path.home() / "Downloads" / f"{ts}_universe_filter_report.md"
    out_path.write_text(report_md, encoding="utf-8")
    print(f"\nReport written to: {out_path}")

    # Print summary to stdout
    print("\n=== LIVE COUNTS ===")
    print(f"  Fundamental-eligible: {live_summary['eligible']:,}")
    print(f"  Fundamental-assessed: {live_summary['fundamental_total']:,}")
    print(f"  Price-only:           {live_summary['price_only']:,}")
    print(f"  Excluded:             {live_summary['excluded']:,}")
    print("\n=== GATE DROPS ===")
    for gate, count in gate_counts.items():
        print(f"  {gate}: {count:,}")
    print("\n=== TIER/REGION RANKING ===")
    for e in rankings:
        score_str = f"{e['score']:.4f}" if e["score"] != -999.0 else "N/A"
        print(
            f"  #{e['rank']} {e['name']:<20} n={e['n_rows']:>6,}  score={score_str}  passed={e['passed']}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
