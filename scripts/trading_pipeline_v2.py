#!/usr/bin/env python3
"""S5 Trading Pipeline — end-to-end decision-support runner.

SHADOW ONLY — this script NEVER places orders.
It produces an action plan (BUY/ADD/HOLD/TRIM/SELL with sizes) for
manual review and execution by the user.

Usage:
    python3 scripts/trading_pipeline_v2.py [--cash-pct 0.20]

Outputs (to ~/Downloads):
    YYYYMMDDHHMM_trading_pipeline_v2_action_plan.md
    YYYYMMDDHHMM_trading_pipeline_v2_action_plan.json

All IO is in this file. The core (trade_modules/pipeline_v2/) is pure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          SHADOW / DECISION-SUPPORT ONLY — S5 PIPELINE           ║
║  This plan is for manual review. No orders are placed.           ║
║  User executes manually. Go-live is user-triggered.              ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ---------------------------------------------------------------------------
# IO helpers — all marked # pragma: no cover
# ---------------------------------------------------------------------------


def _get_timestamp_athens() -> str:  # pragma: no cover
    result = subprocess.run(
        ["bash", "-c", "TZ='Europe/Athens' date '+%Y%m%d%H%M'"], capture_output=True, text=True
    )
    return result.stdout.strip() or datetime.now().strftime("%Y%m%d%H%M")


def _get_generated_at_athens() -> str:  # pragma: no cover
    result = subprocess.run(
        ["bash", "-c", "TZ='Europe/Athens' date '+%Y-%m-%dT%H:%M:%S'"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or datetime.now().isoformat()


def _find_etoro_csv() -> str:  # pragma: no cover
    path = os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/output/etoro.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: etoro.csv not found at {path}")
    return path


def _find_portfolio_csv() -> str | None:  # pragma: no cover
    """Return the first portfolio.csv that exists, or None."""
    candidates = [
        os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/input/portfolio.csv"),
        os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/output/portfolio.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_universe(path: str) -> pd.DataFrame:  # pragma: no cover
    return pd.read_csv(path, dtype=str)


def _load_portfolio(path: str | None) -> pd.DataFrame:  # pragma: no cover
    if path is None:
        print("WARNING: No portfolio.csv found — treating as empty portfolio.")
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)


def _cash_from_portfolio(portfolio_df: pd.DataFrame, fallback: float) -> float:  # pragma: no cover
    """Derive cash % from portfolio totals; fall back to CLI arg."""
    if portfolio_df.empty:
        return fallback
    try:
        # totalInvestmentPct sum gives gross exposure
        invested_pct = (
            pd.to_numeric(portfolio_df["totalInvestmentPct"], errors="coerce").fillna(0.0).sum()
        )
        cash_pct = max(0.0, 1.0 - invested_pct / 100.0)
        return cash_pct
    except Exception:
        return fallback


def _get_regime() -> tuple[float, dict]:  # pragma: no cover
    """Call resolve_regime_multiplier; return (mult, detail). Fail → NEUTRAL."""
    try:
        from trade_modules.riskfirst.regime_state import resolve_regime_multiplier

        mult, detail = resolve_regime_multiplier()
        return float(mult), detail
    except Exception as exc:
        print(f"WARNING: regime resolution failed ({exc}); using NEUTRAL 0.80")
        return 0.80, {"raw_regime": None, "confirmed_regime": "NEUTRAL", "applied_multiplier": 0.80}


def _write_outputs(plan: dict, ts: str) -> tuple[str, str]:  # pragma: no cover
    """Write action plan to ~/Downloads as .md + .json. Return (md_path, json_path)."""
    out_dir = os.path.expanduser("~/Downloads")
    base = f"{ts}_trading_pipeline_v2_action_plan"
    md_path = os.path.join(out_dir, f"{base}.md")
    json_path = os.path.join(out_dir, f"{base}.json")

    # JSON
    with open(json_path, "w") as f:
        json.dump(plan, f, indent=2, default=str)

    # Markdown
    with open(md_path, "w") as f:
        f.write(_render_md(plan))

    return md_path, json_path


def _render_md(plan: dict) -> str:  # pragma: no cover
    lines = [
        "# S5 Trading Pipeline — Action Plan",
        "",
        f"**Generated:** {plan['generated_at']}",
        f"**Regime Multiplier:** {plan['regime_mult']:.2f}",
        f"**Deployable Budget:** {plan['budget_frac']:.2%}",
        f"**Resulting Gross Exposure:** {plan['resulting_gross']:.2%}",
        f"**Resulting Cash:** {plan['resulting_cash']:.2%}",
        "",
        "> **SHADOW / DECISION-SUPPORT ONLY** — No orders are placed by this pipeline.",
        "",
        "## Caveats",
        "",
    ]
    for c in plan.get("caveats", []):
        lines.append(f"- {c}")

    def _fmt_rows(rows: list[dict], label: str) -> list[str]:
        if not rows:
            return [f"\n### {label}\n\n_None_\n"]
        out = [f"\n### {label}\n"]
        out.append("| Ticker | Action | Conviction | Current% | Target% | Delta% | Rationale |")
        out.append("|--------|--------|-----------|----------|---------|--------|-----------|")
        for r in rows:
            ticker = r.get("ticker", "")
            action = r.get("action", "")
            conv = f"{r['conviction']:.1f}" if r.get("conviction") is not None else "—"
            cur = f"{r['current_pct']:.2%}" if r.get("current_pct") is not None else "—"
            tgt = f"{r['target_pct']:.2%}" if r.get("target_pct") is not None else "—"
            dlt = f"{r['delta_pct']:.2%}" if r.get("delta_pct") is not None else "—"
            rat = (r.get("rationale") or "")[:80]
            out.append(f"| {ticker} | {action} | {conv} | {cur} | {tgt} | {dlt} | {rat} |")
        return out

    lines += _fmt_rows(plan.get("actions", []), "Actions (BUY / ADD / TRIM / SELL)")
    lines += _fmt_rows(plan.get("holds", []), "Holds")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="S5 Trading Pipeline (SHADOW)")
    parser.add_argument(
        "--cash-pct",
        type=float,
        default=None,
        help="Override cash % (e.g. 0.20 = 20%%). Auto-derived from portfolio if omitted.",
    )
    args = parser.parse_args()

    print(BANNER)

    # -- Load data
    etoro_path = _find_etoro_csv()
    portfolio_path = _find_portfolio_csv()

    print(f"Loading universe: {etoro_path}")
    universe_df = _load_universe(etoro_path)
    print(f"  {len(universe_df)} instruments loaded.")

    print(f"Loading portfolio: {portfolio_path or '(none)'}")
    portfolio_df = _load_portfolio(portfolio_path)
    if not portfolio_df.empty:
        print(f"  {len(portfolio_df)} positions loaded.")

    # -- Cash %
    if args.cash_pct is not None:
        cash_pct = float(args.cash_pct)
        print(f"Cash % (CLI override): {cash_pct:.2%}")
    else:
        cash_pct = _cash_from_portfolio(portfolio_df, fallback=0.10)
        print(f"Cash % (derived from portfolio): {cash_pct:.2%}")

    # -- Regime
    regime_mult, regime_detail = _get_regime()
    print(f"Regime: {regime_detail.get('confirmed_regime', '?')} (mult={regime_mult:.2f})")

    # -- Timestamp
    ts = _get_timestamp_athens()
    generated_at = _get_generated_at_athens()

    # -- Sector map (offline CSV: market.csv + usindex.csv)
    from trade_modules.pipeline_v2.sectors import load_sector_map

    _sector_csv_paths = [
        os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/input/market.csv"),
        os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/input/usindex.csv"),
    ]
    sector_map = load_sector_map(_sector_csv_paths)
    print(f"Sector map loaded: {len(sector_map)} symbols covered.")

    # Count how many universe/held names resolve to a sector
    if "TKR" in universe_df.columns:
        universe_tkrs = [str(t).strip() for t in universe_df["TKR"].dropna() if str(t).strip()]
        got_sector = sum(1 for t in universe_tkrs if sector_map.get(t.upper()))
        print(
            f"  Universe sector coverage: {got_sector}/{len(universe_tkrs)} names "
            f"have a sector ({len(universe_tkrs) - got_sector} will be excluded from the cap)."
        )
    if not portfolio_df.empty and "symbol" in portfolio_df.columns:
        held_tkrs = [str(t).strip() for t in portfolio_df["symbol"].dropna() if str(t).strip()]
        held_with_sector = sum(1 for t in held_tkrs if sector_map.get(t.upper()))
        print(
            f"  Held sector coverage: {held_with_sector}/{len(held_tkrs)} positions have a sector."
        )

    # -- Run pipeline (pure)
    from trade_modules.pipeline_v2.orchestrator import run_pipeline

    print("\nRunning S1→S2→S3→S4 pipeline...")
    plan = run_pipeline(
        universe_df=universe_df,
        portfolio_df=portfolio_df,
        regime_mult=regime_mult,
        cash_pct=cash_pct,
        generated_at=generated_at,
        sector_map=sector_map,
    )

    # -- Write outputs
    md_path, json_path = _write_outputs(plan, ts)

    # -- Summary
    print(f"\n{'=' * 64}")
    print(f"Actions: {len(plan['actions'])}   Holds: {len(plan['holds'])}")
    print(
        f"Budget: {plan['budget_frac']:.2%}  Gross: {plan['resulting_gross']:.2%}  Cash: {plan['resulting_cash']:.2%}"
    )
    # Log sector cap status: either enforced (with coverage stats) or inoperative.
    for c in plan["caveats"]:
        if "Sector cap" in c:
            if "NOT enforced" in c:
                print(
                    "WARNING: sector cap is NOT enforced (no sector map) — "
                    "review sector concentration manually.",
                    file=sys.stderr,
                )
            else:
                print(f"INFO: {c}")
    for c in plan["caveats"]:
        print(f"  CAVEAT: {c}")
    print("\nOutputs written:")
    print(f"  {md_path}")
    print(f"  {json_path}")
    print(BANNER)


if __name__ == "__main__":
    main()
