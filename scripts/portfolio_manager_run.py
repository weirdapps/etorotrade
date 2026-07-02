#!/usr/bin/env python3
"""S4 Portfolio Manager runner — sizes S3 final universe into concrete action plan.

Decision-support only: prints the plan but does NOT execute trades.

Usage:
    python3 scripts/portfolio_manager_run.py \\
        --universe path/to/s3_universe.json \\
        --cash-pct 0.29 \\
        [--nav 10000]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _athens_now() -> str:
    """Return current Athens time as YYYYMMDDHHMM (via shell TZ)."""  # pragma: no cover
    result = subprocess.run(
        ["date", "+%Y%m%d%H%M"],
        env={**os.environ, "TZ": "Europe/Athens"},
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="S4 Portfolio Manager — ERC action plan")
    parser.add_argument("--universe", required=True, help="Path to S3 final universe JSON file")
    parser.add_argument(
        "--cash-pct", type=float, required=True, help="Current cash as fraction (e.g. 0.29)"
    )
    parser.add_argument(
        "--nav", type=float, default=10_000.0, help="Portfolio NAV in USD (default 10000)"
    )
    args = parser.parse_args()

    # --- Load S3 universe ---
    with open(args.universe) as f:  # pragma: no cover
        final_universe: list[dict] = json.load(f)

    # --- Load current portfolio weights from input/portfolio.csv ---
    current_weights: dict[str, float] = {}
    portfolio_path = os.path.join(os.path.dirname(__file__), "..", "input", "portfolio.csv")
    try:  # pragma: no cover
        import csv

        with open(portfolio_path) as pf:
            reader = csv.DictReader(pf)
            for row in reader:
                is_buy_raw = row.get("isBuy", "").strip().lower()
                is_long = is_buy_raw in ("true", "1", "yes")
                if is_long:
                    ticker = row.get("ticker", "").strip()
                    inv_pct_raw = row.get("investmentPct", "0").strip()
                    try:
                        inv_frac = float(inv_pct_raw) / 100.0
                    except (ValueError, TypeError):
                        inv_frac = 0.0
                    if ticker:
                        current_weights[ticker] = inv_frac
    except FileNotFoundError:  # pragma: no cover
        print(
            f"[WARN] portfolio.csv not found at {portfolio_path}; treating book as empty.",
            file=sys.stderr,
        )

    # --- Resolve regime multiplier ---
    mult: float = 1.0
    regime_detail: dict = {}
    try:  # pragma: no cover
        from trade_modules.riskfirst.regime_state import resolve_regime_multiplier

        mult, regime_detail = resolve_regime_multiplier()  # pragma: no cover
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] regime resolution failed ({exc}); using mult=1.0", file=sys.stderr)

    # --- Compute deployable budget ---
    from trade_modules.portfolio_manager.budget import deployable_budget
    from trade_modules.portfolio_manager.sizer import size_book

    budget_frac = deployable_budget(args.cash_pct, target_cash_pct=0.07, regime_mult=mult)

    # --- Size the book ---
    action_plan = size_book(final_universe, current_weights, budget_frac=budget_frac, nav=args.nav)

    # --- Write output files ---
    ts = _athens_now()  # pragma: no cover
    downloads = os.path.expanduser("~/Downloads")
    md_path = os.path.join(downloads, f"{ts}_s4_action_plan.md")
    json_path = os.path.join(downloads, f"{ts}_s4_action_plan.json")

    try:  # pragma: no cover
        with open(json_path, "w") as jf:
            json.dump(action_plan, jf, indent=2)

        with open(md_path, "w") as mf:
            mf.write(f"# S4 Action Plan — {ts}\n\n")
            mf.write(f"**Cash pct:** {args.cash_pct:.1%}  ")
            mf.write(f"**Budget frac:** {budget_frac:.1%}  ")
            confirmed = regime_detail.get("confirmed_regime", "unknown")
            mf.write(f"**Regime:** {confirmed} (mult={mult:.2f})  ")
            mf.write(f"**NAV:** ${args.nav:,.0f}\n\n")

            mf.write("| Ticker | Action | Current% | Target% | Delta% | Conviction |\n")
            mf.write("|--------|--------|----------|---------|--------|------------|\n")
            for row in action_plan:
                mf.write(
                    f"| {row['ticker']} | {row['action']} "
                    f"| {row['current_pct']:.1%} "
                    f"| {row['target_pct']:.1%} "
                    f"| {row['delta_pct']:+.1%} "
                    f"| {row['conviction']} |\n"
                )
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] failed to write output files: {exc}", file=sys.stderr)

    # --- Summary to stdout ---
    buy_add = [r for r in action_plan if r["action"] in ("BUY", "ADD")]
    trim_sell = [r for r in action_plan if r["action"] in ("TRIM", "SELL")]
    budget_deployed = sum(r["delta_pct"] for r in buy_add)
    confirmed = regime_detail.get("confirmed_regime", "unknown")

    print(f"\n{'=' * 60}")
    print("S4 Portfolio Manager — Action Plan Summary")
    print(f"{'=' * 60}")
    print(f"  Cash pct:       {args.cash_pct:.1%}")
    print(f"  Budget frac:    {budget_frac:.1%}")
    print(f"  Regime:         {confirmed} (mult={mult:.2f})")
    print(f"  BUY/ADD:        {len(buy_add)} names  (deployed {budget_deployed:.1%})")
    print(f"  TRIM/SELL:      {len(trim_sell)} names")
    print(f"  Outputs:        {md_path}")
    print(f"               {json_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()  # pragma: no cover
