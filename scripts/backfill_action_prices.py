#!/usr/bin/env python3
"""
Backfill missing prices in committee action_log.jsonl.

One-time script to fix entries where price_at_recommendation is null or 0.
Uses yfinance to fetch the closing price on the committee_date for each entry.

Usage:
    python scripts/backfill_action_prices.py [--dry-run]
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ACTION_LOG = Path(__file__).parent.parent / "data" / "committee" / "action_log.jsonl"


def backfill(dry_run: bool = False) -> None:
    if not ACTION_LOG.exists():
        print(f"Action log not found: {ACTION_LOG}")
        return

    # Load all entries
    entries = []
    with open(ACTION_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Find entries needing backfill
    needs_fix = []
    for i, entry in enumerate(entries):
        price = entry.get("price_at_recommendation")
        if price is None or price == 0 or price == "0":
            needs_fix.append(i)

    if not needs_fix:
        print("All entries have valid prices. Nothing to backfill.")
        return

    print(f"Found {len(needs_fix)} entries needing price backfill out of {len(entries)} total.")

    if dry_run:
        # Just report what would be fixed
        for i in needs_fix:
            e = entries[i]
            print(f"  {e.get('committee_date')} | {e.get('ticker'):6s} | {e.get('action'):5s} | price={e.get('price_at_recommendation')}")
        print(f"\nDry run: {len(needs_fix)} entries would be backfilled.")
        return

    # Group by (ticker, date) to minimize yfinance calls
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance required. Install with: pip install yfinance")
        return

    ticker_dates = {}
    for i in needs_fix:
        e = entries[i]
        ticker = e.get("ticker", "")
        date = e.get("committee_date", "")
        if ticker and date:
            key = (ticker, date)
            if key not in ticker_dates:
                ticker_dates[key] = []
            ticker_dates[key].append(i)

    print(f"Fetching prices for {len(ticker_dates)} unique (ticker, date) pairs...")

    fixed = 0
    failed = 0

    for (ticker, date), indices in ticker_dates.items():
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            # Fetch a window around the date to handle weekends/holidays
            start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
            end = (dt + timedelta(days=3)).strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end)

            if hist.empty:
                print(f"  FAIL: {ticker} on {date} — no history")
                failed += len(indices)
                continue

            # Find the closest trading day on or before the committee date
            hist.index = hist.index.tz_localize(None)
            valid = hist[hist.index <= dt]
            if valid.empty:
                valid = hist  # Take earliest available
            price = float(valid["Close"].iloc[-1])

            for idx in indices:
                entries[idx]["price_at_recommendation"] = round(price, 4)
                entries[idx]["price_backfilled"] = True
            fixed += len(indices)
            print(f"  OK:   {ticker} on {date} → ${price:.2f} ({len(indices)} entries)")

        except Exception as e:
            print(f"  FAIL: {ticker} on {date} — {e}")
            failed += len(indices)

    # Write back
    if fixed > 0:
        backup = ACTION_LOG.with_suffix(".jsonl.bak")
        with open(backup, "w") as f:
            with open(ACTION_LOG) as src:
                f.write(src.read())
        print(f"\nBackup saved to {backup}")

        with open(ACTION_LOG, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    print(f"\nDone: {fixed} fixed, {failed} failed, {len(entries) - len(needs_fix)} already valid.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    backfill(dry_run=dry)
