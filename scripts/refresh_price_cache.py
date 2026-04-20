"""
Daily Price Cache Refresh (CIO v17 op #8)

Walks every ticker referenced in the most recent concordance + portfolio
+ buy.csv and refreshes its 1y daily price parquet at
~/.weirdapps-trading/price_cache/{ticker}_1y.parquet.

Intended to run from a GitHub Actions cron at ~02:00 UTC after the
daily-signals job. Local-dev users can run it ad hoc.

Usage:
    python scripts/refresh_price_cache.py [--force]
    --force: also refresh entries currently labeled "stale" (last bar
             2-7 days old) instead of only "missing"/"very_stale".
"""

import csv
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.price_cache import (
    cache_stats,
    refresh_if_stale,
    write_health_report,
)

PORTFOLIO_CSV = Path(__file__).parent.parent / "yahoofinance" / "output" / "portfolio.csv"
BUY_CSV = Path(__file__).parent.parent / "yahoofinance" / "output" / "buy.csv"
CONCORDANCE_DIR = Path.home() / ".weirdapps-trading" / "committee" / "history"


def collect_tickers() -> set:
    tickers: set = {"SPY"}  # always cache benchmark

    if PORTFOLIO_CSV.exists():
        with open(PORTFOLIO_CSV) as f:
            for row in csv.DictReader(f):
                t = row.get("TKR", "").strip()
                if t:
                    tickers.add(t)

    if BUY_CSV.exists():
        with open(BUY_CSV) as f:
            for row in csv.DictReader(f):
                t = row.get("TKR", "").strip()
                if t:
                    tickers.add(t)

    # Also collect from the most recent concordance archives.
    if CONCORDANCE_DIR.is_dir():
        for fpath in sorted(CONCORDANCE_DIR.glob("concordance-*.json"))[-5:]:
            try:
                data = json.load(open(fpath))
            except Exception:
                continue
            items = data.get("concordance", []) if isinstance(data, dict) else data
            if not isinstance(items, list):
                continue
            for stock in items:
                if isinstance(stock, dict):
                    t = stock.get("ticker", "")
                    if t:
                        tickers.add(t)

    return tickers


def main():
    force = "--force" in sys.argv

    print("\n" + "=" * 60)
    print("  PRICE CACHE REFRESH")
    print("=" * 60)

    before = cache_stats()
    print(f"\nBefore: {before}")

    tickers = collect_tickers()
    print(f"Tickers to verify: {len(tickers)}")

    results = refresh_if_stale(tickers, force=force)
    if results:
        ok = sum(1 for v in results.values() if v == "ok")
        fail = sum(1 for v in results.values() if v == "fail")
        empty = sum(1 for v in results.values() if v == "empty")
        print(f"Refreshed {len(results)}: {ok} ok, {fail} fail, {empty} empty")
    else:
        print("No refresh needed (all entries fresh)")

    after = cache_stats()
    print(f"After: {after}")

    health = write_health_report()
    print(f"Health report written → {health}")


if __name__ == "__main__":
    main()
