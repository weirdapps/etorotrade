"""Grade one scan shard on how many of its rows came back with no price.

A scan that loses its Yahoo session keeps running, writes a blank row for every later name and
exits 0 (2026-11-15: two of six shards, 2,092 names). This is the check the shard runs on its
own output, so a degraded slice is retried or fails instead of being merged and committed.

Usage:
    python scripts/shard_quality.py etoro_shard_2.csv            # exit 1 when degraded
    python scripts/shard_quality.py --better A.csv B.csv         # print the path with fewer blanks
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from trade_modules.sharding import (  # noqa: E402
    MAX_BLANK_PRICE_SHARE,
    blank_price_counts,
    shard_is_degraded,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="Shard CSV (two with --better)")
    parser.add_argument("--max-blank-share", type=float, default=MAX_BLANK_PRICE_SHARE)
    parser.add_argument(
        "--better", action="store_true", help="Print which of two CSVs has fewer blank prices"
    )
    args = parser.parse_args(argv)

    if args.better:
        if len(args.csv) != 2:
            parser.error("--better takes exactly two CSV paths")
        (rows_a, blank_a), (rows_b, blank_b) = (blank_price_counts(p) for p in args.csv)
        # More PRICED rows wins; a tie keeps the first, so a retry must actually be better.
        print(args.csv[1] if rows_b - blank_b > rows_a - blank_a else args.csv[0])
        return 0

    if len(args.csv) != 1:
        parser.error("exactly one CSV path, or --better with two")
    rows, blank = blank_price_counts(args.csv[0])
    share = blank / rows if rows else 1.0
    degraded = shard_is_degraded(rows, blank, args.max_blank_share)
    print(
        f"{args.csv[0]}: {rows} rows, {blank} with no price ({share:.1%}); "
        f"limit {args.max_blank_share:.0%}: {'DEGRADED' if degraded else 'ok'}"
    )
    return 1 if degraded else 0


if __name__ == "__main__":
    sys.exit(main())
