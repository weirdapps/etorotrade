"""Merge per-shard market-scan outputs into yahoofinance/output/etoro.csv.

Each parallel CI shard produces an ``etoro_shard_<k>.csv`` (a round-robin slice of
the universe). This recombines them — concatenate, dedupe on TKR, sort by market
cap — into the canonical ``output/etoro.csv`` that the downstream trade-filter
steps (``trade.py -o t -t b|s|h``) consume.

Usage:
    python scripts/merge_shards.py
    python scripts/merge_shards.py --shard-dir . --output yahoofinance/output/etoro.csv
"""

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# Running as a script (`python scripts/merge_shards.py`) puts scripts/ on
# sys.path, not the repo root — bootstrap the root so trade_modules imports.
sys.path.insert(0, str(_ROOT))

from trade_modules.sharding import merge_shard_csvs  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = str(_ROOT / "yahoofinance" / "output" / "etoro.csv")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir", default=".", help="Directory containing etoro_shard_*.csv files"
    )
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Merged output path")
    parser.add_argument("--pattern", default="etoro_shard_*.csv", help="Shard filename glob")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = merge_shard_csvs(args.shard_dir, args.output, pattern=args.pattern)
    logger.info("Merged %d tickers into %s", rows, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
