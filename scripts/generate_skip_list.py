"""Generate yfinance_skip.csv — tickers to skip in the daily signal scan.

Two skip criteria:
1. yfinance failures: tickers in input/etoro.csv but absent from output/etoro.csv
2. Low analyst coverage: tickers in output with <3 analysts (structurally INCONCLUSIVE)

Portfolio tickers are NEVER skipped.

Usage:
    python scripts/generate_skip_list.py
    python scripts/generate_skip_list.py --min-analysts 3  (default)
    python scripts/generate_skip_list.py --min-analysts 0  (skip only yfinance failures)
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_DEFAULT_INPUT = str(_ROOT / "yahoofinance" / "input" / "etoro.csv")
_DEFAULT_OUTPUT = str(_ROOT / "yahoofinance" / "output" / "etoro.csv")
_DEFAULT_PORTFOLIO = str(_ROOT / "yahoofinance" / "input" / "portfolio.csv")
_DEFAULT_SKIP_FILE = str(_ROOT / "yahoofinance" / "input" / "yfinance_skip.csv")
_DEFAULT_MIN_ANALYSTS = 3


def _read_symbols(
    path: str, columns: tuple[str, ...] = ("symbol", "Symbol", "SYMBOL", "TKR", "ticker", "Ticker")
) -> set[str]:
    """Read ticker symbols from a CSV, trying multiple column names."""
    if not os.path.exists(path):
        return set()
    symbols: set[str] = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        col_name = None
        for col in columns:
            if col in (reader.fieldnames or []):
                col_name = col
                break
        if not col_name:
            return set()
        for row in reader:
            val = (row.get(col_name) or "").strip().upper()
            if val:
                symbols.add(val)
    return symbols


def _read_symbols_simple(path: str, col: str) -> set[str]:
    """Read symbols from a single known column."""
    if not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            val = (row.get(col) or "").strip().upper()
            if val:
                out.add(val)
    return out


def _read_output_analyst_counts(path: str) -> dict[str, int]:
    """Read {TICKER: analyst_count} from output/etoro.csv."""
    if not os.path.exists(path):
        return {}
    counts: dict[str, int] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            tkr = (row.get("TKR") or "").strip().upper()
            if not tkr:
                continue
            raw = (row.get("#A") or row.get("#T") or "").strip()
            try:
                counts[tkr] = int(float(raw))
            except (ValueError, TypeError):
                counts[tkr] = 0
    return counts


def generate_skip_list(
    input_csv: str = _DEFAULT_INPUT,
    output_csv: str = _DEFAULT_OUTPUT,
    portfolio_csv: str = _DEFAULT_PORTFOLIO,
    skip_file: str = _DEFAULT_SKIP_FILE,
    min_analysts: int = _DEFAULT_MIN_ANALYSTS,
) -> dict[str, int]:
    """Generate the skip list and write to skip_file.

    Returns summary dict with counts.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_symbols = _read_symbols_simple(input_csv, "symbol")
    output_symbols = _read_symbols_simple(output_csv, "TKR")
    portfolio_symbols = _read_symbols(portfolio_csv)
    analyst_counts = _read_output_analyst_counts(output_csv)

    logger.info(
        "Input: %d, Output: %d, Portfolio: %d",
        len(input_symbols),
        len(output_symbols),
        len(portfolio_symbols),
    )

    # Criterion 1: yfinance failures (in input but not in output)
    yfinance_failures = input_symbols - output_symbols
    logger.info("yfinance failures (no output): %d", len(yfinance_failures))

    # Criterion 2: low analyst coverage (in output but <min_analysts)
    low_analyst = set()
    if min_analysts > 0:
        for tkr, count in analyst_counts.items():
            if count < min_analysts:
                low_analyst.add(tkr)
        logger.info("Low analyst (<%d): %d", min_analysts, len(low_analyst))

    # Combine and exclude portfolio
    skip_set = (yfinance_failures | low_analyst) - portfolio_symbols
    logger.info(
        "Combined skip list: %d (after excluding %d portfolio tickers)",
        len(skip_set),
        len(portfolio_symbols),
    )

    # Write
    with open(skip_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol"])
        for sym in sorted(skip_set):
            writer.writerow([sym])

    logger.info("Wrote %s (%d tickers)", skip_file, len(skip_set))

    return {
        "input_count": len(input_symbols),
        "output_count": len(output_symbols),
        "yfinance_failures": len(yfinance_failures),
        "low_analyst": len(low_analyst),
        "portfolio_excluded": len(portfolio_symbols),
        "skip_total": len(skip_set),
        "daily_scan_estimate": len(input_symbols) - len(skip_set),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate yfinance skip list")
    parser.add_argument(
        "--min-analysts",
        type=int,
        default=_DEFAULT_MIN_ANALYSTS,
        help=f"Min analyst count threshold (default: {_DEFAULT_MIN_ANALYSTS})",
    )
    parser.add_argument("--input", default=_DEFAULT_INPUT)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--portfolio", default=_DEFAULT_PORTFOLIO)
    parser.add_argument("--skip-file", default=_DEFAULT_SKIP_FILE)
    args = parser.parse_args()

    result = generate_skip_list(
        input_csv=args.input,
        output_csv=args.output,
        portfolio_csv=args.portfolio,
        skip_file=args.skip_file,
        min_analysts=args.min_analysts,
    )

    print("\nSummary:")
    print(f"  Input universe:      {result['input_count']:,}")
    print(f"  Output (had data):   {result['output_count']:,}")
    print(f"  yfinance failures:   {result['yfinance_failures']:,}")
    print(f"  Low analyst (<{args.min_analysts}):    {result['low_analyst']:,}")
    print(f"  Portfolio excluded:  {result['portfolio_excluded']:,}")
    print(f"  Skip list total:     {result['skip_total']:,}")
    print(
        f"  Daily scan estimate: {result['daily_scan_estimate']:,} tickers (~{result['daily_scan_estimate'] * 2.4 / 3600:.1f}h)"
    )

    sys.exit(0)
