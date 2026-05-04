#!/usr/bin/env python3
"""Validate a midday brief post against a market snapshot.

Extracts every numerical claim from the post text and cross-checks
against the yfinance snapshot JSON. Blocks posting if any market-data
number is unverified.

Usage:
    python scripts/validate_brief.py post.txt snapshot.json
    # exit 0 = PASS, exit 1 = FAIL
"""

import csv
import json
import re
import sys
from pathlib import Path

ETORO_CSV = Path(__file__).resolve().parent.parent / "yahoofinance" / "output" / "etoro.csv"

PCT_TOLERANCE = 0.35
PRICE_TOLERANCE_RATIO = 0.015

NEWS_CONTEXT_PATTERNS = [
    r"revenue",
    r"EPS",
    r"earnings",
    r"YoY",
    r"growth",
    r"\bbid\s+(?:for|of)\b",
    r"cash\s+pile",
    r"takeover",
    r"guidance",
    r"quarter",
    r"margin",
    r"surprise",
    r"beat",
    r"miss",
    r"approval",
    r"chars?\b",
    r"/5000",
    r"copier",
    r"AGM",
    r"net\s+selling",
]

NEWS_CONTEXT_RE = re.compile("|".join(NEWS_CONTEXT_PATTERNS), re.IGNORECASE)

TICKER_RE = re.compile(r"\$([A-Z][A-Z0-9]+(?:\.[A-Z]+)?)")
PCT_RE = re.compile(r"([+-]\d+\.?\d*)%")
PRICE_RE = re.compile(r"\$(\d[\d,]*\.?\d*)")

# Map display names in post text to yfinance tickers for instrument-aware validation
DISPLAY_TO_TICKER = {
    "dax": "^GDAXI",
    "cac": "^FCHI",
    "stoxx 50": "^STOXX50E",
    "stoxx 600": "^STOXX",
    "stoxx50": "^STOXX50E",
    "ftse": "^FTSE",
    "nikkei": "^N225",
    "hang seng": "^HSI",
    "hsi": "^HSI",
    "kospi": "^KS11",
    "asx": "^AXJO",
    "shanghai": "000001.SS",
    " es ": "ES=F",
    " nq ": "NQ=F",
    " ym ": "YM=F",
    "brent": "BZ=F",
    "wti": "CL=F",
    "gold": "GC=F",
    "10y": "^TNX",
    "dxy": "DX-Y.NYB",
    "eur/usd": "EURUSD=X",
    "usd/jpy": "JPY=X",
}


def _context_window(text: str, match_start: int, window: int = 80) -> str:
    start = max(0, match_start - window)
    end = min(len(text), match_start + window)
    return text[start:end]


def _is_news_sourced(text: str, match_start: int) -> bool:
    ctx = _context_window(text, match_start, window=120)
    return bool(NEWS_CONTEXT_RE.search(ctx))


def _is_ticker_price(text: str, match_start: int) -> bool:
    """Check if a $NNN pattern is actually a $TICKER reference."""
    before = text[max(0, match_start - 1) : match_start]
    after_end = match_start + 20
    after = text[match_start:after_end]
    ticker_match = TICKER_RE.match(after)
    if ticker_match:
        candidate = ticker_match.group(1)
        if candidate.isalpha() or "." in candidate:
            return True
    return False


def _find_nearby_instrument(text: str, match_start: int) -> str | None:
    """Find the closest instrument display name preceding the percentage."""
    import unicodedata

    lookback = text[max(0, match_start - 30) : match_start]
    lookback_norm = unicodedata.normalize("NFKD", lookback).lower()

    best_ticker = None
    best_dist = 999

    for display_name, ticker in DISPLAY_TO_TICKER.items():
        idx = lookback_norm.rfind(display_name)
        if idx >= 0:
            dist = len(lookback_norm) - idx - len(display_name)
            if dist < best_dist:
                best_dist = dist
                best_ticker = ticker

    return best_ticker


def extract_market_percentages(text: str) -> list[dict]:
    results = []
    for m in PCT_RE.finditer(text):
        if _is_news_sourced(text, m.start()):
            continue
        results.append(
            {
                "value": float(m.group(1)),
                "raw": m.group(0),
                "context": _context_window(text, m.start(), 60),
                "pos": m.start(),
                "instrument": _find_nearby_instrument(text, m.start()),
            }
        )
    return results


def extract_market_prices(text: str) -> list[dict]:
    results = []
    for m in PRICE_RE.finditer(text):
        if _is_ticker_price(text, m.start()):
            continue
        if _is_news_sourced(text, m.start()):
            continue
        raw_num = m.group(1).replace(",", "")
        try:
            val = float(raw_num)
        except ValueError:
            continue
        if val < 1:
            continue
        results.append(
            {
                "value": val,
                "raw": m.group(0),
                "context": _context_window(text, m.start(), 60),
                "pos": m.start(),
            }
        )
    return results


def extract_tickers(text: str) -> list[str]:
    return list(set(TICKER_RE.findall(text)))


def load_etoro_tickers() -> set[str]:
    if not ETORO_CSV.exists():
        return set()
    tickers = set()
    with open(ETORO_CSV) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                tickers.add(row[0].upper())
    return tickers


def validate(post_text: str, snapshot: dict) -> dict:
    instruments = snapshot.get("instruments", {})

    all_pcts = []
    all_prices = []
    for data in instruments.values():
        all_pcts.append(data["change_pct"])
        all_prices.append(data["price"])
        all_prices.append(data["prev_close"])

    pct_errors = []
    for item in extract_market_percentages(post_text):
        val = item["value"]
        inst_ticker = item.get("instrument")
        if inst_ticker and inst_ticker in instruments:
            ref = instruments[inst_ticker]["change_pct"]
            matched = abs(val - ref) <= PCT_TOLERANCE
        else:
            matched = any(abs(val - ref) <= PCT_TOLERANCE for ref in all_pcts)
        if not matched:
            pct_errors.append(item)

    price_errors = []
    for item in extract_market_prices(post_text):
        val = item["value"]
        matched = any(
            abs(val - ref) / max(ref, 0.01) <= PRICE_TOLERANCE_RATIO for ref in all_prices
        )
        if not matched:
            price_errors.append(item)

    tickers_in_post = extract_tickers(post_text)
    etoro_universe = load_etoro_tickers()
    ticker_errors = []
    if etoro_universe:
        for t in tickers_in_post:
            if t.upper() not in etoro_universe:
                ticker_errors.append(t)

    passed = not pct_errors and not price_errors and not ticker_errors

    return {
        "passed": passed,
        "market_pcts_checked": len(extract_market_percentages(post_text)),
        "market_prices_checked": len(extract_market_prices(post_text)),
        "tickers_checked": len(tickers_in_post),
        "pct_errors": pct_errors,
        "price_errors": price_errors,
        "ticker_errors": ticker_errors,
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <post.txt> <snapshot.json>", file=sys.stderr)
        return 2

    post_path = Path(sys.argv[1])
    snap_path = Path(sys.argv[2])

    if not post_path.exists():
        print(f"Error: {post_path} not found", file=sys.stderr)
        return 2
    if not snap_path.exists():
        print(f"Error: {snap_path} not found", file=sys.stderr)
        return 2

    post_text = post_path.read_text()
    snapshot = json.loads(snap_path.read_text())

    result = validate(post_text, snapshot)

    if result["passed"]:
        print(
            f"PASS: {result['market_pcts_checked']} percentages, "
            f"{result['market_prices_checked']} prices, "
            f"{result['tickers_checked']} tickers verified"
        )
        return 0

    print("FAIL: unverified claims found\n")
    if result["pct_errors"]:
        print("Unverified percentages:")
        for e in result["pct_errors"]:
            print(f"  {e['raw']:>8s}  ...{e['context'].strip()}...")
    if result["price_errors"]:
        print("\nUnverified prices:")
        for e in result["price_errors"]:
            print(f"  {e['raw']:>10s}  ...{e['context'].strip()}...")
    if result["ticker_errors"]:
        print(f"\nTickers not in eToro universe: {', '.join(result['ticker_errors'])}")

    print(
        f"\nSummary: {len(result['pct_errors'])} bad pcts, "
        f"{len(result['price_errors'])} bad prices, "
        f"{len(result['ticker_errors'])} bad tickers"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
