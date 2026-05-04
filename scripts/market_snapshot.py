#!/usr/bin/env python3
"""Deterministic market data snapshot via yfinance.

Outputs a JSON object with verified prices, % changes, and holiday
detection for use by the midday brief pipeline. Zero LLM involvement.

Usage:
    python scripts/market_snapshot.py              # JSON to stdout
    python scripts/market_snapshot.py -o snap.json # JSON to file
"""

import argparse
import json
import sys
from datetime import UTC, datetime

import yfinance as yf

INSTRUMENTS = {
    # European indices
    "^GDAXI": "DAX 40",
    "^FCHI": "CAC 40",
    "^STOXX50E": "Euro Stoxx 50",
    "^STOXX": "Stoxx 600",
    "^FTSE": "FTSE 100",
    # Asian indices
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng",
    "^KS11": "KOSPI",
    "^AXJO": "ASX 200",
    "000001.SS": "Shanghai Composite",
    # US futures
    "ES=F": "S&P 500 futures",
    "NQ=F": "Nasdaq 100 futures",
    "YM=F": "Dow futures",
    # FX
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",
    "DX-Y.NYB": "DXY",
    # Commodities
    "BZ=F": "Brent crude",
    "CL=F": "WTI crude",
    "GC=F": "Gold",
    # Yields
    "^TNX": "10Y UST yield",
}


def _today_athens() -> str:
    """Return today's date in Athens timezone as YYYY-MM-DD."""
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Europe/Athens")).strftime("%Y-%m-%d")


def fetch_snapshot() -> dict:
    """Fetch market data for all instruments. Returns structured dict."""
    today = _today_athens()
    ts = datetime.now(UTC).isoformat()

    instruments = {}
    holidays = []
    errors = []

    for ticker, name in INSTRUMENTS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist.empty:
                errors.append(f"{ticker} ({name}): no data returned")
                continue

            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
            change_pct = round((last - prev) / prev * 100, 2)
            last_date = hist.index[-1].strftime("%Y-%m-%d")
            status = "open" if last_date >= today else "closed"

            if status == "closed":
                holidays.append(ticker)

            instruments[ticker] = {
                "name": name,
                "price": round(last, 2),
                "prev_close": round(prev, 2),
                "change_pct": change_pct,
                "last_date": last_date,
                "status": status,
            }
        except Exception as e:
            errors.append(f"{ticker} ({name}): {e}")

    result = {
        "timestamp": ts,
        "today_athens": today,
        "instruments": instruments,
        "holidays_detected": holidays,
    }
    if errors:
        result["errors"] = errors

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Market data snapshot via yfinance")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    snapshot = fetch_snapshot()

    output = json.dumps(snapshot, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        n = len(snapshot["instruments"])
        h = len(snapshot["holidays_detected"])
        print(f"Snapshot: {n} instruments, {h} holidays → {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
