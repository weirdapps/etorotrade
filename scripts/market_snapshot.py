#!/usr/bin/env python3
"""Deterministic market data snapshot via yfinance.

Outputs a JSON object with verified prices, % changes, and holiday
detection for use by the midday brief pipeline. Zero LLM involvement.

Every change_pct compares two closes of the same series one session apart. Yahoo breaks
that three ways, each handled below: rolling futures switch contract, completed FX daily
bars carry the day's opening tick as their close, and the daily history can skip a session
its intraday bars cover. validate_brief.py checks the post against this snapshot, so a
wrong move here is published as verified.

Usage:
    python scripts/market_snapshot.py              # JSON to stdout
    python scripts/market_snapshot.py -o snap.json # JSON to file
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import yfinance as yf

UTC = timezone.utc  # Python 3.10 compat (datetime.UTC is 3.11+)
ATHENS = ZoneInfo("Europe/Athens")

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

#: The dated contracts behind each rolling future: root, Yahoo exchange suffix, delivery months.
FUTURES_CONTRACTS = {
    "ES=F": ("ES", "CME", "HMUZ"),
    "NQ=F": ("NQ", "CME", "HMUZ"),
    "YM=F": ("YM", "CBT", "HMUZ"),
    "BZ=F": ("BZ", "NYM", "FGHJKMNQUVXZ"),
    "CL=F": ("CL", "NYM", "FGHJKMNQUVXZ"),
    "GC=F": ("GC", "CMX", "GJMQVZ"),
}
MONTH_CODES = "FGHJKMNQUVXZ"
#: The front two contracts of every root above expire within this many months of today.
CONTRACT_WINDOW_MONTHS = 5
#: How far a dated contract's last price may sit from the rolling quote and still be the one
#: behind it: the quotes are fetched seconds apart. Adjacent contracts are usually further apart
#: (Brent Nov/Dec was ~5% on 24 Sep 2026); where they are not, either match is a same-contract move.
CONTRACT_MATCH_TOLERANCE = 0.005

#: Instruments with no bar before their session opens, and its local opening time. At 14:00
#: Athens the latest ^TNX bar is always the previous US session, which is not a holiday.
SESSION_OPEN = {"^TNX": ("America/New_York", 8, 0)}


def _history(symbol: str, **kwargs):
    return yf.Ticker(symbol).history(**kwargs)


def _dated_contracts(ticker: str, on) -> list:
    """Yahoo symbols of the contracts of *ticker*'s root delivering in the months from *on*'s."""
    root, suffix, months = FUTURES_CONTRACTS[ticker]
    symbols = []
    for offset in range(CONTRACT_WINDOW_MONTHS):
        year, month = divmod(on.year * 12 + on.month - 1 + offset, 12)
        if MONTH_CODES[month] in months:
            symbols.append(f"{root}{MONTH_CODES[month]}{year % 100:02d}.{suffix}")
    return symbols


def _same_contract(ticker: str, hist) -> tuple:
    """``(last, prev, contract)`` of the dated contract behind the rolling quote's last price.

    Yahoo's ``=F`` series switches contract around expiry: on 2026-09-23 it moved Brent from
    November to December mid-morning, so its last two closes belonged to two contracts and their
    gap was published as "Brent -3.86%". Raises when no live dated contract matches, because an
    unverified move must not reach the post.
    """
    last = float(hist["Close"].iloc[-1])
    last_date = hist.index[-1].date()
    best = None
    for symbol in _dated_contracts(ticker, last_date):
        dated = _history(symbol, period="5d")
        if len(dated) < 2 or dated.index[-1].date() != last_date:
            continue  # no data, or an expired contract's stale history
        gap = abs(float(dated["Close"].iloc[-1]) - last) / last
        if best is None or gap < best[0]:
            best = (gap, float(dated["Close"].iloc[-1]), float(dated["Close"].iloc[-2]), symbol)
    if best is None or best[0] > CONTRACT_MATCH_TOLERANCE:
        raise ValueError(f"no dated contract matches the rolling quote {last}; move not verified")
    return best[1:]


def _skipped_session_close(ticker: str, prev_date, last_date):
    """Close of a weekday session between the last two daily rows that intraday bars cover.

    On 2026-09-24 Yahoo's daily history had no 22 Sep row for any cash index or for ^TNX, so
    every 23 Sep move was measured from 21 Sep. The last intraday bar can end before the
    closing auction (KOSPI's ran 0.4% off its close that week), which is still far closer
    than a close one session older. A weekday with no intraday bars either is a real holiday
    (Japan, 21-23 Sep 2026), and the older close stands: returns None.
    """
    gaps = [
        prev_date + timedelta(days=n)
        for n in range(1, (last_date - prev_date).days)
        if (prev_date + timedelta(days=n)).weekday() < 5
    ]
    if not gaps:
        return None
    bars = _history(ticker, period="5d", interval="30m")
    if bars.empty:
        return None
    for day in reversed(gaps):
        session = bars[bars.index.date == day]
        if len(session):
            return float(session["Close"].iloc[-1])
    return None


def _status(ticker: str, last_date, today, now: datetime) -> str:
    if last_date >= today:
        return "open"
    if ticker in SESSION_OPEN:
        tz, hour, minute = SESSION_OPEN[ticker]
        local = now.astimezone(ZoneInfo(tz))
        if local.weekday() < 5 and (local.hour, local.minute) < (hour, minute):
            return "pre-open"
    return "closed"


def fetch_snapshot(now: datetime | None = None) -> dict:
    """Fetch market data for all instruments. Returns structured dict."""
    now = now or datetime.now(UTC)
    today = now.astimezone(ATHENS).date()

    instruments = {}
    holidays = []
    errors = []

    for ticker, name in INSTRUMENTS.items():
        try:
            hist = _history(ticker, period="5d")
            if hist.empty:
                errors.append(f"{ticker} ({name}): no data returned")
                continue

            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
            last_date = hist.index[-1].date()
            extra = {}
            if ticker in FUTURES_CONTRACTS:
                last, prev, extra["contract"] = _same_contract(ticker, hist)
            elif ticker.endswith("=X"):
                # Yahoo's completed FX daily bars carry the day's opening tick as their close,
                # so Close[-2] is yesterday's OPEN. In a 24-hour market today's open is
                # yesterday's close.
                prev = float(hist["Open"].iloc[-1])
            elif len(hist) >= 2:
                skipped = _skipped_session_close(ticker, hist.index[-2].date(), last_date)
                if skipped is not None:
                    prev = skipped
            change_pct = round((last - prev) / prev * 100, 2)
            decimals = 4 if ticker.endswith("=X") else 2
            status = _status(ticker, last_date, today, now)

            if status == "closed":
                holidays.append(ticker)

            instruments[ticker] = {
                "name": name,
                "price": round(last, decimals),
                "prev_close": round(prev, decimals),
                "change_pct": change_pct,
                "last_date": last_date.isoformat(),
                "status": status,
                **extra,
            }
        except Exception as e:
            errors.append(f"{ticker} ({name}): {e}")

    result = {
        "timestamp": now.isoformat(),
        "today_athens": today.isoformat(),
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
