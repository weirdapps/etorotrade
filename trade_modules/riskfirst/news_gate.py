"""Event-risk eligibility gate for the risk-first book (entry protection).

Pure decisions + a small JSON reader. The engine cannot call MCP/news APIs
(agent-side), so scandal/litigation risk arrives via an externally-produced
exclusion file, and earnings dates are fetched by the runner and passed in.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta


def _to_date(x):
    """Parse an ISO 'YYYY-MM-DD' string, date, or datetime -> date; None on failure."""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # Accept 'YYYY-MM-DD' and ISO with time component 'YYYY-MM-DDTHH:...'
        try:
            return date.fromisoformat(s[:10])
        except (ValueError, TypeError):
            return None
    return None


def earnings_blackout(earnings_map, as_of, blackout_days: int = 7) -> set:
    """Return the set of tickers whose NEXT earnings date falls within
    [as_of, as_of + blackout_days] inclusive.
    earnings_map: {ticker: date-like (iso str / date / datetime)}.
    as_of: date-like. Tickers with missing/unparseable/None dates are NOT blacked out.
    Past earnings dates (< as_of) are ignored (not upcoming)."""
    start = _to_date(as_of)
    if start is None:
        return set()
    end = start + timedelta(days=blackout_days)

    blacked_out = set()
    for ticker, raw_date in earnings_map.items():
        d = _to_date(raw_date)
        if d is None:
            continue
        if start <= d <= end:
            blacked_out.add(ticker)
    return blacked_out


def apply_exclusions(df, exclude):
    """Return df without rows whose index (ticker) is in `exclude` (a set/iterable).
    Empty/None exclude -> df unchanged. Does not mutate the input."""
    if not exclude:
        return df
    exclude_set = set(exclude)
    mask = ~df.index.isin(exclude_set)
    return df.loc[mask]


def load_event_risk(path) -> set:
    """Read a JSON file of event-risk tickers -> set of upper-cased ticker strings.
    Accepts either a JSON array ['AAPL','MSFT'] or an object {'AAPL': {...}, ...} (keys used).
    Missing/unreadable/invalid file -> empty set (never raises)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return {str(t).upper() for t in data}
        if isinstance(data, dict):
            return {str(k).upper() for k in data}
        return set()
    except Exception:
        return set()
