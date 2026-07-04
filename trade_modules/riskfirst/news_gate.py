"""Event-risk eligibility gate for the risk-first book (entry protection).

Pure decisions + a small JSON reader. The engine cannot call MCP/news APIs
(agent-side), so scandal/litigation risk arrives via an externally-produced
exclusion file, and earnings dates are fetched by the runner and passed in.

## Event-risk file (producer)

``event_risk.json`` is produced OUTSIDE the engine (agent-side).
An agent queries TipRanks ``get_assets_warnings`` / ``get_assets_news`` for
the portfolio + candidates, flags tickers with active scandal/litigation risk,
and writes the array to ``DEFAULT_EVENT_RISK_PATH``.
The engine only consumes it — it never calls news APIs directly.
Format: JSON array of ticker strings, e.g. ``["AAPL", "TSLA"]``.
Alternatively an object whose keys are ticker strings.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import yaml

DEFAULT_EVENT_RISK_PATH = os.path.expanduser("~/.weirdapps-trading/news/event_risk.json")

# config.yaml ships with the repo; resolve it relative to this module so the
# loader works on any host (CI, VPS), not just a checkout under ~/SourceCode.
_REPO_CONFIG_YAML = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
)
_DEFAULT_CONFIG_YAML = (
    _REPO_CONFIG_YAML
    if os.path.exists(_REPO_CONFIG_YAML)
    else os.path.expanduser("~/SourceCode/etorotrade/config.yaml")
)


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
    Matching is case-insensitive; the original index casing is preserved in the
    returned frame. Empty/None exclude -> df unchanged. Does not mutate the input."""
    ex = {str(t).upper() for t in exclude} if exclude else set()
    if not ex:
        return df
    keep = ~df.index.astype(str).str.upper().isin(ex)
    return df.loc[keep]


def load_config(path=None):
    """Load the event_gate section from config.yaml; fall back to defaults."""
    path = path or _DEFAULT_CONFIG_YAML
    try:
        with open(path) as f:
            sec = (yaml.safe_load(f) or {}).get("event_gate") or {}
    except Exception:
        sec = {}
    return {
        "enabled": bool(sec.get("enabled", True)),
        "earnings_blackout_days": int(sec.get("earnings_blackout_days", 7)),
        "event_risk_path": os.path.expanduser(
            sec.get("event_risk_path") or DEFAULT_EVENT_RISK_PATH
        ),
    }


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
