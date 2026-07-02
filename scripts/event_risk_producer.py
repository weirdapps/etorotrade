"""Event-risk exclusion-file producer for the risk-first news gate.

## Flow

1. An **agent** (with MCP access) calls TipRanks ``get_assets_warnings`` for the
   current portfolio + candidate tickers and writes the raw response to a JSON file::

       { "AAPL": [ {"type": "Lawsuit", "date": "2024-01-01", "detail": "..."} ],
         "MSFT": [],
         ... }

2. This producer is then run (CLI or imported) to **classify** those raw warnings
   and write the compact exclusion file that the risk-first engine consumes::

       ["AAPL", "TSLA"]   # sorted, upper-cased array of blocked tickers

3. The engine reads ``DEFAULT_EVENT_RISK_PATH`` (via ``news_gate.load_event_risk``)
   at startup — it never calls news/MCP APIs directly.

The separation exists because the engine runs in a sandboxed context that cannot
reach agent-side tools; the producer runs in the agent session where MCP is
available, then hands off a plain JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from trade_modules.riskfirst.news_gate import DEFAULT_EVENT_RISK_PATH

# ---------------------------------------------------------------------------
# Pure classification logic
# ---------------------------------------------------------------------------


def _warning_type_string(entry) -> str:
    """Extract a normalised type string from a warning entry (dict or str)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        # TipRanks may use 'type', 'warningType', or 'name'
        for field in ("type", "warningType", "name"):
            val = entry.get(field)
            if val:
                return str(val)
    return ""


def classify_event_risk(warnings_by_ticker: dict, block_types=None) -> list:
    """Return sorted, upper-cased list of tickers with >= 1 material warning.

    Parameters
    ----------
    warnings_by_ticker:
        ``{ticker: [warning, ...]}`` where each warning is either a dict (with
        a ``type`` / ``warningType`` / ``name`` key) or a plain string.
        Tickers with ``None`` or empty lists are **not** flagged.
    block_types:
        Optional iterable of warning-type strings to treat as material.
        Matching is case-insensitive substring (``"regul"`` matches
        ``"RegulatoryAction"``).  If ``None``, **any** non-empty warnings list
        flags the ticker.  An **empty** iterable flags nothing.

    Returns
    -------
    list
        Sorted, unique, upper-cased ticker symbols.
    """
    blocked: set[str] = set()

    use_filter = block_types is not None
    if use_filter:
        block_lower = [bt.lower() for bt in block_types]

    for ticker, warnings in warnings_by_ticker.items():
        if not warnings:  # None, empty list, falsy
            continue

        upper = ticker.upper()

        if not use_filter:
            blocked.add(upper)
            continue

        # block_types mode: flag only if >= 1 entry matches
        for entry in warnings:
            type_str = _warning_type_string(entry).lower()
            if any(bt in type_str for bt in block_lower):
                blocked.add(upper)
                break

    return sorted(blocked)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_event_risk(tickers, path=None) -> str:
    """Atomically write a JSON array of unique, upper-cased, sorted tickers.

    Parameters
    ----------
    tickers:
        Iterable of ticker strings (or ``None`` / empty for an empty array).
    path:
        Destination file path.  Defaults to ``DEFAULT_EVENT_RISK_PATH``.

    Returns
    -------
    str
        The path actually written.
    """
    path = path or DEFAULT_EVENT_RISK_PATH
    cleaned: list[str] = sorted({str(t).upper() for t in (tickers or [])})
    payload = json.dumps(cleaned, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return path


def load_raw_warnings(path: str) -> dict:  # pragma: no cover (IO)
    """Read the agent-produced raw-warnings JSON (``{ticker: [warnings...]}``).

    Returns an empty dict on any failure (missing file, bad JSON, wrong type).
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:  # pragma: no cover (IO/CLI)
    """Classify raw TipRanks warnings and write the event_risk exclusion file.

    Usage::

        python -m scripts.event_risk_producer raw_warnings.json \\
            [--block-types lawsuit,regulatory] [--out ~/.weirdapps-trading/news/event_risk.json]
    """
    parser = argparse.ArgumentParser(
        description="Classify TipRanks warnings → event_risk exclusion file"
    )
    parser.add_argument("raw_warnings", help="Path to raw-warnings JSON ({ticker: [...]}")
    parser.add_argument(
        "--block-types",
        default=None,
        help="Comma-separated warning type substrings to flag (default: flag any non-empty)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"Output path (default: {DEFAULT_EVENT_RISK_PATH})",
    )
    args = parser.parse_args(argv)

    raw = load_raw_warnings(args.raw_warnings)
    if not raw:
        print(f"[event_risk_producer] WARNING: no warnings loaded from {args.raw_warnings!r}")

    block_types = (
        [t.strip() for t in args.block_types.split(",") if t.strip()] if args.block_types else None
    )

    tickers = classify_event_risk(raw, block_types=block_types)
    out_path = write_event_risk(tickers, path=args.out)

    print(f"[event_risk_producer] Wrote {len(tickers)} blocked ticker(s) to {out_path}")
    if tickers:
        print("  Blocked:", ", ".join(tickers))
    else:
        print("  No tickers blocked.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
