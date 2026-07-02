"""Offline sector lookup — PURE, no I/O in the lookup functions.

Provides:
  load_sector_map(paths)   — reads CSVs, returns {symbol: sector}
  resolve_sector(ticker, sector_map) — exact-match lookup; None if absent

Only symbols present in market.csv / usindex.csv are covered (~519 US
large-caps).  International tickers (e.g. 0175.HK, DTE.DE, BARC.L) are
not in the source files and resolve to None — they are excluded from the
sector cap, not guessed.
"""

from __future__ import annotations

import csv
import os


def load_sector_map(paths: list[str]) -> dict[str, str]:
    """Read sector CSVs and return ``{symbol_upper: sector}`` mapping.

    Args:
        paths: List of file paths to read (market.csv, usindex.csv, …).
               Missing files are skipped gracefully.

    Rules:
        - Header ``symbol,name,sector,subSector,headQuarter,founded`` is
          expected; columns are located by name, not position.
        - Blank sector values are skipped.
        - *Earlier* files take precedence: a later file does NOT overwrite
          a non-empty entry already recorded (deterministic, predictable).
        - Symbol keys are stored upper-cased for case-insensitive lookup.

    Returns:
        Dict mapping upper-cased ticker → sector string.  Empty dict if
        no readable file is found.
    """
    result: dict[str, str] = {}
    for path in paths:
        expanded = os.path.expanduser(path)
        if not os.path.exists(expanded):
            continue
        try:
            with open(expanded, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    symbol = (row.get("symbol") or "").strip()
                    sector = (row.get("sector") or "").strip()
                    if not symbol or not sector:
                        continue
                    key = symbol.upper()
                    # Earlier file wins — do not overwrite an existing entry.
                    if key not in result:
                        result[key] = sector
        except (OSError, csv.Error):
            continue  # skip unreadable files
    return result


def resolve_sector(ticker: str, sector_map: dict[str, str]) -> str | None:
    """Exact-match lookup of ``ticker`` in ``sector_map``.

    Args:
        ticker:     Instrument symbol (case-insensitive).
        sector_map: Dict as returned by :func:`load_sector_map`.

    Returns:
        Sector string if found; ``None`` if absent.  No heuristics, no
        guessing — absent means absent.
    """
    if not ticker or not sector_map:
        return None
    return sector_map.get(ticker.upper())
