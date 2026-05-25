"""Refresh yahoofinance/input/etoro.csv from the eToro instruments API.

Fetches the public bulk instruments endpoint, filters to stocks + ETFs,
extracts ticker symbols from image URIs, normalizes to Yahoo Finance format,
and atomically overwrites the input file.
"""

import csv
import os
import re
from collections import Counter, defaultdict

_MARKET_AVATAR_RE = re.compile(r"/market-avatars/([^/]+)/")


def extract_symbol(item: dict) -> str | None:
    """Extract eToro ticker from the first image URI matching /market-avatars/SYMBOL/.

    Returns None if no Images, no URI, or no matching pattern.
    """
    for image in item.get("Images", []):
        uri = image.get("Uri", "")
        match = _MARKET_AVATAR_RE.search(uri)
        if match:
            return match.group(1)
    return None



_STOCK_TYPE_ID = 5
_ETF_TYPE_ID = 6
_KEEP_TYPES = {_STOCK_TYPE_ID, _ETF_TYPE_ID}


def is_stock_or_etf(item: dict) -> bool:
    """True if InstrumentTypeID is 5 (stock) or 6 (ETF)."""
    return item.get("InstrumentTypeID") in _KEEP_TYPES


def is_etorian_alias(item: dict) -> bool:
    """True if InstrumentDisplayName starts with 'ETORIAN' (deprecated alias placeholder)."""
    return (item.get("InstrumentDisplayName") or "").startswith("ETORIAN")


def dedupe_by_symbol(rows: list[dict]) -> list[dict]:
    """Remove duplicate rows by 'symbol' key, preserving first occurrence and overall order."""
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        sym = row["symbol"]
        if sym in seen:
            continue
        seen.add(sym)
        out.append(row)
    return out


def build_exchange_map(bulk_data: dict, current_csv_path: str) -> dict[int, str]:
    """Derive {ExchangeID: yahoo_suffix} mapping by cross-referencing the bulk pull
    against the current input/etoro.csv.

    For each (symbol, ExchangeID) in the bulk pull, look up `exchange` column
    in current CSV. The mode (most common) suffix per ExchangeID wins.

    Returns empty dict if current_csv_path doesn't exist.
    """
    if not os.path.exists(current_csv_path):
        return {}

    # symbol (case-insensitive) → exchange suffix from current CSV
    current_suffixes: dict[str, str] = {}
    with open(current_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get("symbol") or "").strip().lower()
            exch = (row.get("exchange") or "").strip()
            if sym:
                current_suffixes[sym] = exch

    # For each bulk item, vote ExchangeID → suffix
    votes: dict[int, Counter] = defaultdict(Counter)
    for item in bulk_data.get("InstrumentDisplayDatas", []):
        exch_id = item.get("ExchangeID")
        sym = extract_symbol(item)
        if exch_id is None or sym is None:
            continue
        suffix = current_suffixes.get(sym.lower())
        if suffix is not None:
            votes[exch_id][suffix] += 1

    # Resolve to mode per ExchangeID
    return {exch_id: counter.most_common(1)[0][0] for exch_id, counter in votes.items()}


_HK_SUFFIX = "HK"
_HK_PAD_WIDTH = 4


def normalize_to_yahoo(
    symbol: str, exchange_id: int, exchange_map: dict[int, str]
) -> tuple[str, bool]:
    """Apply Yahoo Finance suffix conventions.

    Returns (yahoo_symbol, was_unmapped).
    - Uppercases the symbol
    - Adds suffix if not already present
    - Special case: HK base symbol < 4 digits gets zero-padded to 4

    If exchange_id is not in exchange_map, returns the uppercased symbol with no suffix
    and was_unmapped=True so the caller can log it.
    """
    sym_upper = symbol.upper()

    if exchange_id not in exchange_map:
        return sym_upper, True

    suffix = exchange_map[exchange_id]

    # If symbol already carries a suffix (contains "."), leave as-is and uppercase
    if "." in sym_upper:
        base, existing_suffix = sym_upper.rsplit(".", 1)
        if existing_suffix == _HK_SUFFIX and base.isdigit() and len(base) < _HK_PAD_WIDTH:
            base = base.zfill(_HK_PAD_WIDTH)
        return f"{base}.{existing_suffix}", False

    # No dot in symbol — append suffix from mapping (if non-empty)
    if not suffix:
        return sym_upper, False
    return f"{sym_upper}.{suffix}", False
