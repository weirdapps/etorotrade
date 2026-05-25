"""Refresh yahoofinance/input/etoro.csv from the eToro instruments API.

Fetches the public bulk instruments endpoint, filters to stocks + ETFs,
extracts ticker symbols from image URIs, normalizes to Yahoo Finance format,
and atomically overwrites the input file.
"""

import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime

import requests

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


BULK_URL = (
    "https://api.etorostatic.com/sapi/instrumentsmetadata/V1.1/"
    "instruments/bulk?bulkNumber=1&totalBulks=1"
)
_HTTP_TIMEOUT_SEC = 30


def fetch_bulk(url: str = BULK_URL, max_retries: int = 3) -> dict:
    """GET the bulk instruments endpoint with exponential-backoff retry.

    Backoff: 2^attempt seconds between tries (2s, 4s, 8s for 3 attempts).
    Raises RuntimeError if all attempts fail.
    """
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=_HTTP_TIMEOUT_SEC)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < max_retries:
            time.sleep(2**attempt)

    raise RuntimeError(
        f"fetch_bulk: all {max_retries} attempts failed. Last error: {last_error}"
    )


_CSV_COLUMNS = ["symbol", "company", "exchange"]


def write_universe_csv(rows: list[dict], path: str) -> None:
    """Atomically write rows to a CSV at `path`.

    Writes to `path + ".tmp"` first, then os.replace() to final location.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


_SAMPLE_LIMIT = 50


def write_delta_log(
    path: str,
    new_symbols: list[str],
    removed_symbols: list[str],
    total_count: int,
) -> None:
    """Write a JSON snapshot of this run's delta."""
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_count": total_count,
        "new_count": len(new_symbols),
        "removed_count": len(removed_symbols),
        "sample_new": sorted(new_symbols)[:_SAMPLE_LIMIT],
        "sample_removed": sorted(removed_symbols)[:_SAMPLE_LIMIT],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_unmapped_exchanges_log(path: str, unmapped: dict[int, list[str]]) -> None:
    """Write a JSON snapshot of unmapped ExchangeIDs and the symbols affected."""
    payload = {
        str(exch_id): {
            "count": len(symbols),
            "sample_symbols": sorted(symbols)[:_SAMPLE_LIMIT],
        }
        for exch_id, symbols in sorted(unmapped.items())
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
