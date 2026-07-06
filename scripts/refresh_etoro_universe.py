"""Refresh yahoofinance/input/etoro.csv from the eToro market-data API.

Fetches stocks + ETFs from https://www.etoro.com/api/public/v1/market-data/instruments
and atomically overwrites the input file with one row per (symbol, company, exchange).
Symbols come back from the API mostly in Yahoo-Finance format but need
light normalization: strip .US suffix, drop .RTH duplicates, trim HK
5-digit to 4-digit (00001.HK → 0001.HK).
"""

import csv
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

INSTRUMENTS_URL = "https://www.etoro.com/api/public/v1/market-data/instruments"
STOCK_TYPE_ID = 5
ETF_TYPE_ID = 6
MIN_INSTRUMENTS_THRESHOLD = 1000

EXCHANGE_NAMES: dict[int, str] = {
    2: "NYSE",
    4: "Nasdaq",
    5: "NYSE",
    6: "FRA",
    7: "LSE",
    8: "NYSE",
    9: "Euronext Paris",
    10: "Bolsa De Madrid",
    11: "Borsa Italiana",
    12: "SIX",
    14: "Oslo Stock Exchange",
    15: "Stockholm Stock Exchange",
    16: "Copenhagen Stock Exchange",
    17: "Helsinki Stock Exchange",
    19: "OTC Markets",
    20: "CBOE",
    21: "HKEX",
    22: "Euronext Lisbon",
    23: "Euronext Brussels",
    24: "Tadawul",
    30: "Euronext Amsterdam",
    31: "ASX",
    32: "Vienna",
    33: "Xetra",
    34: "Dublin",
    35: "Prague SE",
    36: "Warsaw",
    37: "Budapest",
    38: "Xetra ETFs",
    39: "DFM",
    41: "Abu Dhabi",
    42: "LSE AIM",
    43: "LSE AIM",
    44: "LSE",
    56: "Tokyo Stock Exchange",
}

_HTTP_TIMEOUT_SEC = 30
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
_INTER_PAGE_DELAY_SEC = 0.5

_DEFAULT_OUTPUT_CSV = str(Path(__file__).parent.parent / "yahoofinance" / "input" / "etoro.csv")
_DEFAULT_DELTA_LOG = str(
    Path(__file__).parent.parent / "yahoofinance" / "input" / ".universe-refresh-log.json"
)


def is_etorian_alias(item: dict) -> bool:
    """True if the symbol or displayName starts with 'ETORIAN' (deprecated placeholder)."""
    sym = (item.get("symbol") or "").upper()
    name = (item.get("displayName") or "").upper()
    return sym.startswith("ETORIAN") or name.startswith("ETORIAN")


_HK_PAD_WIDTH = 4

_SUFFIX_REMAP = {
    ".ASX": ".AX",
    ".ZU": ".SW",
    ".NV": ".AS",
    ".LSB": ".LS",
}

_DROP_SUFFIXES = frozenset(
    {
        ".RTH",
        ".DELISTED",
        ".TEST",
        ".OLD",
        ".EXT",
        ".24-7",
        ".CALL1",
        ".CALL2",
        ".PUT1",
        ".PUT2",
        ".TENDER",
        ".CASHRESERVED",
        ".MOEX",
        ".RIGHT",
        ".WS",
        ".PFD",
    }
)


def _is_junk_suffix(suffix: str) -> bool:
    """True if the suffix is a known junk/placeholder (CVR, DUP, or long numeric ID)."""
    if suffix in _DROP_SUFFIXES:
        return True
    s = suffix.lstrip(".")
    if s.startswith(("CVR", "DUP", "ETF", "STOCK")):
        return True
    if s.isdigit() and len(s) > 3:
        return True
    return False


def normalize_symbol(symbol: str) -> str | None:
    """Normalize eToro symbol to Yahoo Finance format.

    Returns None for symbols that should be skipped (RTH variants, junk instruments).
    """
    upper = symbol.upper().strip()

    if "." in upper:
        base, dot_suffix = upper.rsplit(".", 1)
        full_suffix = f".{dot_suffix}"
        if _is_junk_suffix(full_suffix):
            return None
        if full_suffix in _SUFFIX_REMAP:
            upper = base + _SUFFIX_REMAP[full_suffix]

    if upper.endswith(".US"):
        upper = upper[:-3]
    if upper.endswith(".HK"):
        base = upper[:-3]
        if base.isdigit() and len(base) > _HK_PAD_WIDTH:
            upper = base.lstrip("0").zfill(_HK_PAD_WIDTH) + ".HK"
    return upper


_SCANDI_SUFFIXES = frozenset({".ST", ".CO", ".HE", ".OL"})
_SHARE_CLASS_KEYWORDS = ("ser.", "series", "class", "klass", "aktie")


def fix_share_classes(rows: list[dict]) -> list[dict]:
    """Insert hyphen before Scandinavian share-class letters (A/B).

    Detects share classes via two strategies:
    1. A/B pair: both BASEA.ST and BASEB.ST exist → insert hyphen in both
    2. Company name: contains 'ser.', 'Series', 'Class', etc. → insert hyphen

    Already-hyphenated symbols (ASSA-B.ST) are left alone.
    """
    all_symbols = {r["symbol"] for r in rows}

    def _needs_hyphen(sym: str, company: str) -> bool:
        for suf in _SCANDI_SUFFIXES:
            if not sym.endswith(suf):
                continue
            base = sym[: -len(suf)]
            if len(base) < 2 or base[-1] not in ("A", "B") or "-" in base:
                return False
            other_class = "A" if base[-1] == "B" else "B"
            other_sym = base[:-1] + other_class + suf
            other_hyph = base[:-1] + "-" + other_class + suf
            if other_sym in all_symbols or other_hyph in all_symbols:
                return True
            if any(kw in company.lower() for kw in _SHARE_CLASS_KEYWORDS):
                return True
            return False
        return False

    def _insert_hyphen(sym: str) -> str:
        for suf in _SCANDI_SUFFIXES:
            if sym.endswith(suf):
                base = sym[: -len(suf)]
                return base[:-1] + "-" + base[-1] + suf
        return sym

    out: list[dict] = []
    fixed = 0
    for row in rows:
        sym = row["symbol"]
        company = row.get("company", "")
        if _needs_hyphen(sym, company):
            row = {**row, "symbol": _insert_hyphen(sym)}
            fixed += 1
        out.append(row)
    if fixed:
        logger.info("Fixed %d Scandinavian share-class symbols (inserted hyphen)", fixed)
    return out


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


def _get_credential(env_var: str, keychain_service: str) -> str | None:
    """Return credential from environment variable; fall back to macOS keychain on local runs."""
    value = os.environ.get(env_var)
    if value:
        return value
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", "etoro-api", "-s", keychain_service, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_credentials() -> tuple[str, str]:
    """Resolve (api_key, user_key) from env vars or macOS keychain. Raises if either missing."""
    api_key = _get_credential("ETORO_API_KEY", "etoro-public-key")
    user_key = _get_credential("ETORO_USER_KEY", "etoro-user-key")
    if not api_key or not user_key:
        missing = []
        if not api_key:
            missing.append("ETORO_API_KEY")
        if not user_key:
            missing.append("ETORO_USER_KEY")
        raise RuntimeError(
            f"Missing credentials: {', '.join(missing)}. Set env vars or store in macOS keychain (service: etoro-public-key / etoro-user-key)."
        )
    return api_key, user_key


def fetch_all_instruments(
    api_key: str,
    user_key: str,
    max_retries: int = 3,
) -> list[dict]:
    """Fetch all instruments from the market-data API (single call, no pagination).

    Filters to Stocks (type 5) and ETFs (type 6), then maps fields to the
    format expected by the rest of the pipeline.
    """
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "x-api-key": api_key,
            "x-user-key": user_key,
            "x-request-id": str(uuid.uuid4()),
        }
        try:
            response = requests.get(INSTRUMENTS_URL, headers=headers, timeout=_HTTP_TIMEOUT_SEC)
            if response.status_code == 200:
                raw = response.json().get("instrumentDisplayDatas", [])
                filtered = [
                    i for i in raw if i.get("instrumentTypeID") in (STOCK_TYPE_ID, ETF_TYPE_ID)
                ]
                items = []
                for i in filtered:
                    items.append(
                        {
                            "instrumentId": i.get("instrumentID"),
                            "symbol": i.get("symbolFull", ""),
                            "displayName": i.get("instrumentDisplayName", ""),
                            "exchangeName": EXCHANGE_NAMES.get(i.get("exchangeID", 0), ""),
                        }
                    )
                logger.info(
                    "  Fetched %d total, %d Stocks+ETFs after type filter",
                    len(raw),
                    len(items),
                )
                return items
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < max_retries:
            time.sleep(2**attempt)

    raise RuntimeError(
        f"fetch_all_instruments: all {max_retries} attempts failed. Last error: {last_error}"
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_count": total_count,
        "new_count": len(new_symbols),
        "removed_count": len(removed_symbols),
        "sample_new": sorted(new_symbols)[:_SAMPLE_LIMIT],
        "sample_removed": sorted(removed_symbols)[:_SAMPLE_LIMIT],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _read_existing_symbols(path: str) -> set[str]:
    """Return the set of symbols currently in input/etoro.csv (uppercase). Empty if missing."""
    if not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip().upper()
            if sym:
                out.add(sym)
    return out


def main(
    output_csv_path: str = _DEFAULT_OUTPUT_CSV,
    delta_log_path: str = _DEFAULT_DELTA_LOG,
) -> int:
    """Run the refresh pipeline. Returns exit code (0 success, 1 error)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        api_key, user_key = get_credentials()
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    logger.info("Fetching eToro market-data instruments (Stocks + ETFs)...")
    try:
        items = fetch_all_instruments(api_key, user_key)
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    logger.info("Fetched %d Stocks+ETFs from market-data API", len(items))

    if len(items) < MIN_INSTRUMENTS_THRESHOLD:
        logger.error(
            "Refusing to proceed: only %d items returned (threshold %d). API may be broken.",
            len(items),
            MIN_INSTRUMENTS_THRESHOLD,
        )
        return 1

    # Filter ETORIAN aliases + empty symbols + normalize (.US, .RTH, HK 5→4 digit)
    rows: list[dict] = []
    skipped_no_symbol = 0
    skipped_alias = 0
    skipped_rth = 0
    for item in items:
        if is_etorian_alias(item):
            skipped_alias += 1
            continue
        raw_symbol = (item.get("symbol") or "").strip()
        if not raw_symbol:
            skipped_no_symbol += 1
            continue
        normalized = normalize_symbol(raw_symbol)
        if normalized is None:
            skipped_rth += 1
            continue
        rows.append(
            {
                "symbol": normalized,
                "company": item.get("displayName", ""),
                "exchange": item.get("exchangeName", ""),
            }
        )

    logger.info(
        "After filters: %d candidates (skipped %d aliases, %d empty, %d RTH)",
        len(rows),
        skipped_alias,
        skipped_no_symbol,
        skipped_rth,
    )

    rows = fix_share_classes(rows)
    deduped = dedupe_by_symbol(rows)
    logger.info("After dedupe: %d unique symbols", len(deduped))

    existing_symbols = _read_existing_symbols(output_csv_path)
    new_symbols_set = {r["symbol"] for r in deduped}
    new_symbols = sorted(new_symbols_set - existing_symbols)
    removed_symbols = sorted(existing_symbols - new_symbols_set)
    logger.info(
        "Delta: +%d new, -%d removed (vs %d existing)",
        len(new_symbols),
        len(removed_symbols),
        len(existing_symbols),
    )

    write_universe_csv(deduped, output_csv_path)
    write_delta_log(
        delta_log_path,
        new_symbols=new_symbols,
        removed_symbols=removed_symbols,
        total_count=len(deduped),
    )

    logger.info("Wrote %s", output_csv_path)
    logger.info("Delta log: %s", delta_log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
