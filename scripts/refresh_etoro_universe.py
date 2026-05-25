"""Refresh yahoofinance/input/etoro.csv from the eToro Asset Explorer API.

Fetches stocks + ETFs from https://www.etoro.com/api/public/v1/instruments/discover
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
from datetime import UTC, datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DISCOVER_URL = "https://www.etoro.com/api/public/v1/instruments/discover"
ASSET_CLASSES = ("Stocks", "ETF")
DEFAULT_PAGE_SIZE = 1000
DEFAULT_FIELDS = "instrumentId,symbol,displayName,assetClass,exchangeName"
MIN_INSTRUMENTS_THRESHOLD = 1000

_HTTP_TIMEOUT_SEC = 30
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
_INTER_PAGE_DELAY_SEC = 0.5

_DEFAULT_OUTPUT_CSV = str(
    Path(__file__).parent.parent / "yahoofinance" / "input" / "etoro.csv"
)
_DEFAULT_DELTA_LOG = str(
    Path(__file__).parent.parent
    / "yahoofinance"
    / "input"
    / ".universe-refresh-log.json"
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

_DROP_SUFFIXES = frozenset({
    ".RTH", ".DELISTED", ".TEST", ".OLD", ".EXT", ".24-7",
    ".CALL1", ".CALL2", ".PUT1", ".PUT2",
    ".TENDER", ".CASHRESERVED", ".MOEX",
    ".RIGHT", ".WS", ".PFD",
})


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
            capture_output=True, text=True, timeout=5, check=False,
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


def fetch_page(
    asset_class: str,
    page: int,
    api_key: str,
    user_key: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    fields: str = DEFAULT_FIELDS,
    max_retries: int = 3,
) -> dict:
    """GET one page of /instruments/discover with exponential-backoff retry.

    Backoff: 2^attempt seconds between tries (2s, 4s, 8s for 3 attempts).
    Raises RuntimeError if all attempts fail.
    """
    params = {"page": page, "pageSize": page_size, "assetClass": asset_class, "fields": fields}
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
            response = requests.get(
                DISCOVER_URL, headers=headers, params=params, timeout=_HTTP_TIMEOUT_SEC
            )
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < max_retries:
            time.sleep(2**attempt)

    raise RuntimeError(
        f"fetch_page({asset_class}, page={page}): all {max_retries} attempts failed. Last error: {last_error}"
    )


def fetch_all_assets(
    api_key: str,
    user_key: str,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict]:
    """Paginate through all Stocks + ETFs from the Asset Explorer.

    Returns a list of raw API items (with instrumentId, symbol, displayName, assetClass, exchangeName).
    """
    all_items: list[dict] = []
    for asset_class in ASSET_CLASSES:
        page = 1
        while True:
            response = fetch_page(asset_class, page, api_key, user_key, page_size)
            items = response.get("items", [])
            total = response.get("totalItems", 0)
            all_items.extend(items)
            logger.info(
                "  %s page %d: +%d items (running total this class: %d / %d)",
                asset_class, page, len(items), min(page * page_size, total), total,
            )
            if not items or page * page_size >= total:
                break
            page += 1
            time.sleep(_INTER_PAGE_DELAY_SEC)
    return all_items


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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        api_key, user_key = get_credentials()
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    logger.info("Fetching eToro Asset Explorer (Stocks + ETFs)...")
    try:
        items = fetch_all_assets(api_key, user_key)
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    logger.info("Fetched %d total items across all asset classes", len(items))

    if len(items) < MIN_INSTRUMENTS_THRESHOLD:
        logger.error(
            "Refusing to proceed: only %d items returned (threshold %d). "
            "API may be broken.",
            len(items), MIN_INSTRUMENTS_THRESHOLD,
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
        rows.append({
            "symbol": normalized,
            "company": item.get("displayName", ""),
            "exchange": item.get("exchangeName", ""),
        })

    logger.info(
        "After filters: %d candidates (skipped %d aliases, %d empty, %d RTH)",
        len(rows), skipped_alias, skipped_no_symbol, skipped_rth,
    )

    deduped = dedupe_by_symbol(rows)
    logger.info("After dedupe: %d unique symbols", len(deduped))

    existing_symbols = _read_existing_symbols(output_csv_path)
    new_symbols_set = {r["symbol"] for r in deduped}
    new_symbols = sorted(new_symbols_set - existing_symbols)
    removed_symbols = sorted(existing_symbols - new_symbols_set)
    logger.info(
        "Delta: +%d new, -%d removed (vs %d existing)",
        len(new_symbols), len(removed_symbols), len(existing_symbols),
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
