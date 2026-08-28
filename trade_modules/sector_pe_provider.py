"""
Dynamic Sector PE Provider

Fetches live sector PE data from sector ETFs to replace static benchmarks.
This provides more accurate sector-relative valuations.

P1 Improvement - Implemented from HEDGE_FUND_REVIEW.md recommendations.
"""

import logging
import os
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Sector ETF mapping - maps yfinance sector names to SPDR sector ETFs
SECTOR_ETF_MAP: dict[str, str] = {
    # Technology sector variants
    "Technology": "XLK",
    "Information Technology": "XLK",
    # Healthcare sector variants
    "Healthcare": "XLV",
    "Health Care": "XLV",
    # Financials sector variants
    "Financials": "XLF",
    "Financial Services": "XLF",
    "Financial": "XLF",
    # Consumer sectors
    "Consumer Discretionary": "XLY",
    "Consumer Cyclical": "XLY",
    "Consumer Staples": "XLP",
    "Consumer Defensive": "XLP",
    # Other sectors
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

# Default PE values (fallback when ETF data unavailable)
DEFAULT_SECTOR_PE: dict[str, float] = {
    "Technology": 28.0,
    "Healthcare": 22.0,
    "Financials": 12.0,
    "Consumer Discretionary": 20.0,
    "Consumer Staples": 22.0,
    "Energy": 12.0,
    "Industrials": 18.0,
    "Materials": 14.0,
    "Real Estate": 35.0,
    "Utilities": 18.0,
    "Communication Services": 18.0,
}

DEFAULT_MEDIAN_PE = 20.0

# Cache for sector PE values.
#
# Two timestamps, deliberately: one for the last GOOD refresh and one for the
# last ATTEMPT. Caching only success means a failing refresh is retried on
# every single call, and one refresh loops over every distinct sector ETF.
# `get_dynamic_sector_pe` sits on the per-ticker signal path
# (`async_yahoo_finance.calculate_pe_vs_sector` -> `data_normalizer` -> here),
# so with the quote source unreachable that turned a whole-universe scoring run
# into one full ETF sweep per name.
_sector_pe_cache: dict[str, float] = {}
_cache_timestamp: datetime | None = None  # last SUCCESSFUL refresh
_last_attempt_timestamp: datetime | None = None  # last attempt, success or failure
_cache_lock = threading.Lock()
_CACHE_TTL_HOURS = 4  # Refresh every 4 hours
# Negative TTL: how long to back off after a FAILED refresh. Deliberately far
# shorter than the success TTL, because it also bounds the one behavioural cost
# of negative caching. On a TOTAL outage the served value is the same 28.0
# default either way and only the cost changes; but on a PARTIAL or transient
# outage a later ticker that would have succeeded gets the cached default
# instead. Ten minutes keeps that window small while still collapsing a
# multi-thousand-name scan down to a handful of attempts.
_NEGATIVE_TTL_MINUTES = 10


def _fetch_etf_pe(etf_symbol: str) -> float | None:
    """Fetch trailing PE for a sector ETF."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(etf_symbol)
        info = ticker.info
        pe = info.get("trailingPE")
        if pe and pe > 0:
            return round(pe, 2)
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch PE for {etf_symbol}: {e}")
        return None


def _refresh_cache() -> None:
    """Refresh the sector PE cache from ETFs."""
    global _sector_pe_cache, _cache_timestamp

    logger.info("Refreshing sector PE cache from ETFs...")
    new_cache = {}

    # Get unique ETFs to avoid duplicate calls
    unique_etfs = set(SECTOR_ETF_MAP.values())
    etf_pe_values: dict[str, float] = {}

    for etf in unique_etfs:
        pe = _fetch_etf_pe(etf)
        if pe:
            etf_pe_values[etf] = pe
            logger.debug(f"Fetched {etf} PE: {pe}")

    # Map ETF values back to sector names
    for sector, etf in SECTOR_ETF_MAP.items():
        if etf in etf_pe_values:
            new_cache[sector] = etf_pe_values[etf]

    if new_cache:
        # A failed refresh must never clobber good data, so the assignment
        # stays inside this arm.
        _sector_pe_cache = new_cache
        _cache_timestamp = datetime.now()
        logger.info(f"Sector PE cache refreshed with {len(new_cache)} sectors")
        missing = len(unique_etfs) - len(etf_pe_values)
        if missing:
            # A PARTIAL refresh is truthy and takes the full success TTL, so
            # the sectors that did not answer quietly serve static defaults
            # for four hours. That predates negative caching and is not
            # changed here; it is at least no longer silent.
            logger.warning(
                f"Sector PE refresh was PARTIAL: {missing} of {len(unique_etfs)} "
                f"sector ETFs did not answer, so those sectors will serve static "
                f"defaults for up to {_CACHE_TTL_HOURS}h"
            )
    else:
        logger.warning(
            f"Failed to fetch any sector PE data, using defaults; "
            f"not retrying for {_NEGATIVE_TTL_MINUTES} min"
        )


def _is_cache_valid() -> bool:
    """Check if the last SUCCESSFUL refresh is still within its TTL."""
    if not _cache_timestamp:
        return False
    return datetime.now() - _cache_timestamp < timedelta(hours=_CACHE_TTL_HOURS)


def _is_backing_off() -> bool:
    """Check if a recent failed attempt should suppress another refresh."""
    if not _last_attempt_timestamp:
        return False
    return datetime.now() - _last_attempt_timestamp < timedelta(minutes=_NEGATIVE_TTL_MINUTES)


def _should_attempt_refresh() -> bool:
    """Decide whether to hit the network for fresh sector PE data."""
    if _is_cache_valid():
        return False  # Good data, still fresh.
    return not _is_backing_off()  # Otherwise refresh unless a recent try failed.


def _maybe_refresh() -> None:
    """Refresh under the lock, honouring both the success and negative TTLs.

    The attempt is stamped by this function rather than by ``_refresh_cache``
    so that an exception ESCAPING the refresh is negatively cached too. Stamp
    it inside ``_refresh_cache`` and a refresh that raises would never record
    the attempt, leaving exactly the retry-every-call behaviour this fix
    exists to remove.

    The lock is held across the refresh on purpose: concurrent callers queue
    behind a single in-flight sweep instead of sending one each.
    """
    global _last_attempt_timestamp

    with _cache_lock:
        if not _should_attempt_refresh():
            return
        _last_attempt_timestamp = datetime.now()
        try:
            _refresh_cache()
        except Exception as e:
            logger.warning(
                f"Cache refresh failed: {e}; not retrying for {_NEGATIVE_TTL_MINUTES} min"
            )


def get_dynamic_sector_pe(sector: str) -> float:
    """
    Get the current sector PE from ETF data.

    A successful refresh is cached for ``_CACHE_TTL_HOURS`` (4). A FAILED
    refresh is also cached, for the much shorter ``_NEGATIVE_TTL_MINUTES``
    (10), so that an unreachable quote source costs one ETF sweep rather than
    one per caller.

    Falls back to static defaults if ETF data unavailable. A previously
    fetched value survives a failed refresh and is preferred to the default.

    Args:
        sector: Sector name from yfinance (e.g., "Technology", "Financial Services")

    Returns:
        Sector trailing PE value
    """
    _maybe_refresh()

    # Try cache first
    if sector in _sector_pe_cache:
        return _sector_pe_cache[sector]

    # Fall back to defaults
    if sector in DEFAULT_SECTOR_PE:
        return DEFAULT_SECTOR_PE[sector]

    # Ultimate fallback
    return DEFAULT_MEDIAN_PE


def get_all_sector_pe() -> dict[str, float]:
    """
    Get all sector PE values (for display/debugging).

    Shares the success and negative TTLs with :func:`get_dynamic_sector_pe`,
    so a failure seen through either entry point suppresses both.

    Returns:
        Dictionary of sector -> PE values
    """
    _maybe_refresh()

    # Merge cache with defaults
    result = DEFAULT_SECTOR_PE.copy()
    result.update(_sector_pe_cache)
    return result


def invalidate_cache() -> None:
    """Force cache invalidation (for testing).

    Clears the negative backoff as well as the success TTL. Otherwise a forced
    invalidation after a failed refresh would be silently ignored until the
    negative TTL expired, and the tests' autouse fixture would leak backoff
    state from one test into the next.
    """
    global _cache_timestamp, _last_attempt_timestamp
    with _cache_lock:
        _cache_timestamp = None
        _last_attempt_timestamp = None
        _sector_pe_cache.clear()


# Pre-warm cache on module load (optional, disabled by default)
# To enable: set PREWARM_SECTOR_PE=1 environment variable
#
# Routed through _maybe_refresh so a failed pre-warm records its attempt like
# any other. _refresh_cache then has exactly one caller, and there is no second
# ungated path to the network.
if os.environ.get("PREWARM_SECTOR_PE") == "1":
    _maybe_refresh()
