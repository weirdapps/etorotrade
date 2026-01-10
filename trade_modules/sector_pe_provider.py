"""
Dynamic Sector PE Provider

Fetches live sector PE data from sector ETFs to replace static benchmarks.
This provides more accurate sector-relative valuations.

P1 Improvement - Implemented from HEDGE_FUND_REVIEW.md recommendations.
"""

import logging
import os
from typing import Dict, Optional
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

# Sector ETF mapping - maps yfinance sector names to SPDR sector ETFs
SECTOR_ETF_MAP: Dict[str, str] = {
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
DEFAULT_SECTOR_PE: Dict[str, float] = {
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

# Cache for sector PE values
_sector_pe_cache: Dict[str, float] = {}
_cache_timestamp: Optional[datetime] = None
_cache_lock = threading.Lock()
_CACHE_TTL_HOURS = 4  # Refresh every 4 hours


def _fetch_etf_pe(etf_symbol: str) -> Optional[float]:
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
    etf_pe_values: Dict[str, float] = {}

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
        _sector_pe_cache = new_cache
        _cache_timestamp = datetime.now()
        logger.info(f"Sector PE cache refreshed with {len(new_cache)} sectors")
    else:
        logger.warning("Failed to fetch any sector PE data, using defaults")


def _is_cache_valid() -> bool:
    """Check if the cache is still valid."""
    if not _cache_timestamp:
        return False
    return datetime.now() - _cache_timestamp < timedelta(hours=_CACHE_TTL_HOURS)


def get_dynamic_sector_pe(sector: str) -> float:
    """
    Get the current sector PE from ETF data.

    Uses cached values with 4-hour TTL to minimize API calls.
    Falls back to static defaults if ETF data unavailable.

    Args:
        sector: Sector name from yfinance (e.g., "Technology", "Financial Services")

    Returns:
        Sector trailing PE value
    """
    with _cache_lock:
        # Refresh cache if expired or empty
        if not _is_cache_valid():
            try:
                _refresh_cache()
            except Exception as e:
                logger.warning(f"Cache refresh failed: {e}")

    # Try cache first
    if sector in _sector_pe_cache:
        return _sector_pe_cache[sector]

    # Fall back to defaults
    if sector in DEFAULT_SECTOR_PE:
        return DEFAULT_SECTOR_PE[sector]

    # Ultimate fallback
    return DEFAULT_MEDIAN_PE


def get_all_sector_pe() -> Dict[str, float]:
    """
    Get all sector PE values (for display/debugging).

    Returns:
        Dictionary of sector -> PE values
    """
    with _cache_lock:
        if not _is_cache_valid():
            try:
                _refresh_cache()
            except Exception as e:
                logger.warning(f"Cache refresh failed: {e}")

    # Merge cache with defaults
    result = DEFAULT_SECTOR_PE.copy()
    result.update(_sector_pe_cache)
    return result


def invalidate_cache() -> None:
    """Force cache invalidation (for testing)."""
    global _cache_timestamp
    with _cache_lock:
        _cache_timestamp = None
        _sector_pe_cache.clear()


# Pre-warm cache on module load (optional, disabled by default)
# To enable: set PREWARM_SECTOR_PE=1 environment variable
if os.environ.get("PREWARM_SECTOR_PE") == "1":
    try:
        _refresh_cache()
    except Exception:
        pass
