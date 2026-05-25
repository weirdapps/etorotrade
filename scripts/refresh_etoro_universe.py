"""Refresh yahoofinance/input/etoro.csv from the eToro instruments API.

Fetches the public bulk instruments endpoint, filters to stocks + ETFs,
extracts ticker symbols from image URIs, normalizes to Yahoo Finance format,
and atomically overwrites the input file.
"""

import re

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
