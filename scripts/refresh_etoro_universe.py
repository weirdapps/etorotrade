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
