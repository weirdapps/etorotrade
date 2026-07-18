"""BUILD ③ (2026-07-18): offline, cache-backed sector resolution for v3.

v3 previously took ``sector`` ONLY from live yfinance ``.info["sector"]``. When
yfinance throttled, ``sector`` went NaN, ``combine._sector_demean`` collapsed every
NaN name into one ``__NA__`` bucket (global demean = a no-op), and the risk-gate
sector cap became a single-bucket no-op with no diversification semantics.

The v2 static index map (``market.csv`` + ``usindex.csv``, ~519 S&P-scale large-caps)
covers only ~14% of our 3,345-name US universe, so it cannot carry sector-neutrality
on its own. This module layers three tiers, highest trust first:

    static index map  >  persistent cache  >  live yfinance (written back to cache)

The write-back means each run backfills the cache from that run's live sectors, so
coverage grows toward yfinance availability and — critically — a throttled run still
gets stable sectors from the cache instead of silently degrading. Sectors are slow-
changing, so caching carries no look-ahead risk. Taxonomy is yfinance's 11 GICS
sectors, identical to the static CSVs — no taxonomy mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

from trade_modules.pipeline_v2.sectors import load_sector_map

# Repo-anchored so the static files resolve regardless of CWD (cron/VPS/tests).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_PATHS: tuple[str, ...] = (
    str(_REPO_ROOT / "yahoofinance" / "input" / "market.csv"),
    str(_REPO_ROOT / "yahoofinance" / "input" / "usindex.csv"),
)
DEFAULT_CACHE_PATH: str = str(Path("~/.weirdapps-trading/v3_sector_cache.json").expanduser())


def merge_offline_map(static: dict, cache: dict) -> dict:
    """Merge the static index map over the persistent cache — STATIC WINS.

    Blank values are dropped; keys are upper-cased for case-insensitive lookup.
    """
    merged = {str(k).upper(): v for k, v in (cache or {}).items() if v}
    merged.update({str(k).upper(): v for k, v in (static or {}).items() if v})
    return merged


def _read_cache(cache_path: str) -> dict:
    """Read the persistent sector cache; ``{}`` on missing/corrupt/unreadable."""
    try:
        with open(Path(cache_path).expanduser(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k).upper(): str(v) for k, v in data.items() if v}


def load_offline_sector_map(
    static_paths: tuple[str, ...] = DEFAULT_STATIC_PATHS,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> dict:
    """Return the offline sector map (static index CSVs + persistent cache, static wins).

    Keyed by upper-cased ticker. Safe to call with no data present -> ``{}``.
    """
    static = load_sector_map(list(static_paths))
    cache = _read_cache(cache_path)
    return merge_offline_map(static, cache)


def update_sector_cache(
    resolved,
    *,
    cache_path: str = DEFAULT_CACHE_PATH,
    static_paths: tuple[str, ...] = DEFAULT_STATIC_PATHS,
) -> int:
    """Persist live-resolved sectors that the static map does NOT already cover.

    ``resolved`` is any ``{ticker: sector}`` mapping (e.g. ``feats['sector']`` after
    enrichment). Static-covered names are skipped (the static map is authoritative
    and needs no caching). Returns the number of NEW cache entries written. Never
    raises — a failed write returns 0.
    """
    static = load_sector_map(list(static_paths))
    cache = _read_cache(cache_path)
    added = 0
    for ticker, sector in dict(resolved or {}).items():
        if not sector:
            continue
        key = str(ticker).upper()
        if key in static:
            continue  # static is authoritative — do not cache
        if cache.get(key) != str(sector):
            cache[key] = str(sector)
            added += 1
    if added:
        try:
            path = Path(cache_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=0, sort_keys=True)
        except OSError:
            return 0
    return added
