"""
Price Cache (CIO v17 op #8)

Eliminates yfinance as a single point of failure. Every consumer
(backtester, sign calibrator, risk manager, factor attribution) goes
through this module. Hits the cache first; only falls back to yfinance
when the cache is missing or stale.

Storage: parquet files at ~/.weirdapps-trading/price_cache/{ticker}_1y.parquet
Each file holds 1y of daily OHLCV indexed by date.

Refresh policy:
  * fresh        — last bar <2 trading days old
  * stale        — last bar 2-7 days old (still usable, log a warning)
  * very_stale   — last bar >7 days old (force refresh from yfinance)
  * missing      — no file (fetch from yfinance)

The refresh script (refresh_price_cache.py) is intended to run daily
from the GitHub Actions cron at 02:00 UTC after the signal pipeline.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".weirdapps-trading" / "price_cache"
STALE_TRADING_DAYS = 2     # warning threshold
VERY_STALE_DAYS = 7        # force refresh threshold


def _cache_path(ticker: str, cache_dir: Optional[Path] = None) -> Path:
    """Resolve the parquet path for a ticker."""
    cd = cache_dir or DEFAULT_CACHE_DIR
    cd.mkdir(parents=True, exist_ok=True)
    # Sanitise ticker so filesystem accepts it (e.g. "BRK-B", "BTC-USD").
    safe = ticker.replace("/", "_").replace("\\", "_")
    return cd / f"{safe}_1y.parquet"


def _last_bar_date(df: pd.DataFrame) -> Optional[datetime]:
    if df is None or df.empty:
        return None
    last = df.index[-1]
    if isinstance(last, pd.Timestamp):
        return last.to_pydatetime()
    try:
        return datetime.fromisoformat(str(last))
    except ValueError:
        return None


def freshness_status(ticker: str, cache_dir: Optional[Path] = None) -> str:
    """Return one of: missing, fresh, stale, very_stale."""
    p = _cache_path(ticker, cache_dir)
    if not p.exists():
        return "missing"
    try:
        df = pd.read_parquet(p)
    except Exception:
        return "missing"
    last = _last_bar_date(df)
    if last is None:
        return "missing"
    age_days = (datetime.now() - last).days
    if age_days <= STALE_TRADING_DAYS:
        return "fresh"
    if age_days <= VERY_STALE_DAYS:
        return "stale"
    return "very_stale"


def load_prices(
    tickers: Iterable[str],
    cache_dir: Optional[Path] = None,
    allow_stale: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load 1y daily OHLCV per ticker from cache.

    When `allow_stale=True` (default), returns whatever the cache holds
    even if the last bar is up to 7 days old. Set `allow_stale=False` to
    skip stale tickers entirely (the caller decides what to do).

    Returns {ticker: DataFrame[Open,High,Low,Close,Volume]}. Missing
    tickers are simply absent from the dict.
    """
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        status = freshness_status(t, cache_dir)
        if status == "missing":
            continue
        if status == "very_stale":
            logger.warning("price cache very stale for %s — caller should refresh", t)
            if not allow_stale:
                continue
        try:
            df = pd.read_parquet(_cache_path(t, cache_dir))
            out[t] = df
        except Exception as exc:
            logger.warning("Failed to read cache for %s: %s", t, exc)
    return out


def fetch_and_cache(
    tickers: Iterable[str],
    cache_dir: Optional[Path] = None,
    period: str = "1y",
) -> Dict[str, str]:
    """
    Fetch fresh prices from yfinance and persist to parquet.

    Returns {ticker: "ok"|"fail"|"empty"} per ticker.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot refresh cache")
        return {t: "fail" for t in tickers}

    results: Dict[str, str] = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True)
            if df is None or df.empty:
                results[t] = "empty"
                continue
            # Strip timezone for parquet stability across systems.
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.to_parquet(_cache_path(t, cache_dir))
            results[t] = "ok"
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", t, exc)
            results[t] = "fail"
    return results


def refresh_if_stale(
    tickers: Iterable[str],
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, str]:
    """
    Refresh cache entries for tickers whose status is `missing`,
    `very_stale`, or — when `force=True` — also `stale`.

    Returns the same {ticker: status} dict as fetch_and_cache.
    """
    needs_refresh: List[str] = []
    for t in tickers:
        st = freshness_status(t, cache_dir)
        if st in ("missing", "very_stale"):
            needs_refresh.append(t)
        elif st == "stale" and force:
            needs_refresh.append(t)
    if not needs_refresh:
        return {}
    logger.info("Refreshing %d stale/missing cache entries", len(needs_refresh))
    return fetch_and_cache(needs_refresh, cache_dir)


def cache_stats(cache_dir: Optional[Path] = None) -> Dict[str, int]:
    """Health snapshot of the entire cache."""
    cd = cache_dir or DEFAULT_CACHE_DIR
    if not cd.is_dir():
        return {"total": 0, "fresh": 0, "stale": 0, "very_stale": 0}
    counts = {"total": 0, "fresh": 0, "stale": 0, "very_stale": 0}
    for f in cd.glob("*_1y.parquet"):
        ticker = f.stem.replace("_1y", "")
        st = freshness_status(ticker, cd)
        counts["total"] += 1
        if st in counts:
            counts[st] += 1
    return counts


def write_health_report(
    cache_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Persist a JSON snapshot for downstream observability."""
    cd = cache_dir or DEFAULT_CACHE_DIR
    out = output_path or cd / "_health.json"
    stats = cache_stats(cd)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "cache_dir": str(cd),
        "stats": stats,
    }
    out.write_text(json.dumps(payload, indent=2))
    return out
