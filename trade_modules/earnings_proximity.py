"""
Earnings Proximity Module

Checks if a stock is within N trading days of an earnings release.
Signals near earnings carry higher binary event risk and should be
treated differently — either flagged or forced to HOLD.

CIO Review Finding M4: No earnings proximity adjustment in signal engine.
"""

import logging
import threading
from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc  # Python 3.10 compat (datetime.UTC is 3.11+)
from typing import Any

logger = logging.getLogger(__name__)

# Proximity thresholds (calendar days)
EARNINGS_IMMINENT_DAYS = 7  # Within 7 calendar days of earnings
EARNINGS_RECENT_DAYS = 5  # Within 5 calendar days AFTER earnings
POST_EARNINGS_BOOST_DAYS = 14  # Signals within 14 days after earnings are freshest
POST_EARNINGS_NORMAL_DAYS = 45  # 14-45 days: moderate accuracy
# Beyond 45 days: estimates getting stale (CIO v3 F4)

# Cache for earnings dates
_earnings_cache: dict[str, tuple[datetime | None, datetime]] = {}
_earnings_cache_lock = threading.Lock()
_EARNINGS_CACHE_TTL_HOURS = 12  # Earnings dates don't change often


def _fetch_next_earnings(ticker: str) -> datetime | None:
    """
    Fetch next earnings date for a ticker from yfinance.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Next earnings datetime, or None if unavailable
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        # Try calendar first
        try:
            cal = stock.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    earnings_date = cal.get("Earnings Date")
                    if earnings_date:
                        if isinstance(earnings_date, list) and len(earnings_date) > 0:
                            return earnings_date[0]
                        elif isinstance(earnings_date, datetime):
                            return earnings_date
                elif isinstance(cal, list) and len(cal) > 0:
                    return cal[0] if isinstance(cal[0], datetime) else None
        except Exception:
            pass

        # Fallback: try earnings_dates attribute
        try:
            dates = stock.earnings_dates
            if dates is not None and not dates.empty:
                future_dates = dates.index[dates.index >= datetime.now()]
                if len(future_dates) > 0:
                    return future_dates[0].to_pydatetime()
        except Exception:
            pass

        return None
    except Exception as e:
        logger.debug(f"Failed to fetch earnings date for {ticker}: {e}")
        return None


def get_next_earnings(ticker: str) -> datetime | None:
    """
    Get next earnings date with caching.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Next earnings datetime, or None
    """
    with _earnings_cache_lock:
        if ticker in _earnings_cache:
            value, cached_at = _earnings_cache[ticker]
            if datetime.now() - cached_at < timedelta(hours=_EARNINGS_CACHE_TTL_HOURS):
                return value

    earnings_date = _fetch_next_earnings(ticker)

    with _earnings_cache_lock:
        _earnings_cache[ticker] = (earnings_date, datetime.now())

    return earnings_date


def check_earnings_proximity(ticker: str) -> dict[str, Any]:
    """
    Check if a stock is near its earnings date.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with:
            earnings_date: datetime or None
            days_until: int or None (negative = days since)
            status: 'imminent' | 'recent' | 'post_earnings_window' | 'clear' | 'unknown'
            should_hold: bool - whether to force HOLD due to earnings proximity
            conviction_boost: bool - whether post-earnings window boosts conviction
    """
    earnings_date = get_next_earnings(ticker)

    if earnings_date is None:
        return {
            "earnings_date": None,
            "days_until": None,
            "status": "unknown",
            "should_hold": False,
            "conviction_boost": False,
        }

    now = datetime.now()
    # Handle date vs datetime — yfinance may return either
    if isinstance(earnings_date, date) and not isinstance(earnings_date, datetime):
        earnings_date = datetime(earnings_date.year, earnings_date.month, earnings_date.day)
    # Handle timezone-aware dates
    if hasattr(earnings_date, "tzinfo") and earnings_date.tzinfo is not None:
        now = now.replace(tzinfo=UTC)

    delta = (earnings_date - now).days

    # CIO v3 F4: Earnings recency decay factor
    # Analyst estimates are most accurate right after earnings and decay over the quarter.
    # conviction_adjustment: positive = boost, negative = penalty, 0 = neutral
    if 0 <= delta <= EARNINGS_IMMINENT_DAYS:
        status = "imminent"
        should_hold = True
        conviction_boost = False
        conviction_adjustment = 0  # Don't adjust — holding anyway
    elif -EARNINGS_RECENT_DAYS <= delta < 0:
        status = "recent"
        should_hold = False  # Just reported — signal may be adjusting
        conviction_boost = False
        conviction_adjustment = 0  # Too soon for estimates to update
    elif -POST_EARNINGS_BOOST_DAYS <= delta < -EARNINGS_RECENT_DAYS:
        status = "post_earnings_window"
        should_hold = False
        conviction_boost = True  # Post-earnings signals are highest quality
        conviction_adjustment = 5  # Fresh estimates boost
    elif -POST_EARNINGS_NORMAL_DAYS <= delta < -POST_EARNINGS_BOOST_DAYS:
        status = "normal_window"
        should_hold = False
        conviction_boost = False
        conviction_adjustment = 0  # Normal accuracy
    elif delta < -POST_EARNINGS_NORMAL_DAYS:
        status = "stale_estimates"
        should_hold = False
        conviction_boost = False
        conviction_adjustment = -3  # Estimates getting stale
    else:
        status = "clear"
        should_hold = False
        conviction_boost = False
        conviction_adjustment = 0

    return {
        "earnings_date": earnings_date,
        "days_until": delta,
        "status": status,
        "should_hold": should_hold,
        "conviction_boost": conviction_boost,
        "conviction_adjustment": conviction_adjustment,
    }


def get_earnings_flags(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """
    Get earnings proximity flags for multiple tickers.

    Args:
        tickers: List of ticker symbols

    Returns:
        Dict mapping ticker to earnings proximity info
    """
    results = {}
    for ticker in tickers:
        results[ticker] = check_earnings_proximity(ticker)
    return results


def get_imminent_earnings(tickers: list[str]) -> list[dict[str, Any]]:
    """
    Get list of tickers with imminent earnings (for briefing/committee).

    Args:
        tickers: List of ticker symbols

    Returns:
        List of dicts with ticker and earnings info, sorted by days_until
    """
    imminent = []
    for ticker in tickers:
        info = check_earnings_proximity(ticker)
        if info["status"] == "imminent":
            imminent.append(
                {
                    "ticker": ticker,
                    "earnings_date": info["earnings_date"],
                    "days_until": info["days_until"],
                }
            )

    imminent.sort(key=lambda x: x["days_until"] if x["days_until"] is not None else 999)
    return imminent


def invalidate_cache() -> None:
    """Clear earnings cache (for testing)."""
    global _earnings_cache
    with _earnings_cache_lock:
        _earnings_cache = {}
