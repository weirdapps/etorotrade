"""
Sector-Relative Momentum Provider

Calculates momentum relative to sector ETFs to identify stocks
outperforming or underperforming their sector.

CIO Review Finding #N: Sector-relative momentum factor.
"""

import logging
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import SECTOR_ETF_MAP from sector_pe_provider
from trade_modules.sector_pe_provider import SECTOR_ETF_MAP

# Cache for momentum calculations
_momentum_cache: dict[str, dict[str, float]] = {}
_cache_timestamp: datetime | None = None
_cache_lock = threading.Lock()
_CACHE_TTL_HOURS = 4  # 4-hour cache matching other providers

# Underperformance threshold
UNDERPERFORMANCE_THRESHOLD = 15.0  # >15% below sector = underperforming


def _fetch_return(ticker: str, period_days: int = 252) -> float | None:
    """
    Fetch total return for a ticker over a period.

    Args:
        ticker: Stock ticker or ETF symbol
        period_days: Lookback period in trading days (default 252 = 1 year)

    Returns:
        Total return as percentage, or None if unable to calculate
    """
    try:
        import yfinance as yf

        yticker = yf.Ticker(ticker)

        # Calculate date range
        end_date = datetime.now()
        # Add buffer days to account for weekends/holidays
        start_date = end_date - timedelta(days=int(period_days * 1.5))

        # Fetch historical data
        hist = yticker.history(start=start_date, end=end_date)

        if hist is None or hist.empty or len(hist) < 2:
            logger.debug(f"Insufficient history for {ticker}")
            return None

        # Get first and last close prices
        first_price = float(hist["Close"].iloc[0])
        last_price = float(hist["Close"].iloc[-1])

        if first_price <= 0:
            return None

        # Calculate return as percentage
        total_return = ((last_price - first_price) / first_price) * 100
        return round(total_return, 2)

    except Exception as e:
        logger.debug(f"Failed to fetch return for {ticker}: {e}")
        return None


def calculate_relative_momentum(
    ticker: str, sector: str | None, period_days: int = 252
) -> float | None:
    """
    Calculate stock momentum relative to its sector ETF.

    Args:
        ticker: Stock ticker symbol
        sector: Sector name from yfinance
        period_days: Lookback period in trading days (default 252 = 1 year)

    Returns:
        Relative momentum as percentage (stock return - sector return),
        or None if calculation not possible
    """
    if not sector or sector not in SECTOR_ETF_MAP:
        logger.debug(f"No sector ETF mapping for sector: {sector}")
        return None

    # Get sector ETF
    sector_etf = SECTOR_ETF_MAP[sector]

    # Fetch returns
    stock_return = _fetch_return(ticker, period_days)
    sector_return = _fetch_return(sector_etf, period_days)

    if stock_return is None or sector_return is None:
        return None

    # Calculate relative momentum
    relative_momentum = stock_return - sector_return
    return round(relative_momentum, 2)


def get_relative_momentum_flags(
    tickers_with_sectors: list[tuple[str, str | None]], period_days: int = 252
) -> dict[str, dict[str, float | None]]:
    """
    Calculate relative momentum for multiple tickers (batch processing).

    Args:
        tickers_with_sectors: List of (ticker, sector) tuples
        period_days: Lookback period in trading days

    Returns:
        Dictionary mapping ticker to {
            'relative_momentum': float or None,
            'underperforming': bool (True if >15% below sector)
        }
    """
    global _cache_timestamp

    results = {}

    with _cache_lock:
        # Check cache validity
        cache_valid = (
            _cache_timestamp is not None
            and datetime.now() - _cache_timestamp < timedelta(hours=_CACHE_TTL_HOURS)
        )

        for ticker, sector in tickers_with_sectors:
            # Check cache first if valid
            if cache_valid and ticker in _momentum_cache:
                results[ticker] = _momentum_cache[ticker].copy()
                logger.debug(f"Using cached momentum for {ticker}")
                continue

            # Calculate fresh
            relative_momentum = calculate_relative_momentum(ticker, sector, period_days)

            underperforming = False
            if relative_momentum is not None:
                underperforming = relative_momentum < -UNDERPERFORMANCE_THRESHOLD

            result = {"relative_momentum": relative_momentum, "underperforming": underperforming}

            results[ticker] = result
            _momentum_cache[ticker] = result.copy()

        # Update cache timestamp after batch
        if not cache_valid:
            _cache_timestamp = datetime.now()

    return results


def is_underperforming_sector(
    ticker: str, sector: str | None, threshold: float = UNDERPERFORMANCE_THRESHOLD
) -> bool:
    """
    Check if a stock is significantly underperforming its sector.

    Args:
        ticker: Stock ticker symbol
        sector: Sector name from yfinance
        threshold: Underperformance threshold in percentage points (default 15%)

    Returns:
        True if stock is underperforming sector by more than threshold
    """
    relative_momentum = calculate_relative_momentum(ticker, sector)

    if relative_momentum is None:
        return False

    return relative_momentum < -threshold


def invalidate_cache() -> None:
    """Force cache invalidation (for testing)."""
    global _cache_timestamp
    with _cache_lock:
        _cache_timestamp = None
        _momentum_cache.clear()
