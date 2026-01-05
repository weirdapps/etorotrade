"""
Data normalization utilities for Yahoo Finance API.

This module provides functions for formatting and calculating financial data
from Yahoo Finance API responses.
"""

import pandas as pd
from typing import Any, Optional
from yahoofinance.core.logging import get_logger

logger = get_logger(__name__)


def format_market_cap(value: Optional[float]) -> Optional[str]:
    """
    Format market cap value to human-readable string.

    Args:
        value: Market cap value in dollars

    Returns:
        Formatted string (e.g., "3.14T", "297B", "5.2M") or None
    """
    if value is None:
        return None
    try:
        val = float(value)
        if val >= 1e12:
            return f"{val / 1e12:.1f}T" if val >= 10e12 else f"{val / 1e12:.2f}T"
        elif val >= 1e9:
            if val >= 100e9:
                return f"{int(val / 1e9)}B"
            elif val >= 10e9:
                return f"{val / 1e9:.1f}B"
            else:
                return f"{val / 1e9:.2f}B"
        elif val >= 1e6:
            if val >= 100e6:
                return f"{int(val / 1e6)}M"
            elif val >= 10e6:
                return f"{val / 1e6:.1f}M"
            else:
                return f"{val / 1e6:.2f}M"
        else:
            return f"{int(val):,}"
    except (ValueError, TypeError):
        return str(value)


def calculate_upside_potential(
    current_price: Optional[float], target_price: Optional[float]
) -> Optional[float]:
    """
    Calculate upside potential percentage.

    Args:
        current_price: Current stock price
        target_price: Analyst target price

    Returns:
        Upside percentage or None if calculation not possible
    """
    if current_price is not None and target_price is not None and current_price > 0:
        try:
            return ((float(target_price) / float(current_price)) - 1) * 100
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    return None


def format_date(date: Any) -> Optional[str]:
    """
    Format date object to YYYY-MM-DD string.

    Args:
        date: Date object (datetime, date, str, or timestamp)

    Returns:
        Formatted date string or None
    """
    if date is None:
        return None
    if isinstance(date, str):
        return date[:10]

    # Handle datetime/date objects without retaining references that could cause tzinfo leaks
    if hasattr(date, "strftime"):
        try:
            # Convert to simple string immediately to avoid holding timezone references
            formatted = date.strftime("%Y-%m-%d")
            # Force date object to be garbage collected
            date = None
            return formatted
        except AttributeError:
            # If strftime fails for some reason
            pass

    # Fall back to string conversion
    try:
        result = str(date)[:10]
        # Explicitly remove reference to the original object
        date = None
        return result
    except Exception as e:
        from yahoofinance.utils.error_handling import translate_error
        # Translate standard exception to our error hierarchy
        custom_error = translate_error(e, context={"location": __name__})
        raise custom_error
    return None


def calculate_earnings_growth(ticker: str) -> Optional[float]:
    """
    Calculate earnings growth by comparing recent quarters.

    Uses quarterly income statement data since quarterly_earnings is deprecated.
    Tries year-over-year growth first, falls back to quarter-over-quarter if needed.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Earnings growth as percentage, or None if unable to calculate
    """
    try:
        import yfinance as yf

        yticker = yf.Ticker(ticker)

        # Use quarterly income statement (quarterly_earnings is deprecated)
        quarterly_income = yticker.quarterly_income_stmt
        if quarterly_income is None or quarterly_income.empty:
            logger.debug(f"No quarterly income statement data for {ticker}")
            return None

        # Look for net income fields in order of preference
        earnings_row = None
        potential_keys = [
            'Net Income From Continuing Operation Net Minority Interest',
            'Net Income Common Stockholders',
            'Net Income',
            'Net Income Including Noncontrolling Interests',
            'Net Income Continuous Operations'
        ]

        for potential_key in potential_keys:
            if potential_key in quarterly_income.index:
                earnings_row = quarterly_income.loc[potential_key]
                logger.debug(f"Using earnings field '{potential_key}' for {ticker}")
                break

        if earnings_row is None:
            logger.debug(f"No earnings row found in quarterly income statement for {ticker}")
            return None

        # Remove NaN values and sort by date descending (most recent first)
        earnings_data = earnings_row.dropna().sort_index(ascending=False)

        if len(earnings_data) < 2:
            logger.debug(f"Insufficient income statement data for {ticker} (only {len(earnings_data)} quarters)")
            return None

        # Convert to numeric and handle any string/object types
        earnings_data = pd.to_numeric(earnings_data, errors='coerce').dropna()

        if len(earnings_data) < 2:
            logger.debug(f"Insufficient numeric earnings data for {ticker}")
            return None

        # Try year-over-year calculation first (preferred)
        if len(earnings_data) >= 4:
            current_quarter = float(earnings_data.iloc[0])  # Most recent
            year_ago_quarter = float(earnings_data.iloc[3])  # 4 quarters ago

            if year_ago_quarter != 0 and abs(year_ago_quarter) > 1000:  # Avoid division by small numbers
                # Year-over-year growth
                yoy_growth = ((current_quarter - year_ago_quarter) / abs(year_ago_quarter)) * 100
                logger.debug(f"Calculated YoY earnings growth for {ticker}: {yoy_growth:.1f}% (current: {current_quarter:,.0f}, year ago: {year_ago_quarter:,.0f})")
                return round(yoy_growth, 1)

        # Fall back to quarter-over-quarter calculation
        current_quarter = float(earnings_data.iloc[0])  # Most recent
        previous_quarter = float(earnings_data.iloc[1])  # Previous quarter

        if previous_quarter != 0 and abs(previous_quarter) > 1000:  # Avoid division by small numbers
            # Quarter-over-quarter growth (not annualized for display)
            qoq_growth = ((current_quarter - previous_quarter) / abs(previous_quarter)) * 100
            logger.debug(f"Calculated QoQ earnings growth for {ticker}: {qoq_growth:.1f}% (current: {current_quarter:,.0f}, previous: {previous_quarter:,.0f})")
            return round(qoq_growth, 1)

        logger.debug(f"Unable to calculate earnings growth for {ticker} - zero or insufficient base earnings")
        return None

    except Exception as e:
        logger.debug(f"Error calculating earnings growth for {ticker}: {e}")
        return None


def is_us_ticker(ticker: str) -> bool:
    """
    Check if a ticker is a US ticker based on suffix.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if ticker is for a US-listed security
    """
    try:
        from yahoofinance.utils.market.ticker_utils import is_us_ticker as util_is_us_ticker
        return util_is_us_ticker(ticker)
    except ImportError:
        logger.warning("Could not import is_us_ticker utility, using inline logic.")
        if ticker in ["BRK.A", "BRK.B", "BF.A", "BF.B"]:
            return True
        if "." not in ticker:
            return True
        if ticker.endswith(".US"):
            return True
        return False
