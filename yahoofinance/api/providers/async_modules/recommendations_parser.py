"""
Analyst recommendations parsing utilities for Yahoo Finance API.

This module provides functions for parsing and analyzing analyst recommendations,
ratings, and earnings-related data from Yahoo Finance.
"""

import datetime
import pandas as pd
from typing import Any, Dict, Optional
from yahoofinance.core.logging import get_logger
from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.config import COLUMN_NAMES

logger = get_logger(__name__)

# Positive analyst grades
POSITIVE_GRADES = [
    "Buy",
    "Overweight",
    "Outperform",
    "Strong Buy",
    "Long-Term Buy",
    "Positive",
    "Market Outperform",
    "Add",
    "Sector Outperform",
]


def parse_analyst_recommendations(ticker_info: Dict[str, Any], yticker) -> Dict[str, Any]:
    """
    Parse analyst recommendations from ticker data.

    Attempts multiple sources in order of reliability:
    1. Recommendations DataFrame (most accurate)
    2. Recommendation key/mean from ticker info
    3. Upgrades/downgrades data

    Args:
        ticker_info: Dictionary from yticker.info
        yticker: yfinance Ticker object

    Returns:
        Dictionary with analyst_count, total_ratings, and buy_percentage
    """
    result = {
        "analyst_count": 0,
        "total_ratings": 0,
        "buy_percentage": None
    }

    try:
        # First check if ticker_info has analyst data
        number_of_analysts = ticker_info.get("numberOfAnalystOpinions", 0)
        if number_of_analysts > 0:
            result["analyst_count"] = number_of_analysts
            result["total_ratings"] = number_of_analysts

            # Try to get recommendation key first
            rec_key = ticker_info.get("recommendationKey", "").lower()
            rec_mean = ticker_info.get("recommendationMean", None)

            # If we have a recommendation key, map it to a buy percentage
            if rec_key:
                mapped_pct = _map_recommendation_key_to_percentage(rec_key, yticker.ticker)
                if mapped_pct is not None:
                    result["buy_percentage"] = int(mapped_pct)
            # If no key but we have recommendation mean, use it to estimate buy percentage
            elif rec_mean is not None:
                # Convert 1-5 scale to percentage (1=Strong Buy, 5=Sell)
                # 1 = 90%, 3 = 50%, 5 = 10%
                result["buy_percentage"] = int(max(0, min(100, 110 - (rec_mean * 20))))
                logger.debug(
                    f"Using recommendationMean {rec_mean} to set buy_percentage={result['buy_percentage']} for {yticker.ticker}"
                )

        # Try to get detailed recommendations from recommendations DataFrame (most accurate)
        recommendations_df = getattr(yticker, "recommendations", None)
        if recommendations_df is not None and not recommendations_df.empty:
            try:
                # Get the most recent recommendations
                latest_date = recommendations_df.index.max()
                latest_recs = recommendations_df.loc[latest_date]

                # Extract counts
                strong_buy = int(latest_recs.get("strongBuy", 0))
                buy = int(latest_recs.get("buy", 0))
                hold = int(latest_recs.get("hold", 0))
                sell = int(latest_recs.get("sell", 0))
                strong_sell = int(latest_recs.get("strongSell", 0))

                # Calculate the total
                total = strong_buy + buy + hold + sell + strong_sell

                # Only update if we have valid data
                if total > 0:
                    buy_count = strong_buy + buy
                    buy_percentage = (buy_count / total) * 100

                    # This is the most accurate source of data, always update
                    result["buy_percentage"] = int(buy_percentage)
                    result["total_ratings"] = total
                    result["analyst_count"] = total

                    logger.debug(
                        f"Using recommendations DataFrame to set analyst data for {yticker.ticker}: "
                        f"buy_percentage={buy_percentage}, total_ratings={total}"
                    )
            except (IndexError, KeyError, ValueError, TypeError) as e:
                logger.warning(
                    f"Error extracting recommendations data for {yticker.ticker}: {e}. Using fallback values."
                )

        # Fall back to analyzing upgrades_downgrades if needed
        if result["buy_percentage"] is None:
            try:
                upgrades_downgrades = getattr(yticker, "upgrades_downgrades", None)
                if upgrades_downgrades is not None and not upgrades_downgrades.empty:
                    # Count grades
                    if "ToGrade" in upgrades_downgrades.columns:
                        total_grades = len(upgrades_downgrades)
                        positive_count = upgrades_downgrades[
                            upgrades_downgrades["ToGrade"].isin(POSITIVE_GRADES)
                        ].shape[0]

                        if total_grades > 0:
                            buy_percentage = (positive_count / total_grades) * 100
                            result["buy_percentage"] = int(buy_percentage)
                            result["total_ratings"] = total_grades
                            result["analyst_count"] = total_grades

                            logger.debug(
                                f"Using upgrades_downgrades to set analyst data for {yticker.ticker}: "
                                f"buy_percentage={buy_percentage}, total_ratings={total_grades}"
                            )
            except (KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
                logger.warning(
                    f"Error analyzing upgrades_downgrades for {yticker.ticker}: {e}. Using fallback values."
                )

        # If we still don't have a buy percentage but have analyst count,
        # use recommendationMean as a last resort
        if (
            result["analyst_count"] > 0
            and result["buy_percentage"] is None
            and rec_mean is not None
        ):
            # Convert 1-5 scale to percentage (1=Strong Buy, 5=Sell)
            result["buy_percentage"] = max(0, min(100, 110 - (rec_mean * 20)))
            logger.debug(
                f"Final fallback: Using recommendationMean {rec_mean} to set buy_percentage={result['buy_percentage']} for {yticker.ticker}"
            )

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Error processing analyst data for {yticker.ticker}: {e}", exc_info=False)

    return result


def _map_recommendation_key_to_percentage(rec_key: str, ticker: str) -> Optional[float]:
    """Map recommendation key to buy percentage estimate."""
    mapping = {
        ("buy", "strongbuy"): 85.0,
        ("outperform",): 75.0,
        ("hold",): 50.0,
        ("underperform",): 25.0,
        ("sell",): 15.0,
    }

    rec_key_lower = rec_key.lower()
    for keys, percentage in mapping.items():
        if rec_key_lower in keys:
            logger.debug(
                f"Using recommendationKey '{rec_key}' to set buy_percentage={percentage} for {ticker}"
            )
            return percentage

    return None


def get_last_earnings_date(yticker) -> Optional[str]:
    """
    Get the last earnings date with optimized memory handling.

    Tries multiple sources in order:
    1. Quarterly income statement (most reliable)
    2. Calendar
    3. Earnings dates attribute

    Args:
        yticker: yfinance Ticker object

    Returns:
        Last earnings date as YYYY-MM-DD string or None
    """
    # Try quarterly income statement first - this is the most reliable source
    try:
        quarterly_income = getattr(yticker, "quarterly_income_stmt", None)
        if quarterly_income is not None and not quarterly_income.empty:
            # Get the most recent quarter date
            latest_date = quarterly_income.columns[0]  # Most recent is first column
            result = latest_date.strftime("%Y-%m-%d")
            logger.debug(f"Found last earnings date from quarterly_income_stmt for {yticker.ticker}: {result}")
            return result
    except (KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
        logger.debug(f"Error getting earnings from quarterly_income_stmt for {yticker.ticker}: {e}")

    # Try calendar second
    try:
        calendar = None
        earnings_date_list = None
        today = None
        result = None

        try:
            calendar = getattr(yticker, "calendar", None)
            if isinstance(calendar, dict) and COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                earnings_date_list = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                    today = pd.Timestamp.now().date()
                    # Filter for past dates
                    latest_date = None
                    for d in earnings_date_list:
                        if isinstance(d, datetime.date) and d < today:
                            if latest_date is None or d > latest_date:
                                latest_date = d

                    if latest_date is not None:
                        result = latest_date.strftime("%Y-%m-%d")
                        return result
        finally:
            # Explicitly delete references to free memory
            del calendar
            del earnings_date_list
            del today

    except YFinanceError as e:
        logger.debug(
            f"Error getting earnings from calendar for {yticker.ticker}: {e}", exc_info=False
        )

    # Try earnings_dates attribute
    try:
        earnings_dates = None
        today = None
        result = None

        try:
            earnings_dates = getattr(yticker, "earnings_dates", None)
            if earnings_dates is not None and not earnings_dates.empty:
                today = pd.Timestamp.now()

                # Find the latest past date
                latest_date = None
                for date in earnings_dates.index:
                    # Handle timezone differences
                    compare_date = date
                    compare_today = today

                    # Make timestamps comparable
                    if date.tzinfo is not None and today.tzinfo is None:
                        try:
                            compare_today = today.tz_localize(date.tzinfo)
                        except (ValueError, TypeError):
                            compare_today = today.tz_localize('UTC')
                            compare_date = date.tz_convert('UTC')
                    elif date.tzinfo is None and today.tzinfo is not None:
                        try:
                            compare_date = date.tz_localize(today.tzinfo)
                        except (ValueError, TypeError):
                            compare_today = today.tz_convert('UTC')
                            compare_date = pd.Timestamp(date).tz_localize('UTC')
                    elif date.tzinfo is not None and today.tzinfo is not None:
                        compare_today = today.tz_convert('UTC')
                        compare_date = date.tz_convert('UTC')

                    # Compare and keep the latest
                    if compare_date < compare_today:
                        if latest_date is None or compare_date > latest_date:
                            latest_date = compare_date

                # Format the date if found
                if latest_date is not None:
                    result = latest_date.strftime("%Y-%m-%d")
                    return result
        finally:
            # Explicitly delete references to free memory
            del earnings_dates
            del today

    except YFinanceError as e:
        logger.debug(
            f"Error getting earnings from earnings_dates for {yticker.ticker}: {e}",
            exc_info=False,
        )

    return None


def has_post_earnings_ratings(ticker: str, yticker, is_us: bool, positive_grades: list) -> Dict[str, Any]:
    """
    Check if there are ratings available since the last earnings date.

    Args:
        ticker: Stock ticker symbol
        yticker: yfinance Ticker object
        is_us: Whether ticker is US-based
        positive_grades: List of positive grade strings

    Returns:
        Dictionary with has_ratings (bool) and optional ratings_data (dict)
    """
    try:
        if not is_us:
            return {"has_ratings": False}

        last_earnings = get_last_earnings_date(yticker)
        if last_earnings is None:
            return {"has_ratings": False}

        try:
            # Get upgrades_downgrades but ensure we free memory afterward
            upgrades_downgrades = None
            try:
                upgrades_downgrades = getattr(yticker, "upgrades_downgrades", None)
                if upgrades_downgrades is None or upgrades_downgrades.empty:
                    return {"has_ratings": False}

                # Create a copy of the data to avoid reference issues
                if hasattr(upgrades_downgrades, "reset_index"):
                    df = upgrades_downgrades.reset_index().copy()
                else:
                    df = upgrades_downgrades.copy()

                # Process the GradeDate column
                if "GradeDate" not in df.columns:
                    if "index" in df.columns and pd.api.types.is_datetime64_any_dtype(df["index"]):
                        df.rename(columns={"index": "GradeDate"}, inplace=True)
                    elif hasattr(upgrades_downgrades, "index") and isinstance(
                        upgrades_downgrades.index, pd.DatetimeIndex
                    ):
                        df["GradeDate"] = upgrades_downgrades.index
                    else:
                        logger.warning(f"Could not find GradeDate column for {ticker}")
                        return {"has_ratings": False}

                # Convert dates
                df["GradeDate"] = pd.to_datetime(df["GradeDate"], errors="coerce")
                df.dropna(subset=["GradeDate"], inplace=True)

                # Get post-earnings data
                earnings_date = pd.to_datetime(last_earnings)

                # Handle timezone compatibility
                if "GradeDate" in df.columns:
                    if any(date.tzinfo is not None for date in df["GradeDate"] if hasattr(date, "tzinfo")):
                        if earnings_date.tzinfo is None:
                            for date in df["GradeDate"]:
                                if hasattr(date, "tzinfo") and date.tzinfo is not None:
                                    earnings_date = earnings_date.tz_localize(date.tzinfo)
                                    break
                    elif earnings_date.tzinfo is not None:
                        earnings_date = earnings_date.tz_localize(None)

                # Filter post-earnings ratings
                post_earnings_df = df[df["GradeDate"] >= earnings_date]

                # Calculate statistics if we have data
                if not post_earnings_df.empty:
                    total_ratings = len(post_earnings_df)
                    positive_ratings = post_earnings_df[
                        post_earnings_df["ToGrade"].isin(positive_grades)
                    ].shape[0]

                    ratings_data = {
                        "buy_percentage": (
                            (positive_ratings / total_ratings * 100) if total_ratings > 0 else 0
                        ),
                        "total_ratings": total_ratings,
                        "ratings_type": "E",
                    }
                    return {"has_ratings": True, "ratings_data": ratings_data}

                return {"has_ratings": False}

            finally:
                # Explicitly clear references to free memory
                del upgrades_downgrades

        except YFinanceError as e:
            logger.warning(
                f"Error getting post-earnings ratings for {ticker}: {e}", exc_info=False
            )
        return {"has_ratings": False}
    except YFinanceError as e:
        logger.warning(f"Error in has_post_earnings_ratings for {ticker}: {e}", exc_info=False)
        return {"has_ratings": False}
