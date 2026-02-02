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


def calculate_analyst_momentum(yticker) -> Dict[str, Any]:
    """
    Calculate analyst momentum by comparing current vs 3-month-ago recommendations.

    Analyst momentum measures whether analysts are becoming more bullish (positive)
    or more bearish (negative) on a stock over time.

    Args:
        yticker: yfinance Ticker object

    Returns:
        Dictionary with:
        - analyst_momentum: Change in buy% over 3 months (positive = upgrading)
        - analyst_count_trend: "increasing", "stable", or "decreasing"
    """
    result: Dict[str, Any] = {
        "analyst_momentum": None,
        "analyst_count_trend": None
    }

    try:
        recommendations_df = getattr(yticker, "recommendations", None)
        if recommendations_df is None or len(recommendations_df) < 2:
            return result  # Need at least 2 months of data

        # Sort by date and get rows
        sorted_df = recommendations_df.sort_index()

        # Get latest and approximately 3 months ago (or earliest available)
        latest = sorted_df.iloc[-1]
        # Try to get 3 months ago, otherwise use earliest
        past_idx = max(0, len(sorted_df) - 3)
        past = sorted_df.iloc[past_idx]

        # Calculate buy percentages
        def calc_buy_pct(row):
            total = (
                row.get("strongBuy", 0) +
                row.get("buy", 0) +
                row.get("hold", 0) +
                row.get("sell", 0) +
                row.get("strongSell", 0)
            )
            if total == 0:
                return None
            return ((row.get("strongBuy", 0) + row.get("buy", 0)) / total) * 100

        current_pct = calc_buy_pct(latest)
        past_pct = calc_buy_pct(past)

        if current_pct is not None and past_pct is not None:
            result["analyst_momentum"] = round(current_pct - past_pct, 1)

        # Calculate count trend
        current_count = sum([
            latest.get("strongBuy", 0),
            latest.get("buy", 0),
            latest.get("hold", 0),
            latest.get("sell", 0),
            latest.get("strongSell", 0)
        ])
        past_count = sum([
            past.get("strongBuy", 0),
            past.get("buy", 0),
            past.get("hold", 0),
            past.get("sell", 0),
            past.get("strongSell", 0)
        ])

        if past_count > 0:
            if current_count > past_count * 1.1:
                result["analyst_count_trend"] = "increasing"
            elif current_count < past_count * 0.9:
                result["analyst_count_trend"] = "decreasing"
            else:
                result["analyst_count_trend"] = "stable"

        logger.debug(
            f"Analyst momentum for {yticker.ticker}: momentum={result['analyst_momentum']}, "
            f"trend={result['analyst_count_trend']}"
        )

    except Exception as e:
        logger.warning(f"Error calculating analyst momentum for {yticker.ticker}: {e}")

    return result


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
