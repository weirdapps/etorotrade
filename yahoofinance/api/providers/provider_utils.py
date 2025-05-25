"""
Utilities for Yahoo Finance data providers.

This module provides common utility functions used by provider implementations
to reduce duplication and promote consistent behavior across providers.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ...utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ...core.errors import APIError, RateLimitError, ValidationError
from ...core.logging import get_logger


logger = get_logger(__name__)


def is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an error is related to rate limiting.

    Args:
        error: Exception to check

    Returns:
        True if error is related to rate limiting, False otherwise
    """
    error_str = str(error).lower()
    rate_limit_patterns = ["rate limit", "too many requests", "429", "quota exceeded", "throttled"]
    return any(pattern in error_str for pattern in rate_limit_patterns)


def safe_extract_value(obj: Any, key: str, default: Any = 0) -> Any:
    """
    Safely extract a value from an object, converting to float with fallback.

    Args:
        obj: Object to extract from
        key: Key to extract
        default: Default value if extraction fails

    Returns:
        Extracted value or default
    """
    try:
        if hasattr(obj, "get"):
            value = obj.get(key, default)
        elif hasattr(obj, key):
            value = getattr(obj, key)
        else:
            return default

        # Try to convert to float
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default


@with_retry
def handle_api_error(func_name: str, ticker: str, error: Exception) -> None:
    """
    Handle API errors consistently across providers.

    Args:
        func_name: Name of the function where the error occurred
        ticker: Ticker symbol being processed
        error: Exception that occurred

    Raises:
        RateLimitError: If the error is related to rate limiting
        APIError: For all other API errors
    """
    error_msg = f"Error in {func_name} for {ticker}: {str(error)}"

    if is_rate_limit_error(error):
        logger.warning(f"Rate limit error: {error_msg}")
        raise RateLimitError(f"Rate limit exceeded for {ticker}")
    else:
        logger.error(error_msg)
        raise YFinanceError("An error occurred")


def process_analyst_ratings(ratings_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process analyst ratings DataFrame to extract consistent metrics.

    Args:
        ratings_df: DataFrame containing analyst ratings

    Returns:
        Processed analyst ratings with standardized metrics
    """
    if ratings_df is None or ratings_df.empty:
        return {
            "buy_percentage": None,
            "total_ratings": 0,
            "ratings_type": "unknown",
            "recommendations": {},
        }

    try:
        # Get most recent ratings
        recent = ratings_df.iloc[-1]

        # Extract recommendation counts
        buy_count = recent.get("strongBuy", 0) + recent.get("buy", 0)
        hold_count = recent.get("hold", 0)
        sell_count = recent.get("sell", 0) + recent.get("strongSell", 0)

        total = buy_count + hold_count + sell_count

        # Calculate buy percentage
        buy_percentage = (buy_count / total) * 100 if total > 0 else None

        return {
            "buy_percentage": buy_percentage,
            "total_ratings": total,
            "ratings_type": "buy_sell_hold",
            "recommendations": {"buy": buy_count, "hold": hold_count, "sell": sell_count},
        }
    except YFinanceError as e:
        logger.warning(f"Error processing analyst ratings: {str(e)}")
        return {
            "buy_percentage": None,
            "total_ratings": 0,
            "ratings_type": "unknown",
            "recommendations": {},
        }


def truncate_ticker_lists(tickers: List[str], batch_size: int) -> List[List[str]]:
    """
    Split a list of tickers into batches for efficient API requests.

    Args:
        tickers: List of ticker symbols
        batch_size: Maximum size of each batch

    Returns:
        List of ticker batches
    """
    return [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]


def merge_ticker_results(
    results: Dict[str, Any], failed_tickers: List[str], errors: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Merge ticker results, failed tickers, and errors into a consistent format.

    Args:
        results: Dictionary of successful ticker results
        failed_tickers: List of tickers that failed to process
        errors: Dictionary mapping tickers to error messages

    Returns:
        Dictionary mapping all tickers to their results or errors
    """
    # Start with successful results
    merged = {ticker: data for ticker, data in results.items()}

    # Add failed tickers
    for ticker in failed_tickers:
        merged[ticker] = {"symbol": ticker, "error": "Failed to process"}

    # Add error information for failed tickers
    for ticker, error in errors.items():
        merged[ticker] = {"symbol": ticker, "error": error}

    return merged
