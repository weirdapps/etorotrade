"""
Utilities for providers across the codebase.

This module provides utilities for providers to reduce duplication
and promote consistent behavior.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from yahoofinance.utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..core.logging_config import get_logger


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
