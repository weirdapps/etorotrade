"""
Utilities for working with yfinance and preventing memory leaks.

This module contains utilities for working with the yfinance library,
especially focused on preventing memory leaks caused by the library's
internal caching mechanisms and object tracking.
"""

import gc
import logging
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger(__name__)


def clean_ticker_object(ticker_obj: Any) -> None:
    """
    Clean up a yfinance.Ticker object to prevent memory leaks.

    The yfinance.Ticker class maintains several caches and properties that
    can cause memory leaks, especially when many Ticker objects are created
    and destroyed. This function attempts to clear these caches and break
    any circular references.

    Args:
        ticker_obj: A yfinance.Ticker object to clean up
    """
    if ticker_obj is None:
        return

    # Clear known problematic attributes
    problematic_attrs = [
        "_earnings",
        "_financials",
        "_balance_sheet",
        "_cashflow",
        "_recommendations",
        "_isin",
        "_major_holders",
        "_institutional_holders",
        "_mutualfund_holders",
        "_info",
        "_sustainability",
        "_calendar",
        "_expirations",
        "_options",
        "_earnings_dates",
        "_history",
        "_actions",
        "_dividends",
        "_splits",
        "_capital_gains",
        "_shares",
        "_quarterly_earnings",
        "_quarterly_financials",
        "_quarterly_balance_sheet",
        "_quarterly_cashflow",
    ]

    for attr in problematic_attrs:
        if hasattr(ticker_obj, attr):
            try:
                delattr(ticker_obj, attr)
            except (AttributeError, TypeError):
                # Some attrs might be properties that can't be deleted
                pass

    # Try to clear the session
    if hasattr(ticker_obj, "session") and ticker_obj.session is not None:
        try:
            ticker_obj.session.close()
            ticker_obj.session = None
        except Exception as e:
            logger.debug(f"Error closing ticker session: {e}")


def safe_create_ticker(ticker_symbol: str) -> Any:
    """
    Safely create a yfinance.Ticker object with memory leak prevention.

    This function creates a yfinance.Ticker object with additional
    safeguards to prevent memory leaks.

    Args:
        ticker_symbol: The ticker symbol to create a Ticker object for

    Returns:
        A yfinance.Ticker object
    """
    try:
        import yfinance as yf

        # Create the ticker object
        ticker_obj = yf.Ticker(ticker_symbol)

        # Immediately disable history caching
        if hasattr(ticker_obj, "_history_metadata"):
            ticker_obj._history_metadata = None

        # Force garbage collection right after creating the object
        # to clean up any immediate leaks
        gc.collect()

        return ticker_obj
    except ImportError:
        logger.error("yfinance package not installed")
        return None
    except Exception as e:
        logger.error(f"Error creating ticker object for {ticker_symbol}: {e}")
        return None


def with_safe_ticker(ticker_arg_name: str = "ticker") -> Callable:
    """
    Decorator to safely handle yfinance.Ticker objects in functions.

    This decorator ensures that yfinance.Ticker objects are properly
    cleaned up after use to prevent memory leaks.

    Args:
        ticker_arg_name: The name of the argument that contains the ticker symbol

    Returns:
        A decorator function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Extract the ticker symbol from args or kwargs
            ticker_symbol = None
            if ticker_arg_name in kwargs:
                ticker_symbol = kwargs[ticker_arg_name]
            elif len(args) > 0:
                ticker_symbol = args[0]

            if not ticker_symbol:
                return func(*args, **kwargs)

            import yfinance as yf

            ticker_obj = None

            try:
                # Create the ticker object
                ticker_obj = safe_create_ticker(ticker_symbol)

                # Call the function with the ticker object
                result = func(ticker_obj, *args[1:], **kwargs)

                return result
            finally:
                # Clean up the ticker object
                if ticker_obj is not None:
                    clean_ticker_object(ticker_obj)
                    ticker_obj = None
                    gc.collect()

        return wrapper

    return decorator


def extract_info_safely(ticker_obj: Any) -> Dict[str, Any]:
    """
    Safely extract info from a ticker object to prevent memory leaks.

    This function extracts the info dictionary from a yfinance.Ticker object
    in a way that prevents memory leaks by making deep copies of the data
    and immediately cleaning up references.

    Args:
        ticker_obj: A yfinance.Ticker object

    Returns:
        A dictionary of ticker information
    """
    if ticker_obj is None:
        return {}

    try:
        import copy

        # Get the info dictionary
        info = getattr(ticker_obj, "info", {})

        # Make a deep copy to break references
        if info is not None:
            result = copy.deepcopy(info)
        else:
            result = {}

        # Force garbage collection
        info = None
        gc.collect()

        return result
    except Exception as e:
        logger.error(f"Error extracting info from ticker object: {e}")
        return {}
