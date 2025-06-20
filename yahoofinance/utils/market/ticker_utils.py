"""
Ticker-related utilities for Yahoo Finance data.

This module provides functions for validating and transforming ticker symbols
to ensure they are properly formatted for API calls.
"""

import re
from typing import Any, Dict, List, Optional, Set

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ...utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ...core.config import SPECIAL_TICKERS
from ...core.errors import ValidationError


def validate_ticker(ticker: str) -> bool:
    """
    Validate a ticker symbol format.

    Args:
        ticker: Ticker symbol to validate

    Returns:
        True if valid, False otherwise

    Raises:
        ValidationError: If ticker format is invalid
    """
    if not ticker or not isinstance(ticker, str):
        raise ValidationError(f"Invalid ticker: {ticker} - must be a non-empty string")

    # Check basic ticker format
    if len(ticker) > 20:
        raise ValidationError(f"Invalid ticker: {ticker} - length exceeds 20 characters")

    # Check for invalid characters (allow alphanumeric, dots, hyphens, and equals for futures)
    if re.search(r"[^\w\.\-=]", ticker):
        raise ValidationError(f"Invalid ticker: {ticker} - contains invalid characters")

    return True


def is_us_ticker(ticker: str) -> bool:
    """
    Check if a ticker is a US stock.

    US stocks have either:
    1. No suffix (like "AAPL")
    2. A .US suffix (like "INTC.US")
    3. Are in the special cases list (like "BRK.A", "BRK.B")

    Args:
        ticker: Ticker symbol to check

    Returns:
        True if US ticker, False otherwise
    """
    # Check special cases first (US stocks with dots like BRK.A)
    if ticker in SPECIAL_TICKERS["US_SPECIAL_CASES"]:
        return True

    # Check for .US suffix
    if ticker.endswith(".US"):
        return True

    # Check for non-US exchange suffixes
    non_us_suffixes = {
        ".L",
        ".TO",
        ".V",
        ".PA",
        ".DE",
        ".HK",
        ".LN",
        ".SZ",
        ".SS",
        ".TW",
        ".AX",
        ".SA",
        ".F",
        ".MI",
        ".BR",
        ".SW",
        ".MC",
        ".AS",
        ".CO",
        ".OL",
        ".ST",
        ".LS",
        ".MX",
        ".KS",
        ".KQ",
        ".VX",
        ".IR",
        ".JK",
        ".SI",
        ".IL",
        ".NZ",
        ".TA",
    }

    for suffix in non_us_suffixes:
        if ticker.endswith(suffix):
            return False

    # If no exchange suffix, assume US
    if "." not in ticker:
        return True

    # Check for crypto (not US stock)
    if ticker.endswith("-USD") or ticker.endswith("-EUR"):
        return False

    # Default to assuming US if no other patterns matched
    return True


def normalize_hk_ticker(ticker: str) -> str:
    """
    Normalize Hong Kong ticker format.

    eToro uses different formats for HK stocks, this function
    standardizes them for API calls.

    Args:
        ticker: Ticker symbol to normalize

    Returns:
        Normalized ticker
    """
    # Check if it's a HK ticker
    if not ticker.endswith(".HK"):
        return ticker

    # Extract the numerical part
    ticker_num = ticker.split(".")[0]

    # If the ticker starts with a zero, strip leading zeros
    if ticker_num.startswith("0"):
        normalized_num = ticker_num.lstrip("0")
        # If all digits were zeros, keep one
        if not normalized_num:
            normalized_num = "0"
        return f"{normalized_num}.HK"

    # For non-zero starting tickers, keep as is
    return ticker


def is_stock_ticker(ticker: str) -> bool:
    """
    Check if a ticker represents a stock (as opposed to ETF, commodity, or cryptocurrency).
    
    Stocks have insider trading data available, while ETFs, commodities, and 
    cryptocurrencies typically do not.

    Args:
        ticker: Ticker symbol to check

    Returns:
        True if likely a stock ticker, False for ETFs/commodities/crypto
    """
    ticker_upper = ticker.upper()
    
    # Cryptocurrency patterns
    if ticker_upper.endswith("-USD") or ticker_upper.endswith("-EUR"):
        return False
    
    # Common ETF patterns (many ETFs are 3-4 letters)
    # This is a heuristic - not all 3-letter tickers are ETFs
    common_etf_tickers = {
        "TLT", "FXI", "INDA", "SPY", "QQQ", "IWM", "EFA", "EEM", "VTI", "VEA",
        "VWO", "AGG", "LQD", "HYG", "GLD", "SLV", "OIL", "USO", "UNG", "GDXJ",
        "XLF", "XLE", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLB", "XLRE",
        "SQQQ", "TQQQ", "UVXY", "VXX", "VIXY"
    }
    
    if ticker_upper in common_etf_tickers:
        return False
    
    # Common commodity/futures patterns
    commodity_patterns = {
        "GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZS=F", "ZW=F"  # Futures contracts
    }
    
    if ticker_upper in commodity_patterns:
        return False
    
    # Default to assuming it's a stock if we can't definitively classify it
    return True


def filter_valid_tickers(
    tickers: List[str], excluded_tickers: Optional[Set[str]] = None
) -> List[str]:
    """
    Filter out invalid tickers and excluded tickers.

    Args:
        tickers: List of ticker symbols to filter
        excluded_tickers: Set of tickers to exclude

    Returns:
        List of valid tickers
    """
    if excluded_tickers is None:
        excluded_tickers = set()

    valid_tickers = []

    for ticker in tickers:
        # Skip empty or None values
        if not ticker:
            continue

        # Skip excluded tickers
        if ticker in excluded_tickers:
            continue

        try:
            # Validate ticker format
            valid = validate_ticker(ticker)
            if valid:
                # Normalize HK tickers
                normalized = normalize_hk_ticker(ticker)
                valid_tickers.append(normalized)
        except ValidationError:
            # Skip invalid tickers
            continue

    return valid_tickers
