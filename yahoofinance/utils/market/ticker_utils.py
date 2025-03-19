"""
Market-specific ticker utility functions.

This module provides utilities for working with ticker symbols,
including normalization and validation functions.

CANONICAL SOURCE:
This is the canonical source for ticker and market utilities. Other modules
that provide similar functionality are compatibility layers that import from 
this module. Always prefer to import directly from this module in new code:

    from yahoofinance.utils.market.ticker_utils import (
        is_us_ticker, normalize_hk_ticker, filter_valid_tickers
    )

Key Components:
- is_us_ticker: Check if a ticker is a US stock
- normalize_hk_ticker: Normalize Hong Kong stock tickers
- filter_valid_tickers: Filter valid ticker symbols from a list

Example usage:
    # Normalize a Hong Kong ticker symbol
    normalized = normalize_hk_ticker('03690.HK')  # Returns '3690.HK'
    
    # Check if a ticker is a US stock
    is_us = is_us_ticker('AAPL')  # Returns True
"""

from typing import List, Set


# US tickers that have dots but are still US stocks
US_SPECIAL_CASES: Set[str] = {"BRK.A", "BRK.B", "BF.A", "BF.B"}


def is_us_ticker(ticker: str) -> bool:
    """
    Determine if a ticker is from a US exchange.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        bool: True if US ticker, False otherwise
    """
    # Handle special cases like "BRK.B" which are US tickers
    if ticker in US_SPECIAL_CASES:
        return True
        
    # US tickers generally have no suffix or .US suffix
    return '.' not in ticker or ticker.endswith('.US')


def normalize_hk_ticker(ticker: str) -> str:
    """
    Normalize Hong Kong stock tickers to standard format.
    
    Hong Kong stock tickers follow these rules:
    1. Remove all leading zeros
    2. If the numeric part is less than 4 digits, pad with leading zeros to make it 4 digits
    3. If the numeric part is 4 or more digits, leave it as is
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        str: Normalized ticker symbol
    """
    if not ticker or not isinstance(ticker, str) or not ticker.endswith('.HK'):
        return ticker
    
    # Handle case with class designations (e.g., '0700-A.HK')
    if '-' in ticker:
        base_part, class_part = ticker.split('-', 1)
        suffix = f"-{class_part}"
    else:
        base_part = ticker.split('.')[0]
        suffix = ".HK"
    
    # Remove all leading zeros
    normalized_base = base_part.lstrip('0')
    
    # Handle the case where it was all zeros
    if not normalized_base:
        normalized_base = '0'
    
    # If less than 4 digits, pad with leading zeros
    if len(normalized_base) < 4:
        normalized_base = normalized_base.zfill(4)
    
    return f"{normalized_base}{suffix}"


def filter_valid_tickers(tickers: List[str]) -> List[str]:
    """
    Filter out invalid ticker formats.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        List[str]: Filtered list of valid ticker symbols
    """
    valid_tickers = []
    for ticker in tickers:
        if not isinstance(ticker, str):
            continue
            
        ticker = ticker.strip().upper()
        if not ticker:
            continue
            
        # Skip numeric-only tickers
        if ticker.isdigit():
            continue
            
        # Check length constraints based on exchange suffix
        has_exchange_suffix = '.' in ticker
        max_length = 20 if has_exchange_suffix else 10
        
        if len(ticker) <= max_length:
            valid_tickers.append(ticker)
            
    return valid_tickers