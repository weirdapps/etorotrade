"""Market-specific utility functions."""

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
    Normalize Hong Kong stock tickers by removing leading zeros for 5+ digit tickers.
    
    eToro represents HK tickers differently than Yahoo Finance:
    - eToro: 03690.HK (leading zero)
    - Yahoo Finance: 3690.HK (no leading zero)
    
    This function converts the eToro format to Yahoo Finance format.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        str: Normalized ticker symbol
    """
    if not ticker.endswith('.HK'):
        return ticker
        
    # Extract the numeric part
    numeric_part = ticker.split('.')[0]
    
    # Remove leading zeros if numeric part has 5 or more digits
    if len(numeric_part) >= 5 and numeric_part.startswith('0'):
        normalized_numeric = numeric_part.lstrip('0')
        return f"{normalized_numeric}.HK"
    
    return ticker


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