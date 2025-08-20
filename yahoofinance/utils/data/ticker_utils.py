"""
Ticker utilities for handling dual-listed stocks and ticker normalization.

This module provides utility functions for working with ticker symbols,
especially for handling dual-listed stocks that trade on multiple exchanges.
"""

from typing import Dict, List, Optional, Set, Tuple
from ..ticker_mappings import (
    DUAL_LISTED_MAPPINGS,
    REVERSE_MAPPINGS,
    DUAL_LISTED_TICKERS,
    TICKER_GEOGRAPHY,
    get_normalized_ticker,
    get_us_ticker,
    is_dual_listed,
    get_display_ticker,
    get_data_fetch_ticker,
    get_ticker_geography,
    are_equivalent_tickers,
    get_all_equivalent_tickers
)

def normalize_ticker(ticker: str) -> str:
    """
    Normalize a ticker to its canonical form (original exchange ticker).
    
    This is the main function to use throughout the application to ensure
    consistent ticker handling.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Normalized ticker symbol (standardized and mapped to canonical form)
    """
    if not ticker:
        return ticker
    
    # First standardize format (uppercase, clean whitespace, handle special cases)
    standardized = standardize_ticker_format(ticker)
    
    # Then get the canonical ticker (handle dual-listings)
    normalized = get_normalized_ticker(standardized)
    
    return normalized

def normalize_ticker_list(tickers: List[str]) -> List[str]:
    """
    Normalize a list of tickers to their canonical forms.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        List of normalized ticker symbols
    """
    return [normalize_ticker(ticker) for ticker in tickers if ticker]

def get_ticker_for_display(ticker: str) -> str:
    """
    Get the ticker symbol that should be used for display purposes.
    
    This always returns the original exchange ticker for consistency.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Display ticker symbol
    """
    return get_display_ticker(ticker)

def get_ticker_for_data_fetch(ticker: str) -> str:
    """
    Get the ticker symbol that should be used for data fetching.
    
    This may return a different ticker than the display ticker if
    a different exchange provides better data quality.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Data fetch ticker symbol
    """
    return get_data_fetch_ticker(ticker)

def get_geographic_region(ticker: str) -> str:
    """
    Get the geographic region for a ticker symbol.
    
    This is used for applying geographic risk multipliers in position sizing.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Geographic region code (HK, EU, UK, US, JP, AU, etc.)
    """
    return get_ticker_geography(ticker)

def is_ticker_dual_listed(ticker: str) -> bool:
    """
    Check if a ticker has dual listings on multiple exchanges.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        True if ticker has dual listings, False otherwise
    """
    return is_dual_listed(ticker)

def get_all_ticker_variants(ticker: str) -> List[str]:
    """
    Get all known variants of a ticker (US and original exchange).
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        List of all known ticker variants
    """
    normalized = normalize_ticker(ticker)
    us_ticker = get_us_ticker(normalized)
    
    variants = [normalized]
    if us_ticker != normalized:
        variants.append(us_ticker)
    
    return list(set(variants))  # Remove duplicates

def validate_ticker_format(ticker: str) -> bool:
    """
    Validate that a ticker has a reasonable format.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        True if ticker format is valid, False otherwise
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    ticker = ticker.strip()
    if not ticker:
        return False
    
    # Basic format validation
    # Allow letters, numbers, dots, hyphens, and common suffixes
    # But prevent trailing dots, hyphens, or underscores
    import re
    pattern = r'^[A-Za-z0-9][A-Za-z0-9\.\-_]*[A-Za-z0-9]$'
    return bool(re.match(pattern, ticker))

def get_ticker_exchange_suffix(ticker: str) -> Optional[str]:
    """
    Extract the exchange suffix from a ticker symbol.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Exchange suffix (e.g., '.HK', '.L', '.PA') or None if no suffix
    """
    if not ticker:
        return None
    
    # Look for common exchange suffixes
    suffixes = ['.HK', '.L', '.PA', '.DE', '.NV', '.MI', '.BR', '.T', '.AX', '.CO', '-USD']
    
    for suffix in suffixes:
        if ticker.upper().endswith(suffix.upper()):
            return suffix.upper()
    
    return None

def standardize_ticker_format(ticker: str) -> str:
    """
    Standardize ticker format (uppercase, clean whitespace).
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Standardized ticker symbol
    """
    if not ticker:
        return ticker
    
    # Clean and standardize
    cleaned = ticker.strip().upper()
    
    # Handle common format inconsistencies
    # Hong Kong tickers: ensure 4-digit format (e.g., '700.HK' -> '0700.HK', '1.HK' -> '0001.HK')
    if cleaned.endswith('.HK'):
        base_ticker = cleaned.split('.')[0]
        if base_ticker.isdigit():
            # Remove leading zeros, then pad to 4 digits
            normalized_base = base_ticker.lstrip('0') or '0'  # Handle all zeros case
            cleaned = normalized_base.zfill(4) + '.HK'
    
    # VIX tickers: normalize all VIX variants to ^VIX for proper data fetching
    # This handles VIX, VIX.CBE, VIX.SEP25, etc. -> ^VIX
    elif cleaned.startswith('VIX') and (cleaned == 'VIX' or cleaned.startswith('VIX.')):
        cleaned = '^VIX'
    
    # Crypto tickers: ensure -USD suffix for major cryptos
    elif cleaned in ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK', 'XLM', 'DOGE', 'SOL', 'HBAR']:
        if not cleaned.endswith('-USD'):
            cleaned = cleaned + '-USD'
    
    # Copenhagen Stock Exchange (.CO) tickers: normalize format for dual-listed stocks
    elif cleaned.endswith('.CO'):
        base_ticker = cleaned.split('.')[0]
        # Handle specific patterns for Copenhagen stocks with dashes
        # Many Copenhagen stocks use the pattern XXXB.CO but should be XXX-B.CO
        if base_ticker.endswith('B') and len(base_ticker) > 1:
            # Check for known patterns that need dash insertion
            if base_ticker in ['MAERSKB', 'NOVOB', 'COLOB']:
                if base_ticker == 'MAERSKB':
                    cleaned = 'MAERSK-B.CO'
                elif base_ticker == 'NOVOB':
                    cleaned = 'NOVO-B.CO'
                elif base_ticker == 'COLOB':
                    cleaned = 'COLO-B.CO'
    
    return cleaned

def process_ticker_input(ticker: str) -> str:
    """
    Process ticker input through the complete normalization pipeline.
    
    This is the main function to call when processing ticker input from
    any source (user input, file parsing, etc.).
    
    Args:
        ticker: Raw ticker input
        
    Returns:
        Processed and normalized ticker symbol
    """
    if not ticker:
        return ticker
    
    # Step 1: Standardize format
    standardized = standardize_ticker_format(ticker)
    
    # Step 2: Normalize to canonical form
    normalized = normalize_ticker(standardized)
    
    return normalized

def get_ticker_info_summary(ticker: str) -> Dict[str, str]:
    """
    Get a summary of information about a ticker symbol.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Dictionary with ticker information
    """
    normalized = normalize_ticker(ticker)
    us_ticker = get_us_ticker(normalized)
    geography = get_geographic_region(normalized)
    dual_listed = is_ticker_dual_listed(ticker)
    
    return {
        "input_ticker": ticker,
        "normalized_ticker": normalized,
        "display_ticker": normalized,
        "us_ticker": us_ticker if us_ticker != normalized else None,
        "geography": geography,
        "is_dual_listed": str(dual_listed),
        "exchange_suffix": get_ticker_exchange_suffix(normalized)
    }

# Convenience functions for backward compatibility
def fix_ticker_format(ticker: str) -> str:
    """Legacy function name for ticker format standardization."""
    return standardize_ticker_format(ticker)

def get_canonical_ticker(ticker: str) -> str:
    """Legacy function name for ticker normalization."""
    return normalize_ticker(ticker)

def check_equivalent_tickers(ticker1: str, ticker2: str) -> bool:
    """
    Check if two tickers represent the same underlying asset.
    
    This is essential for portfolio filtering to prevent suggesting
    buying the same stock that's already owned under a different ticker.
    
    Args:
        ticker1: First ticker symbol
        ticker2: Second ticker symbol
        
    Returns:
        True if tickers are equivalent, False otherwise
    """
    return are_equivalent_tickers(ticker1, ticker2)

def get_ticker_equivalents(ticker: str) -> Set[str]:
    """
    Get all known ticker variants for the same underlying asset.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Set of all equivalent ticker symbols
    """
    return get_all_equivalent_tickers(ticker)