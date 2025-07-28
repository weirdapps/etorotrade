"""
Ticker mapping configuration for dual-listed stocks.

This module provides centralized mapping for stocks that trade on multiple exchanges,
ensuring consistent handling throughout the system. When a US ticker (ADR or cross-listing)
is encountered, it will be normalized to the original exchange ticker for display,
position sizing, and all other operations.
"""

from typing import Dict, Set

# US Ticker -> Original Exchange Ticker mappings
DUAL_LISTED_MAPPINGS: Dict[str, str] = {
    # European stocks with US ADRs/cross-listings
    "NVO": "NOVO-B.CO",      # Novo Nordisk ADR → Copenhagen
    "SNY": "SAN.PA",         # Sanofi ADR → Paris  
    "ASML": "ASML.NV",       # ASML NASDAQ → Netherlands
    "SHEL": "SHEL.L",        # Shell ADR → London
    "UL": "ULVR.L",          # Unilever ADR → London
    "RDS.A": "SHEL.L",       # Shell (old ticker) → London
    "RDS.B": "SHEL.L",       # Shell (old ticker) → London
    "SAP": "SAP.DE",         # SAP ADR → Germany
    "TM": "7203.T",          # Toyota ADR → Tokyo
    "SONY": "6758.T",        # Sony ADR → Tokyo
    "NTT": "9432.T",         # NTT ADR → Tokyo
    
    # Hong Kong stocks with US ADRs
    "JD": "9618.HK",         # JD.com ADR → Hong Kong
    "JD.US": "9618.HK",      # JD.com US listing → Hong Kong
    "BABA": "9988.HK",       # Alibaba ADR → Hong Kong
    "TCEHY": "0700.HK",      # Tencent ADR → Hong Kong  
    "BYDDY": "1211.HK",      # BYD ADR → Hong Kong
    "MEITX": "3690.HK",      # Meituan ADR → Hong Kong
    "YUMC": "YUMC",          # Yum China (already US-based, no mapping needed)
    
    # Google share classes (GOOG is the main ticker)
    "GOOGL": "GOOG",         # Google Class A → Google Class C (main ticker)
    
    # Other cross-listings
    "TSM": "TSM",            # Taiwan Semiconductor (already properly listed)
    "RIO": "RIO.L",          # Rio Tinto ADR → London primary
    "BHP": "BHP.AX",         # BHP ADR → Australia primary
}

# Reverse mapping: Original Exchange Ticker -> US Ticker
REVERSE_MAPPINGS: Dict[str, str] = {v: k for k, v in DUAL_LISTED_MAPPINGS.items()}

# Set of all tickers that have dual listings (for quick lookup)
DUAL_LISTED_TICKERS: Set[str] = set(DUAL_LISTED_MAPPINGS.keys()) | set(DUAL_LISTED_MAPPINGS.values())

def get_normalized_ticker(ticker: str) -> str:
    """
    Get the normalized (original exchange) ticker for a given ticker.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Normalized ticker (original exchange ticker if dual-listed, otherwise unchanged)
    """
    if not ticker:
        return ticker
        
    # Convert to uppercase for consistent matching
    ticker_upper = ticker.upper()
    
    # If this is a US ticker with a mapped original, return the original
    if ticker_upper in DUAL_LISTED_MAPPINGS:
        return DUAL_LISTED_MAPPINGS[ticker_upper]
    
    # If the ticker is already in its normalized form (uppercase), return uppercase version
    # Otherwise return the ticker in standardized format (uppercase)
    return ticker_upper

def get_us_ticker(ticker: str) -> str:
    """
    Get the US ticker equivalent for a given ticker.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        US ticker if available, otherwise the original ticker
    """
    if not ticker:
        return ticker
        
    # Convert to uppercase for consistent matching
    ticker_upper = ticker.upper()
    
    # If this is an original ticker with a US equivalent, return the US ticker
    if ticker_upper in REVERSE_MAPPINGS:
        return REVERSE_MAPPINGS[ticker_upper]
    
    # Return the ticker in standardized format (uppercase)
    return ticker_upper

def is_dual_listed(ticker: str) -> bool:
    """
    Check if a ticker has dual listings.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        True if ticker has dual listings, False otherwise
    """
    if not ticker:
        return False
        
    return ticker.upper() in DUAL_LISTED_TICKERS

def get_display_ticker(ticker: str) -> str:
    """
    Get the preferred display ticker (always the original exchange ticker).
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Preferred display ticker (original exchange ticker)
    """
    return get_normalized_ticker(ticker)

def get_data_fetch_ticker(ticker: str) -> str:
    """
    Get the best ticker for data fetching (may use US ticker for better data availability).
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Best ticker for data fetching
    """
    # For data fetching, we might want to use US tickers when available
    # as they often have better data coverage, but we'll normalize the results
    normalized = get_normalized_ticker(ticker)
    us_ticker = get_us_ticker(normalized)
    
    # For now, prefer the normalized ticker, but this can be customized
    # based on data quality preferences per ticker
    return normalized

# Geographic region mapping for dual-listed stocks
# This ensures correct geographic risk multipliers are applied
TICKER_GEOGRAPHY: Dict[str, str] = {
    # European tickers
    "NOVO-B.CO": "EU",
    "SAN.PA": "EU", 
    "ASML.NV": "EU",
    "SHEL.L": "UK",
    "ULVR.L": "UK",
    "SAP.DE": "EU",
    
    # Asian tickers
    "9618.HK": "HK",
    "9988.HK": "HK",
    "0700.HK": "HK",
    "1211.HK": "HK",
    "3690.HK": "HK",
    "7203.T": "JP",
    "6758.T": "JP",
    "9432.T": "JP",
    
    # Other regions
    "RIO.L": "UK",
    "BHP.AX": "AU",
}

def are_equivalent_tickers(ticker1: str, ticker2: str) -> bool:
    """
    Check if two tickers represent the same underlying asset.
    
    This is critical for portfolio filtering - if someone owns NVO, 
    they shouldn't see NOVO-B.CO as a buy opportunity (and vice versa).
    
    Args:
        ticker1: First ticker symbol
        ticker2: Second ticker symbol
        
    Returns:
        True if tickers represent the same underlying asset, False otherwise
    """
    if not ticker1 or not ticker2:
        return False
    
    # Normalize both tickers to their canonical forms
    normalized1 = get_normalized_ticker(ticker1.upper())
    normalized2 = get_normalized_ticker(ticker2.upper())
    
    # If normalized forms are the same, they're equivalent
    if normalized1 == normalized2:
        return True
    
    # Check if one is the US version of the other
    us_ticker1 = get_us_ticker(normalized1)
    us_ticker2 = get_us_ticker(normalized2)
    
    # If either ticker maps to the other's US equivalent, they're the same
    return (normalized1 == us_ticker2 or normalized2 == us_ticker1 or 
            us_ticker1 == normalized2 or us_ticker2 == normalized1)

def get_all_equivalent_tickers(ticker: str) -> Set[str]:
    """
    Get all known ticker variants for the same underlying asset.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Set of all equivalent ticker symbols (including the input)
    """
    if not ticker:
        return set()
    
    ticker_upper = ticker.upper()
    normalized = get_normalized_ticker(ticker_upper)
    us_ticker = get_us_ticker(normalized)
    
    # Start with the normalized form
    equivalents = {normalized}
    
    # Add the US ticker if different
    if us_ticker != normalized:
        equivalents.add(us_ticker)
    
    # Add the original input if different from normalized
    if ticker_upper != normalized:
        equivalents.add(ticker_upper)
    
    return equivalents

def get_ticker_geography(ticker: str) -> str:
    """
    Get the geographic region for a ticker.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Geographic region code (HK, EU, UK, US, JP, AU, etc.)
    """
    if not ticker:
        return "US"  # Default to US
        
    normalized_ticker = get_normalized_ticker(ticker)
    
    # Check explicit mapping first
    if normalized_ticker in TICKER_GEOGRAPHY:
        return TICKER_GEOGRAPHY[normalized_ticker]
    
    # Infer from ticker suffix
    if normalized_ticker.endswith('.HK'):
        return "HK"
    elif normalized_ticker.endswith('.L'):
        return "UK"  
    elif normalized_ticker.endswith(('.PA', '.DE', '.NV', '.MI', '.BR')):
        return "EU"
    elif normalized_ticker.endswith('.T'):
        return "JP"
    elif normalized_ticker.endswith('.AX'):
        return "AU"
    else:
        return "US"  # Default to US for unrecognized patterns