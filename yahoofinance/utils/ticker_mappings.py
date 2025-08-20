"""
Ticker Mappings Compatibility Module

This module provides backward compatibility for the old ticker_mappings imports.
All functionality is now provided by the unified ConfigManager.
"""

from trade_modules.config_manager import get_config

# Get config instance
_config = get_config()

# Export the mappings as module-level constants for backward compatibility
DUAL_LISTED_MAPPINGS = _config.dual_listed_mappings
REVERSE_MAPPINGS = _config.reverse_mappings
DUAL_LISTED_TICKERS = _config.dual_listed_tickers
TICKER_GEOGRAPHY = _config.ticker_geography

# Export functions that delegate to ConfigManager
def get_normalized_ticker(ticker: str) -> str:
    """Get the normalized (original exchange) ticker for a given ticker."""
    return _config.get_normalized_ticker(ticker)

def get_us_ticker(ticker: str) -> str:
    """Get the US ticker equivalent for a given ticker."""
    return _config.get_us_ticker(ticker)

def is_dual_listed(ticker: str) -> bool:
    """Check if a ticker has dual listings."""
    return _config.is_dual_listed(ticker)

def get_display_ticker(ticker: str) -> str:
    """Get the preferred display ticker (always the original exchange ticker)."""
    return _config.get_display_ticker(ticker)

def get_data_fetch_ticker(ticker: str) -> str:
    """Get the best ticker for data fetching."""
    return _config.get_data_fetch_ticker(ticker)

def get_ticker_geography(ticker: str) -> str:
    """Get the geographic region for a ticker."""
    return _config.get_ticker_geography(ticker)

def are_equivalent_tickers(ticker1: str, ticker2: str) -> bool:
    """Check if two tickers represent the same underlying asset."""
    return _config.are_equivalent_tickers(ticker1, ticker2)

def get_all_equivalent_tickers(ticker: str):
    """Get all known ticker variants for the same underlying asset."""
    return _config.get_all_equivalent_tickers(ticker)