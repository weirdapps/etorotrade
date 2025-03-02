"""
Market-specific utility functions.

This module is a compatibility layer that re-exports market utilities
from the structured 'market' package to maintain backward compatibility.
"""

# Import from the market.ticker_utils module
from .market.ticker_utils import (
    US_SPECIAL_CASES,
    is_us_ticker,
    normalize_hk_ticker,
    filter_valid_tickers
)

# For documentation purposes
"""
This module provides backward compatibility for:
- US_SPECIAL_CASES constant
- is_us_ticker function
- normalize_hk_ticker function 
- filter_valid_tickers function

These are now maintained in market.ticker_utils module.
"""