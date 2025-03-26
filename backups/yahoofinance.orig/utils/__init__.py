"""
Central utilities package for Yahoo Finance data processing.

This package provides various utilities organized by category:
- network: Rate limiting, pagination, and API request handling
- data: Data formatting and transformation
- market: Market-specific utilities for ticker handling
- date: Date/time utilities for financial data
"""

# Import submodules to make them accessible
from . import network
from . import data
from . import market
from . import date

# Re-export commonly used components for convenience
from .network import (
    global_rate_limiter,
    rate_limited,
    batch_process,
    paginated_request,
    bulk_fetch,
)

from .data import (
    FormatUtils,
    format_number,
    format_table,
    format_market_metrics,
    generate_market_html,
    format_for_csv,
)

from .market import (
    is_us_ticker,
    normalize_hk_ticker,
    filter_valid_tickers,
    US_SPECIAL_CASES,
)

from .date import (
    DateUtils,
    validate_date_format,
    get_user_dates,
    get_date_range,
    format_date_for_api,
    format_date_for_display,
)

__all__ = [
    # Modules
    'network',
    'data',
    'market',
    'date',
    
    # Rate limiting and network utilities
    'global_rate_limiter',
    'rate_limited',
    'batch_process',
    'paginated_request',
    'bulk_fetch',
    
    # Data formatting utilities
    'FormatUtils',
    'format_number',
    'format_table',
    'format_market_metrics',
    'generate_market_html',
    'format_for_csv',
    
    # Market utilities
    'is_us_ticker',
    'normalize_hk_ticker',
    'filter_valid_tickers',
    'US_SPECIAL_CASES',
    
    # Date utilities
    'DateUtils',
    'validate_date_format',
    'get_user_dates',
    'get_date_range',
    'format_date_for_api',
    'format_date_for_display',
]