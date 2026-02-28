"""
Utility modules for Yahoo Finance data processing.

This package provides various utilities organized by category:
- data: Data formatting utilities
- date: Date handling utilities
- market: Market-specific utilities
- network: Network and rate limiting utilities
- error_handling: Standardized error handling utilities
"""

from .data import (
    format_for_csv,
    format_market_cap,
    format_market_metrics,
    format_number,
    format_table,
    generate_market_html,
)
from .date import (
    format_date_for_api,
    format_date_for_display,
    get_date_range,
    validate_date_format,
)
from .error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_error_context,
    with_retry,
)
from .market import (
    filter_valid_tickers,
    is_us_ticker,
    normalize_hk_ticker,
)
from .network import (
    global_rate_limiter,
    rate_limited,
)


__all__ = [
    # Rate limiting and network utilities
    "global_rate_limiter",
    "rate_limited",
    # Data formatting utilities
    "format_number",
    "format_table",
    "format_market_cap",
    "format_market_metrics",
    "generate_market_html",
    "format_for_csv",
    # Market utilities
    "is_us_ticker",
    "normalize_hk_ticker",
    "filter_valid_tickers",
    # Date utilities
    "validate_date_format",
    "get_date_range",
    "format_date_for_api",
    "format_date_for_display",
    # Error handling utilities
    "enrich_error_context",
    "translate_error",
    "with_error_context",
    "with_retry",
    "safe_operation",
]
