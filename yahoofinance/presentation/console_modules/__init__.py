"""
Console presentation modules - Split from console.py.

This package contains the refactored console presentation utilities,
split into focused modules for better maintainability.
"""

from .console_display import ConsoleDisplay
from .data_manager import (
    display_report,
    filter_by_trade_action,
    load_tickers,
    save_to_csv,
)
from .progress import display_console_error_summary, process_tickers_with_progress
from .rate_limiter import RateLimitTracker
from .table_renderer import (
    add_position_size_column,
    calculate_actions,
    display_stock_table,
    format_dataframe,
    parse_market_cap_value,
    parse_percentage_value,
    sort_market_data,
)

__all__ = [
    # Rate limiting
    "RateLimitTracker",
    # Progress tracking
    "process_tickers_with_progress",
    "display_console_error_summary",
    # Table rendering
    "format_dataframe",
    "calculate_actions",
    "sort_market_data",
    "display_stock_table",
    "add_position_size_column",
    "parse_market_cap_value",
    "parse_percentage_value",
    # Data management
    "load_tickers",
    "filter_by_trade_action",
    "save_to_csv",
    "display_report",
    # Console display
    "ConsoleDisplay",
]
