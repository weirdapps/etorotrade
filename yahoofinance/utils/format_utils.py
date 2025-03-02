"""
Formatting utilities for HTML output and other display formats.

This module is a compatibility layer that re-exports formatting utilities
from the structured 'data' package to maintain backward compatibility.
"""

# Import everything from the data.format_utils module
from .data.format_utils import (
    FormatUtils,
    format_number,
    format_table,
    format_market_metrics,
    generate_market_html,
    format_for_csv
)

# Explicitly add the module-level functions for backward compatibility
# These are automatically imported from the data module above
"""
# For compatibility, these functions are already imported:
format_number = FormatUtils.format_number
format_table = FormatUtils.format_table
format_market_metrics = FormatUtils.format_market_metrics
generate_market_html = FormatUtils.generate_market_html
format_for_csv = FormatUtils.format_for_csv
"""