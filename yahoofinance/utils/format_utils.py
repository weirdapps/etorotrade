"""Formatting utilities for HTML output and other display formats.

This module is a compatibility layer that re-exports formatting utilities
from the structured 'data' package to maintain backward compatibility.

WARNING: This module is deprecated. Import from yahoofinance.utils.data.format_utils instead.
"""

import warnings

warnings.warn(
    "The yahoofinance.utils.format_utils module is deprecated. "
    "Use yahoofinance.utils.data.format_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

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