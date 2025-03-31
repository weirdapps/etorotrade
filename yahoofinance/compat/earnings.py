"""
Compatibility module for earnings calendar classes from v1.

This module provides the EarningsCalendar class that mirrors the interface of
the v1 earnings calendar class but uses the v2 implementation under the hood.

DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
Use the canonical import path instead:
from yahoofinance.analysis.earnings import EarningsCalendar, format_earnings_table
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union

# Show deprecation warning
warnings.warn(
    "The yahoofinance.compat.earnings module is deprecated and will be removed in a future version. "
    "Use 'from yahoofinance.analysis.earnings import EarningsCalendar, format_earnings_table' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location
from ..analysis.earnings import EarningsCalendar, format_earnings_table

logger = logging.getLogger(__name__)

# Export the necessary functions/classes for backward compatibility
__all__ = ['EarningsCalendar', 'format_earnings_table']