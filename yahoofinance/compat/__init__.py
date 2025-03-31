"""
Compatibility layer for v1 API.

This module provides compatibility with the v1 API for easier transition
to the v2 framework. It maps v1 function calls and objects to their v2
equivalents, allowing existing code to work without modifications.

DEPRECATION WARNING: This package is deprecated and will be removed in a future version.
Use the canonical import paths instead, as documented in MIGRATION_STATUS.md.
"""

import warnings

# Show deprecation warning
warnings.warn(
    "The yahoofinance.compat package is deprecated and will be removed in a future version. "
    "Please migrate to the canonical import paths as documented in MIGRATION_STATUS.md.",
    DeprecationWarning,
    stacklevel=2
)

from .client import YFinanceClient
from .display import MarketDisplay
from .formatting import DisplayFormatter, DisplayConfig
from .analyst import AnalystData
from .pricing import PricingAnalyzer

__all__ = [
    # Client compatibility
    'YFinanceClient',
    
    # Display compatibility
    'MarketDisplay',
    
    # Formatting compatibility
    'DisplayFormatter',
    'DisplayConfig',
    
    # Analysis compatibility
    'AnalystData',
    'PricingAnalyzer',
]