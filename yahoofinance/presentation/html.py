"""
HTML output generation - Backward Compatibility Layer.

This module provides backward compatibility for existing code that imports
from yahoofinance.presentation.html. All core functionality has been
moved to yahoofinance.presentation.html_modules.

New code should import from yahoofinance.presentation.html_modules directly.
"""

# Import everything from the new split modules for backward compatibility
from .html_modules import (
    FormatUtils,
    HTMLGenerator,
    DEFAULT_CSS,
    DEFAULT_JS,
)

# Re-export all for backward compatibility
__all__ = [
    "FormatUtils",
    "HTMLGenerator",
    "DEFAULT_CSS",
    "DEFAULT_JS",
]
