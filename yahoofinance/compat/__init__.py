"""
Compatibility layer for v1 API.

This module provides compatibility with the v1 API for easier transition
to the v2 framework. It maps v1 function calls and objects to their v2
equivalents, allowing existing code to work without modifications.
"""

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