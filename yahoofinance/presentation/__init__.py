"""
Presentation modules for Yahoo Finance data.

This package contains modules for formatting and displaying data including:
- Console: Console output utilities
- Formatter: Data formatting utilities
- HTML: HTML output generation
- Templates: HTML templates
"""

from .console import MarketDisplay, RateLimitTracker
from .formatter import Color, DisplayConfig, DisplayFormatter
from .html import FormatUtils, HTMLGenerator
from .templates import TemplateEngine, Templates


__all__ = [
    # Formatter
    "DisplayFormatter",
    "DisplayConfig",
    "Color",
    # Console
    "MarketDisplay",
    "RateLimitTracker",
    # HTML
    "FormatUtils",
    "HTMLGenerator",
    # Templates
    "Templates",
    "TemplateEngine",
]
