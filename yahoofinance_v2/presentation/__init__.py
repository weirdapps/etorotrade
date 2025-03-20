"""
Presentation modules for Yahoo Finance data.

This package contains modules for formatting and displaying data including:
- Console: Console output utilities
- Formatter: Data formatting utilities
- HTML: HTML output generation
- Templates: HTML templates
"""

from .formatter import DisplayFormatter, DisplayConfig, Color
from .console import MarketDisplay, RateLimitTracker
from .html import FormatUtils, HTMLGenerator
from .templates import Templates, TemplateEngine

__all__ = [
    # Formatter
    'DisplayFormatter',
    'DisplayConfig',
    'Color',
    
    # Console
    'MarketDisplay',
    'RateLimitTracker',
    
    # HTML
    'FormatUtils',
    'HTMLGenerator',
    
    # Templates
    'Templates',
    'TemplateEngine',
]
