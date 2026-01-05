"""
HTML modules for financial data visualization.

This package contains utilities for generating HTML reports from financial data.
"""

from .formatters import FormatUtils
from .generator import HTMLGenerator
from .styles import DEFAULT_CSS, DEFAULT_JS

__all__ = [
    "FormatUtils",
    "HTMLGenerator",
    "DEFAULT_CSS",
    "DEFAULT_JS",
]
