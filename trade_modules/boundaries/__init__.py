"""
Module boundary interfaces for trade modules.

This package defines clear boundaries between trade_modules and yahoofinance
packages, providing stable interfaces and preventing direct coupling.
"""

from .yahoo_finance_boundary import YahooFinanceBoundary
from .trade_modules_boundary import TradeModulesBoundary  
from .config_boundary import ConfigBoundary
from .data_boundary import DataBoundary

__all__ = [
    'YahooFinanceBoundary',
    'TradeModulesBoundary',
    'ConfigBoundary', 
    'DataBoundary'
]