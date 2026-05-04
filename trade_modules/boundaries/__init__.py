"""
Module boundary interfaces for trade modules.

This package defines clear boundaries between trade_modules and yahoofinance
packages, providing stable interfaces and preventing direct coupling.
"""

from .config_boundary import ConfigBoundary
from .data_boundary import DataBoundary
from .trade_modules_boundary import TradeModulesBoundary
from .yahoo_finance_boundary import YahooFinanceBoundary

__all__ = ["YahooFinanceBoundary", "TradeModulesBoundary", "ConfigBoundary", "DataBoundary"]
