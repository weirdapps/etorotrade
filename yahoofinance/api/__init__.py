"""
Yahoo Finance API package.

This package provides a standardized interface for accessing financial data,
including both synchronous and asynchronous interfaces.
"""

from .providers.base import FinanceDataProvider
from .providers.yahoo_finance import YahooFinanceProvider
from .providers.async_base import AsyncFinanceDataProvider
from .providers.async_yahoo_finance import AsyncYahooFinanceProvider

# Create default provider instances
default_provider = YahooFinanceProvider()
default_async_provider = AsyncYahooFinanceProvider()

# Export key components
__all__ = [
    'FinanceDataProvider',
    'YahooFinanceProvider',
    'AsyncFinanceDataProvider',
    'AsyncYahooFinanceProvider',
    'default_provider',
    'default_async_provider',
]