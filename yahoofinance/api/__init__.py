"""
Finance API module for accessing financial data.

This module provides access to financial data through provider interfaces,
which abstract the implementation details of different data sources.
"""

from typing import Any, Dict, Optional, Type, Union

# Import the provider registry
from .provider_registry import get_all_providers, get_default_provider, get_provider
from .providers.async_hybrid_provider import AsyncHybridProvider
from .providers.async_yahoo_finance import AsyncYahooFinanceProvider
from .providers.async_yahooquery_provider import AsyncYahooQueryProvider
from .providers.base_provider import AsyncFinanceDataProvider, FinanceDataProvider
from .providers.hybrid_provider import HybridProvider
from .providers.yahoo_finance import YahooFinanceProvider
from .providers.yahooquery_provider import YahooQueryProvider


# Export provider classes
__all__ = [
    # Provider factory functions
    "get_provider",
    "get_all_providers",
    "get_default_provider",
    # Provider interfaces
    "FinanceDataProvider",
    "AsyncFinanceDataProvider",
    # Provider implementations
    "YahooFinanceProvider",
    "AsyncYahooFinanceProvider",
    "YahooQueryProvider",
    "AsyncYahooQueryProvider",
    "HybridProvider",
    "AsyncHybridProvider",
]
