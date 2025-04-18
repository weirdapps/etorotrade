"""
Finance API module for accessing financial data.

This module provides access to financial data through provider interfaces,
which abstract the implementation details of different data sources.
"""

from typing import Dict, Any, Optional, Type, Union

from .providers.base_provider import FinanceDataProvider, AsyncFinanceDataProvider
from .providers.yahoo_finance import YahooFinanceProvider
from .providers.async_yahoo_finance import AsyncYahooFinanceProvider
from .providers.enhanced_async_yahoo_finance import EnhancedAsyncYahooFinanceProvider
from .providers.yahooquery_provider import YahooQueryProvider
from .providers.async_yahooquery_provider import AsyncYahooQueryProvider
from .providers.hybrid_provider import HybridProvider
from .providers.async_hybrid_provider import AsyncHybridProvider
from .providers.optimized_async_yfinance import OptimizedAsyncYFinanceProvider

# Import the provider registry
from .provider_registry import get_provider, get_all_providers, get_default_provider

# Export provider classes
__all__ = [
    # Provider factory functions
    'get_provider',
    'get_all_providers',
    'get_default_provider',
    
    # Provider interfaces
    'FinanceDataProvider',
    'AsyncFinanceDataProvider',
    
    # Provider implementations (for backward compatibility)
    'YahooFinanceProvider',
    'AsyncYahooFinanceProvider',
    'EnhancedAsyncYahooFinanceProvider',
    'YahooQueryProvider',
    'AsyncYahooQueryProvider',
    'HybridProvider',
    'AsyncHybridProvider',
    'OptimizedAsyncYFinanceProvider',
]