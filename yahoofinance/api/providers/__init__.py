"""
Provider module for different finance data sources.

This module provides a standardized interface for accessing financial data
from different sources such as Yahoo Finance or other providers.

Importing:
    from yahoofinance.api.providers import get_provider, FinanceDataProvider, AsyncFinanceDataProvider
    
    # Get default provider
    provider = get_provider()
    
    # Get specific provider
    async_provider = get_provider(async_api=True)
    
    # Custom provider implementation
    custom_provider = get_provider(provider_name="custom")
"""

from typing import Optional, Union, Dict, Any, List

from .base_provider import (
    FinanceDataProvider, 
    AsyncFinanceDataProvider
)
from .yahoo_finance_base import YahooFinanceBaseProvider
from .yahoo_finance import YahooFinanceProvider
from .async_yahoo_finance import AsyncYahooFinanceProvider
from .yahooquery_provider import YahooQueryProvider
from .async_yahooquery_provider import AsyncYahooQueryProvider
from .hybrid_provider import HybridProvider
from .async_hybrid_provider import AsyncHybridProvider
from .optimized_async_yfinance import OptimizedAsyncYFinanceProvider

__all__ = [
    'get_provider',
    'FinanceDataProvider',
    'AsyncFinanceDataProvider',
    'YahooFinanceBaseProvider',
    'YahooFinanceProvider',
    'AsyncYahooFinanceProvider',
    'YahooQueryProvider',
    'AsyncYahooQueryProvider',
    'HybridProvider',
    'AsyncHybridProvider',
    'OptimizedAsyncYFinanceProvider',
]

_PROVIDERS = {
    'yahoo': YahooFinanceProvider,
    'yahoo_async': AsyncYahooFinanceProvider,
    'yahooquery': YahooQueryProvider,
    'yahooquery_async': AsyncYahooQueryProvider,
    'hybrid': HybridProvider,
    'hybrid_async': AsyncHybridProvider,
    'optimized': OptimizedAsyncYFinanceProvider,
}

def get_provider(
    provider_name: str = 'hybrid',  # Updated default to hybrid
    async_api: bool = False,
    **kwargs
) -> Union[FinanceDataProvider, AsyncFinanceDataProvider]:
    """
    Get an instance of a finance data provider.
    
    Args:
        provider_name: Name of the provider to use (default: 'hybrid')
        async_api: Whether to return an async provider
        **kwargs: Additional arguments to pass to the provider constructor
        
    Returns:
        An instance of a FinanceDataProvider or AsyncFinanceDataProvider
        
    Raises:
        ValueError: If an invalid provider name is provided
    """
    if async_api:
        # Check if there's a direct async provider first
        if f"{provider_name}_async" in _PROVIDERS:
            return _PROVIDERS[f"{provider_name}_async"](**kwargs)
        # For special case of hybrid_async
        elif provider_name == 'hybrid':
            return _PROVIDERS["hybrid_async"](**kwargs)
        # For other cases, try to find a matching async provider
        else:
            # Try alternative registrations
            if provider_name == 'optimized':
                return _PROVIDERS['optimized'](**kwargs)
            else:
                raise ValueError(f"Async provider '{provider_name}' not found")
    
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Provider '{provider_name}' not found")
    
    return _PROVIDERS[provider_name](**kwargs)
