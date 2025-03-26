"""
Provider module for different finance data sources.

This module provides a standardized interface for accessing financial data
from different sources such as Yahoo Finance or other providers.

Importing:
    from yahoofinance_v2.api.providers import get_provider, FinanceDataProvider, AsyncFinanceDataProvider
    
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
from .yahoo_finance import YahooFinanceProvider
from .async_yahoo_finance import AsyncYahooFinanceProvider

__all__ = [
    'get_provider',
    'FinanceDataProvider',
    'AsyncFinanceDataProvider',
    'YahooFinanceProvider',
    'AsyncYahooFinanceProvider',
]

_PROVIDERS = {
    'yahoo': YahooFinanceProvider,
    'yahoo_async': AsyncYahooFinanceProvider,
}

def get_provider(
    provider_name: str = 'yahoo',
    async_api: bool = False,
    **kwargs
) -> Union[FinanceDataProvider, AsyncFinanceDataProvider]:
    """
    Get an instance of a finance data provider.
    
    Args:
        provider_name: Name of the provider to use (default: 'yahoo')
        async_api: Whether to return an async provider
        **kwargs: Additional arguments to pass to the provider constructor
        
    Returns:
        An instance of a FinanceDataProvider or AsyncFinanceDataProvider
        
    Raises:
        ValueError: If an invalid provider name is provided
    """
    if async_api:
        key = f"{provider_name}_async"
        if key not in _PROVIDERS:
            raise ValueError(f"Async provider '{provider_name}' not found")
        return _PROVIDERS[key](**kwargs)
    
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Provider '{provider_name}' not found")
    
    return _PROVIDERS[provider_name](**kwargs)
