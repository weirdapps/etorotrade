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

# Factory function to get appropriate provider
def get_provider(
    provider_type: str = "yahoo",
    async_mode: bool = False,
    enhanced: bool = False,
    **kwargs
) -> Union[FinanceDataProvider, AsyncFinanceDataProvider]:
    """
    Get a provider instance for accessing financial data.
    
    Args:
        provider_type: Type of provider (yahoo, etc.)
        async_mode: Whether to use asynchronous API
        enhanced: Whether to use enhanced async implementation (only applicable when async_mode=True)
        **kwargs: Additional arguments to pass to provider constructor
        
    Returns:
        Provider instance
    
    Raises:
        ValueError: When provider_type is invalid
    """
    if provider_type.lower() == "yahoo":
        if async_mode:
            if enhanced:
                return EnhancedAsyncYahooFinanceProvider(**kwargs)
            else:
                return AsyncYahooFinanceProvider(**kwargs)
        else:
            return YahooFinanceProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

# Export provider classes
__all__ = [
    'get_provider',
    'FinanceDataProvider',
    'AsyncFinanceDataProvider',
    'YahooFinanceProvider',
    'AsyncYahooFinanceProvider',
    'EnhancedAsyncYahooFinanceProvider',
]