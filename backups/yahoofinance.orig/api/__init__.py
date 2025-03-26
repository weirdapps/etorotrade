"""
Yahoo Finance API package.

This package provides a standardized interface for accessing financial data,
including both synchronous and asynchronous interfaces.
"""

from .providers.base import FinanceDataProvider
from .providers.async_base import AsyncFinanceDataProvider

# Import providers lazily to avoid circular dependencies
_default_provider = None
_default_async_provider = None

def get_provider(async_mode: bool = False):
    """
    Get appropriate finance data provider.
    
    Args:
        async_mode: Whether to return an async provider
        
    Returns:
        Finance data provider instance
    """
    global _default_provider, _default_async_provider
    
    if async_mode:
        if _default_async_provider is None:
            from .providers.async_yahoo_finance import AsyncYahooFinanceProvider
            _default_async_provider = AsyncYahooFinanceProvider()
        return _default_async_provider
    
    if _default_provider is None:
        from .providers.yahoo_finance import YahooFinanceProvider
        _default_provider = YahooFinanceProvider()
    return _default_provider

# Export key components
__all__ = [
    'FinanceDataProvider',
    'AsyncFinanceDataProvider',
    'get_provider',
]