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
"""

from typing import Union

from .async_hybrid_provider import AsyncHybridProvider
from .async_yahoo_finance import AsyncYahooFinanceProvider
from .async_yahooquery_provider import AsyncYahooQueryProvider
from .base_provider import AsyncFinanceDataProvider, FinanceDataProvider


__all__ = [
    "get_provider",
    "FinanceDataProvider",
    "AsyncFinanceDataProvider",
    "AsyncYahooFinanceProvider",
    "AsyncYahooQueryProvider",
    "AsyncHybridProvider",
]

_PROVIDERS = {
    "yahoo_async": AsyncYahooFinanceProvider,
    "yahooquery_async": AsyncYahooQueryProvider,
    "hybrid_async": AsyncHybridProvider,
}


def get_provider(
    provider_name: str = "hybrid", async_api: bool = True, **kwargs
) -> Union[FinanceDataProvider, AsyncFinanceDataProvider]:
    """
    Get an instance of a finance data provider.

    Args:
        provider_name: Name of the provider to use (default: 'hybrid')
        async_api: Whether to return an async provider (default: True)
        **kwargs: Additional arguments to pass to the provider constructor

    Returns:
        An instance of a FinanceDataProvider or AsyncFinanceDataProvider

    Raises:
        ValueError: If an invalid provider name is provided
    """
    key = f"{provider_name}_async" if async_api else provider_name

    if key not in _PROVIDERS:
        raise ValueError(
            f"Provider '{key}' not found. Available: {list(_PROVIDERS.keys())}"
        )

    return _PROVIDERS[key](**kwargs)
