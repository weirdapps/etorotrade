"""
Base analysis service functionality.

This module provides a base class for analysis services that share common functionality
such as provider initialization and async/sync detection. Analysis services like
AnalystRatingsService and EarningsAnalyzer should inherit from this base class
to reduce code duplication.
"""

from typing import Optional, Union

from ..api import AsyncFinanceDataProvider, FinanceDataProvider, get_provider
from ..core.logging import get_logger


logger = get_logger(__name__)


class BaseAnalysisService:
    """
    Base class for analysis services that use data providers.

    This class provides common provider initialization and async/sync detection
    that is used in multiple analysis services to reduce code duplication.

    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
    """

    def __init__(
        self, provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None
    ):
        """
        Initialize the analysis service.

        Args:
            provider: Data provider (sync or async), if None, a default provider is created
        """
        self.provider = provider if provider is not None else get_provider()

        # Check if the provider is async using introspection
        self._check_provider_async()

    def _check_provider_async(self):
        """
        Check if the provider is asynchronous by checking for coroutine functions.
        """
        # Determine if provider supports async operations
        # This is a common pattern used in multiple services
        self.is_async = (
            hasattr(self.provider, "get_ticker_info")
            and callable(self.provider.get_ticker_info)
            and hasattr(self.provider.get_ticker_info, "__await__")
        )

    def _verify_sync_provider(self, method_name: str):
        """
        Verify that the provider is synchronous when using sync methods.

        Args:
            method_name: Name of the async method to use instead

        Raises:
            TypeError: When trying to use a sync method with async provider
        """
        if self.is_async:
            raise TypeError(
                f"Cannot use sync method with async provider. Use {method_name} instead."
            )

    def _verify_async_provider(self, method_name: str):
        """
        Verify that the provider is asynchronous when using async methods.

        Args:
            method_name: Name of the sync method to use instead

        Raises:
            TypeError: When trying to use an async method with sync provider
        """
        if not self.is_async:
            raise TypeError(
                f"Cannot use async method with sync provider. Use {method_name} instead."
            )
