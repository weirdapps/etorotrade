"""
Dependency Injection Container for trade modules.

This module provides a simple dependency injection container that manages
the creation and lifecycle of services. It helps break circular dependencies
by centralizing object creation and providing lazy initialization.

Usage:
    from trade_modules.container import get_container

    container = get_container()
    provider = container.get_provider()
    engine = container.get_trading_engine()
"""

from typing import Any, Dict, Optional, TYPE_CHECKING
import logging

from .protocols import (
    FinanceDataProviderProtocol,
    LoggerProtocol,
    AnalysisServiceProtocol,
    FilterServiceProtocol,
    PortfolioServiceProtocol,
    DataProcessingServiceProtocol,
)

if TYPE_CHECKING:
    from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
    from .analysis_service import AnalysisService
    from .filter_service import FilterService
    from .portfolio_service import PortfolioService
    from .data_processing_service import DataProcessingService
    from .trade_engine import TradingEngine


# Singleton container instance
_container: Optional["Container"] = None


class Container:
    """
    Dependency injection container for trade modules.

    This container manages service instances and provides lazy initialization
    to break circular dependencies. Services are created on first access
    and cached for subsequent requests.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the container.

        Args:
            config: Optional configuration dictionary
        """
        self._config = config or {}
        self._instances: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration dictionary."""
        return self._config

    def get_logger(self, name: str = __name__) -> logging.Logger:
        """
        Get a logger instance.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        return logging.getLogger(name)

    def get_provider(self, max_concurrency: Optional[int] = None) -> "AsyncHybridProvider":
        """
        Get the finance data provider instance.

        Uses lazy initialization to avoid circular imports.

        Args:
            max_concurrency: Maximum concurrent requests

        Returns:
            AsyncHybridProvider instance
        """
        if "provider" not in self._instances:
            # Import here to avoid circular imports
            from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
            from yahoofinance.core.config import get_max_concurrent_requests

            concurrency = max_concurrency or get_max_concurrent_requests()
            self._instances["provider"] = AsyncHybridProvider(max_concurrency=concurrency)
            self._logger.debug(f"Created AsyncHybridProvider with max_concurrency={concurrency}")

        return self._instances["provider"]

    def get_analysis_service(self) -> "AnalysisService":
        """
        Get the analysis service instance.

        Returns:
            AnalysisService instance
        """
        if "analysis_service" not in self._instances:
            from .analysis_service import AnalysisService

            self._instances["analysis_service"] = AnalysisService(
                self._config,
                self.get_logger("trade_modules.analysis_service")
            )
            self._logger.debug("Created AnalysisService")

        return self._instances["analysis_service"]

    def get_filter_service(self) -> "FilterService":
        """
        Get the filter service instance.

        Returns:
            FilterService instance
        """
        if "filter_service" not in self._instances:
            from .filter_service import FilterService

            self._instances["filter_service"] = FilterService(
                self.get_logger("trade_modules.filter_service")
            )
            self._logger.debug("Created FilterService")

        return self._instances["filter_service"]

    def get_portfolio_service(self) -> "PortfolioService":
        """
        Get the portfolio service instance.

        Returns:
            PortfolioService instance
        """
        if "portfolio_service" not in self._instances:
            from .portfolio_service import PortfolioService

            self._instances["portfolio_service"] = PortfolioService(
                self.get_logger("trade_modules.portfolio_service")
            )
            self._logger.debug("Created PortfolioService")

        return self._instances["portfolio_service"]

    def get_data_processing_service(
        self,
        provider: Optional[FinanceDataProviderProtocol] = None
    ) -> "DataProcessingService":
        """
        Get the data processing service instance.

        Args:
            provider: Optional provider instance (uses default if not provided)

        Returns:
            DataProcessingService instance
        """
        if "data_processing_service" not in self._instances:
            from .data_processing_service import DataProcessingService

            actual_provider = provider or self.get_provider()
            self._instances["data_processing_service"] = DataProcessingService(
                actual_provider,
                self.get_logger("trade_modules.data_processing_service")
            )
            self._logger.debug("Created DataProcessingService")

        return self._instances["data_processing_service"]

    def get_trading_engine(
        self,
        provider: Optional[FinanceDataProviderProtocol] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> "TradingEngine":
        """
        Get the trading engine instance.

        Args:
            provider: Optional provider instance
            config: Optional configuration dictionary

        Returns:
            TradingEngine instance
        """
        if "trading_engine" not in self._instances:
            from .trade_engine import TradingEngine

            actual_provider = provider or self.get_provider()
            actual_config = config or self._config
            self._instances["trading_engine"] = TradingEngine(
                provider=actual_provider,
                config=actual_config
            )
            self._logger.debug("Created TradingEngine")

        return self._instances["trading_engine"]

    def clear(self) -> None:
        """Clear all cached instances."""
        self._instances.clear()
        self._logger.debug("Cleared all container instances")

    def reset(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Reset the container with new configuration.

        Args:
            config: New configuration dictionary
        """
        self.clear()
        if config is not None:
            self._config = config
        self._logger.debug("Reset container")


def get_container(config: Optional[Dict[str, Any]] = None) -> Container:
    """
    Get the singleton container instance.

    Args:
        config: Optional configuration dictionary (only used on first call)

    Returns:
        Container singleton instance
    """
    global _container
    if _container is None:
        _container = Container(config)
    return _container


def reset_container(config: Optional[Dict[str, Any]] = None) -> Container:
    """
    Reset and return the container instance.

    This is useful for testing or when configuration needs to change.

    Args:
        config: New configuration dictionary

    Returns:
        Reset container instance
    """
    global _container
    if _container is not None:
        _container.clear()
    _container = Container(config)
    return _container
