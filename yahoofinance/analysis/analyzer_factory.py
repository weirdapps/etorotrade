"""
Factory for creating analyzer instances.

This module provides factory functions for creating StockAnalyzer instances
with appropriate dependencies injected.
"""

from typing import Any, Dict, List, Optional, Union

from ..api.providers.base_provider import AsyncFinanceDataProvider, FinanceDataProvider
from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..utils.dependency_injection import inject, registry


# Set up logging
logger = get_logger(__name__)

# Import the analyzer class
from .stock import StockAnalyzer


# Register factory for creating StockAnalyzer instances
@registry.register("stock_analyzer")
def create_stock_analyzer(
    provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None,
    async_mode: bool = False,
    enhanced: bool = False,
    **kwargs,
) -> StockAnalyzer:
    """
    Create a StockAnalyzer instance with appropriate dependencies.

    This factory function creates a StockAnalyzer instance with the
    appropriate provider injected.

    Args:
        provider: Provider instance to use (optional)
        async_mode: Whether to use asynchronous API if no provider specified
        enhanced: Whether to use enhanced provider if no provider specified
        **kwargs: Additional arguments to pass to StockAnalyzer constructor

    Returns:
        StockAnalyzer instance

    Raises:
        ValidationError: When provider creation fails
    """
    # Use the provided provider or create a new one
    if provider is None:
        try:
            get_provider = registry.resolve("get_provider")
            provider = get_provider(async_mode=async_mode, enhanced=enhanced)
        except Exception as e:
            logger.error(f"Failed to create provider for StockAnalyzer: {str(e)}")
            raise ValidationError(f"Failed to create provider for StockAnalyzer: {str(e)}") from e

    # Create the analyzer with the provider
    return StockAnalyzer(provider=provider, **kwargs)


# Decorator for injecting a StockAnalyzer
def with_analyzer(**kwargs):
    """
    Decorator for injecting a StockAnalyzer instance.

    This decorator adds a 'analyzer' parameter to the function that is
    resolved from the registry.

    Args:
        **kwargs: Additional arguments to pass to create_stock_analyzer()

    Returns:
        Decorated function

    Example:
        ```
        @with_analyzer(async_mode=True)
        def analyze_stock(ticker, analyzer=None):
            return analyzer.analyze_ticker(ticker)
        ```
    """
    return inject("stock_analyzer", **kwargs)


# Register factory for creating PortfolioAnalyzer instances
@registry.register("portfolio_analyzer")
def create_portfolio_analyzer(
    provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None,
    stock_analyzer: Optional[StockAnalyzer] = None,
    async_mode: bool = False,
    enhanced: bool = False,
    **kwargs,
) -> Any:  # Return type is PortfolioAnalyzer, but we use Any to avoid circular imports
    """
    Create a PortfolioAnalyzer instance with appropriate dependencies.

    Args:
        provider: Provider instance to use (optional)
        stock_analyzer: StockAnalyzer instance to use (optional)
        async_mode: Whether to use asynchronous API if no provider specified
        enhanced: Whether to use enhanced provider if no provider specified
        **kwargs: Additional arguments to pass to PortfolioAnalyzer constructor

    Returns:
        PortfolioAnalyzer instance

    Raises:
        ValidationError: When provider or analyzer creation fails
    """
    # Lazy import to avoid circular dependencies
    from .portfolio import PortfolioAnalyzer

    # Use the provided provider or create a new one
    if provider is None:
        try:
            from ..api.provider_registry import get_provider

            provider = get_provider(async_mode=async_mode, enhanced=enhanced)
        except Exception as e:
            logger.error(f"Failed to create provider for PortfolioAnalyzer: {str(e)}")
            raise ValidationError(
                f"Failed to create provider for PortfolioAnalyzer: {str(e)}"
            ) from e

    # Use the provided analyzer or create a new one
    if stock_analyzer is None:
        stock_analyzer = create_stock_analyzer(provider=provider, **kwargs)

    # Create the portfolio analyzer
    return PortfolioAnalyzer(provider=provider, stock_analyzer=stock_analyzer)


# Decorator for injecting a PortfolioAnalyzer
def with_portfolio_analyzer(**kwargs):
    """
    Decorator for injecting a PortfolioAnalyzer instance.

    This decorator adds a 'portfolio_analyzer' parameter to the function that is
    resolved from the registry.

    Args:
        **kwargs: Additional arguments to pass to create_portfolio_analyzer()

    Returns:
        Decorated function

    Example:
        ```
        @with_portfolio_analyzer(async_mode=True)
        def analyze_portfolio(portfolio_file, portfolio_analyzer=None):
            portfolio_analyzer.load_portfolio_from_csv(portfolio_file)
            return portfolio_analyzer.analyze_portfolio()
        ```
    """
    return inject("portfolio_analyzer", **kwargs)


# Initialize registry
registry.register("get_analyzer", create_stock_analyzer)
registry.register("get_portfolio_analyzer", create_portfolio_analyzer)
