"""
Provider registry for managing multiple data provider fallbacks.

This module implements a registry pattern that tries providers in order:
1. yfinance (primary)
2. yahooquery (secondary)
3. Alpha Vantage (tertiary)
4. Polygon.io (quaternary)

Usage statistics are tracked for monitoring provider reliability.
"""

from typing import Any, Dict, List, Optional

from ...core.logging import get_logger
from .alpha_vantage_provider import AlphaVantageProvider
from .async_yahoo_finance import AsyncYahooFinanceProvider
from .async_yahooquery_provider import AsyncYahooQueryProvider
from .polygon_provider import PolygonProvider


logger = get_logger(__name__)


class ProviderRegistry:
    """
    Registry for managing multiple data providers with automatic fallback.

    This class maintains a list of providers and tries them in order until one succeeds.
    It also tracks usage statistics for each provider.

    Attributes:
        providers: List of provider instances in order of preference
        stats: Dict tracking success/failure counts per provider
    """

    def __init__(self):
        """Initialize the provider registry with all available providers."""
        self.providers: List[tuple] = [
            ("yfinance", AsyncYahooFinanceProvider()),
            ("yahooquery", AsyncYahooQueryProvider()),
            ("alpha_vantage", AlphaVantageProvider()),
            ("polygon", PolygonProvider()),
        ]

        # Statistics tracking
        self.stats: Dict[str, Dict[str, int]] = {
            name: {"success": 0, "failure": 0}
            for name, _ in self.providers
        }

        logger.info(
            f"Provider registry initialized with {len(self.providers)} providers: "
            f"{', '.join(name for name, _ in self.providers)}"
        )

    async def get_stock_data(
        self, ticker: str, skip_insider_metrics: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get stock data for a ticker, trying providers in order.

        This method tries each provider in sequence until one returns valid data.
        Provider usage is logged and statistics are tracked.

        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict containing stock information, or None if all providers fail
        """
        last_error = None

        for provider_name, provider in self.providers:
            try:
                logger.debug(f"Trying {provider_name} for {ticker}")
                data = await provider.get_ticker_info(ticker, skip_insider_metrics)

                # Check if we got valid data
                if data and self._is_valid_data(data):
                    self.stats[provider_name]["success"] += 1
                    logger.info(f"Successfully fetched {ticker} from {provider_name}")
                    return data
                else:
                    logger.debug(f"{provider_name} returned empty/invalid data for {ticker}")
                    self.stats[provider_name]["failure"] += 1

            except Exception as e:
                last_error = e
                self.stats[provider_name]["failure"] += 1
                logger.warning(f"{provider_name} failed for {ticker}: {e}")
                # Continue to next provider
                continue

        # All providers failed
        logger.error(
            f"All providers failed for {ticker}. Last error: {last_error}"
        )
        return None

    async def get_price_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get price data for a ticker, trying providers in order.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing price data, or None if all providers fail
        """
        last_error = None

        for provider_name, provider in self.providers:
            try:
                logger.debug(f"Trying {provider_name} for price data: {ticker}")
                data = await provider.get_price_data(ticker)

                # Check if we got valid price data
                if data and data.get("current_price") is not None:
                    self.stats[provider_name]["success"] += 1
                    logger.info(f"Successfully fetched price for {ticker} from {provider_name}")
                    return data
                else:
                    logger.debug(f"{provider_name} returned no price for {ticker}")
                    self.stats[provider_name]["failure"] += 1

            except Exception as e:
                last_error = e
                self.stats[provider_name]["failure"] += 1
                logger.warning(f"{provider_name} price fetch failed for {ticker}: {e}")
                continue

        # All providers failed
        logger.error(
            f"All providers failed for price data: {ticker}. Last error: {last_error}"
        )
        return None

    @staticmethod
    def _is_valid_data(data: Dict[str, Any]) -> bool:
        """
        Check if data dict contains valid information.

        Args:
            data: Data dict from provider

        Returns:
            True if data contains at least one valid field
        """
        if not data:
            return False

        # Check for at least symbol or name
        if not (data.get("symbol") or data.get("name")):
            return False

        # Check that we have at least some data beyond symbol/name
        non_empty_fields = sum(
            1 for key, value in data.items()
            if key not in ("symbol", "name") and value is not None
        )

        return non_empty_fields > 0

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get provider usage statistics.

        Returns:
            Dict mapping provider names to their success/failure counts
        """
        return self.stats.copy()

    def get_stats_summary(self) -> str:
        """
        Get a formatted summary of provider statistics.

        Returns:
            Human-readable statistics summary
        """
        lines = ["Provider Usage Statistics:", "=" * 50]

        for provider_name, counts in self.stats.items():
            success = counts["success"]
            failure = counts["failure"]
            total = success + failure

            if total > 0:
                success_rate = (success / total) * 100
                lines.append(
                    f"{provider_name:15s}: {success:4d} success, {failure:4d} failure "
                    f"({success_rate:5.1f}% success rate)"
                )
            else:
                lines.append(f"{provider_name:15s}: No requests")

        return "\n".join(lines)

    def reset_stats(self) -> None:
        """Reset all provider statistics to zero."""
        for provider_name in self.stats:
            self.stats[provider_name] = {"success": 0, "failure": 0}
        logger.info("Provider statistics reset")

    def clear_caches(self) -> None:
        """Clear caches for all providers."""
        for provider_name, provider in self.providers:
            try:
                provider.clear_cache()
                logger.debug(f"Cleared cache for {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to clear cache for {provider_name}: {e}")


# Global singleton instance
_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """
    Get the global provider registry singleton.

    Returns:
        ProviderRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


async def get_stock_data(ticker: str, skip_insider_metrics: bool = False) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get stock data using the global registry.

    Args:
        ticker: Stock ticker symbol
        skip_insider_metrics: If True, skip fetching insider trading metrics

    Returns:
        Dict containing stock information, or None if all providers fail
    """
    registry = get_provider_registry()
    return await registry.get_stock_data(ticker, skip_insider_metrics)


async def get_price_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get price data using the global registry.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict containing price data, or None if all providers fail
    """
    registry = get_provider_registry()
    return await registry.get_price_data(ticker)


def get_provider_stats() -> Dict[str, Dict[str, int]]:
    """
    Get provider usage statistics from the global registry.

    Returns:
        Dict mapping provider names to their success/failure counts
    """
    registry = get_provider_registry()
    return registry.get_stats()


def print_provider_stats() -> None:
    """Print provider usage statistics to logger."""
    registry = get_provider_registry()
    logger.info("\n" + registry.get_stats_summary())


def reset_provider_stats() -> None:
    """Reset all provider statistics in the global registry."""
    registry = get_provider_registry()
    registry.reset_stats()
