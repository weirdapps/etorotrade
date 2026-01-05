"""
Resilient provider with fallback strategy.

This wraps AsyncHybridProvider with cascading fallbacks for maximum uptime.
"""
from datetime import datetime
from typing import Dict, Any, List

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.api.providers.async_yahooquery_provider import AsyncYahooQueryProvider
from yahoofinance.api.providers.fallback_strategy import (
    CascadingFallbackStrategy,
    FetchResult,
    DataSource
)
from yahoofinance.core.logging import get_logger
from yahoofinance.data.cache import get_cache_manager


logger = get_logger(__name__)


class ResilientProvider:
    """
    Provider with intelligent fallback hierarchy for 99.9% uptime.

    Fallback order:
    1. Yahoo Finance (via AsyncHybridProvider)
    2. YahooQuery (alternative API)
    3. Stale cache (up to 7 days old)

    Example:
        ```python
        provider = ResilientProvider()

        # Fetch with automatic fallback
        result = await provider.get_ticker_info("AAPL")

        if result.get('_is_stale'):
            print(f"⚠️  Using {result['_data_source']} data")

        # Check reliability
        stats = provider.get_reliability_stats()
        print(f"Primary success rate: {stats['primary_success_rate']:.1f}%")
        ```
    """

    def __init__(
        self,
        enable_fallback: bool = True,
        enable_stale_cache: bool = True
    ):
        """
        Initialize resilient provider.

        Args:
            enable_fallback: Enable YahooQuery fallback
            enable_stale_cache: Allow stale cache data
        """
        # Primary provider
        self.primary = AsyncHybridProvider()

        # Fallback provider (only if enabled)
        self.fallback = AsyncYahooQueryProvider() if enable_fallback else None

        # Cache
        self.cache = get_cache_manager()

        # Fallback strategy
        self.strategy = CascadingFallbackStrategy(
            primary_provider=self.primary,
            fallback_provider=self.fallback,
            cache=self.cache if enable_stale_cache else None
        )

    async def get_ticker_info(
        self,
        ticker: str,
        skip_insider_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Get ticker info with automatic fallback.

        Returns:
            Ticker data with additional metadata:
            - _data_source: Where data came from
            - _is_stale: Whether data is stale
            - _fetched_at: When data was fetched
        """
        result: FetchResult = await self.strategy.fetch(ticker)

        if not result.success:
            logger.error(f"Failed to fetch {ticker}: {result.error}")
            return {
                "symbol": ticker,
                "ticker": ticker,
                "error": str(result.error),
                "_data_source": DataSource.ERROR.value
            }

        # Add metadata
        data = result.data.copy()
        data['_data_source'] = result.source.value
        data['_is_stale'] = result.is_stale
        data['_fetched_at'] = result.timestamp.isoformat()
        data['_latency_ms'] = result.latency_ms

        # Log warning for stale data
        if result.is_stale:
            logger.warning(
                f"Using stale data for {ticker} from {result.source.value} "
                f"(age: {(datetime.now() - result.timestamp).total_seconds()/3600:.1f}h)"
            )

        return data

    async def batch_get_ticker_info(
        self,
        tickers: List[str],
        skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch fetch with fallback for each ticker.

        This ensures one ticker's failure doesn't block others.
        """
        import asyncio

        async def fetch_one(ticker: str) -> tuple:
            data = await self.get_ticker_info(ticker, skip_insider_metrics)
            return ticker, data

        results = await asyncio.gather(*[
            fetch_one(ticker) for ticker in tickers
        ])

        return dict(results)

    def get_reliability_stats(self) -> Dict[str, Any]:
        """
        Get reliability statistics.

        Returns:
            Dict with success rates and fallback usage
        """
        stats = self.strategy.get_stats()
        total_success = stats.get(DataSource.PRIMARY.value, 0) + stats.get(DataSource.FALLBACK.value, 0)

        return {
            "primary_success_rate": stats.get(DataSource.PRIMARY.value, 0),
            "fallback_usage_rate": stats.get(DataSource.FALLBACK.value, 0),
            "stale_cache_usage_rate": stats.get(DataSource.CACHE_STALE.value, 0),
            "total_success_rate": total_success,
            "error_rate": stats.get(DataSource.ERROR.value, 0),
            "uptime_estimate": (total_success + stats.get(DataSource.CACHE_STALE.value, 0)),
        }

    async def close(self):
        """Close all providers"""
        await self.primary.close()
        if self.fallback:
            await self.fallback.close()
