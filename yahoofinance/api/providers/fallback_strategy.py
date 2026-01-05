"""
Fallback strategy for resilient data fetching.

Implements cascading fallback hierarchy:
1. Primary provider
2. Fallback provider
3. Stale cache
4. Graceful error
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

from yahoofinance.core.logging import get_logger


logger = get_logger(__name__)


class DataSource(str, Enum):
    """Source of fetched data"""
    PRIMARY = "primary"
    FALLBACK = "fallback"
    CACHE_FRESH = "cache_fresh"
    CACHE_STALE = "cache_stale"
    ERROR = "error"


@dataclass
class FetchResult:
    """Result of data fetch attempt with metadata"""
    data: Optional[Dict[str, Any]]
    source: DataSource
    timestamp: datetime
    is_stale: bool = False
    error: Optional[Exception] = None
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Whether fetch was successful"""
        return self.data is not None


class FallbackStrategy(ABC):
    """Base class for fallback strategies"""

    @abstractmethod
    async def fetch(self, ticker: str) -> FetchResult:
        """Fetch data with fallback logic"""
        pass


class CascadingFallbackStrategy(FallbackStrategy):
    """
    Cascading fallback through multiple data sources.

    Tries sources in order:
    1. Primary provider (with circuit breaker)
    2. Fallback provider (if primary fails)
    3. Cache (if both providers fail)
    4. Error (if everything fails)

    Example:
        ```python
        strategy = CascadingFallbackStrategy(
            primary=yahoo_finance_provider,
            fallback=yahooquery_provider,
            cache=redis_cache
        )

        result = await strategy.fetch("AAPL")

        if result.success:
            if result.is_stale:
                print(f"⚠️  Using stale data from {result.source}")
            return result.data
        else:
            print(f"❌ All providers failed: {result.error}")
        ```
    """

    def __init__(
        self,
        primary_provider,
        fallback_provider=None,
        cache=None,
        stale_threshold: timedelta = timedelta(hours=48),
        max_stale_age: timedelta = timedelta(days=7)
    ):
        """
        Initialize cascading fallback strategy.

        Args:
            primary_provider: Primary data provider
            fallback_provider: Optional fallback provider
            cache: Optional cache for stale data
            stale_threshold: Age threshold for marking data as stale
            max_stale_age: Maximum age for cached data (older = rejected)
        """
        self.primary = primary_provider
        self.fallback = fallback_provider
        self.cache = cache
        self.stale_threshold = stale_threshold
        self.max_stale_age = max_stale_age

        # Metrics
        self._stats = {
            DataSource.PRIMARY: 0,
            DataSource.FALLBACK: 0,
            DataSource.CACHE_FRESH: 0,
            DataSource.CACHE_STALE: 0,
            DataSource.ERROR: 0,
        }

    async def fetch(self, ticker: str) -> FetchResult:
        """
        Fetch data with cascading fallback.

        Returns:
            FetchResult with data and metadata
        """
        import time

        # Try primary provider
        try:
            logger.debug(f"Fetching {ticker} from primary provider")
            start = time.perf_counter()

            data = await self.primary.get_ticker_info(ticker)
            latency = (time.perf_counter() - start) * 1000

            if data and 'error' not in data:
                self._stats[DataSource.PRIMARY] += 1
                logger.debug(f"✅ Primary provider success for {ticker} ({latency:.1f}ms)")

                # Store in cache for future use
                if self.cache:
                    await self._store_in_cache(ticker, data)

                return FetchResult(
                    data=data,
                    source=DataSource.PRIMARY,
                    timestamp=datetime.now(),
                    latency_ms=latency
                )

        except Exception as e:
            logger.warning(f"Primary provider failed for {ticker}: {e}")

        # Try fallback provider
        if self.fallback:
            try:
                logger.debug(f"Trying fallback provider for {ticker}")
                start = time.perf_counter()

                data = await self.fallback.get_ticker_info(ticker)
                latency = (time.perf_counter() - start) * 1000

                if data and 'error' not in data:
                    self._stats[DataSource.FALLBACK] += 1
                    logger.info(f"✅ Fallback provider success for {ticker} ({latency:.1f}ms)")

                    # Store in cache
                    if self.cache:
                        await self._store_in_cache(ticker, data)

                    return FetchResult(
                        data=data,
                        source=DataSource.FALLBACK,
                        timestamp=datetime.now(),
                        latency_ms=latency
                    )

            except Exception as e:
                logger.warning(f"Fallback provider failed for {ticker}: {e}")

        # Try cache (stale data better than no data)
        if self.cache:
            try:
                logger.debug(f"Trying cache for {ticker}")
                cached = await self._fetch_from_cache(ticker)

                if cached:
                    cache_age = datetime.now() - cached['_cached_at']

                    # Reject if too old
                    if cache_age > self.max_stale_age:
                        logger.warning(
                            f"Cache too old for {ticker}: "
                            f"{cache_age.total_seconds()/3600:.1f}h "
                            f"(max: {self.max_stale_age.total_seconds()/3600:.0f}h)"
                        )
                    else:
                        is_stale = cache_age > self.stale_threshold
                        source = DataSource.CACHE_STALE if is_stale else DataSource.CACHE_FRESH
                        self._stats[source] += 1

                        logger.info(
                            f"{'⚠️  Stale' if is_stale else '✅ Fresh'} cache hit for {ticker} "
                            f"(age: {cache_age.total_seconds()/3600:.1f}h)"
                        )

                        return FetchResult(
                            data=cached,
                            source=source,
                            timestamp=cached['_cached_at'],
                            is_stale=is_stale
                        )

            except Exception as e:
                logger.warning(f"Cache fetch failed for {ticker}: {e}")

        # All sources failed
        self._stats[DataSource.ERROR] += 1
        logger.error(f"❌ All providers failed for {ticker}")

        return FetchResult(
            data=None,
            source=DataSource.ERROR,
            timestamp=datetime.now(),
            error=Exception(f"All providers failed for {ticker}")
        )

    async def _store_in_cache(self, ticker: str, data: Dict[str, Any]):
        """Store data in cache with timestamp"""
        try:
            data_copy = data.copy()
            data_copy['_cached_at'] = datetime.now()
            # Sync cache call - run in executor to avoid blocking
            if hasattr(self.cache, 'set'):
                self.cache.set(ticker, data_copy)
        except Exception as e:
            logger.warning(f"Failed to cache data for {ticker}: {e}")

    async def _fetch_from_cache(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch data from cache"""
        try:
            # Sync cache call - run in executor to avoid blocking
            if hasattr(self.cache, 'get'):
                return self.cache.get(ticker)
            return None
        except Exception as e:
            logger.warning(f"Cache fetch error for {ticker}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        total = sum(self._stats.values())

        if total == 0:
            return {source.value: 0.0 for source in DataSource}

        return {
            source.value: (count / total) * 100
            for source, count in self._stats.items()
        }

    def reset_stats(self):
        """Reset statistics"""
        for source in DataSource:
            self._stats[source] = 0
