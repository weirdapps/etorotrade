"""
Cache warming strategy for improved performance.

Pre-loads frequently accessed data into cache to reduce latency
and improve user experience.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Set, Dict, Any
from pathlib import Path

from yahoofinance.core.cache import default_cache_manager


logger = logging.getLogger(__name__)


class CacheWarmer:
    """
    Intelligent cache warming for trading data.

    Pre-fetches portfolio stocks and popular tickers to ensure
    fast response times when user requests them.
    """

    def __init__(
        self,
        provider,
        cache_manager=None,
        enable_background_refresh: bool = True,
        refresh_interval_minutes: int = 30
    ):
        """
        Initialize cache warmer.

        Args:
            provider: Data provider for fetching ticker info
            cache_manager: Cache manager instance (defaults to global)
            enable_background_refresh: Enable automatic background refresh
            refresh_interval_minutes: Minutes between background refreshes
        """
        self.provider = provider
        self.cache = cache_manager or default_cache_manager
        self.enable_background_refresh = enable_background_refresh
        self.refresh_interval = timedelta(minutes=refresh_interval_minutes)

        # Track warming status
        self._last_warm_time: Optional[datetime] = None
        self._warming_in_progress: bool = False
        self._background_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "total_warmed": 0,
            "total_refreshed": 0,
            "cache_hits_saved": 0,
            "warm_duration_seconds": 0.0,
        }

    async def warm_portfolio(self, portfolio_path: str) -> Dict[str, Any]:
        """
        Warm cache with portfolio stocks.

        Args:
            portfolio_path: Path to portfolio CSV file

        Returns:
            Dictionary with warming statistics
        """
        if self._warming_in_progress:
            logger.warning("Cache warming already in progress, skipping")
            return {"status": "already_in_progress"}

        self._warming_in_progress = True
        start_time = datetime.now()

        try:
            # Read portfolio tickers
            tickers = self._read_portfolio_tickers(portfolio_path)

            if not tickers:
                logger.info("No tickers found in portfolio, skipping cache warming")
                return {"status": "no_tickers", "count": 0}

            # Warm cache
            results = await self._warm_tickers(tickers, reason="portfolio")

            # Update stats
            duration = (datetime.now() - start_time).total_seconds()
            self.stats["warm_duration_seconds"] = duration
            self._last_warm_time = datetime.now()

            logger.info(
                f"Cache warming complete: {results['warmed']}/{results['total']} "
                f"tickers in {duration:.2f}s"
            )

            return {
                "status": "completed",
                "total": results["total"],
                "warmed": results["warmed"],
                "failed": results["failed"],
                "duration_seconds": duration,
            }

        finally:
            self._warming_in_progress = False

    async def warm_popular_stocks(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Warm cache with popular/frequently accessed stocks.

        Args:
            tickers: List of ticker symbols to warm

        Returns:
            Dictionary with warming statistics
        """
        if self._warming_in_progress:
            logger.warning("Cache warming already in progress, skipping")
            return {"status": "already_in_progress"}

        self._warming_in_progress = True
        start_time = datetime.now()

        try:
            results = await self._warm_tickers(tickers, reason="popular")
            duration = (datetime.now() - start_time).total_seconds()

            # Update stats
            self.stats["warm_duration_seconds"] = duration
            self._last_warm_time = datetime.now()

            logger.info(
                f"Popular stocks warming complete: {results['warmed']}/{results['total']} "
                f"tickers in {duration:.2f}s"
            )

            return {
                "status": "completed",
                "total": results["total"],
                "warmed": results["warmed"],
                "failed": results["failed"],
                "duration_seconds": duration,
            }

        finally:
            self._warming_in_progress = False

    async def _warm_tickers(
        self,
        tickers: List[str],
        reason: str = "manual"
    ) -> Dict[str, int]:
        """
        Internal method to warm cache for list of tickers.

        Args:
            tickers: List of ticker symbols
            reason: Reason for warming (for logging)

        Returns:
            Dictionary with success/failure counts
        """
        warmed = 0
        failed = 0

        # Process in batches to avoid overwhelming the API
        batch_size = 25
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            # Fetch data for batch concurrently
            tasks = [self.provider.get_ticker_info(ticker) for ticker in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes/failures
            for ticker, result in zip(batch, results):
                if isinstance(result, Exception):
                    failed += 1
                    logger.debug(f"Failed to warm {ticker}: {result}")
                elif result and "error" not in result:
                    warmed += 1
                    # Data is already cached by provider
                    self.stats["total_warmed"] += 1
                else:
                    failed += 1

            # Small delay between batches to respect rate limits
            if i + batch_size < len(tickers):
                await asyncio.sleep(0.5)

        return {
            "total": len(tickers),
            "warmed": warmed,
            "failed": failed,
        }

    def _read_portfolio_tickers(self, portfolio_path: str) -> List[str]:
        """
        Read ticker symbols from portfolio CSV.

        Args:
            portfolio_path: Path to portfolio CSV file

        Returns:
            List of unique ticker symbols
        """
        try:
            import pandas as pd

            path = Path(portfolio_path)
            if not path.exists():
                logger.warning(f"Portfolio file not found: {portfolio_path}")
                return []

            # Read CSV
            df = pd.read_csv(portfolio_path)

            # Extract ticker column (try common column names)
            ticker_columns = ["ticker", "Ticker", "symbol", "Symbol", "TICKER"]
            ticker_col = None

            for col in ticker_columns:
                if col in df.columns:
                    ticker_col = col
                    break

            if ticker_col is None:
                logger.warning(f"No ticker column found in {portfolio_path}")
                return []

            # Get unique tickers
            tickers = df[ticker_col].dropna().unique().tolist()

            # Remove any empty strings
            tickers = [t for t in tickers if t and isinstance(t, str)]

            logger.info(f"Found {len(tickers)} tickers in portfolio")
            return tickers

        except Exception as e:
            logger.error(f"Error reading portfolio: {e}")
            return []

    async def start_background_refresh(self, portfolio_path: str):
        """
        Start background task to periodically refresh cache.

        Args:
            portfolio_path: Path to portfolio CSV file
        """
        if not self.enable_background_refresh:
            logger.info("Background refresh is disabled")
            return

        if self._background_task and not self._background_task.done():
            logger.warning("Background refresh already running")
            return

        logger.info(
            f"Starting background cache refresh "
            f"(interval: {self.refresh_interval.total_seconds()/60:.0f} minutes)"
        )

        self._background_task = asyncio.create_task(
            self._background_refresh_loop(portfolio_path)
        )

    async def _background_refresh_loop(self, portfolio_path: str):
        """
        Background loop to periodically refresh cache.

        Args:
            portfolio_path: Path to portfolio CSV file
        """
        while self.enable_background_refresh:
            try:
                # Wait for refresh interval
                await asyncio.sleep(self.refresh_interval.total_seconds())

                # Refresh cache
                logger.info("Starting background cache refresh")
                results = await self.warm_portfolio(portfolio_path)

                if results["status"] == "completed":
                    self.stats["total_refreshed"] += results["warmed"]
                    logger.info(
                        f"Background refresh complete: {results['warmed']} tickers"
                    )

            except asyncio.CancelledError:
                logger.info("Background refresh cancelled")
                break
            except Exception as e:
                logger.error(f"Error in background refresh: {e}")
                # Continue loop despite error

    async def stop_background_refresh(self):
        """Stop background refresh task."""
        if self._background_task and not self._background_task.done():
            logger.info("Stopping background cache refresh")
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache warming statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "last_warm_time": self._last_warm_time.isoformat() if self._last_warm_time else None,
            "warming_in_progress": self._warming_in_progress,
            "background_refresh_enabled": self.enable_background_refresh,
            "refresh_interval_minutes": self.refresh_interval.total_seconds() / 60,
        }

    def is_cache_warm(self) -> bool:
        """
        Check if cache has been warmed recently.

        Returns:
            True if cache was warmed within refresh interval
        """
        if self._last_warm_time is None:
            return False

        age = datetime.now() - self._last_warm_time
        return age < self.refresh_interval
