"""
Async Yahoo Query provider implementation.

This module implements the AsyncFinanceDataProvider interface using the yahooquery library
with asyncio for concurrent processing, enabling efficient asynchronous data fetching.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
from yahooquery import Ticker

from ...core.config import CACHE_CONFIG, COLUMN_NAMES, POSITIVE_GRADES
from ...core.errors import APIError, RateLimitError, ValidationError, YFinanceError
from ...core.logging import get_logger
from ...utils.async_utils.enhanced import async_rate_limited
from ...utils.market.ticker_utils import is_us_ticker
from .base_provider import AsyncFinanceDataProvider
from .yahoo_finance_base import YahooFinanceBaseProvider
from .yahooquery_provider import YahooQueryProvider


logger = get_logger(__name__)


class AsyncYahooQueryProvider(YahooFinanceBaseProvider, AsyncFinanceDataProvider):
    """
    Async Yahoo Query data provider implementation.

    This provider wraps the yahooquery library with proper rate limiting,
    error handling, and caching to provide reliable access to financial data asynchronously.

    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
        _ticker_cache: Cache of ticker information to avoid repeated fetches
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, **kwargs):
        """
        Initialize the Async Yahoo Query provider.

        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries
            **kwargs: Additional keyword arguments (for factory compatibility)
        """
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        # Create a sync provider to reuse methods
        self._sync_provider = YahooQueryProvider(max_retries=max_retries, retry_delay=retry_delay, **kwargs)

    async def _handle_delay(self, delay: float):
        """
        Handle delaying execution for retry logic using asyncio.sleep().

        Args:
            delay: Time in seconds to delay
        """
        await asyncio.sleep(delay)

    async def _run_in_executor(self, func, *args, **kwargs):
        """
        Run a synchronous function in an executor to make it non-blocking.

        Args:
            func: Synchronous function to run
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function call
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # Temporarily remove decorator for debugging
    # @async_rate_limited
    async def get_ticker_info(
        self, ticker: str, skip_insider_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker asynchronously.

        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict containing stock information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Run the synchronous version in an executor to make it non-blocking
        return await self._run_in_executor(
            self._sync_provider.get_ticker_info, ticker, skip_insider_metrics
        )

    @async_rate_limited
    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker asynchronously.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing price data including current price, target price, and upside potential

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Run the synchronous version in an executor to make it non-blocking
        return await self._run_in_executor(self._sync_provider.get_price_data, ticker)

    @async_rate_limited
    async def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data for a ticker asynchronously.

        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")

        Returns:
            DataFrame containing historical data

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Run the synchronous version in an executor to make it non-blocking
        return await self._run_in_executor(
            self._sync_provider.get_historical_data, ticker, period, interval
        )

    @async_rate_limited
    async def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the last two earnings dates for a stock asynchronously.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple containing:
                - most_recent_date: The most recent earnings date in YYYY-MM-DD format
                - previous_date: The second most recent earnings date in YYYY-MM-DD format
                Both values will be None if no earnings dates are found

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Run the synchronous version in an executor to make it non-blocking
        return await self._run_in_executor(self._sync_provider.get_earnings_dates, ticker)

    @async_rate_limited
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker asynchronously.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing analyst ratings information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Run the synchronous version in an executor to make it non-blocking
        return await self._run_in_executor(self._sync_provider.get_analyst_ratings, ticker)

    @async_rate_limited
    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a ticker asynchronously.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of dicts containing insider transaction information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Run the synchronous version in an executor to make it non-blocking
        return await self._run_in_executor(self._sync_provider.get_insider_transactions, ticker)

    @async_rate_limited
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query asynchronously.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching tickers with metadata

        Raises:
            YFinanceError: When an error occurs while searching
        """
        # Run the synchronous version in an executor to make it non-blocking
        return await self._run_in_executor(self._sync_provider.search_tickers, query, limit)

    @async_rate_limited
    async def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers in a batch asynchronously.

        Args:
            tickers: List of stock ticker symbols
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict mapping ticker symbols to their information dicts

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Run the synchronous version in an executor to make it non-blocking
        # This is more efficient than splitting the batch into individual async calls
        return await self._run_in_executor(
            self._sync_provider.batch_get_ticker_info, tickers, skip_insider_metrics
        )
