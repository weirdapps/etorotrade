"""
Base provider interface for Yahoo Finance data access.

This module defines the abstract base classes that all providers must implement,
ensuring a consistent interface regardless of the underlying data source.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ...core.errors import YFinanceError
from ...core.logging import get_logger


logger = get_logger(__name__)


class FinanceDataProvider(ABC):
    """
    Abstract base class for finance data providers.

    This interface defines the contract that all synchronous providers must implement.
    It provides a consistent interface for retrieving financial data from any source.
    """

    @abstractmethod
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker.

        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict containing stock information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    @abstractmethod
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing price data including current price, target price, and upside potential

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    @abstractmethod
    def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")

        Returns:
            DataFrame containing historical data

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    @abstractmethod
    def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the last two earnings dates for a stock.

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
        pass

    @abstractmethod
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing analyst ratings information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    @abstractmethod
    def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of dicts containing insider transaction information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    @abstractmethod
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching tickers with metadata

        Raises:
            YFinanceError: When an error occurs while searching
        """
        pass

    @abstractmethod
    def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers in a batch.

        Args:
            tickers: List of stock ticker symbols
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict mapping ticker symbols to their information dicts

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    def clear_cache(self) -> None:
        """
        Clear any cached data.
        """
        pass

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.

        Returns:
            Dict containing cache information
        """
        return {}


class AsyncFinanceDataProvider(ABC):
    """
    Abstract base class for asynchronous finance data providers.

    This interface defines the contract that all asynchronous providers must implement.
    It provides a consistent interface for retrieving financial data from any source.
    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")

        Returns:
            DataFrame containing historical data

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    @abstractmethod
    async def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the last two earnings dates for a stock.

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
        pass

    @abstractmethod
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing analyst ratings information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    @abstractmethod
    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of dicts containing insider transaction information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    @abstractmethod
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching tickers with metadata

        Raises:
            YFinanceError: When an error occurs while searching
        """
        pass

    @abstractmethod
    async def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers in a batch.

        Args:
            tickers: List of stock ticker symbols
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict mapping ticker symbols to their information dicts

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        pass

    def clear_cache(self) -> None:
        """
        Clear any cached data.
        """
        pass

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.

        Returns:
            Dict containing cache information
        """
        return {}
