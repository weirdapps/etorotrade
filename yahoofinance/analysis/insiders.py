"""
Insider trading analysis module.

This module provides functionality for analyzing insider transactions,
including buys, sells, and net activity.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..api import AsyncFinanceDataProvider, FinanceDataProvider, get_provider
from ..core.errors import ValidationError, YFinanceError
from ..core.logging import get_logger
from ..utils.market import is_us_ticker


logger = get_logger(__name__)


@dataclass
class InsiderTransaction:
    """
    Container for a single insider transaction.

    Attributes:
        name: Name of the insider
        title: Title or position of the insider
        date: Date of the transaction
        transaction_type: Type of transaction (e.g., 'Buy', 'Sell')
        shares: Number of shares involved
        value: Dollar value of the transaction
        share_price: Price per share
    """

    name: str
    title: Optional[str] = None
    date: Optional[str] = None
    transaction_type: Optional[str] = None
    shares: Optional[int] = None
    value: Optional[float] = None
    share_price: Optional[float] = None


@dataclass
class InsiderSummary:
    """
    Summary of insider trading activity.

    Attributes:
        transactions: List of insider transactions
        buy_count: Number of buy transactions
        sell_count: Number of sell transactions
        total_buy_value: Total value of buy transactions
        total_sell_value: Total value of sell transactions
        net_value: Net value (buy - sell)
        net_share_count: Net shares (buy - sell)
        average_buy_price: Average price per share for buys
        average_sell_price: Average price per share for sells
    """

    transactions: Optional[List[InsiderTransaction]] = None
    buy_count: int = 0
    sell_count: int = 0
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    net_value: float = 0.0
    net_share_count: int = 0
    average_buy_price: Optional[float] = None
    average_sell_price: Optional[float] = None

    def __post_init__(self):
        if self.transactions is None:
            self.transactions = []


class InsiderAnalyzer:
    """
    Service for retrieving and analyzing insider trading data.

    This service uses a data provider to retrieve insider transactions
    and provides methods for calculating insider trading summaries.

    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
    """

    def __init__(
        self, provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None
    ):
        """
        Initialize the InsiderAnalyzer.

        Args:
            provider: Data provider (sync or async), if None, a default provider is created
        """
        self.provider = provider if provider is not None else get_provider()

        # Check if the provider is async
        self.is_async = (
            hasattr(self.provider, "get_insider_transactions")
            and callable(self.provider.get_insider_transactions)
            and hasattr(self.provider.get_insider_transactions, "__await__")
        )

    def get_transactions(self, ticker: str, days: int = 90) -> InsiderSummary:
        """
        Get insider transactions for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            InsiderSummary object containing transaction data and summary metrics

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError(
                "Cannot use sync method with async provider. Use get_transactions_async instead."
            )

        try:
            # Skip API call for non-US tickers to avoid unnecessary errors
            if not is_us_ticker(ticker):
                logger.info(f"Skipping insider transactions for non-US ticker: {ticker}")
                return InsiderSummary()

            # Fetch insider transactions data
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            transactions_data = self.provider.get_insider_transactions(ticker, start_date)

            # Process the data into InsiderSummary object
            return self._process_transactions_data(transactions_data)

        except YFinanceError as e:
            logger.error(f"Error fetching insider transactions for {ticker}: {str(e)}")
            return InsiderSummary()

    async def get_transactions_async(self, ticker: str, days: int = 90) -> InsiderSummary:
        """
        Get insider transactions for a ticker asynchronously.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            InsiderSummary object containing transaction data and summary metrics

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError(
                "Cannot use async method with sync provider. Use get_transactions instead."
            )

        try:
            # Skip API call for non-US tickers to avoid unnecessary errors
            if not is_us_ticker(ticker):
                logger.info(f"Skipping insider transactions for non-US ticker: {ticker}")
                return InsiderSummary()

            # Fetch insider transactions data asynchronously
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            transactions_data = await self.provider.get_insider_transactions(ticker, start_date)

            # Process the data into InsiderSummary object
            return self._process_transactions_data(transactions_data)

        except YFinanceError as e:
            logger.error(f"Error fetching insider transactions for {ticker}: {str(e)}")
            return InsiderSummary()

    def get_transactions_batch(
        self, tickers: List[str], days: int = 90
    ) -> Dict[str, InsiderSummary]:
        """
        Get insider transactions for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            days: Number of days to look back

        Returns:
            Dictionary mapping ticker symbols to InsiderSummary objects

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError(
                "Cannot use sync method with async provider. Use get_transactions_batch_async instead."
            )

        # Filter out non-US tickers first to avoid unnecessary API calls
        us_tickers = [ticker for ticker in tickers if is_us_ticker(ticker)]

        results = {}

        # Get data for US tickers
        for ticker in us_tickers:
            try:
                results[ticker] = self.get_transactions(ticker, days)
            except YFinanceError as e:
                logger.error(f"Error fetching insider transactions for {ticker}: {str(e)}")
                results[ticker] = InsiderSummary()

        # Add empty results for non-US tickers
        for ticker in tickers:
            if ticker not in results:
                results[ticker] = InsiderSummary()

        return results

    async def get_transactions_batch_async(
        self, tickers: List[str], days: int = 90
    ) -> Dict[str, InsiderSummary]:
        """
        Get insider transactions for multiple tickers asynchronously.

        Args:
            tickers: List of stock ticker symbols
            days: Number of days to look back

        Returns:
            Dictionary mapping ticker symbols to InsiderSummary objects

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError(
                "Cannot use async method with sync provider. Use get_transactions_batch instead."
            )

        import asyncio

        # Filter out non-US tickers first to avoid unnecessary API calls
        us_tickers = [ticker for ticker in tickers if is_us_ticker(ticker)]

        # Create tasks for US tickers
        tasks = [self.get_transactions_async(ticker, days) for ticker in us_tickers]

        # Wait for all tasks to complete
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = {}
        for ticker, result in zip(us_tickers, results_list):
            if isinstance(result, Exception):
                logger.error(f"Error fetching insider transactions for {ticker}: {str(result)}")
                results[ticker] = InsiderSummary()
            else:
                results[ticker] = result

        # Add empty results for non-US tickers
        for ticker in tickers:
            if ticker not in results:
                results[ticker] = InsiderSummary()

        return results

    def _process_transactions_data(self, transactions_data: List[Dict[str, Any]]) -> InsiderSummary:
        """
        Process insider transactions data into InsiderSummary object.

        Args:
            transactions_data: List of dictionaries containing transaction data

        Returns:
            InsiderSummary object with processed transactions and summary metrics
        """
        if not transactions_data:
            return InsiderSummary()

        # Convert raw transactions to InsiderTransaction objects
        transactions = []
        buy_values = []
        sell_values = []
        buy_share_count = 0
        sell_share_count = 0
        total_buy_value = 0.0
        total_sell_value = 0.0

        for trans in transactions_data:
            transaction = InsiderTransaction(
                name=trans.get("name", "Unknown"),
                title=trans.get("title"),
                date=trans.get("date"),
                transaction_type=trans.get("transaction_type"),
                shares=trans.get("shares"),
                value=trans.get("value"),
                share_price=trans.get("share_price"),
            )
            transactions.append(transaction)

            # Categorize as buy or sell and track metrics
            transaction_type = trans.get("transaction_type", "").lower()
            shares = trans.get("shares", 0)
            value = trans.get("value", 0.0)
            share_price = trans.get("share_price")

            if "buy" in transaction_type:
                buy_values.append(value)
                buy_share_count += shares
                total_buy_value += value
                if share_price:
                    buy_values.append(share_price)
            elif "sell" in transaction_type:
                sell_values.append(value)
                sell_share_count += shares
                total_sell_value += value
                if share_price:
                    sell_values.append(share_price)

        # Calculate summary metrics
        average_buy_price = sum(buy_values) / len(buy_values) if buy_values else None
        average_sell_price = sum(sell_values) / len(sell_values) if sell_values else None

        return InsiderSummary(
            transactions=transactions,
            buy_count=len(buy_values),
            sell_count=len(sell_values),
            total_buy_value=total_buy_value,
            total_sell_value=total_sell_value,
            net_value=total_buy_value - total_sell_value,
            net_share_count=buy_share_count - sell_share_count,
            average_buy_price=average_buy_price,
            average_sell_price=average_sell_price,
        )

    def _calculate_sentiment_metrics(self, summary: InsiderSummary, days: int) -> Dict[str, Any]:
        """
        Calculate sentiment metrics from an InsiderSummary.

        Args:
            summary: InsiderSummary object
            days: Number of days looked back

        Returns:
            Dictionary with sentiment metrics
        """
        # Calculate sentiment metrics
        sentiment = "NEUTRAL"
        confidence = "LOW"

        # Determine sentiment based on net value and transaction counts
        if summary.net_value > 0 and summary.buy_count > summary.sell_count:
            sentiment = "BULLISH"
        elif summary.net_value < 0 and summary.sell_count > summary.buy_count:
            sentiment = "BEARISH"

        # Determine confidence based on transaction counts and size
        total_transactions = summary.buy_count + summary.sell_count
        if total_transactions > 5:
            confidence = "HIGH" if abs(summary.net_value) > 1000000 else "MEDIUM"
        elif total_transactions > 2:
            confidence = "MEDIUM"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "buy_count": summary.buy_count,
            "sell_count": summary.sell_count,
            "net_value": summary.net_value,
            "net_share_count": summary.net_share_count,
            "recent_transactions": len(summary.transactions),
            "lookback_days": days,
        }

    def analyze_insider_sentiment(self, ticker: str, days: int = 90) -> Dict[str, Any]:
        """
        Analyze insider sentiment based on recent transactions.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            Dictionary with insider sentiment metrics

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError(
                "Cannot use sync method with async provider. Use analyze_insider_sentiment_async instead."
            )

        try:
            # Get insider transactions
            summary = self.get_transactions(ticker, days)

            # Calculate sentiment metrics using the shared method
            return self._calculate_sentiment_metrics(summary, days)

        except YFinanceError as e:
            logger.error(f"Error analyzing insider sentiment for {ticker}: {str(e)}")
            return {"sentiment": "UNKNOWN", "confidence": "NONE", "error": str(e)}

    async def analyze_insider_sentiment_async(self, ticker: str, days: int = 90) -> Dict[str, Any]:
        """
        Analyze insider sentiment based on recent transactions asynchronously.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            Dictionary with insider sentiment metrics

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError(
                "Cannot use async method with sync provider. Use analyze_insider_sentiment instead."
            )

        try:
            # Get insider transactions asynchronously
            summary = await self.get_transactions_async(ticker, days)

            # Calculate sentiment metrics using the shared method
            return self._calculate_sentiment_metrics(summary, days)

        except YFinanceError as e:
            logger.error(f"Error analyzing insider sentiment for {ticker}: {str(e)}")
            return {"sentiment": "UNKNOWN", "confidence": "NONE", "error": str(e)}
