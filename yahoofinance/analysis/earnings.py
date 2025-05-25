"""
Earnings analysis module.

This module provides functionality for analyzing earnings data,
including earnings dates, surprises, and trends.

The module includes both synchronous and asynchronous implementations
of earnings data analysis functionality, sharing common business logic
across both APIs through private helper methods like _calculate_trend_metrics.
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
from .base_analysis import BaseAnalysisService


logger = get_logger(__name__)


# The EarningsCalendar class below provides backward compatibility with the v1 API
class EarningsCalendar:
    """
    Class for retrieving and displaying upcoming earnings dates.
    """

    def __init__(self):
        """Initialize the EarningsCalendar."""
        # List of major stocks to monitor for earnings
        self.major_stocks = [
            # Technology
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "INTC",
            "CSCO",
            "ORCL",
            "IBM",
            # Communication Services
            "NFLX",
            "CMCSA",
            "T",
            "VZ",
            "DIS",
            # Consumer Discretionary
            "TSLA",
            "HD",
            "MCD",
            "NKE",
            "SBUX",
            # Financials
            "JPM",
            "BAC",
            "WFC",
            "C",
            "GS",
            # Healthcare
            "JNJ",
            "UNH",
            "PFE",
            "MRK",
            "ABT",
            # Industrials
            "BA",
            "CAT",
            "GE",
            "HON",
            "UPS",
            # Energy
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "EOG",
        ]

    def validate_date_format(self, date_str: str) -> bool:
        """
        Validate date format (YYYY-MM-DD).

        Args:
            date_str: Date string to validate

        Returns:
            True if valid, False otherwise
        """
        if not date_str:
            return False

        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def _format_market_cap(self, market_cap: Optional[float]) -> str:
        """
        Format market cap in billions (B) or trillions (T).

        Args:
            market_cap: Market cap value

        Returns:
            Formatted market cap string
        """
        if market_cap is None or market_cap <= 0:
            return "N/A"

        # Convert to billions
        billions = market_cap / 1_000_000_000

        # Check if it's in trillions range
        if billions >= 1000:
            trillions = billions / 1000

            # Format with appropriate precision
            if trillions >= 10:
                return f"${trillions:.1f}T"  # ≥ 10T: 1 decimal
            else:
                return f"${trillions:.2f}T"  # < 10T: 2 decimals

        # Format billions with appropriate precision
        if billions >= 100:
            return f"${int(billions)}B"  # ≥ 100B: 0 decimals
        elif billions >= 10:
            return f"${billions:.1f}B"  # ≥ 10B: 1 decimal
        else:
            return f"${billions:.2f}B"  # < 10B: 2 decimals

    def _format_eps(self, eps: Optional[float]) -> str:
        """
        Format EPS value with 2 decimal places.

        Args:
            eps: EPS value

        Returns:
            Formatted EPS string
        """
        if eps is None or pd.isna(eps):
            return "N/A"
        return f"{eps:.2f}"

    def get_trading_date(self, timestamp: pd.Timestamp) -> str:
        """
        Get trading date from timestamp, adjusting for after-market reporting.

        Args:
            timestamp: Timestamp of earnings release

        Returns:
            Trading date (YYYY-MM-DD)
        """
        # After 4 PM, the trading date is the next day
        if timestamp.hour >= 16:
            return (timestamp + timedelta(days=1)).strftime("%Y-%m-%d")

        return timestamp.strftime("%Y-%m-%d")

    def _process_earnings_row(
        self, ticker: str, date: pd.Timestamp, row: pd.Series, info: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Process a single earnings row.

        Args:
            ticker: Ticker symbol
            date: Earnings date timestamp
            row: Row data from earnings calendar
            info: Company info dictionary

        Returns:
            Processed earnings row as a dictionary
        """
        # Format market cap
        market_cap = info.get("marketCap")
        market_cap_formatted = self._format_market_cap(market_cap)

        # Format EPS estimate
        eps_estimate = row.get("EPS Estimate")
        eps_formatted = self._format_eps(eps_estimate)

        # Format the trading date
        trading_date = self.get_trading_date(date)

        return {
            "Symbol": ticker,
            "Market Cap": market_cap_formatted,
            "Date": trading_date,
            "EPS Est": eps_formatted,
        }

    def get_earnings_calendar(
        self, start_date: str, end_date: Optional[str] = None, tickers: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get earnings calendar for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to start_date + 7 days
            tickers: Optional list of tickers to filter by

        Returns:
            DataFrame with earnings calendar or None if error/no data
        """
        # Validate date format
        if not self.validate_date_format(start_date):
            logger.error(f"Invalid date format: {start_date}")
            return None

        # Set default end date if not provided
        if end_date is None:
            end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=7)).strftime(
                "%Y-%m-%d"
            )
        elif not self.validate_date_format(end_date):
            logger.error(f"Invalid date format: {end_date}")
            return None

        try:
            # Use the provided tickers or default to major stocks
            tickers_to_check = tickers if tickers else self.major_stocks

            # Start date and end date as datetime objects
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            # Import yfinance here to avoid circular imports
            import yfinance as yf

            # List to store processed earnings data
            earnings_data = []

            # Process each ticker
            for ticker in tickers_to_check:
                try:
                    # Get the ticker object
                    ticker_obj = yf.Ticker(ticker)

                    # Get earnings dates
                    earnings_dates = ticker_obj.earnings_dates

                    # Skip if no earnings dates available
                    if earnings_dates is None or earnings_dates.empty:
                        continue

                    # Get company info
                    info = ticker_obj.info

                    # Process each earnings date within the range
                    for date, row in earnings_dates.iterrows():
                        # Convert date to datetime for comparison
                        trading_date = self.get_trading_date(date)
                        trading_date_dt = datetime.strptime(trading_date, "%Y-%m-%d")

                        # Check if date is within range
                        if start <= trading_date_dt <= end:
                            # Process the earnings row
                            processed_row = self._process_earnings_row(ticker, date, row, info)
                            earnings_data.append(processed_row)

                except YFinanceError as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
                    continue

            # Return None if no earnings found
            if not earnings_data:
                return None

            # Create DataFrame from processed data
            df = pd.DataFrame(earnings_data)

            # Sort by date
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date")
                df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

            return df

        except YFinanceError as e:
            logger.error(f"Error fetching earnings calendar: {str(e)}")
            return None


def format_earnings_table(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """
    Format and print earnings calendar table.

    Args:
        df: DataFrame with earnings calendar
        start_date: Start date
        end_date: End date
    """
    if df is None or df.empty:
        return

    # Print header
    print(f"\nEarnings Calendar ({start_date} to {end_date})")

    try:
        # Use tabulate if available for nicer formatting
        from tabulate import tabulate

        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    except ImportError:
        # Fall back to simple formatting if tabulate is not available
        print("=" * 60)
        print(f"{'Symbol':<6} {'Market Cap':<10} {'Date':<12} {'EPS Est':<8}")
        print("-" * 60)

        # Print each row
        for _, row in df.iterrows():
            print(
                f"{row['Symbol']:<6} {row['Market Cap']:<10} {row['Date']:<12} {row['EPS Est']:<8}"
            )

        # Print footer
        print("=" * 60)
        print(f"Total: {len(df)} companies reporting earnings")


@dataclass
class EarningsData:
    """
    Container for earnings data.

    Attributes:
        next_date: Next earnings date (if available)
        previous_date: Previous earnings date (if available)
        earnings_history: List of previous earnings results
        eps_estimate: Latest EPS estimate
        eps_actual: Latest actual EPS
        eps_surprise: Latest EPS surprise (percentage)
        revenue_estimate: Latest revenue estimate
        revenue_actual: Latest actual revenue
        revenue_surprise: Latest revenue surprise (percentage)
    """

    next_date: Optional[str] = None
    previous_date: Optional[str] = None
    earnings_history: List[Dict[str, Any]] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    eps_surprise: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    revenue_surprise: Optional[float] = None

    def __post_init__(self):
        if self.earnings_history is None:
            self.earnings_history = []


class EarningsAnalyzer(BaseAnalysisService):
    """
    Service for retrieving and analyzing earnings data.

    This service uses a data provider to retrieve earnings data
    and provides methods for calculating earnings trends.

    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
    """

    def get_earnings_data(self, ticker: str) -> EarningsData:
        """
        Get earnings data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            EarningsData object containing earnings information

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        self._verify_sync_provider("get_earnings_data_async")

        try:
            # Fetch earnings dates
            earnings_dates = self.provider.get_earnings_dates(ticker)

            # Fetch earnings history (if available)
            earnings_history = self.provider.get_earnings_history(ticker)

            # Process the data into EarningsData object
            return self._process_earnings_data(earnings_dates, earnings_history)

        except YFinanceError as e:
            logger.error(f"Error fetching earnings data for {ticker}: {str(e)}")
            return EarningsData()

    async def get_earnings_data_async(self, ticker: str) -> EarningsData:
        """
        Get earnings data for a ticker asynchronously.

        Args:
            ticker: Stock ticker symbol

        Returns:
            EarningsData object containing earnings information

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        self._verify_async_provider("get_earnings_data")

        try:
            # Fetch earnings dates asynchronously
            earnings_dates = await self.provider.get_earnings_dates(ticker)

            # Fetch earnings history (if available) asynchronously
            earnings_history = await self.provider.get_earnings_history(ticker)

            # Process the data into EarningsData object
            return self._process_earnings_data(earnings_dates, earnings_history)

        except YFinanceError as e:
            logger.error(f"Error fetching earnings data for {ticker}: {str(e)}")
            return EarningsData()

    def get_earnings_batch(self, tickers: List[str]) -> Dict[str, EarningsData]:
        """
        Get earnings data for multiple tickers.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker symbols to EarningsData objects

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        self._verify_sync_provider("get_earnings_batch_async")

        results = {}

        for ticker in tickers:
            try:
                results[ticker] = self.get_earnings_data(ticker)
            except YFinanceError as e:
                logger.error(f"Error fetching earnings data for {ticker}: {str(e)}")
                results[ticker] = EarningsData()

        return results

    async def get_earnings_batch_async(self, tickers: List[str]) -> Dict[str, EarningsData]:
        """
        Get earnings data for multiple tickers asynchronously.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker symbols to EarningsData objects

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        self._verify_async_provider("get_earnings_batch")

        import asyncio

        # Create tasks for all tickers
        tasks = [self.get_earnings_data_async(ticker) for ticker in tickers]

        # Wait for all tasks to complete
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = {}
        for ticker, result in zip(tickers, results_list):
            if isinstance(result, Exception):
                logger.error(f"Error fetching earnings data for {ticker}: {str(result)}")
                results[ticker] = EarningsData()
            else:
                results[ticker] = result

        return results

    def _process_earnings_data(
        self,
        earnings_dates: Optional[Tuple[str, str]],
        earnings_history: Optional[List[Dict[str, Any]]],
    ) -> EarningsData:
        """
        Process earnings data into EarningsData object.

        Args:
            earnings_dates: Tuple containing (next_date, previous_date)
            earnings_history: List of earnings history dictionaries

        Returns:
            EarningsData object with processed earnings data
        """
        # Initialize basic earnings data
        earnings_data = EarningsData()

        # Process earnings dates
        if earnings_dates:
            earnings_data.next_date = earnings_dates[0]
            earnings_data.previous_date = earnings_dates[1] if len(earnings_dates) > 1 else None

        # Process earnings history
        if earnings_history and len(earnings_history) > 0:
            earnings_data.earnings_history = earnings_history

            # Get the latest earnings report
            latest_report = earnings_history[
                0
            ]  # We already checked that earnings_history is not empty

            # Extract EPS data
            earnings_data.eps_estimate = latest_report.get("eps_estimate")
            earnings_data.eps_actual = latest_report.get("eps_actual")
            earnings_data.eps_surprise = latest_report.get("eps_surprise_pct")

            # Extract revenue data
            earnings_data.revenue_estimate = latest_report.get("revenue_estimate")
            earnings_data.revenue_actual = latest_report.get("revenue_actual")
            earnings_data.revenue_surprise = latest_report.get("revenue_surprise_pct")

        return earnings_data

    def _calculate_trend_metrics(
        self, earnings_data: EarningsData, ticker: str, quarters: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate earnings trend metrics from earnings data.
        This is a helper method used by both sync and async trend calculation methods.

        Args:
            earnings_data: The earnings data to analyze
            ticker: Stock ticker symbol (for logging)
            quarters: Number of quarters to analyze

        Returns:
            Dictionary with earnings trend metrics or None if insufficient data
        """
        # Check if we have enough history
        if not earnings_data.earnings_history or len(earnings_data.earnings_history) < quarters:
            logger.info(f"Insufficient earnings history for {ticker} to calculate trend")
            return None

        # Limit to specified number of quarters
        history = earnings_data.earnings_history[:quarters]

        # Extract EPS values
        eps_values = [
            report.get("eps_actual") for report in history if report.get("eps_actual") is not None
        ]

        # Extract revenue values
        revenue_values = [
            report.get("revenue_actual")
            for report in history
            if report.get("revenue_actual") is not None
        ]

        # Calculate trends
        if len(eps_values) < 2 or len(revenue_values) < 2:
            logger.info(f"Insufficient valid data points for {ticker} to calculate trend")
            return None

        # Calculate growth rates
        eps_growth = [
            (
                (eps_values[i] - eps_values[i + 1]) / abs(eps_values[i + 1]) * 100
                if eps_values[i + 1] != 0
                else 0
            )
            for i in range(len(eps_values) - 1)
        ]

        revenue_growth = [
            (
                (revenue_values[i] - revenue_values[i + 1]) / revenue_values[i + 1] * 100
                if revenue_values[i + 1] != 0
                else 0
            )
            for i in range(len(revenue_values) - 1)
        ]

        # Calculate averages
        avg_eps_growth = sum(eps_growth) / len(eps_growth) if eps_growth else None
        avg_revenue_growth = sum(revenue_growth) / len(revenue_growth) if revenue_growth else None

        # Calculate consistency (how many quarters showed growth)
        eps_beat_count = sum(1 for x in eps_growth if x > 0)
        revenue_beat_count = sum(1 for x in revenue_growth if x > 0)

        return {
            "eps_values": eps_values,
            "revenue_values": revenue_values,
            "eps_growth": eps_growth,
            "revenue_growth": revenue_growth,
            "avg_eps_growth": avg_eps_growth,
            "avg_revenue_growth": avg_revenue_growth,
            "eps_beat_count": eps_beat_count,
            "revenue_beat_count": revenue_beat_count,
            "total_quarters": len(eps_growth) + 1,
            "eps_consistency": eps_beat_count / len(eps_growth) * 100 if eps_growth else None,
            "revenue_consistency": (
                revenue_beat_count / len(revenue_growth) * 100 if revenue_growth else None
            ),
        }

    def calculate_earnings_trend(self, ticker: str, quarters: int = 4) -> Optional[Dict[str, Any]]:
        """
        Calculate earnings trend over multiple quarters.

        Args:
            ticker: Stock ticker symbol
            quarters: Number of quarters to analyze

        Returns:
            Dictionary with earnings trend metrics or None if insufficient data

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        self._verify_sync_provider("calculate_earnings_trend_async")

        try:
            # Get earnings data
            earnings_data = self.get_earnings_data(ticker)

            # Calculate trend metrics using the shared helper method
            return self._calculate_trend_metrics(earnings_data, ticker, quarters)

        except YFinanceError as e:
            logger.error(f"Error calculating earnings trend for {ticker}: {str(e)}")
            return None

    async def calculate_earnings_trend_async(
        self, ticker: str, quarters: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate earnings trend over multiple quarters asynchronously.

        Args:
            ticker: Stock ticker symbol
            quarters: Number of quarters to analyze

        Returns:
            Dictionary with earnings trend metrics or None if insufficient data

        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        self._verify_async_provider("calculate_earnings_trend")

        try:
            # Get earnings data asynchronously
            earnings_data = await self.get_earnings_data_async(ticker)

            # Calculate trend metrics using the shared helper method
            return self._calculate_trend_metrics(earnings_data, ticker, quarters)

        except YFinanceError as e:
            logger.error(f"Error calculating earnings trend for {ticker}: {str(e)}")
            return None
