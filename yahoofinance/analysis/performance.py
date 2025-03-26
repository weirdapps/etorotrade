"""
Performance tracking module for financial data.

This module provides functions for tracking market index performance
and portfolio performance from external sources, including web scraping
for additional performance metrics not available through the API.
"""

import logging
import asyncio
import pytz
import os
import json
import aiohttp
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path

from ..api import get_provider, FinanceDataProvider, AsyncFinanceDataProvider
from ..core.errors import YFinanceError, ValidationError, NetworkError
from ..core.config import FILE_PATHS, PATHS
from ..presentation.html import HTMLGenerator, FormatUtils
from ..utils.network.circuit_breaker import circuit_protected, async_circuit_protected

logger = logging.getLogger(__name__)

# Define the indices to track with carets (^)
INDICES = {
    'DJI30': '^DJI',
    'SP500': '^GSPC',
    'NQ100': '^NDX',
    'VIX': '^VIX'
}

# Timezone for Athens, Greece
athens_tz = pytz.timezone('Europe/Athens')

@dataclass
class IndexPerformance:
    """
    Represents performance metrics for a market index.
    
    Attributes:
        index_name: Name of the index
        ticker: Ticker symbol
        current_value: Current value of the index
        previous_value: Previous value of the index
        change_percent: Percentage change
        start_date: Start date for the comparison period
        end_date: End date for the comparison period
        period_type: Type of period (weekly, monthly, etc.)
    """
    
    index_name: str
    ticker: str
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    change_percent: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    period_type: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HTML generation."""
        result = {
            "index": self.index_name,
            "ticker": self.ticker,
            "current": self.current_value,
            "previous": self.previous_value,
            "change": self.change_percent
        }
        
        # Add formatted dates if available
        if self.start_date:
            result["start_date"] = self.start_date.strftime("%Y-%m-%d")
        if self.end_date:
            result["end_date"] = self.end_date.strftime("%Y-%m-%d")
            
        return result


@dataclass
class PortfolioPerformance:
    """
    Represents portfolio performance metrics.
    
    Attributes:
        this_month: Performance for the current month (%)
        year_to_date: Performance year to date (%)
        two_years: Performance over two years (%)
        beta: Portfolio beta
        sharpe: Sharpe ratio
        alpha: Alpha (Jensen's Alpha)
        sortino: Sortino ratio
        cash: Cash percentage
        source: Source of the data
        last_updated: When the data was last updated
    """
    
    this_month: Optional[float] = None
    year_to_date: Optional[float] = None
    two_years: Optional[float] = None
    beta: Optional[float] = None
    sharpe: Optional[float] = None
    alpha: Optional[float] = None
    sortino: Optional[float] = None
    cash: Optional[float] = None
    source: str = "unknown"
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to dictionary for HTML generation."""
        result = {}
        
        # Performance metrics
        if self.this_month is not None:
            result["This Month"] = {
                "value": self.this_month,
                "label": "This Month",
                "is_percentage": True
            }
            
        if self.year_to_date is not None:
            result["Year To Date"] = {
                "value": self.year_to_date,
                "label": "Year To Date",
                "is_percentage": True
            }
            
        if self.two_years is not None:
            result["2 Years"] = {
                "value": self.two_years,
                "label": "2 Years",
                "is_percentage": True
            }
        
        # Risk metrics
        if self.beta is not None:
            result["Beta"] = {
                "value": self.beta,
                "label": "Beta",
                "is_percentage": False
            }
            
        if self.sharpe is not None:
            result["Sharpe"] = {
                "value": self.sharpe,
                "label": "Sharpe",
                "is_percentage": False
            }
            
        if self.cash is not None:
            result["Cash"] = {
                "value": self.cash,
                "label": "Cash",
                "is_percentage": True
            }
            
        if self.alpha is not None:
            result["Alpha"] = {
                "value": self.alpha,
                "label": "Jensen's Alpha",
                "is_percentage": False
            }
            
        if self.sortino is not None:
            result["Sortino"] = {
                "value": self.sortino,
                "label": "Sortino",
                "is_percentage": False
            }
            
        return result


class PerformanceTracker:
    """
    Tracks performance of market indices and portfolios.
    
    This class provides methods for tracking market indices and portfolio
    performance, including web scraping for external performance data.
    
    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
        html_generator: HTML generator for creating dashboards
    """
    
    def __init__(
        self,
        provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the PerformanceTracker.
        
        Args:
            provider: Data provider (sync or async), if None, a default provider is created
            output_dir: Directory for output files (defaults to config)
        """
        self.provider = provider if provider is not None else get_provider()
        self.output_dir = output_dir or PATHS["OUTPUT_DIR"]
        self.html_generator = HTMLGenerator(output_dir=self.output_dir)
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if the provider is async
        self.is_async = hasattr(self.provider, 'batch_get_ticker_info') and \
                        callable(self.provider.batch_get_ticker_info) and \
                        hasattr(self.provider.batch_get_ticker_info, '__await__')
    
    @staticmethod
    def calculate_weekly_dates() -> Tuple[datetime, datetime]:
        """
        Calculate last Friday and the previous Friday.
        
        Returns:
            Tuple of (previous_friday, last_friday)
        """
        today = datetime.today()
        # Calculate last Friday
        days_since_friday = (today.weekday() - 4) % 7
        last_friday = today - timedelta(days=days_since_friday)
        # Calculate previous Friday
        previous_friday = last_friday - timedelta(days=7)
        return previous_friday, last_friday
    
    @staticmethod
    def calculate_monthly_dates() -> Tuple[datetime, datetime]:
        """
        Calculate last business days of previous and current month.
        
        Returns:
            Tuple of (previous_month_end, last_month_end)
        """
        today = datetime.today()
        # Get the last day of the previous month
        last_month = today.replace(day=1) - timedelta(days=1)
        last_month_end = last_month.date()
        # Get the last day of the previous previous month
        previous_month = last_month.replace(day=1) - timedelta(days=1)
        previous_month_end = previous_month.date()
        return previous_month.replace(hour=0, minute=0, second=0, microsecond=0), \
               last_month.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def get_previous_trading_day_close(self, ticker: str, date: datetime) -> Tuple[float, datetime]:
        """
        Get the closing price for the last trading day before the given date.
        
        Args:
            ticker: Ticker symbol
            date: Date to find the closest trading day before
            
        Returns:
            Tuple of (close_price, actual_date)
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_previous_trading_day_close_async instead.")
        
        # Loop to handle potential empty data
        attempts = 0
        while attempts < 3:  # Limit attempts to avoid infinite loop
            try:
                # Use yfinance directly for market indices
                import yfinance as yf
                
                # Calculate date range
                start_date = date - timedelta(days=7 + attempts*3)  # Move back further on each attempt
                end_date = date + timedelta(days=1)
                
                # Download data
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                # Check if we have data
                if not data.empty:
                    # Get the last close price
                    close_price = float(data['Close'].iloc[-1])  # Explicitly convert to float
                    close_date = data.index[-1].to_pydatetime()  # Convert to Python datetime
                    return close_price, close_date
                
                # Increment attempts
                attempts += 1
                
            except Exception as e:
                logger.error(f"Error getting previous trading day close for {ticker}: {str(e)}")
                raise YFinanceError(f"Failed to get previous trading day close: {str(e)}")
    
    async def get_previous_trading_day_close_async(self, ticker: str, date: datetime) -> Tuple[float, datetime]:
        """
        Get the closing price for the last trading day before the given date asynchronously.
        
        Args:
            ticker: Ticker symbol
            date: Date to find the closest trading day before
            
        Returns:
            Tuple of (close_price, actual_date)
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_previous_trading_day_close instead.")
        
        # Loop to handle potential empty data
        attempts = 0
        while attempts < 3:  # Limit attempts to avoid infinite loop
            try:
                # Use yfinance directly with asyncio.to_thread
                import asyncio
                import yfinance as yf
                
                # Calculate date range
                start_date = date - timedelta(days=7 + attempts*3)  # Move back further on each attempt
                end_date = date + timedelta(days=1)
                
                # Use asyncio.to_thread to run yf.download in a separate thread
                data = await asyncio.to_thread(
                    yf.download, 
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    progress=False
                )
                
                # Check if we have data
                if not data.empty:
                    # Convert to primitive types
                    close_price = float(data['Close'].iloc[-1])  # Explicitly convert to float
                    close_date = data.index[-1].to_pydatetime()  # Convert to Python datetime
                    return close_price, close_date
                
                # Increment attempts
                attempts += 1
                
            except Exception as e:
                logger.error(f"Error getting previous trading day close async for {ticker}: {str(e)}")
                raise YFinanceError(f"Failed to get previous trading day close async: {str(e)}")
    
    def get_index_performance(self, period_type: str = "weekly") -> List[IndexPerformance]:
        """
        Get performance of market indices for the specified period.
        
        Args:
            period_type: 'weekly' or 'monthly'
            
        Returns:
            List of IndexPerformance objects
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_index_performance_async instead.")
        
        # Calculate dates based on period type
        if period_type.lower() == "weekly":
            start_date, end_date = self.calculate_weekly_dates()
        elif period_type.lower() == "monthly":
            start_date, end_date = self.calculate_monthly_dates()
        else:
            raise ValueError(f"Invalid period_type: {period_type}. Must be 'weekly' or 'monthly'.")
        
        # Get performance for each index
        performances = []
        for name, ticker in INDICES.items():
            try:
                # Get closing prices
                previous_price, previous_date = self.get_previous_trading_day_close(ticker, start_date)
                current_price, current_date = self.get_previous_trading_day_close(ticker, end_date)
                
                # Calculate change
                change_percent = ((current_price - previous_price) / previous_price) * 100
                
                # Create performance object
                performance = IndexPerformance(
                    index_name=name,
                    ticker=ticker,
                    current_value=current_price,
                    previous_value=previous_price,
                    change_percent=change_percent,
                    start_date=previous_date,
                    end_date=current_date,
                    period_type=period_type
                )
                
                performances.append(performance)
                
            except Exception as e:
                logger.error(f"Error getting {period_type} performance for {name} ({ticker}): {str(e)}")
                # Still include the index with None values
                performances.append(IndexPerformance(
                    index_name=name,
                    ticker=ticker,
                    period_type=period_type
                ))
        
        return performances
    
    async def get_index_performance_async(self, period_type: str = "weekly") -> List[IndexPerformance]:
        """
        Get performance of market indices for the specified period asynchronously.
        
        Args:
            period_type: 'weekly' or 'monthly'
            
        Returns:
            List of IndexPerformance objects
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_index_performance instead.")
        
        # Calculate dates based on period type
        if period_type.lower() == "weekly":
            start_date, end_date = self.calculate_weekly_dates()
        elif period_type.lower() == "monthly":
            start_date, end_date = self.calculate_monthly_dates()
        else:
            raise ValueError(f"Invalid period_type: {period_type}. Must be 'weekly' or 'monthly'.")
        
        # Create tasks for getting each index's performance
        tasks = []
        for name, ticker in INDICES.items():
            tasks.append(self._get_index_performance_single_async(name, ticker, start_date, end_date, period_type))
        
        # Run tasks concurrently
        performances = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and replace with None values
        result = []
        for item in performances:
            if isinstance(item, Exception):
                logger.error(f"Error in index performance task: {str(item)}")
                continue
            result.append(item)
            
        return result
    
    async def _get_index_performance_single_async(
        self,
        name: str,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        period_type: str
    ) -> IndexPerformance:
        """
        Get performance for a single index asynchronously.
        
        Args:
            name: Index name
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            period_type: 'weekly' or 'monthly'
            
        Returns:
            IndexPerformance object
        """
        try:
            # Get closing prices
            previous_price, previous_date = await self.get_previous_trading_day_close_async(ticker, start_date)
            current_price, current_date = await self.get_previous_trading_day_close_async(ticker, end_date)
            
            # Calculate change
            change_percent = ((current_price - previous_price) / previous_price) * 100
            
            # Create performance object
            return IndexPerformance(
                index_name=name,
                ticker=ticker,
                current_value=current_price,
                previous_value=previous_price,
                change_percent=change_percent,
                start_date=previous_date,
                end_date=current_date,
                period_type=period_type
            )
            
        except Exception as e:
            logger.error(f"Error getting {period_type} performance for {name} ({ticker}): {str(e)}")
            # Return index with None values on error
            return IndexPerformance(
                index_name=name,
                ticker=ticker,
                period_type=period_type
            )
    
    @staticmethod
    def _format_percentage_value(value: str) -> float:
        """Format a percentage value as float."""
        try:
            # Remove % symbol but keep signs
            clean_value = value.replace('%', '').strip()
            return float(clean_value)
        except ValueError:
            return None
    
    @circuit_protected("web_scraping")
    def get_portfolio_performance_web(self, url: str = "https://bullaware.com/etoro/plessas") -> PortfolioPerformance:
        """
        Get portfolio performance data from a web source.
        
        Args:
            url: URL to scrape for portfolio performance data
            
        Returns:
            PortfolioPerformance object
        """
        try:
            # Fetch and parse HTML content
            soup = self._get_soup(url)
            
            # Extract data
            performance = PortfolioPerformance(source=url, last_updated=datetime.now())
            
            # Extract summary metrics (TODAY, MTD, YTD, 2YR)
            summary_data = self._extract_summary_data(soup)
            
            # Map the scraped data fields to performance fields
            field_mapping = {
                'This Month': 'this_month',
                'MTD': 'this_month',
                'Today': 'this_month',
                'Year To Date': 'year_to_date',
                'YTD': 'year_to_date',
                '2 Years': 'two_years',
                '2YR': 'two_years',
                'Beta': 'beta',
                "Jensen's Alpha": 'alpha',
                'Alpha': 'alpha',
                'Sharpe': 'sharpe',
                'Sharpe Ratio': 'sharpe',
                'Sortino': 'sortino',
                'Sortino Ratio': 'sortino',
                'Cash': 'cash'
            }
            
            # Process each data item
            for key, value in summary_data.items():
                # Find matching field in performance object
                for pattern, field_name in field_mapping.items():
                    if pattern in key:
                        # Parse the value
                        if isinstance(value, str) and '%' in value:
                            # Convert percentage string to float
                            parsed_value = self._format_percentage_value(value)
                            if parsed_value is not None:
                                setattr(performance, field_name, parsed_value)
                        else:
                            # Try to convert to float
                            try:
                                parsed_value = float(value)
                                setattr(performance, field_name, parsed_value)
                            except (ValueError, TypeError):
                                # Keep as string if conversion fails
                                setattr(performance, field_name, value)
                        
                        # Found a match, move to next item
                        break
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance from web: {str(e)}")
            # Return empty performance object on error
            return PortfolioPerformance(source=url, last_updated=datetime.now())
    
    @async_circuit_protected("web_scraping")
    async def get_portfolio_performance_web_async(self, url: str = "https://bullaware.com/etoro/plessas") -> PortfolioPerformance:
        """
        Get portfolio performance data from a web source asynchronously.
        
        Args:
            url: URL to scrape for portfolio performance data
            
        Returns:
            PortfolioPerformance object
        """
        try:
            # Fetch and parse HTML content asynchronously
            soup = await self._get_soup_async(url)
            
            # Extract data
            performance = PortfolioPerformance(source=url, last_updated=datetime.now())
            
            # Extract summary metrics (TODAY, MTD, YTD, 2YR)
            summary_data = self._extract_summary_data(soup)
            
            # Map the scraped data fields to performance fields
            field_mapping = {
                'This Month': 'this_month',
                'MTD': 'this_month',
                'Today': 'this_month',
                'Year To Date': 'year_to_date',
                'YTD': 'year_to_date',
                '2 Years': 'two_years',
                '2YR': 'two_years',
                'Beta': 'beta',
                "Jensen's Alpha": 'alpha',
                'Alpha': 'alpha',
                'Sharpe': 'sharpe',
                'Sharpe Ratio': 'sharpe',
                'Sortino': 'sortino',
                'Sortino Ratio': 'sortino',
                'Cash': 'cash'
            }
            
            # Process each data item
            for key, value in summary_data.items():
                # Find matching field in performance object
                for pattern, field_name in field_mapping.items():
                    if pattern in key:
                        # Parse the value
                        if isinstance(value, str) and '%' in value:
                            # Convert percentage string to float
                            parsed_value = self._format_percentage_value(value)
                            if parsed_value is not None:
                                setattr(performance, field_name, parsed_value)
                        else:
                            # Try to convert to float
                            try:
                                parsed_value = float(value)
                                setattr(performance, field_name, parsed_value)
                            except (ValueError, TypeError):
                                # Keep as string if conversion fails
                                setattr(performance, field_name, value)
                        
                        # Found a match, move to next item
                        break
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance from web async: {str(e)}")
            # Return empty performance object on error
            return PortfolioPerformance(source=url, last_updated=datetime.now())
    
    def _get_soup(self, url: str) -> BeautifulSoup:
        """
        Fetch and parse HTML content from a URL.
        
        Args:
            url: The URL to fetch data from
            
        Returns:
            BeautifulSoup: Parsed HTML content
            
        Raises:
            NetworkError: If the request fails
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Connection': 'keep-alive',
        }
        
        session = requests.Session()
        try:
            # First attempt with default SSL verification
            response = session.get(url, headers=headers, verify=True, timeout=30)
            response.raise_for_status()
            
            # Force response encoding to UTF-8
            response.encoding = 'utf-8'
            return BeautifulSoup(response.text, "html.parser")
            
        except requests.exceptions.SSLError:
            try:
                # Second attempt with SSL verification disabled
                session.verify = False
                response = session.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Force response encoding to UTF-8
                response.encoding = 'utf-8'
                return BeautifulSoup(response.text, "html.parser")
                
            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Failed to fetch data from {url} (SSL retry failed): {str(e)}")
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Failed to fetch data from {url}: {str(e)}")
        finally:
            session.close()
    
    async def _get_soup_async(self, url: str) -> BeautifulSoup:
        """
        Fetch and parse HTML content from a URL asynchronously.
        
        Args:
            url: The URL to fetch data from
            
        Returns:
            BeautifulSoup: Parsed HTML content
            
        Raises:
            NetworkError: If the request fails
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Connection': 'keep-alive',
        }
        
        # Configure client SSL context
        ssl_context = None  # Default
        
        async with aiohttp.ClientSession() as session:
            try:
                # First attempt with default SSL verification
                async with session.get(url, headers=headers, ssl=ssl_context, timeout=30) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return BeautifulSoup(html_content, "html.parser")
                    else:
                        response.raise_for_status()
                        
            except aiohttp.ClientSSLError:
                try:
                    # Second attempt with SSL verification disabled
                    async with session.get(url, headers=headers, ssl=False, timeout=30) as response:
                        if response.status == 200:
                            html_content = await response.text()
                            return BeautifulSoup(html_content, "html.parser")
                        else:
                            response.raise_for_status()
                            
                except aiohttp.ClientError as e:
                    raise NetworkError(f"Failed to fetch data from {url} (SSL retry failed): {str(e)}")
            except aiohttp.ClientError as e:
                raise NetworkError(f"Failed to fetch data from {url}: {str(e)}")
    
    def _extract_summary_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract summary metrics from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary of extracted metrics
        """
        data = {}
        
        # Extract summary items (like MTD, YTD)
        summary_items = soup.select("div.relative div.flex.flex-col.items-center")
        
        if summary_items:
            for item in summary_items:
                value_span = (item.find("span", class_="font-semibold text-green-600") or
                             item.find("span", class_="font-semibold text-red-600"))
                label_div = item.find("div", class_="text-sm text-slate-400")
                
                if label_div and value_span:
                    label = label_div.text.strip()
                    value = value_span.text.strip()
                    data[label] = value
        
        # Extract other metrics (Beta, Alpha, etc.)
        metrics = [
            ("Beta", "Beta"),
            ("Alpha", "Jensen's Alpha"),
            ("Sharpe", "Sharpe Ratio"),
            ("Sortino", "Sortino Ratio")
        ]
        
        for label, contains_text in metrics:
            result = self._extract_metric(soup, label, contains_text)
            if result:
                data[result[0]] = result[1]
        
        # Extract cash percentage
        cash_result = self._extract_cash_percentage(soup)
        if cash_result:
            data[cash_result[0]] = cash_result[1]
        
        return data
    
    def _extract_metric(self, soup: BeautifulSoup, label: str, contains_text: str) -> Optional[Tuple[str, str]]:
        """
        Extract a metric value given its label and containing text.
        
        Args:
            soup: BeautifulSoup object
            label: Label for the metric
            contains_text: Text to search for in the container
            
        Returns:
            Tuple of (label, value) or None if not found
        """
        container = soup.find('h2',
                            class_=['font-semibold', 'text-slate-100'],
                            string=lambda s: contains_text in str(s))
        if container:
            value_span = container.find_next("span", class_="text-5xl")
            if value_span:
                return label, value_span.text.strip()
        return None
    
    def _extract_cash_percentage(self, soup: BeautifulSoup) -> Optional[Tuple[str, str]]:
        """
        Extract cash percentage value.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Tuple of ('Cash', value) or None if not found
        """
        # Two ways to try finding the cash element
        # Method 1: Using CSS selector
        cash_container = soup.select_one("div.relative.flex.justify-between.space-x-2:-soup-contains('Cash')")
        if cash_container:
            cash_value_span = cash_container.find("div", class_="font-medium")
            if cash_value_span:
                return "Cash", cash_value_span.text.strip()
        
        # Method 2: Look for text containing Cash
        cash_containers = soup.find_all(lambda tag: tag.name == 'div' and 'Cash' in tag.text)
        for container in cash_containers:
            value_span = container.find("div", class_="font-medium")
            if value_span:
                return "Cash", value_span.text.strip()
                
        return None
    
    def generate_index_performance_html(self, performances: List[IndexPerformance], title: str = "Market Performance") -> Optional[str]:
        """
        Generate HTML for index performance.
        
        Args:
            performances: List of IndexPerformance objects
            title: Title for the HTML document
            
        Returns:
            Path to generated HTML file or None if failed
        """
        try:
            if not performances:
                logger.warning("No performance data provided for index HTML")
                return None
            
            # Create metrics dictionary for formatting
            metrics = {}
            
            # Get the date range for display
            date_range = ""
            if performances and performances[0].start_date and performances[0].end_date:
                date_range = f"({performances[0].start_date.strftime('%Y-%m-%d')} to {performances[0].end_date.strftime('%Y-%m-%d')})"
            
            # Format each index performance
            for perf in performances:
                if perf.change_percent is not None:
                    metrics[perf.index_name] = {
                        'value': perf.change_percent,
                        'label': perf.index_name,
                        'is_percentage': True
                    }
            
            # Format metrics using FormatUtils
            formatted_metrics = FormatUtils.format_market_metrics(metrics)
            
            # Generate the HTML using HTMLGenerator
            sections = [{
                'title': title,
                'metrics': formatted_metrics,
                'columns': 2,
                'rows': 2,
                'width': "800px",
                'date_range': date_range
            }]
            
            html_content = self.html_generator.generate_market_html(
                title=title,
                sections=sections
            )
            
            # Write to file
            output_path = os.path.join(self.output_dir, "index.html")
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(html_content)
                
            logger.info(f"Generated index performance HTML at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating index performance HTML: {str(e)}")
            return None
    
    def generate_portfolio_performance_html(self, performance: PortfolioPerformance) -> Optional[str]:
        """
        Generate HTML for portfolio performance.
        
        Args:
            performance: PortfolioPerformance object
            
        Returns:
            Path to generated HTML file or None if failed
        """
        try:
            # Convert performance metrics to dictionary for HTML generation
            performance_metrics = performance.to_dict()
            
            # Split into performance and risk metrics
            perf_metrics = {}
            risk_metrics = {}
            
            # Performance metrics
            for key in ["This Month", "Year To Date", "2 Years"]:
                if key in performance_metrics:
                    perf_metrics[key] = performance_metrics[key]
            
            # Risk metrics
            for key in ["Beta", "Alpha", "Sharpe", "Sortino", "Cash"]:
                if key in performance_metrics:
                    risk_metrics[key] = performance_metrics[key]
            
            # Generate HTML using the HTML generator
            output_path = self.html_generator.generate_portfolio_dashboard(
                performance_metrics=perf_metrics,
                risk_metrics=risk_metrics
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating portfolio performance HTML: {str(e)}")
            return None
    
    def save_performance_data(self, data: Union[List[IndexPerformance], PortfolioPerformance], file_name: str) -> Optional[str]:
        """
        Save performance data to a JSON file.
        
        Args:
            data: Performance data to save
            file_name: Name of the output file
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            output_path = os.path.join(self.output_dir, file_name)
            
            # Convert data to serializable dictionary
            if isinstance(data, list):
                # List of IndexPerformance objects
                serialized = []
                for item in data:
                    serialized.append({
                        "index_name": item.index_name,
                        "ticker": item.ticker,
                        "current_value": item.current_value,
                        "previous_value": item.previous_value,
                        "change_percent": item.change_percent,
                        "start_date": item.start_date.isoformat() if item.start_date else None,
                        "end_date": item.end_date.isoformat() if item.end_date else None,
                        "period_type": item.period_type
                    })
            else:
                # PortfolioPerformance object
                serialized = {
                    "this_month": data.this_month,
                    "year_to_date": data.year_to_date,
                    "two_years": data.two_years,
                    "beta": data.beta,
                    "sharpe": data.sharpe,
                    "alpha": data.alpha,
                    "sortino": data.sortino,
                    "cash": data.cash,
                    "source": data.source,
                    "last_updated": data.last_updated.isoformat() if data.last_updated else None
                }
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(serialized, file, indent=2)
                
            logger.info(f"Saved performance data to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
            return None


def track_index_performance(period_type: str = "weekly"):
    """
    Track and display market index performance.
    
    Args:
        period_type: 'weekly' or 'monthly'
    """
    try:
        # Create performance tracker
        tracker = PerformanceTracker()
        
        # Get index performance
        performances = tracker.get_index_performance(period_type=period_type)
        
        # Display in console
        print(f"\n{period_type.capitalize()} Market Performance:")
        
        # Convert to DataFrame for display
        data = []
        for perf in performances:
            change_str = f"{perf.change_percent:+.2f}%" if perf.change_percent is not None else "N/A"
            
            # Add color to change percentage
            if perf.change_percent is not None:
                if perf.change_percent > 0:
                    change_str = f"\033[92m{change_str}\033[0m"  # Green for positive
                elif perf.change_percent < 0:
                    change_str = f"\033[91m{change_str}\033[0m"  # Red for negative
            
            data.append({
                'Index': perf.index_name,
                f'Previous ({perf.start_date.strftime("%Y-%m-%d") if perf.start_date else "N/A"})': 
                    f"{perf.previous_value:,.2f}" if perf.previous_value else "N/A",
                f'Current ({perf.end_date.strftime("%Y-%m-%d") if perf.end_date else "N/A"})': 
                    f"{perf.current_value:,.2f}" if perf.current_value else "N/A",
                'Change Percent': change_str
            })
        
        # Create DataFrame with proper column alignment
        df = pd.DataFrame(data)
        
        # Use tabulate for better table formatting (like in v1)
        from tabulate import tabulate
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', 
                     colalign=["left", "right", "right", "right"], 
                     showindex=False))
        print(f"\nCurrent time in Athens: {datetime.now(athens_tz).strftime('%Y-%m-%d %H:%M')}")
        
        # Generate HTML
        tracker.generate_index_performance_html(
            performances,
            title=f"{period_type.capitalize()} Market Performance"
        )
        
        # Save performance data
        tracker.save_performance_data(
            performances,
            file_name=f"{period_type.lower()}_performance.json"
        )
        
    except Exception as e:
        logger.error(f"Error tracking index performance: {str(e)}")
        print(f"Error: {str(e)}")


def track_portfolio_performance(url: str = "https://bullaware.com/etoro/plessas"):
    """
    Track and display portfolio performance from web source.
    
    Args:
        url: URL to scrape for portfolio performance data
    """
    try:
        # Create performance tracker
        tracker = PerformanceTracker()
        
        # Get portfolio performance
        performance = tracker.get_portfolio_performance_web(url=url)
        
        # Display in console
        print(f"\nPortfolio Performance (from {url}):")
        
        # Create table of metrics
        data = []
        for field in ["this_month", "year_to_date", "two_years", "beta", "sharpe", "alpha", "sortino", "cash"]:
            value = getattr(performance, field, None)
            # Format value
            if value is not None:
                if field in ["this_month", "year_to_date", "two_years", "cash"]:
                    formatted_value = f"{value:+.2f}%" if value >= 0 else f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.2f}"
                
                # Color positive/negative values
                if value > 0 and field in ["this_month", "year_to_date", "two_years"]:
                    color_code = "\033[92m"  # Green for positive
                elif value < 0 and field in ["this_month", "year_to_date", "two_years"]:
                    color_code = "\033[91m"  # Red for negative
                else:
                    color_code = ""
                
                # Reset color
                reset_code = "\033[0m" if color_code else ""
                
                # Format field name
                field_name = field.replace("_", " ").title()
                
                data.append([field_name, f"{color_code}{formatted_value}{reset_code}"])
        
        # Display table using tabulate for better formatting (like in v1)
        from tabulate import tabulate
        print(tabulate(data, headers=['Metric', 'Value'], tablefmt='fancy_grid', showindex=False))
        print(f"\nLast updated: {performance.last_updated.strftime('%Y-%m-%d %H:%M') if performance.last_updated else 'N/A'}")
        
        # Generate HTML
        tracker.generate_portfolio_performance_html(performance)
        
        # Save performance data
        tracker.save_performance_data(
            performance,
            file_name="portfolio_performance.json"
        )
        
    except Exception as e:
        logger.error(f"Error tracking portfolio performance: {str(e)}")
        print(f"Error: {str(e)}")


async def track_performance_async(period_type: str = "weekly", portfolio_url: str = "https://bullaware.com/etoro/plessas"):
    """
    Track both index and portfolio performance asynchronously.
    
    Args:
        period_type: 'weekly' or 'monthly'
        portfolio_url: URL to scrape for portfolio performance data
    """
    try:
        # Create performance tracker with async provider
        from ..api import get_async_provider
        provider = get_async_provider()
        tracker = PerformanceTracker(provider=provider)
        
        # Create tasks for both operations
        index_task = tracker.get_index_performance_async(period_type=period_type)
        portfolio_task = tracker.get_portfolio_performance_web_async(url=portfolio_url)
        
        # Run both tasks concurrently
        index_perf, portfolio_perf = await asyncio.gather(index_task, portfolio_task, return_exceptions=True)
        
        # Process index performance results
        if isinstance(index_perf, Exception):
            logger.error(f"Error getting index performance: {str(index_perf)}")
            print(f"Error getting index performance: {str(index_perf)}")
        else:
            # Generate HTML and save data
            tracker.generate_index_performance_html(
                index_perf,
                title=f"{period_type.capitalize()} Market Performance"
            )
            tracker.save_performance_data(
                index_perf,
                file_name=f"{period_type.lower()}_performance.json"
            )
            
            # Display in console
            print(f"\n{period_type.capitalize()} Market Performance:")
            data = []
            for perf in index_perf:
                change_str = f"{perf.change_percent:+.2f}%" if perf.change_percent is not None else "N/A"
                
                # Add color to change percentage
                if perf.change_percent is not None:
                    if perf.change_percent > 0:
                        change_str = f"\033[92m{change_str}\033[0m"  # Green for positive
                    elif perf.change_percent < 0:
                        change_str = f"\033[91m{change_str}\033[0m"  # Red for negative
                
                data.append({
                    'Index': perf.index_name,
                    f'Previous ({perf.start_date.strftime("%Y-%m-%d") if perf.start_date else "N/A"})': 
                        f"{perf.previous_value:,.2f}" if perf.previous_value else "N/A",
                    f'Current ({perf.end_date.strftime("%Y-%m-%d") if perf.end_date else "N/A"})': 
                        f"{perf.current_value:,.2f}" if perf.current_value else "N/A",
                    'Change Percent': change_str
                })
            
            # Use tabulate for better table formatting (like in v1)
            from tabulate import tabulate
            print(tabulate(data, headers='keys', tablefmt='fancy_grid', 
                         colalign=["left", "right", "right", "right"], 
                         showindex=False))
        
        # Process portfolio performance results
        if isinstance(portfolio_perf, Exception):
            logger.error(f"Error getting portfolio performance: {str(portfolio_perf)}")
            print(f"Error getting portfolio performance: {str(portfolio_perf)}")
        else:
            # Generate HTML and save data
            tracker.generate_portfolio_performance_html(portfolio_perf)
            tracker.save_performance_data(
                portfolio_perf,
                file_name="portfolio_performance.json"
            )
            
            # Display in console
            print(f"\nPortfolio Performance (from {portfolio_url}):")
            data = []
            for field in ["this_month", "year_to_date", "two_years", "beta", "sharpe", "alpha", "sortino", "cash"]:
                value = getattr(portfolio_perf, field, None)
                # Format value
                if value is not None:
                    if field in ["this_month", "year_to_date", "two_years", "cash"]:
                        formatted_value = f"{value:+.2f}%" if value >= 0 else f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:.2f}"
                    
                    # Color positive/negative values
                    if value > 0 and field in ["this_month", "year_to_date", "two_years"]:
                        color_code = "\033[92m"  # Green for positive
                    elif value < 0 and field in ["this_month", "year_to_date", "two_years"]:
                        color_code = "\033[91m"  # Red for negative
                    else:
                        color_code = ""
                    
                    # Reset color
                    reset_code = "\033[0m" if color_code else ""
                    
                    # Format field name
                    field_name = field.replace("_", " ").title()
                    
                    data.append([field_name, f"{color_code}{formatted_value}{reset_code}"])
            
            # Display table using tabulate for better formatting (like in v1)
            from tabulate import tabulate
            print(tabulate(data, headers=['Metric', 'Value'], tablefmt='fancy_grid', showindex=False))
        
        print(f"\nCurrent time in Athens: {datetime.now(athens_tz).strftime('%Y-%m-%d %H:%M')}")
        
    except Exception as e:
        logger.error(f"Error tracking performance asynchronously: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        # Close the provider session
        if hasattr(provider, 'close') and callable(provider.close):
            await provider.close()


if __name__ == "__main__":
    import sys
    
    # Default option
    option = "portfolio"
    
    # Parse command line args
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
    
    if option in ["weekly", "w"]:
        print("Tracking weekly market performance...")
        track_index_performance(period_type="weekly")
    elif option in ["monthly", "m"]:
        print("Tracking monthly market performance...")
        track_index_performance(period_type="monthly")
    elif option in ["portfolio", "p"]:
        print("Tracking portfolio performance...")
        track_portfolio_performance()
    elif option in ["all", "a"]:
        print("Tracking both market and portfolio performance asynchronously...")
        period_type = "weekly"
        asyncio.run(track_performance_async(period_type=period_type))
    else:
        print(f"Usage: python -m yahoofinance_v2.analysis.performance [option]")
        print(f"Options:")
        print(f"  weekly (w)    - Track weekly market performance")
        print(f"  monthly (m)   - Track monthly market performance")
        print(f"  portfolio (p) - Track portfolio performance")
        print(f"  all (a)       - Track both weekly market and portfolio performance")
        print(f"If no option is provided, portfolio performance is tracked by default.")