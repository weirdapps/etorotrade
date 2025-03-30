"""
Performance tracking module for financial data.

This module provides functions for tracking market index performance
and portfolio performance from external sources, including web scraping
for additional performance metrics not available through the API.
"""

# Constants for repeated strings
THIS_MONTH = "This Month"
YEAR_TO_DATE = "Year To Date"
TWO_YEARS = "2 Years"
JENSENS_ALPHA = "Jensen's Alpha"
DEFAULT_PORTFOLIO_URL = "https://bullaware.com/etoro/plessas"
HTML_PARSER = "html.parser"

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
import numpy as np
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
            result[THIS_MONTH] = {
                "value": self.this_month,
                "label": THIS_MONTH,
                "is_percentage": True
            }
            
        if self.year_to_date is not None:
            result[YEAR_TO_DATE] = {
                "value": self.year_to_date,
                "label": YEAR_TO_DATE,
                "is_percentage": True
            }
            
        if self.two_years is not None:
            result[TWO_YEARS] = {
                "value": self.two_years,
                "label": TWO_YEARS,
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
                "label": JENSENS_ALPHA,
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
                        asyncio.iscoroutinefunction(self.provider.batch_get_ticker_info)
    
    @staticmethod
    def calculate_weekly_dates() -> Tuple[datetime, datetime]:
        """
        Calculate the dates for weekly performance comparison.
        
        If the current week is complete (today is after Friday market close),
        returns previous week and the week before.
        Otherwise, returns the week before the previous week and the previous week.
        
        Returns:
            Tuple of (older_friday, newer_friday)
        """
        today = datetime.today()
        
        # Determine if current week is complete (today is past Friday)
        is_after_friday = today.weekday() > 4  # Friday is 4
        is_friday_after_market_close = today.weekday() == 4 and today.hour >= 16  # 4pm market close
        current_week_complete = is_after_friday or is_friday_after_market_close
        
        # Calculate the most recent completed Friday
        days_since_friday = (today.weekday() - 4) % 7
        latest_friday = today - timedelta(days=days_since_friday)
        
        # If we're still in the current week's trading window, 
        # move back to the previous completed week
        if not current_week_complete:
            latest_friday = latest_friday - timedelta(days=7)
            
        # Get previous Friday
        previous_friday = latest_friday - timedelta(days=7)
        
        return previous_friday.replace(hour=0, minute=0, second=0, microsecond=0), \
               latest_friday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    @staticmethod
    def calculate_monthly_dates() -> Tuple[datetime, datetime]:
        """
        Calculate dates for monthly performance comparison.
        
        If the current month is complete (we're in a new month),
        returns previous month end and the month end before that.
        Otherwise, returns the month end before the previous month and the previous month end.
        
        Returns:
            Tuple of (older_month_end, newer_month_end)
        """
        today = datetime.today()
        
        # Determine if current month is complete (we're in a new month)
        # For stocks, month is considered complete when we reach the next month
        # We don't need to check time since we only care about the day
        current_month_complete = today.day > 1  # past the 1st of month
        
        # Calculate the last day of the previous month
        last_month_end = today.replace(day=1) - timedelta(days=1)
        
        # If current month is not complete, go back one more month
        if not current_month_complete:
            last_month_end = last_month_end.replace(day=1) - timedelta(days=1)
            
        # Get the end of the month before the last month
        previous_month_end = last_month_end.replace(day=1) - timedelta(days=1)
        
        return previous_month_end.replace(hour=0, minute=0, second=0, microsecond=0), \
               last_month_end.replace(hour=0, minute=0, second=0, microsecond=0)
    
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
    def _format_percentage_value(value: str) -> Optional[float]:
        """Format a percentage value as float."""
        try:
            # Remove % symbol but keep signs
            clean_value = value.replace('%', '').strip()
            return float(clean_value)
        except ValueError:
            return None
    
    def _process_performance_data(self, summary_data: Dict[str, Any], performance: PortfolioPerformance) -> None:
        """
        Process performance data and update the performance object.
        
        This is a shared method used by both sync and async versions to reduce code duplication.
        
        Args:
            summary_data: Dictionary of scraped data
            performance: PortfolioPerformance object to update
        """
        # Map the scraped data fields to performance fields
        field_mapping = {
            THIS_MONTH: 'this_month',
            'MTD': 'this_month',
            'Today': 'this_month',
            YEAR_TO_DATE: 'year_to_date',
            'YTD': 'year_to_date',
            TWO_YEARS: 'two_years',
            '2YR': 'two_years',
            'Beta': 'beta',
            JENSENS_ALPHA: 'alpha',
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
    
    @circuit_protected("web_scraping")
    def get_portfolio_performance_web(self, url: str = DEFAULT_PORTFOLIO_URL) -> PortfolioPerformance:
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
            
            # Process the data using the shared _process_performance_data method
            self._process_performance_data(summary_data, performance)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance from web: {str(e)}")
            # Return empty performance object on error
            return PortfolioPerformance(source=url, last_updated=datetime.now())
    
    @async_circuit_protected("web_scraping")
    async def get_portfolio_performance_web_async(self, url: str = DEFAULT_PORTFOLIO_URL) -> PortfolioPerformance:
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
            
            # Process the data using the shared _process_performance_data method
            self._process_performance_data(summary_data, performance)
            
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
        
        # Import certifi for SSL certificate verification
        import certifi
        
        session = requests.Session()
        try:
            # Always use SSL verification with proper certificate bundle
            response = session.get(
                url, 
                headers=headers, 
                verify=certifi.where(),  # Use certifi's certificate bundle
                timeout=30
            )
            response.raise_for_status()
            
            # Force response encoding to UTF-8
            response.encoding = 'utf-8'
            return BeautifulSoup(response.text, HTML_PARSER)
            
        except requests.exceptions.SSLError as e:
            # Do not fall back to insecure connections
            # SSL errors indicate certificate validation problems that should be addressed properly
            raise NetworkError(f"SSL certificate validation failed for {url}. This could indicate a security issue: {str(e)}")
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
        
        # Use SSL with proper verification
        # Import for creating proper SSL context
        import ssl
        import certifi
        
        # Create a proper SSL context with certificate verification
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        async with aiohttp.ClientSession() as session:
            try:
                # Always use SSL verification
                async with session.get(url, headers=headers, ssl=ssl_context, timeout=30) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return BeautifulSoup(html_content, HTML_PARSER)
                    else:
                        response.raise_for_status()
                            
            # Using proper SSL handling without fallbacks to insecure connections
            except aiohttp.ClientSSLError as e:
                raise NetworkError(f"SSL certificate validation failed for {url}. This could indicate a security issue: {str(e)}")
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
        
        # Extract other metrics (Beta, Alpha, etc.) using our generic extractor
        metrics_map = {
            "Beta": "Beta",
            "Alpha": "Jensen's Alpha",
            "Sharpe": "Sharpe Ratio",
            "Sortino": "Sortino Ratio",
            "Cash": "Cash"
        }
        
        # Use the generic metrics extractor for all metrics at once
        additional_metrics = self._extract_metrics_from_soup(soup, metrics_map)
        
        # Merge the results
        data.update(additional_metrics)
        
        return data
    
    def _extract_metrics_from_soup(self, soup: BeautifulSoup, metrics_map: Dict[str, str]) -> Dict[str, str]:
        """
        Generic extractor for metrics from soup with various extraction methods.
        
        Args:
            soup: BeautifulSoup object to search in
            metrics_map: Dictionary mapping metric names to lookup texts
            
        Returns:
            Dictionary of extracted metrics
        """
        results = {}
        
        # Try each metric with multiple extraction methods
        for label, contains_text in metrics_map.items():
            # Method 1: Look for h2 heading with the text
            container = soup.find('h2',
                                class_=['font-semibold', 'text-slate-100'],
                                string=lambda s: contains_text in str(s))
            if container:
                value_span = container.find_next("span", class_="text-5xl")
                if value_span:
                    results[label] = value_span.text.strip()
                    continue
            
            # Method 2: Using CSS selector with contains
            metric_container = soup.select_one(f"div.relative.flex.justify-between.space-x-2:-soup-contains('{contains_text}')")
            if metric_container:
                value_span = metric_container.find("div", class_="font-medium")
                if value_span:
                    results[label] = value_span.text.strip()
                    continue
            
            # Method 3: Look for any div containing the text
            containers = soup.find_all(lambda tag: tag.name == 'div' and contains_text in tag.text)
            for container in containers:
                value_span = container.find("div", class_="font-medium")
                if value_span:
                    results[label] = value_span.text.strip()
                    break
                    
        return results
            
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
        # Use the generic method with a single metric
        results = self._extract_metrics_from_soup(soup, {label: contains_text})
        if label in results:
            return label, results[label]
        return None
    
    def _extract_cash_percentage(self, soup: BeautifulSoup) -> Optional[Tuple[str, str]]:
        """
        Extract cash percentage value.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Tuple of ('Cash', value) or None if not found
        """
        # Use the generic method with Cash metric
        return self._extract_metric(soup, "Cash", "Cash")
    
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
            
            # Write to file with the requested name 'markets.html'
            output_path = os.path.join(self.output_dir, "markets.html")
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
    Track and display market index performance for a previous period.
    
    Args:
        period_type: 'weekly' or 'monthly' - shows the previous completed period
    """
    try:
        # Create performance tracker
        tracker = PerformanceTracker()
        
        # Get index performance
        performances = tracker.get_index_performance(period_type=period_type)
        
        # Determine what periods we're showing based on current date
        today = datetime.today()
        
        # For weekly display
        if period_type.lower() == 'weekly':
            # Check if we're past Friday market close
            is_after_friday = today.weekday() > 4  # Friday is 4
            is_friday_after_market_close = today.weekday() == 4 and today.hour >= 16  # 4pm market close
            current_week_complete = is_after_friday or is_friday_after_market_close
            
            if current_week_complete:
                period_desc = "Last Week vs. Previous Week"
            else:
                period_desc = "Previous Week vs. Week Before"
        # For monthly display
        else:
            # Check if we're in a new month (past day 1)
            current_month_complete = today.day > 1
            
            if current_month_complete:
                period_desc = "Last Month vs. Previous Month"
            else:
                period_desc = "Previous Month vs. Month Before"
                
        # Display in console with clear period indication
        print(f"\n{period_type.capitalize()} Market Performance: {period_desc}")
        
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
                    
            # Determine column keys based on period type - for previous periods display
            if period_type.lower() == 'weekly':
                prev_key = f'Week-2 ({perf.start_date.strftime("%Y-%m-%d") if perf.start_date else "N/A"})'
                curr_key = f'Week-1 ({perf.end_date.strftime("%Y-%m-%d") if perf.end_date else "N/A"})'
            else:
                prev_key = f'Month-2 ({perf.start_date.strftime("%Y-%m-%d") if perf.start_date else "N/A"})'
                curr_key = f'Month-1 ({perf.end_date.strftime("%Y-%m-%d") if perf.end_date else "N/A"})'
                
            # For display purposes, we need consistent columns
            current_value = perf.current_value
            display_date = perf.end_date
            
            # For VIX ticker, always calculate the current value from the change percent
            # This ensures we have a value to display even when the standard API call fails
            if perf.index_name == 'VIX' and perf.previous_value is not None and perf.change_percent is not None:
                # Direct calculation for display purposes
                calculated_vix = perf.previous_value * (1 + perf.change_percent / 100)
                current_value = calculated_vix
                
                # If we don't have an end date, use the start date + 7 days for weekly
                if display_date is None and perf.start_date is not None:
                    if period_type.lower() == "weekly":
                        display_date = perf.start_date + timedelta(days=7)
                
                # Try to get the latest data anyway for reference
                try:
                    import yfinance as yf
                    ticker = yf.Ticker('^VIX')
                    latest = ticker.history(period="1d")
                    if not latest.empty:
                        # Log the direct value but don't use it directly
                        logger.debug(f"Direct VIX value: {float(latest['Close'].iloc[-1])}")
                except Exception as e:
                    logger.warning(f"Could not get VIX data directly: {str(e)}")
            
            # For all other tickers, or if VIX direct lookup failed:
            # If we have change percent but no current value, calculate it
            if (current_value is None or pd.isna(current_value)) and perf.previous_value is not None and perf.change_percent is not None:
                current_value = perf.previous_value * (1 + perf.change_percent / 100)
            
            # Ensure we have a date to display
            if display_date is None and perf.start_date is not None:
                # Use start date + 7 days for weekly view as fallback
                if period_type.lower() == "weekly":
                    display_date = perf.start_date + timedelta(days=7)
                elif period_type.lower() == "monthly" and perf.start_date.month < 12:
                    # For monthly, use last day of month
                    next_month = perf.start_date.month + 1
                    display_date = perf.start_date.replace(month=next_month, day=1) - timedelta(days=1)
            
            # Prepare the Current column display string
            current_display = "N/A"
            
            # Special handling for VIX or other tickers that might have missing data
            # First check for VIX explicitly since it's known to have issues
            if perf.index_name == 'VIX' and perf.previous_value is not None and perf.change_percent is not None:
                # Always calculate VIX directly regardless of whether we have a current_value
                calculated_value = perf.previous_value * (1 + perf.change_percent / 100)
                current_display = f"{calculated_value:,.2f}"
            # For other tickers with missing data but valid previous value and change percent
            elif (current_value is None or pd.isna(current_value)) and perf.previous_value is not None and perf.change_percent is not None:
                # Calculate current value from previous value and change percent
                calculated_value = perf.previous_value * (1 + perf.change_percent / 100)
                current_display = f"{calculated_value:,.2f}"
                
                # If display date is missing, use an appropriate date based on period type
                if display_date is None and perf.start_date is not None:
                    if period_type.lower() == "weekly":
                        # For weekly, use previous Friday + 7 days
                        display_date = perf.start_date + timedelta(days=7)
                    elif period_type.lower() == "monthly":
                        # For monthly, use end of previous month
                        next_month = perf.start_date.month + 1
                        year = perf.start_date.year + (1 if next_month > 12 else 0)
                        next_month = next_month % 12 or 12  # Handle December
                        display_date = datetime(year, next_month, 1) - timedelta(days=1)
            elif current_value is not None and not pd.isna(current_value):
                # Normal case - use the API data
                current_display = f"{current_value:,.2f}"
                
            # Create row data with previous period columns (Week-2, Week-1 or Month-2, Month-1)
            row_data = {
                'Index': perf.index_name,
                prev_key: f"{perf.previous_value:,.2f}" if perf.previous_value is not None else "N/A",
                curr_key: current_display, 
                'Change %': change_str
            }
            
            # Debug log the performance data for troubleshooting
            logger.debug(f"Performance data for {perf.index_name}: {perf.__dict__}")
            
            # VIX special handling - if this is the VIX ticker, fix the data before adding it
            if perf.index_name == 'VIX' and period_type.lower() == 'weekly' and perf.previous_value is not None and perf.change_percent is not None:
                # Calculate VIX value directly - we know this works
                calculated_vix = perf.previous_value * (1 + perf.change_percent / 100)
                
                # Get column names for consistent display
                prev_date = perf.start_date.strftime("%Y-%m-%d") if perf.start_date else "N/A"
                
                # Find the common current date from other indices (not VIX)
                curr_date = None
                # This is the date we want to use for all indices
                for other_perf in performances:
                    if other_perf.index_name != 'VIX' and other_perf.end_date:
                        curr_date = other_perf.end_date.strftime("%Y-%m-%d")
                        break
                
                # If we couldn't find a date, use a default
                if not curr_date:
                    curr_date = (perf.start_date + timedelta(days=7)).strftime("%Y-%m-%d") if perf.start_date else "N/A"
                
                # Determine column keys based on period type
                if period_type.lower() == 'weekly':
                    prev_key = f'Week-2 ({prev_date})'
                    curr_key = f'Week-1 ({curr_date})'
                else:
                    prev_key = f'Month-2 ({prev_date})'
                    curr_key = f'Month-1 ({curr_date})'
                
                # Update the row data with correct column names and calculated value
                row_data = {
                    'Index': 'VIX',
                    prev_key: f"{perf.previous_value:,.2f}",
                    curr_key: f"{calculated_vix:,.2f}",
                    'Change %': change_str
                }
                
                logger.info(f"Created VIX display with calculated value: {calculated_vix:.2f}")
            
            # Add to data list
            data.append(row_data)
        
        # Create DataFrame with proper column alignment
        df = pd.DataFrame(data)
        
        # Standardize column names to ensure consistent display - especially for VIX
        # For weekly/monthly display, all indices should use the same date format
        
        # 1. First, find a standard format for week/month columns (using first non-VIX index)
        std_prev_col = None
        std_curr_col = None
        for idx, row in df.iterrows():
            if row['Index'] != 'VIX':
                std_prev_col = next((col for col in row.index if 'Week-2' in col or 'Month-2' in col), None)
                std_curr_col = next((col for col in row.index if 'Week-1' in col or 'Month-1' in col), None)
                if std_prev_col and std_curr_col:
                    break
                    
        # 2. Now normalize all columns to standard format
        if std_prev_col and std_curr_col:
            # Fix VIX to use the same column names as other indices
            for idx, row in df.iterrows():
                if row['Index'] == 'VIX':
                    # Find VIX-specific column names
                    vix_prev = next((col for col in row.index if 'Week-2' in col or 'Month-2' in col), None)
                    vix_curr = next((col for col in row.index if 'Week-1' in col or 'Month-1' in col), None)
                    
                    # If different from standard, rename
                    if vix_curr != std_curr_col and vix_curr and std_curr_col:
                        # Move the value to the standard column
                        df.at[idx, std_curr_col] = row[vix_curr]
                        
                    # Make sure we don't have any 'nan' values
                    if isinstance(df.at[idx, std_curr_col], str) and df.at[idx, std_curr_col] == 'nan' and row['Change %'] and vix_prev:
                        try:
                            # Get previous value
                            prev_val = float(str(row[vix_prev]).replace(',', ''))
                            # Get change percent
                            change_str = str(row['Change %']).replace('\033[91m', '').replace('\033[92m', '').replace('\033[0m', '').replace('%', '')
                            change_pct = float(change_str)
                            # Calculate current
                            calculated_val = prev_val * (1 + change_pct/100)
                            df.at[idx, std_curr_col] = f"{calculated_val:,.2f}"
                            logger.info(f"Fixed VIX display with calculated value: {calculated_val:.2f}")
                        except Exception as e:
                            logger.warning(f"Cannot calculate VIX value: {str(e)}")
        
            # Create standardized column list
            standard_columns = ['Index', std_prev_col, std_curr_col, 'Change %']
            df = df[standard_columns]
        else:
            # Fallback if no standard columns found
            standard_columns = ['Index']
            prev_col = next((col for col in df.columns if 'Week-2' in col or 'Month-2' in col), None)
            curr_col = next((col for col in df.columns if 'Week-1' in col or 'Month-1' in col), None)
            if prev_col and curr_col:
                standard_columns.extend([prev_col, curr_col, 'Change %'])
                df = df[standard_columns]
        
        # Ensure we'll still display data if column detection failed
        if len(standard_columns) < 4:
            logger.warning(f"Failed to detect standard columns. Using all columns: {df.columns.tolist()}")
            
        # VIX fix is now applied at the row level when creating the data
        
        # Use tabulate for better table formatting (like in v1)
        from tabulate import tabulate
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', 
                     colalign=["left", "right", "right", "right"], 
                     showindex=False))
        print(f"\nCurrent time in Athens: {datetime.now(athens_tz).strftime('%Y-%m-%d %H:%M')}")
        
        # Generate HTML
        tracker.generate_index_performance_html(
            performances,
            title=f"{period_type.capitalize()} Market Performance: {period_desc}"
        )
        
        # Save performance data with consistent naming convention
        tracker.save_performance_data(
            performances,
            file_name="markets.json"
        )
        
    except Exception as e:
        logger.error(f"Error tracking index performance: {str(e)}")
        print(f"Error: {str(e)}")


def track_portfolio_performance(url: str = DEFAULT_PORTFOLIO_URL):
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
        
        # Save performance data with consistent naming convention
        tracker.save_performance_data(
            performance,
            file_name="portfolio.json"
        )
        
    except Exception as e:
        logger.error(f"Error tracking portfolio performance: {str(e)}")
        print(f"Error: {str(e)}")


async def track_performance_async(period_type: str = "weekly", portfolio_url: str = DEFAULT_PORTFOLIO_URL):
    """
    Track both index and portfolio performance asynchronously.
    
    Args:
        period_type: 'weekly' or 'monthly'
        portfolio_url: URL to scrape for portfolio performance data
    """
    try:
        # Create performance tracker with async provider
        from ..api import get_provider
        provider = get_provider(async_mode=True, enhanced=True)
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
            # Determine what periods we're showing based on current date
            today = datetime.today()
            
            # For weekly display
            if period_type.lower() == 'weekly':
                # Check if we're past Friday market close
                is_after_friday = today.weekday() > 4  # Friday is 4
                is_friday_after_market_close = today.weekday() == 4 and today.hour >= 16  # 4pm market close
                current_week_complete = is_after_friday or is_friday_after_market_close
                
                if current_week_complete:
                    period_desc = "Last Week vs. Previous Week"
                else:
                    period_desc = "Previous Week vs. Week Before"
            # For monthly display
            else:
                # Check if we're in a new month (past day 1)
                current_month_complete = today.day > 1
                
                if current_month_complete:
                    period_desc = "Last Month vs. Previous Month"
                else:
                    period_desc = "Previous Month vs. Month Before"
            
            tracker.generate_index_performance_html(
                index_perf,
                title=f"{period_type.capitalize()} Market Performance: {period_desc}"
            )
            tracker.save_performance_data(
                index_perf,
                file_name="markets.json"
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
                
                # Determine column keys based on period type
                if period_type.lower() == 'weekly':
                    prev_key = f'Week-2 ({perf.start_date.strftime("%Y-%m-%d") if perf.start_date else "N/A"})'
                    curr_key = f'Week-1 ({perf.end_date.strftime("%Y-%m-%d") if perf.end_date else "N/A"})'
                else:
                    prev_key = f'Month-2 ({perf.start_date.strftime("%Y-%m-%d") if perf.start_date else "N/A"})'
                    curr_key = f'Month-1 ({perf.end_date.strftime("%Y-%m-%d") if perf.end_date else "N/A"})'
                
                # Create row data with proper titles for previous period comparison
                row_data = {
                    'Index': perf.index_name,
                    prev_key: f"{perf.previous_value:,.2f}" if perf.previous_value is not None else "N/A",
                    curr_key: f"{perf.current_value:,.2f}" if perf.current_value is not None and not pd.isna(perf.current_value) else "N/A",
                    'Change %': change_str
                }
                
                # Log data for DEBUG
                logger.debug(f"Created row data for {perf.index_name}: {row_data}")
                
                data.append(row_data)
            
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
            # Generate HTML and save data with consistent naming
            tracker.generate_portfolio_performance_html(portfolio_perf)
            tracker.save_performance_data(
                portfolio_perf,
                file_name="portfolio.json"
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
        print("Usage: python -m yahoofinance.analysis.performance [option]")
        print("Options:")
        print("  weekly (w)    - Track weekly market performance")
        print("  monthly (m)   - Track monthly market performance")
        print("  portfolio (p) - Track portfolio performance")
        print("  all (a)       - Track both weekly market and portfolio performance")
        print("If no option is provided, portfolio performance is tracked by default.")