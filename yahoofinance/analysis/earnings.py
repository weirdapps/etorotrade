"""
Earnings analysis module.

This module provides functionality for analyzing earnings data,
including earnings dates, surprises, and trends.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..api import get_provider, FinanceDataProvider, AsyncFinanceDataProvider
from ..core.errors import YFinanceError, ValidationError

logger = logging.getLogger(__name__)

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


class EarningsAnalyzer:
    """
    Service for retrieving and analyzing earnings data.
    
    This service uses a data provider to retrieve earnings data
    and provides methods for calculating earnings trends.
    
    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
    """
    
    def __init__(self, provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None):
        """
        Initialize the EarningsAnalyzer.
        
        Args:
            provider: Data provider (sync or async), if None, a default provider is created
        """
        self.provider = provider if provider is not None else get_provider()
        
        # Check if the provider is async
        self.is_async = hasattr(self.provider, 'get_earnings_dates') and \
                        callable(self.provider.get_earnings_dates) and \
                        hasattr(self.provider.get_earnings_dates, '__await__')
    
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
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_earnings_data_async instead.")
        
        try:
            # Fetch earnings dates
            earnings_dates = self.provider.get_earnings_dates(ticker)
            
            # Fetch earnings history (if available)
            earnings_history = self.provider.get_earnings_history(ticker)
            
            # Process the data into EarningsData object
            return self._process_earnings_data(earnings_dates, earnings_history)
        
        except Exception as e:
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
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_earnings_data instead.")
        
        try:
            # Fetch earnings dates asynchronously
            earnings_dates = await self.provider.get_earnings_dates(ticker)
            
            # Fetch earnings history (if available) asynchronously
            earnings_history = await self.provider.get_earnings_history(ticker)
            
            # Process the data into EarningsData object
            return self._process_earnings_data(earnings_dates, earnings_history)
        
        except Exception as e:
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
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_earnings_batch_async instead.")
        
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.get_earnings_data(ticker)
            except Exception as e:
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
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_earnings_batch instead.")
        
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
        earnings_history: Optional[List[Dict[str, Any]]]
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
            latest_report = earnings_history[0]  # We already checked that earnings_history is not empty
            
            # Extract EPS data
            earnings_data.eps_estimate = latest_report.get('eps_estimate')
            earnings_data.eps_actual = latest_report.get('eps_actual')
            earnings_data.eps_surprise = latest_report.get('eps_surprise_pct')
            
            # Extract revenue data
            earnings_data.revenue_estimate = latest_report.get('revenue_estimate')
            earnings_data.revenue_actual = latest_report.get('revenue_actual')
            earnings_data.revenue_surprise = latest_report.get('revenue_surprise_pct')
        
        return earnings_data
    
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
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use calculate_earnings_trend_async instead.")
        
        try:
            # Get earnings data
            earnings_data = self.get_earnings_data(ticker)
            
            # Check if we have enough history
            if not earnings_data.earnings_history or len(earnings_data.earnings_history) < quarters:
                logger.info(f"Insufficient earnings history for {ticker} to calculate trend")
                return None
            
            # Limit to specified number of quarters
            history = earnings_data.earnings_history[:quarters]
            
            # Extract EPS values
            eps_values = [report.get('eps_actual') for report in history if report.get('eps_actual') is not None]
            
            # Extract revenue values
            revenue_values = [report.get('revenue_actual') for report in history if report.get('revenue_actual') is not None]
            
            # Calculate trends
            if len(eps_values) < 2 or len(revenue_values) < 2:
                logger.info(f"Insufficient valid data points for {ticker} to calculate trend")
                return None
            
            # Calculate growth rates
            eps_growth = [(eps_values[i] - eps_values[i+1]) / abs(eps_values[i+1]) * 100 if eps_values[i+1] != 0 else 0 
                         for i in range(len(eps_values)-1)]
            
            revenue_growth = [(revenue_values[i] - revenue_values[i+1]) / revenue_values[i+1] * 100 if revenue_values[i+1] != 0 else 0 
                            for i in range(len(revenue_values)-1)]
            
            # Calculate averages
            avg_eps_growth = sum(eps_growth) / len(eps_growth) if eps_growth else None
            avg_revenue_growth = sum(revenue_growth) / len(revenue_growth) if revenue_growth else None
            
            # Calculate consistency (how many quarters showed growth)
            eps_beat_count = sum(1 for x in eps_growth if x > 0)
            revenue_beat_count = sum(1 for x in revenue_growth if x > 0)
            
            return {
                'eps_values': eps_values,
                'revenue_values': revenue_values,
                'eps_growth': eps_growth,
                'revenue_growth': revenue_growth,
                'avg_eps_growth': avg_eps_growth,
                'avg_revenue_growth': avg_revenue_growth,
                'eps_beat_count': eps_beat_count,
                'revenue_beat_count': revenue_beat_count,
                'total_quarters': len(eps_growth) + 1,
                'eps_consistency': eps_beat_count / len(eps_growth) * 100 if eps_growth else None,
                'revenue_consistency': revenue_beat_count / len(revenue_growth) * 100 if revenue_growth else None
            }
        
        except Exception as e:
            logger.error(f"Error calculating earnings trend for {ticker}: {str(e)}")
            return None
    
    async def calculate_earnings_trend_async(self, ticker: str, quarters: int = 4) -> Optional[Dict[str, Any]]:
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
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use calculate_earnings_trend instead.")
        
        try:
            # Get earnings data asynchronously
            earnings_data = await self.get_earnings_data_async(ticker)
            
            # Check if we have enough history
            if not earnings_data.earnings_history or len(earnings_data.earnings_history) < quarters:
                logger.info(f"Insufficient earnings history for {ticker} to calculate trend")
                return None
            
            # Limit to specified number of quarters
            history = earnings_data.earnings_history[:quarters]
            
            # Extract EPS values
            eps_values = [report.get('eps_actual') for report in history if report.get('eps_actual') is not None]
            
            # Extract revenue values
            revenue_values = [report.get('revenue_actual') for report in history if report.get('revenue_actual') is not None]
            
            # Calculate trends
            if len(eps_values) < 2 or len(revenue_values) < 2:
                logger.info(f"Insufficient valid data points for {ticker} to calculate trend")
                return None
            
            # Calculate growth rates
            eps_growth = [(eps_values[i] - eps_values[i+1]) / abs(eps_values[i+1]) * 100 if eps_values[i+1] != 0 else 0 
                         for i in range(len(eps_values)-1)]
            
            revenue_growth = [(revenue_values[i] - revenue_values[i+1]) / revenue_values[i+1] * 100 if revenue_values[i+1] != 0 else 0 
                            for i in range(len(revenue_values)-1)]
            
            # Calculate averages
            avg_eps_growth = sum(eps_growth) / len(eps_growth) if eps_growth else None
            avg_revenue_growth = sum(revenue_growth) / len(revenue_growth) if revenue_growth else None
            
            # Calculate consistency (how many quarters showed growth)
            eps_beat_count = sum(1 for x in eps_growth if x > 0)
            revenue_beat_count = sum(1 for x in revenue_growth if x > 0)
            
            return {
                'eps_values': eps_values,
                'revenue_values': revenue_values,
                'eps_growth': eps_growth,
                'revenue_growth': revenue_growth,
                'avg_eps_growth': avg_eps_growth,
                'avg_revenue_growth': avg_revenue_growth,
                'eps_beat_count': eps_beat_count,
                'revenue_beat_count': revenue_beat_count,
                'total_quarters': len(eps_growth) + 1,
                'eps_consistency': eps_beat_count / len(eps_growth) * 100 if eps_growth else None,
                'revenue_consistency': revenue_beat_count / len(revenue_growth) * 100 if revenue_growth else None
            }
        
        except Exception as e:
            logger.error(f"Error calculating earnings trend for {ticker}: {str(e)}")
            return None