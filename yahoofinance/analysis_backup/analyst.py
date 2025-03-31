"""
Analyst ratings and recommendations module.

This module provides functionality for analyzing analyst ratings,
recommendations, and price targets for stocks.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..api import get_provider, FinanceDataProvider, AsyncFinanceDataProvider
from ..core.errors import YFinanceError, ValidationError
from ..utils.market import is_us_ticker

logger = logging.getLogger(__name__)

@dataclass
class AnalystData:
    """
    Container for analyst ratings and recommendations data.
    
    Attributes:
        buy_percentage: Percentage of analysts with buy ratings
        total_ratings: Total number of analyst ratings
        strong_buy: Number of strong buy ratings
        buy: Number of buy ratings
        hold: Number of hold ratings
        sell: Number of sell ratings
        strong_sell: Number of strong sell ratings
        date: Date of the latest analyst recommendation
        average_price_target: Average price target
        median_price_target: Median price target
        highest_price_target: Highest price target
        lowest_price_target: Lowest price target
        analyst_count: Number of analysts with price targets
    """
    
    buy_percentage: Optional[float] = None
    total_ratings: Optional[int] = None
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    date: Optional[str] = None
    average_price_target: Optional[float] = None
    median_price_target: Optional[float] = None
    highest_price_target: Optional[float] = None
    lowest_price_target: Optional[float] = None
    analyst_count: Optional[int] = None


class AnalystRatingsService:
    """
    Service for retrieving and analyzing analyst ratings data.
    
    This service uses a data provider to retrieve analyst ratings
    and provides methods for calculating ratings summaries.
    
    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
    """
    
    def __init__(self, provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None):
        """
        Initialize the AnalystRatingsService.
        
        Args:
            provider: Data provider (sync or async), if None, a default provider is created
        """
        self.provider = provider if provider is not None else get_provider()
        
        # Check if the provider is async
        self.is_async = hasattr(self.provider, 'get_analyst_ratings') and \
                        callable(self.provider.get_analyst_ratings) and \
                        hasattr(self.provider.get_analyst_ratings, '__await__')
    
    def get_ratings(self, ticker: str) -> AnalystData:
        """
        Get analyst ratings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            AnalystData object containing analyst ratings
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_ratings_async instead.")
        
        try:
            # Skip API call for non-US tickers to avoid unnecessary errors
            if not is_us_ticker(ticker):
                logger.info(f"Skipping analyst ratings for non-US ticker: {ticker}")
                return AnalystData()
            
            # Fetch analyst ratings data
            ratings_data = self.provider.get_analyst_ratings(ticker)
            
            # Process the data into AnalystData object
            return self._process_ratings_data(ratings_data)
        
        except Exception as e:
            logger.error(f"Error fetching analyst ratings for {ticker}: {str(e)}")
            return AnalystData()
    
    async def get_ratings_async(self, ticker: str) -> AnalystData:
        """
        Get analyst ratings for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            AnalystData object containing analyst ratings
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_ratings instead.")
        
        try:
            # Skip API call for non-US tickers to avoid unnecessary errors
            if not is_us_ticker(ticker):
                logger.info(f"Skipping analyst ratings for non-US ticker: {ticker}")
                return AnalystData()
            
            # Fetch analyst ratings data asynchronously
            ratings_data = await self.provider.get_analyst_ratings(ticker)
            
            # Process the data into AnalystData object
            return self._process_ratings_data(ratings_data)
        
        except Exception as e:
            logger.error(f"Error fetching analyst ratings for {ticker}: {str(e)}")
            return AnalystData()
    
    def get_ratings_batch(self, tickers: List[str]) -> Dict[str, AnalystData]:
        """
        Get analyst ratings for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to AnalystData objects
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_ratings_batch_async instead.")
        
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.get_ratings(ticker)
            except Exception as e:
                logger.error(f"Error fetching analyst ratings for {ticker}: {str(e)}")
                results[ticker] = AnalystData()
        
        return results
    
    async def get_ratings_batch_async(self, tickers: List[str]) -> Dict[str, AnalystData]:
        """
        Get analyst ratings for multiple tickers asynchronously.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to AnalystData objects
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_ratings_batch instead.")
        
        import asyncio
        
        # Filter out non-US tickers first to avoid unnecessary API calls
        us_tickers = [ticker for ticker in tickers if is_us_ticker(ticker)]
        
        # Create tasks for US tickers
        tasks = [self.get_ratings_async(ticker) for ticker in us_tickers]
        
        # Wait for all tasks to complete
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {}
        for ticker, result in zip(us_tickers, results_list):
            if isinstance(result, Exception):
                logger.error(f"Error fetching analyst ratings for {ticker}: {str(result)}")
                results[ticker] = AnalystData()
            else:
                results[ticker] = result
        
        # Add empty results for non-US tickers
        for ticker in tickers:
            if ticker not in results:
                results[ticker] = AnalystData()
        
        return results
    
    def _process_ratings_data(self, ratings_data: Dict[str, Any]) -> AnalystData:
        """
        Process analyst ratings data into AnalystData object.
        
        Args:
            ratings_data: Dictionary containing analyst ratings data
            
        Returns:
            AnalystData object with processed ratings
        """
        if not ratings_data:
            return AnalystData()
        
        # Extract ratings data
        return AnalystData(
            buy_percentage=ratings_data.get('buy_percentage'),
            total_ratings=ratings_data.get('recommendations'),
            strong_buy=ratings_data.get('strong_buy', 0),
            buy=ratings_data.get('buy', 0),
            hold=ratings_data.get('hold', 0),
            sell=ratings_data.get('sell', 0),
            strong_sell=ratings_data.get('strong_sell', 0),
            date=ratings_data.get('date'),
            average_price_target=ratings_data.get('average_price_target'),
            median_price_target=ratings_data.get('median_price_target'),
            highest_price_target=ratings_data.get('highest_price_target'),
            lowest_price_target=ratings_data.get('lowest_price_target'),
            analyst_count=ratings_data.get('analyst_count')
        )
    
    def get_recent_changes(self, ticker: str, days: int = 30) -> List[Dict[str, str]]:
        """
        Get recent rating changes for a stock.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of dictionaries containing recent rating changes
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_recent_changes_async instead.")
        
        if not isinstance(days, int) or days < 1:
            raise ValidationError("Days must be a positive integer")
        
        try:
            # Skip API call for non-US tickers to avoid unnecessary errors
            if not is_us_ticker(ticker):
                logger.info(f"Skipping recent changes for non-US ticker: {ticker}")
                return []
            
            # Calculate start date
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Fetch recent changes data
            changes_data = self.provider.get_rating_changes(ticker, start_date)
            
            return self._process_changes_data(changes_data)
        
        except Exception as e:
            logger.error(f"Error fetching recent changes for {ticker}: {str(e)}")
            return []
    
    async def get_recent_changes_async(self, ticker: str, days: int = 30) -> List[Dict[str, str]]:
        """
        Get recent rating changes for a stock asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of dictionaries containing recent rating changes
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_recent_changes instead.")
        
        if not isinstance(days, int) or days < 1:
            raise ValidationError("Days must be a positive integer")
        
        try:
            # Skip API call for non-US tickers to avoid unnecessary errors
            if not is_us_ticker(ticker):
                logger.info(f"Skipping recent changes for non-US ticker: {ticker}")
                return []
            
            # Calculate start date
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Fetch recent changes data asynchronously
            changes_data = await self.provider.get_rating_changes(ticker, start_date)
            
            return self._process_changes_data(changes_data)
        
        except Exception as e:
            logger.error(f"Error fetching recent changes for {ticker}: {str(e)}")
            return []
    
    def _process_changes_data(self, changes_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Process rating changes data.
        
        Args:
            changes_data: List of dictionaries with rating changes data
            
        Returns:
            List of dictionaries with processed rating changes
        """
        if not changes_data:
            return []
        
        processed_changes = []
        for change in changes_data:
            processed_change = {
                "date": change.get("date", ""),
                "firm": change.get("firm", ""),
                "from_grade": change.get("from_grade", ""),
                "to_grade": change.get("to_grade", ""),
                "action": change.get("action", "")
            }
            processed_changes.append(processed_change)
        
        return processed_changes