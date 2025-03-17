"""
Async base provider interface for Yahoo Finance API client.

This module defines the abstract base class that all async API providers must implement,
ensuring a consistent interface for asynchronous operations regardless of the underlying data source.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class AsyncFinanceDataProvider(ABC):
    """
    Abstract base class for asynchronous finance data providers.
    
    All async data providers must implement these methods to ensure
    consistent behavior across different data sources.
    """
    
    @abstractmethod
    async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic information for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing stock information
        """
        pass
    
    @abstractmethod
    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing price information
        """
        pass
    
    @abstractmethod
    async def get_historical_data(self, 
                                ticker: str, 
                                period: Optional[str] = "1y", 
                                interval: Optional[str] = "1d") -> Any:
        """
        Get historical price data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            Object containing historical data (typically DataFrame)
        """
        pass
    
    @abstractmethod
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing analyst ratings
        """
        pass
    
    @abstractmethod
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing earnings data
        """
        pass
    
    @abstractmethod
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query asynchronously.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching tickers with metadata
        """
        pass