"""
Base provider interface for Yahoo Finance API client.

This module defines the abstract base class that all API providers must implement,
ensuring a consistent interface regardless of the underlying data source.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class FinanceDataProvider(ABC):
    """
    Abstract base class for finance data providers.
    
    All data providers must implement these methods to ensure
    consistent behavior across different data sources.
    """
    
    @abstractmethod
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic information for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing stock information
        """
        pass
    
    @abstractmethod
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing price information
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, 
                          ticker: str, 
                          period: Optional[str] = "1y", 
                          interval: Optional[str] = "1d") -> Any:
        """
        Get historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            Object containing historical data (typically DataFrame)
        """
        pass
    
    @abstractmethod
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing analyst ratings
        """
        pass
    
    @abstractmethod
    def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing earnings data
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
        """
        pass
    
    def batch_get_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple symbols in a batch.
        
        This is a convenience method with a default implementation that
        calls get_ticker_info for each ticker. Subclasses can override
        this method to provide a more efficient batch implementation.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to their information
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_ticker_info(ticker)
            except Exception as e:
                # Log error and continue with next ticker
                import logging
                logging.getLogger(__name__).warning(f"Error getting data for {ticker}: {str(e)}")
                results[ticker] = None
        return results