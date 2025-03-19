"""Base provider classes for both sync and async implementations.

This module provides the base classes that all finance data providers should inherit from,
consolidating common functionality and reducing duplication between sync and async
implementations.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Protocol, TypeVar, runtime_checkable
import pandas as pd
from abc import ABC, abstractmethod

from ...core.errors import YFinanceError

T = TypeVar('T')
logger = logging.getLogger(__name__)


@runtime_checkable
class BaseProviderProtocol(Protocol):
    """Protocol defining the basic structure all providers must implement."""
    
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic information for a ticker."""
        ...
    
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get current price data for a ticker."""
        ...
    
    def get_historical_data(self, ticker: str, period: Optional[str] = "1y", 
                          interval: Optional[str] = "1d") -> pd.DataFrame:
        """Get historical price data for a ticker."""
        ...
    
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings for a ticker."""
        ...
    
    def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data for a ticker."""
        ...
    
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers matching a query."""
        ...
    
    def batch_get_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get ticker information for multiple symbols in a batch."""
        ...


class FinanceDataProvider(ABC):
    """Abstract base class for finance data providers.
    
    This class defines the interface that all finance data providers must implement
    and provides some common utility methods.
    """
    
    @abstractmethod
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic information for a ticker."""
        raise NotImplementedError("Providers must implement get_ticker_info")
    
    @abstractmethod
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get current price data for a ticker."""
        raise NotImplementedError("Providers must implement get_price_data")
    
    @abstractmethod
    def get_historical_data(self, ticker: str, period: Optional[str] = "1y", 
                          interval: Optional[str] = "1d") -> pd.DataFrame:
        """Get historical price data for a ticker."""
        raise NotImplementedError("Providers must implement get_historical_data")
    
    @abstractmethod
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings for a ticker."""
        raise NotImplementedError("Providers must implement get_analyst_ratings")
    
    @abstractmethod
    def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data for a ticker."""
        raise NotImplementedError("Providers must implement get_earnings_data")
    
    @abstractmethod
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers matching a query."""
        raise NotImplementedError("Providers must implement search_tickers")
    
    def batch_get_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get ticker information for multiple symbols in a batch.
        
        Default implementation calls get_ticker_info for each ticker.
        Providers should override this method with a more efficient implementation
        if possible.
        
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
                logger.warning(f"Error getting data for {ticker}: {str(e)}")
                results[ticker] = None
        return results
    
    def _format_market_cap(self, value: Any) -> Optional[str]:
        """Format market cap with T/B suffix according to standard rules.
        
        Args:
            value: Market cap value
            
        Returns:
            Formatted market cap string
        """
        if value is None:
            return None
            
        try:
            # Convert to a float
            value_float = float(str(value).replace(',', ''))
            
            # Format in trillions or billions with correct precision
            if value_float >= 1_000_000_000_000:  # >= 1T
                value_trillions = value_float / 1_000_000_000_000
                if value_trillions >= 10:  # >= 10T: 1 decimal place
                    return f"{value_trillions:.1f}T"
                else:  # < 10T: 2 decimal places
                    return f"{value_trillions:.2f}T"
            else:  # < 1T (in billions)
                value_billions = value_float / 1_000_000_000
                if value_billions >= 100:  # >= 100B: no decimals
                    return f"{value_billions:.0f}B"
                elif value_billions >= 10:  # >= 10B: 1 decimal place
                    return f"{value_billions:.1f}B"
                else:  # < 10B: 2 decimal places
                    return f"{value_billions:.2f}B"
        except (ValueError, TypeError):
            return None
    
    def _calculate_upside_potential(self, current_price: float, target_price: float) -> Optional[float]:
        """Calculate upside potential as a percentage.
        
        Args:
            current_price: Current stock price
            target_price: Target price
            
        Returns:
            Upside potential as a percentage or None if inputs are invalid
        """
        if current_price is None or target_price is None or current_price <= 0:
            return None
        return ((target_price / current_price) - 1) * 100
    
    def _format_date(self, date) -> Optional[str]:
        """Format date as YYYY-MM-DD string.
        
        Args:
            date: Date object or string
            
        Returns:
            Formatted date string or None if invalid
        """
        if not date:
            return None
            
        try:
            # Handle pandas Timestamp
            if hasattr(date, 'strftime'):
                return date.strftime('%Y-%m-%d')
                
            # Handle string date
            date_obj = pd.to_datetime(date)
            return date_obj.strftime('%Y-%m-%d')
        except Exception:
            # Return as-is if we can't format it
            return str(date)
    
    def _validate_input(self, value: str, field_name: str) -> None:
        """Validate input strings.
        
        Args:
            value: Input value to validate
            field_name: Name of the field for error messages
            
        Raises:
            YFinanceError: If validation fails
        """
        if not isinstance(value, str):
            raise YFinanceError(f"{field_name} must be a string")
        if not value.strip():
            raise YFinanceError(f"{field_name} cannot be empty")


@runtime_checkable
class AsyncProviderProtocol(Protocol):
    """Protocol defining the basic structure all async providers must implement."""
    
    async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic information for a ticker asynchronously."""
        ...
    
    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get current price data for a ticker asynchronously."""
        ...
    
    async def get_historical_data(self, ticker: str, period: Optional[str] = "1y", 
                               interval: Optional[str] = "1d") -> pd.DataFrame:
        """Get historical price data for a ticker asynchronously."""
        ...
    
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings for a ticker asynchronously."""
        ...
    
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data for a ticker asynchronously."""
        ...
    
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers matching a query asynchronously."""
        ...
    
    async def batch_get_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get ticker information for multiple symbols in a batch asynchronously."""
        ...


class AsyncFinanceDataProvider(ABC):
    """Abstract base class for async finance data providers.
    
    This class defines the interface that all async finance data providers must implement
    and provides some common utility methods.
    """
    
    @abstractmethod
    async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic information for a ticker asynchronously."""
        raise NotImplementedError("Async providers must implement get_ticker_info")
    
    @abstractmethod
    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get current price data for a ticker asynchronously."""
        raise NotImplementedError("Async providers must implement get_price_data")
    
    @abstractmethod
    async def get_historical_data(self, ticker: str, period: Optional[str] = "1y", 
                               interval: Optional[str] = "1d") -> pd.DataFrame:
        """Get historical price data for a ticker asynchronously."""
        raise NotImplementedError("Async providers must implement get_historical_data")
    
    @abstractmethod
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings for a ticker asynchronously."""
        raise NotImplementedError("Async providers must implement get_analyst_ratings")
    
    @abstractmethod
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data for a ticker asynchronously."""
        raise NotImplementedError("Async providers must implement get_earnings_data")
    
    @abstractmethod
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers matching a query asynchronously."""
        raise NotImplementedError("Async providers must implement search_tickers")
    
    @abstractmethod
    async def batch_get_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get ticker information for multiple symbols in a batch asynchronously.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to their information
        """
        raise NotImplementedError("Async providers must implement batch_get_ticker_info")
    
    # Utility methods are the same as the sync version
    def _format_market_cap(self, value: Any) -> Optional[str]:
        """Format market cap with T/B suffix according to standard rules.
        
        Args:
            value: Market cap value
            
        Returns:
            Formatted market cap string
        """
        if value is None:
            return None
            
        try:
            # Convert to a float
            value_float = float(str(value).replace(',', ''))
            
            # Format in trillions or billions with correct precision
            if value_float >= 1_000_000_000_000:  # >= 1T
                value_trillions = value_float / 1_000_000_000_000
                if value_trillions >= 10:  # >= 10T: 1 decimal place
                    return f"{value_trillions:.1f}T"
                else:  # < 10T: 2 decimal places
                    return f"{value_trillions:.2f}T"
            else:  # < 1T (in billions)
                value_billions = value_float / 1_000_000_000
                if value_billions >= 100:  # >= 100B: no decimals
                    return f"{value_billions:.0f}B"
                elif value_billions >= 10:  # >= 10B: 1 decimal place
                    return f"{value_billions:.1f}B"
                else:  # < 10B: 2 decimal places
                    return f"{value_billions:.2f}B"
        except (ValueError, TypeError):
            return None
    
    def _calculate_upside_potential(self, current_price: float, target_price: float) -> Optional[float]:
        """Calculate upside potential as a percentage.
        
        Args:
            current_price: Current stock price
            target_price: Target price
            
        Returns:
            Upside potential as a percentage or None if inputs are invalid
        """
        if current_price is None or target_price is None or current_price <= 0:
            return None
        return ((target_price / current_price) - 1) * 100
    
    def _format_date(self, date) -> Optional[str]:
        """Format date as YYYY-MM-DD string.
        
        Args:
            date: Date object or string
            
        Returns:
            Formatted date string or None if invalid
        """
        if not date:
            return None
            
        try:
            # Handle pandas Timestamp
            if hasattr(date, 'strftime'):
                return date.strftime('%Y-%m-%d')
                
            # Handle string date
            date_obj = pd.to_datetime(date)
            return date_obj.strftime('%Y-%m-%d')
        except Exception:
            # Return as-is if we can't format it
            return str(date)
    
    def _validate_input(self, value: str, field_name: str) -> None:
        """Validate input strings.
        
        Args:
            value: Input value to validate
            field_name: Name of the field for error messages
            
        Raises:
            YFinanceError: If validation fails
        """
        if not isinstance(value, str):
            raise YFinanceError(f"{field_name} must be a string")
        if not value.strip():
            raise YFinanceError(f"{field_name} cannot be empty")