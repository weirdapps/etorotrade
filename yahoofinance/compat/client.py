"""
Compatibility module for YFinanceClient from v1.

This module provides a YFinanceClient class that mirrors the interface of
the v1 client but uses the v2 provider implementation under the hood.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from ..api import get_provider, FinanceDataProvider
from ..core.config import CACHE_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Compatibility class for v1 StockData"""
    # Required to maintain v1 interface
    ticker: str
    name: Optional[str] = None
    price: Optional[float] = None
    price_change: Optional[float] = None
    price_change_percentage: Optional[float] = None
    market_cap: Optional[float] = None
    analyst_count: Optional[int] = None
    target_price: Optional[float] = None
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    peg_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    short_float_pct: Optional[float] = None
    last_earnings: Optional[str] = None
    insider_buy_pct: Optional[float] = None
    insider_transactions: Optional[int] = None
    total_ratings: Optional[int] = None
    hold_pct: Optional[float] = None
    buy_pct: Optional[float] = None
    # Add other fields used in v1 here
    
    @classmethod
    def from_provider_data(cls, ticker: str, data: Dict[str, Any]) -> 'StockData':
        """
        Create StockData from provider data dictionary.
        
        Args:
            ticker: Stock ticker symbol
            data: Provider data dictionary
            
        Returns:
            StockData instance
        """
        return cls(
            ticker=ticker,
            name=data.get("name"),
            price=data.get("price"),
            price_change=data.get("price_change"),
            price_change_percentage=data.get("price_change_percentage"),
            market_cap=data.get("market_cap"),
            analyst_count=data.get("analyst_count"),
            target_price=data.get("target_price"),
            pe_trailing=data.get("pe_trailing"),
            pe_forward=data.get("pe_forward"),
            peg_ratio=data.get("peg_ratio"),
            dividend_yield=data.get("dividend_yield"),
            beta=data.get("beta"),
            short_float_pct=data.get("short_float_pct"),
            last_earnings=data.get("last_earnings"),
            insider_buy_pct=data.get("insider_buy_pct"),
            insider_transactions=data.get("insider_transactions"),
            total_ratings=data.get("total_ratings"),
            hold_pct=data.get("hold_percentage"),
            buy_pct=data.get("buy_percentage")
        )

class YFinanceClient:
    """
    Compatibility class for v1 YFinanceClient.
    
    Uses v2 provider pattern under the hood.
    """
    
    def __init__(self, cache_timeout: int = None):
        """
        Initialize v1-compatible client.
        
        Args:
            cache_timeout: Cache timeout in seconds (v1 compatibility)
        """
        # Get v2 provider
        self.provider = get_provider()
        
        # Set up cache timeout (for v1 compatibility)
        if cache_timeout is not None:
            self.cache_timeout = cache_timeout
        else:
            self.cache_timeout = CACHE_CONFIG.get("MARKET_CACHE_TTL", 300)
    
    def get_ticker_info(self, ticker: str) -> Optional[StockData]:
        """
        Get stock information with v1-compatible interface.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            StockData object with ticker information
        """
        try:
            # Get data from v2 provider
            data = self.provider.get_ticker_info(ticker)
            if not data:
                return None
                
            # Convert to v1-compatible StockData
            return StockData.from_provider_data(ticker, data)
            
        except Exception as e:
            logger.error(f"Error getting ticker info for {ticker}: {str(e)}")
            return None
    
    def get_analyst_ratings(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get analyst ratings with v1-compatible interface.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with analyst ratings
        """
        try:
            # Get data from v2 provider
            data = self.provider.get_analyst_ratings(ticker)
            return data
            
        except Exception as e:
            logger.error(f"Error getting analyst ratings for {ticker}: {str(e)}")
            return None
    
    def get_price_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get price data with v1-compatible interface.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with price data
        """
        try:
            # Get data from v2 provider
            data = self.provider.get_ticker_info(ticker)
            return {
                "current_price": data.get("price"),
                "target_price": data.get("target_price"),
                "upside": data.get("upside"),
                "ticker": ticker
            }
            
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {str(e)}")
            return None
    
    def get_price_targets(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get price targets with v1-compatible interface.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with price targets
        """
        try:
            # Get data from v2 provider
            data = self.provider.get_ticker_info(ticker)
            return {
                "target_price": data.get("target_price"),
                "upside": data.get("upside"),
                "ticker": ticker
            }
            
        except Exception as e:
            logger.error(f"Error getting price targets for {ticker}: {str(e)}")
            return None
    
    def get_insider_transactions(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get insider transactions with v1-compatible interface.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of insider transactions
        """
        try:
            # Get data from v2 provider
            data = self.provider.get_insider_transactions(ticker)
            return data
            
        except Exception as e:
            logger.error(f"Error getting insider transactions for {ticker}: {str(e)}")
            return None
    
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> Optional[Dict[str, Any]]:
        """
        Get historical price data with v1-compatible interface.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary with historical price data
        """
        try:
            # Get data from v2 provider
            data = self.provider.get_historical_data(ticker, period, interval)
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            return None
    
    def get_news(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get news articles with v1-compatible interface.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of news articles
        """
        try:
            # Get data from v2 provider
            data = self.provider.get_news(ticker)
            return data
            
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {str(e)}")
            return None
    
    def get_institutional_holders(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get institutional holders with v1-compatible interface.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of institutional holders
        """
        try:
            # Get data from v2 provider
            data = self.provider.get_institutional_holders(ticker)
            return data
            
        except Exception as e:
            logger.error(f"Error getting institutional holders for {ticker}: {str(e)}")
            return None