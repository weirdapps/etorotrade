"""
Yahoo Finance API provider implementation.

This module implements the FinanceDataProvider interface for Yahoo Finance data.
It wraps the existing functionality to provide a consistent interface.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from .base import FinanceDataProvider
from ...core.client import YFinanceClient
from ...core.errors import YFinanceError

logger = logging.getLogger(__name__)


class YahooFinanceProvider(FinanceDataProvider):
    """
    Yahoo Finance data provider implementation.
    
    This provider uses the core YFinanceClientCore to fetch data
    and adapts it to the provider interface.
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance provider with a client instance."""
        self.client = YFinanceClient()
    
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic information for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing stock information
        """
        try:
            stock_data = self.client.get_ticker_info(ticker)
            # Convert StockData object to dictionary
            return {
                'ticker': stock_data.symbol,
                'name': stock_data.name,
                'sector': stock_data.sector,
                'industry': stock_data.industry,
                'market_cap': stock_data.market_cap,
                'beta': stock_data.beta,
                'pe_trailing': stock_data.pe_trailing,
                'pe_forward': stock_data.pe_forward,
                'dividend_yield': stock_data.dividend_yield,
                'price': stock_data.current_price,
                'currency': stock_data.currency,
                'exchange': stock_data.exchange,
                'analyst_count': stock_data.analyst_count,
                'peg_ratio': stock_data.peg_ratio,
                'short_float_pct': stock_data.short_float_pct,
                'last_earnings': stock_data.last_earnings,
                'previous_earnings': stock_data.previous_earnings,
            }
        except Exception as e:
            logger.error(f"Error getting ticker info for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get ticker info: {str(e)}")
    
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing price information
        """
        try:
            # Use existing pricing analyzer functionality
            from ...pricing import PricingAnalyzer
            
            pricing = PricingAnalyzer(self.client)
            metrics = pricing.calculate_price_metrics(ticker)
            
            if not metrics:
                return {}
                
            return {
                'current_price': metrics.get('current_price'),
                'target_price': metrics.get('target_price'),
                'upside_potential': metrics.get('upside_potential'),
                'price_change': metrics.get('price_change'),
                'price_change_percentage': metrics.get('price_change_percentage'),
            }
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get price data: {str(e)}")
    
    def get_historical_data(self, 
                          ticker: str, 
                          period: Optional[str] = "1y", 
                          interval: Optional[str] = "1d") -> pd.DataFrame:
        """
        Get historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            DataFrame containing historical data
        """
        try:
            return self.client.get_historical_data(ticker, period, interval)
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get historical data: {str(e)}")
    
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing analyst ratings
        """
        try:
            # Use existing analyst functionality
            from ...analyst import AnalystData
            
            analyst = AnalystData(self.client)
            ratings = analyst.get_ratings_summary(ticker)
            
            if not ratings:
                return {}
                
            return {
                'positive_percentage': ratings.get('positive_percentage'),
                'total_ratings': ratings.get('total_ratings'),
                'ratings_type': ratings.get('ratings_type', ''),
                'recommendations': ratings.get('recommendations', {}),
            }
        except Exception as e:
            logger.error(f"Error getting analyst ratings for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get analyst ratings: {str(e)}")
    
    def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing earnings data
        """
        try:
            # Get earnings data
            stock_data = self.client.get_ticker_info(ticker)
            
            # Basic earnings information
            earnings_data = {
                'last_earnings': stock_data.last_earnings,
                'previous_earnings': stock_data.previous_earnings,
            }
            
            # Add detailed earnings from earnings module if needed
            from ...earnings import EarningsAnalyzer
            
            try:
                earnings_analyzer = EarningsAnalyzer(self.client)
                detailed_earnings = earnings_analyzer.get_earnings_data(ticker)
                if detailed_earnings:
                    earnings_data.update(detailed_earnings)
            except Exception:
                # Fallback to basic earnings data if detailed fetch fails
                pass
                
            return earnings_data
        except Exception as e:
            logger.error(f"Error getting earnings data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get earnings data: {str(e)}")
    
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching tickers with metadata
        """
        try:
            # Use core search functionality
            results = self.client.search_tickers(query, limit)
            
            # Format results according to common interface
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'symbol': result.get('symbol'),
                    'name': result.get('shortname', result.get('longname', '')),
                    'exchange': result.get('exchange', ''),
                    'type': result.get('quoteType', ''),
                    'score': result.get('score', 0),
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching tickers for '{query}': {str(e)}")
            raise YFinanceError(f"Failed to search tickers: {str(e)}")