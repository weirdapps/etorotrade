"""
Async Yahoo Finance API provider implementation.

This module implements the AsyncFinanceDataProvider interface for Yahoo Finance data.
It provides asynchronous access to financial data with proper rate limiting.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from .async_base import AsyncFinanceDataProvider
from ...core.client import YFinanceClient
from ...core.errors import YFinanceError
from ...utils.async_helpers import AsyncRateLimiter, async_rate_limited, retry_async

logger = logging.getLogger(__name__)


class AsyncYahooFinanceProvider(AsyncFinanceDataProvider):
    """
    Async Yahoo Finance data provider implementation.
    
    This provider uses the core YFinanceClientCore to fetch data asynchronously
    and adapts it to the async provider interface with proper rate limiting.
    """
    
    def __init__(self, max_concurrency: int = 4):
        """
        Initialize the async Yahoo Finance provider.
        
        Args:
            max_concurrency: Maximum number of concurrent requests
        """
        self.client = YFinanceClient()
        self.limiter = AsyncRateLimiter(max_concurrency=max_concurrency)
    
    async def _run_sync_in_executor(self, func, *args, **kwargs):
        """
        Run a synchronous function in an executor to make it async.
        
        Args:
            func: Synchronous function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: func(*args, **kwargs)
        )
    
    @async_rate_limited(ticker_param='ticker')
    async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic information for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing stock information
        """
        try:
            # Run synchronous client method in executor
            stock_data = await self._run_sync_in_executor(
                self.client.get_ticker_info,
                ticker
            )
            
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
                'current_price': stock_data.current_price,
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
    
    @async_rate_limited(ticker_param='ticker')
    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing price information
        """
        try:
            # Use existing pricing analyzer functionality
            from ...pricing import PricingAnalyzer
            
            pricing = PricingAnalyzer(self.client)
            metrics = await self._run_sync_in_executor(
                pricing.calculate_price_metrics,
                ticker
            )
            
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
    
    @async_rate_limited(ticker_param='ticker')
    async def get_historical_data(self, 
                               ticker: str, 
                               period: Optional[str] = "1y", 
                               interval: Optional[str] = "1d") -> pd.DataFrame:
        """
        Get historical price data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            DataFrame containing historical data
        """
        try:
            return await self._run_sync_in_executor(
                self.client.get_historical_data,
                ticker,
                period,
                interval
            )
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get historical data: {str(e)}")
    
    @async_rate_limited(ticker_param='ticker')
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing analyst ratings
        """
        try:
            # Use existing analyst functionality
            from ...analyst import AnalystData
            
            analyst = AnalystData(self.client)
            ratings = await self._run_sync_in_executor(
                analyst.get_ratings_summary,
                ticker
            )
            
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
    
    @async_rate_limited(ticker_param='ticker')
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing earnings data
        """
        try:
            # Get earnings data
            stock_data = await self._run_sync_in_executor(
                self.client.get_ticker_info,
                ticker
            )
            
            # Basic earnings information
            earnings_data = {
                'last_earnings': stock_data.last_earnings,
                'previous_earnings': stock_data.previous_earnings,
            }
            
            # Add detailed earnings from earnings module if needed
            from ...earnings import EarningsAnalyzer
            
            try:
                earnings_analyzer = EarningsAnalyzer(self.client)
                detailed_earnings = await self._run_sync_in_executor(
                    earnings_analyzer.get_earnings_data,
                    ticker
                )
                if detailed_earnings:
                    earnings_data.update(detailed_earnings)
            except Exception as e:
                # Fallback to basic earnings data if detailed fetch fails
                logger.debug(f"Error getting detailed earnings for {ticker}: {str(e)}")
                
            return earnings_data
        except Exception as e:
            logger.error(f"Error getting earnings data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get earnings data: {str(e)}")
    
    @async_rate_limited()
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query asynchronously.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching tickers with metadata
        """
        try:
            # Use core search functionality
            results = await self._run_sync_in_executor(
                self.client.search_tickers,
                query,
                limit
            )
            
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
    
    async def batch_get_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple symbols in a batch.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to their information
        """
        # Create async tasks for each ticker
        tasks = [self.get_ticker_info(ticker) for ticker in tickers]
        
        # Use gather_with_rate_limit from async utilities
        from ...utils.async_helpers import gather_with_rate_limit
        
        results = await gather_with_rate_limit(
            tasks,
            max_concurrent=self.limiter.semaphore._value,
            return_exceptions=True
        )
        
        # Process results
        ticker_data = {}
        for i, ticker in enumerate(tickers):
            if isinstance(results[i], Exception):
                logger.warning(f"Error getting data for {ticker}: {str(results[i])}")
                ticker_data[ticker] = None
            else:
                ticker_data[ticker] = results[i]
        
        return ticker_data