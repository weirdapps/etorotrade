"""
Yahoo Finance provider implementation.

This module implements the FinanceDataProvider interface for Yahoo Finance data.
It provides a consistent API for retrieving financial information with 
appropriate rate limiting, caching, and error handling.
"""

from ...core.logging_config import get_logger

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation
import time
from typing import Dict, Any, Optional, List, Tuple, cast
import pandas as pd
import yfinance as yf

from .base_provider import FinanceDataProvider
from .yahoo_finance_base import YahooFinanceBaseProvider
from ...core.errors import YFinanceError, APIError, ValidationError, RateLimitError
from ...utils.market.ticker_utils import is_us_ticker
from ...utils.network.rate_limiter import rate_limited
from ...core.config import CACHE_CONFIG, COLUMN_NAMES

logger = get_logger(__name__)

class YahooFinanceProvider(YahooFinanceBaseProvider, FinanceDataProvider):
    """
    Yahoo Finance data provider implementation.
    
    This provider wraps the yfinance library with proper rate limiting,
    error handling, and caching to provide reliable access to financial data.
    
    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
        _ticker_cache: Cache of ticker information to avoid repeated fetches
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Yahoo Finance provider.
        
        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries
        """
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        
    def _handle_delay(self, delay: float):
        """
        Handle delaying execution for retry logic using synchronous time.sleep().
        
        Args:
            delay: Delay time in seconds
        """
        time.sleep(delay)
    
    @rate_limited
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing price data
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        logger.debug(f"Getting price data for {ticker}")
        info = self.get_ticker_info(ticker)
        
        # Extract price-related fields
        return {
            "ticker": ticker,
            "current_price": info.get("price"),
            "target_price": info.get("target_price"),
            "upside": self._calculate_upside_potential(info.get("price"), info.get("target_price")),
            "fifty_two_week_high": info.get("fifty_two_week_high"),
            "fifty_two_week_low": info.get("fifty_two_week_low"),
            "fifty_day_avg": info.get("fifty_day_avg"),
            "two_hundred_day_avg": info.get("two_hundred_day_avg")
        }
    
    @rate_limited
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics
            
        Returns:
            Dict containing stock information
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        logger.debug(f"Getting ticker info for {ticker}")
        ticker_obj = self._get_ticker_object(ticker)
        
        # Basic information with proper rate limiting
        result = {}
        
        for attempt in range(self.max_retries):
            try:
                # Get basic info
                info = ticker_obj.info
                if not info:
                    raise DataError(f"No information found for ticker {ticker}")
                
                # Extract key metrics using the base class helper
                result = self._extract_common_ticker_info(info)
                result["symbol"] = ticker  # Ensure the symbol is set correctly
                
                # Additional metrics for US stocks
                if is_us_ticker(ticker) and not skip_insider_metrics:
                    try:
                        # Get insider metrics
                        insider_data = self.get_insider_transactions(ticker)
                        if insider_data:
                            # Calculate insider metrics
                            total_buys = sum(1 for tx in insider_data if tx.get("shares", 0) > 0)
                            total_sells = sum(1 for tx in insider_data if tx.get("shares", 0) < 0)
                            
                            result["insider_transactions"] = len(insider_data)
                            result["insider_buys"] = total_buys
                            result["insider_sells"] = total_sells
                            result["insider_ratio"] = total_buys / (total_buys + total_sells) if (total_buys + total_sells) > 0 else 0
                    except YFinanceError as e:
                        logger.warning(f"Failed to get insider data for {ticker}: {str(e)}")
                
                break
            except RateLimitError as rate_error:
                # Use the shared retry logic handler from the base class
                raise rate_error
            except YFinanceError as e:
                # Use the shared retry logic handler from the base class
                delay = self._handle_retry_logic(e, attempt, ticker, "ticker info")
                time.sleep(delay)
        
        return result
    
    @rate_limited
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            DataFrame containing historical data
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        ticker_obj = self._get_ticker_object(ticker)
        
        for attempt in range(self.max_retries):
            try:
                # Use the shared _extract_historical_data method from the base class
                return self._extract_historical_data(ticker, ticker_obj, period, interval)
            except RateLimitError as rate_error:
                # Specific handling for rate limits - just re-raise YFinanceError("An error occurred")
                raise rate_error
            except YFinanceError as e:
                # Use the shared retry logic handler from the base class
                delay = self._handle_retry_logic(e, attempt, ticker, "historical data")
                time.sleep(delay)
    
    @rate_limited
    def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the last two earnings dates for a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple containing:
                - most_recent_date: The most recent earnings date in YYYY-MM-DD format
                - previous_date: The second most recent earnings date in YYYY-MM-DD format
                Both values will be None if no earnings dates are found
                
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        ticker_obj = self._get_ticker_object(ticker)
        
        for attempt in range(self.max_retries):
            try:
                # Get earnings data
                calendar = ticker_obj.calendar
                
                # Handle cases where calendar might be None or not have earnings date
                if calendar is None or COLUMN_NAMES['EARNINGS_DATE'] not in calendar:
                    logger.debug(f"No earnings dates found for {ticker}")
                    return None, None
                    
                earnings_date = calendar[COLUMN_NAMES['EARNINGS_DATE']]
                
                # Convert to list even if there's only one date
                if not isinstance(earnings_date, list):
                    earnings_date = [earnings_date]
                
                # Format dates
                formatted_dates = [self._format_date(date) for date in earnings_date if date is not None]
                
                # Sort dates in descending order
                formatted_dates.sort(reverse=True)
                
                # Return the last two earnings dates
                if len(formatted_dates) >= 2:
                    return formatted_dates[0], formatted_dates[1]
                elif len(formatted_dates) == 1:
                    return formatted_dates[0], None
                else:
                    return None, None
                    
            except RateLimitError as rate_error:
                # Specific handling for rate limits - just re-raise YFinanceError("An error occurred")
                raise rate_error
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} earnings dates: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise YFinanceError(f"Failed to get data after {self.max_retries} attempts")
    
    @rate_limited
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing analyst ratings information
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Skip analyst ratings for non-US tickers using the base method
        if not is_us_ticker(ticker):
            return self._get_empty_analyst_data(ticker)
        
        ticker_obj = self._get_ticker_object(ticker)
        
        for attempt in range(self.max_retries):
            try:
                # Get analyst consensus data using the enhanced base method
                consensus = self._get_analyst_consensus(ticker_obj, ticker)
                
                # Get the recommendations
                recommendations = ticker_obj.recommendations
                
                # Add symbol and date if available
                result = {"symbol": ticker}
                
                # Add date if we have recommendations
                if recommendations is not None and not recommendations.empty:
                    latest_date = recommendations.index.max()
                    result["date"] = self.format_date(latest_date)
                else:
                    result["date"] = None
                
                # Add recommendation count
                result["recommendations"] = consensus["total_ratings"]
                
                # Add buy percentage
                result["buy_percentage"] = consensus["buy_percentage"]
                
                # Add individual counts
                if "recommendations" in consensus and consensus["recommendations"]:
                    result["strong_buy"] = consensus["recommendations"].get("strong_buy", 0)
                    result["buy"] = consensus["recommendations"].get("buy", 0)
                    result["hold"] = consensus["recommendations"].get("hold", 0)
                    result["sell"] = consensus["recommendations"].get("sell", 0)
                    result["strong_sell"] = consensus["recommendations"].get("strong_sell", 0)
                else:
                    result["strong_buy"] = 0
                    result["buy"] = 0
                    result["hold"] = 0
                    result["sell"] = 0
                    result["strong_sell"] = 0
                
                return result
                
            except RateLimitError as rate_error:
                # Specific handling for rate limits - just re-raise YFinanceError("An error occurred")
                raise rate_error
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} analyst ratings: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise YFinanceError(f"Failed to get data after {self.max_retries} attempts")
    
    @rate_limited
    def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of dicts containing insider transaction information
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        ticker_obj = self._get_ticker_object(ticker)
        
        # Skip insider transactions for non-US tickers
        if not is_us_ticker(ticker):
            logger.debug(f"Skipping insider transactions for non-US ticker {ticker}")
            return []
        
        for attempt in range(self.max_retries):
            try:
                # Get insider transactions
                insiders = ticker_obj.institutional_holders
                
                # Handle case where there are no insider transactions
                if insiders is None or insiders.empty:
                    logger.debug(f"No insider transactions found for {ticker}")
                    return []
                
                # Convert to list of dicts
                result = []
                for _, row in insiders.iterrows():
                    transaction = {
                        "name": row.get("Holder", ""),
                        "shares": row.get("Shares", 0),
                        "date": str(row.get("Date Reported", "")),
                        "value": row.get("Value", 0),
                        "pct_out": row.get("% Out", 0) * 100 if row.get("% Out") else 0,
                    }
                    result.append(transaction)
                
                return result
                
            except RateLimitError as rate_error:
                # Specific handling for rate limits - just re-raise YFinanceError("An error occurred")
                raise rate_error
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} insider transactions: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise YFinanceError(f"Failed to get data after {self.max_retries} attempts")
    
    @rate_limited
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching tickers with metadata
            
        Raises:
            YFinanceError: When an error occurs while searching
        """
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        
        for attempt in range(self.max_retries):
            try:
                # Search for tickers
                search_results = yf.Ticker(query).search()
                
                # Handle case where there are no search results
                if not search_results or 'quotes' not in search_results or not search_results['quotes']:
                    logger.debug(f"No search results found for query '{query}'")
                    return []
                
                # Format results
                results = []
                for quote in search_results['quotes'][:limit]:
                    result = {
                        "symbol": quote.get("symbol", ""),
                        "name": quote.get("longname", quote.get("shortname", "")),
                        "exchange": quote.get("exchange", ""),
                        "type": quote.get("quoteType", ""),
                    }
                    results.append(result)
                
                return results
                
            except RateLimitError as rate_error:
                # Specific handling for rate limits - just re-raise YFinanceError("An error occurred")
                raise rate_error
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for search query '{query}': {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise YFinanceError(f"Failed to get data after {self.max_retries} attempts")
    
    @with_retry
    def batch_get_ticker_info(self, tickers: List[str], skip_insider_metrics: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers in a batch.
        
        Args:
            tickers: List of stock ticker symbols
            skip_insider_metrics: If True, skip fetching insider trading metrics
            
        Returns:
            Dict mapping ticker symbols to their information dicts
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        if not tickers:
            return {}
        
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_ticker_info(ticker, skip_insider_metrics)
            except YFinanceError as e:
                results[ticker] = self._process_error_for_batch(ticker, e)
        
        return results
    
    def clear_cache(self) -> None:
        """
        Clear the ticker object cache.
        """
        self._ticker_cache.clear()
        
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            Dict containing cache information
        """
        return {
            "ticker_cache_size": len(self._ticker_cache),
            "ticker_cache_keys": list(self._ticker_cache.keys())
        }