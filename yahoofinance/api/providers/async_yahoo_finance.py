"""
Asynchronous Yahoo Finance provider implementation.

This module implements the AsyncFinanceDataProvider interface for Yahoo Finance data.
It provides a consistent async API for retrieving financial information with 
appropriate rate limiting, caching, and error handling.
"""

from ...core.logging import get_logger

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError, RateLimitError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, cast, TypeVar, Callable, Awaitable, Union
import pandas as pd
import yfinance as yf
from functools import wraps
import concurrent.futures
from datetime import datetime

from .base_provider import AsyncFinanceDataProvider
from .yahoo_finance_base import YahooFinanceBaseProvider
from ...utils.market.ticker_utils import validate_ticker, is_us_ticker
from ...utils.async_utils.helpers import async_rate_limited, gather_with_concurrency
from ...core.config import CACHE_CONFIG, COLUMN_NAMES

logger = get_logger(__name__)

# Constants
ERROR_GENERIC = "An error occurred"

T = TypeVar('T')  # Return type for async functions

class AsyncYahooFinanceProvider(YahooFinanceBaseProvider, AsyncFinanceDataProvider):
    """
    Asynchronous Yahoo Finance data provider implementation.
    
    This provider wraps the yfinance library with proper rate limiting,
    error handling, and caching to provide asynchronous access to financial data.
    
    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
        max_concurrency: Maximum number of concurrent operations
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, max_concurrency: int = 4):
        """
        Initialize the Async Yahoo Finance provider.
        
        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries (exponential backoff applied)
            max_concurrency: Maximum number of concurrent operations
        """
        # Call the base class constructor
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        
        # Add async-specific attributes
        self.max_concurrency = max_concurrency
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency)
        
    async def _handle_delay(self, delay: float):
        """
        Handle delaying execution for retry logic using asynchronous asyncio.sleep().
        
        Args:
            delay: Delay time in seconds
        """
        await asyncio.sleep(delay)
        
    # These methods are now inherited from YahooFinanceBaseProvider
    
    async def _run_sync_in_executor(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Run a synchronous function in an executor to make it async.
        
        Args:
            func: The synchronous function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, 
            lambda: func(*args, **kwargs)
        )
    
    @with_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
    async def _get_ticker_object(self, ticker: str) -> yf.Ticker:
        """
        Get a yfinance Ticker object for the given symbol with caching.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            yf.Ticker: Ticker object for the given symbol (do not store this long-term)
            
        Raises:
            ValidationError: If the ticker is invalid
        """
        # Validate the ticker format
        validate_ticker(ticker)
        
        # Create new ticker object every time - don't cache the actual objects to prevent memory leaks
        try:
            # Use safe_create_ticker from yfinance_utils if available
            try:
                from ...utils.yfinance_utils import safe_create_ticker
                ticker_obj = await self._run_sync_in_executor(safe_create_ticker, ticker)
            except ImportError:
                # Fall back to regular creation if the utility isn't available
                ticker_obj = await self._run_sync_in_executor(yf.Ticker, ticker)
                
            return ticker_obj
        except YFinanceError as e:
            logger.error(f"Error creating ticker object for {ticker}: {str(e)}")
            raise
    
    @async_rate_limited
    async def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics
            
        Returns:
            Dict containing stock information
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        logger.debug(f"Getting ticker info for {ticker}")
        ticker_obj = await self._get_ticker_object(ticker)
        
        # Basic information with proper rate limiting
        result = {}
        
        for attempt in range(self.max_retries):
            try:
                # Get basic info
                info = await self._run_sync_in_executor(lambda: ticker_obj.info)
                if not info:
                    raise DataError(f"No information available for {ticker}")
                
                # Use shared _process_ticker_info method from base class but with async adjustment for insider data
                result = await self._run_sync_in_executor(
                    lambda data=info: self._extract_common_ticker_info(data)
                )
                result["symbol"] = ticker  # Ensure the symbol is set correctly
                
                # Additional metrics for US stocks
                if is_us_ticker(ticker) and not skip_insider_metrics:
                    try:
                        # Get insider metrics
                        insider_data = await self.get_insider_transactions(ticker)
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
            except RateLimitError:
                # Specific handling for rate limits - just re-raise e
                raise e
            except YFinanceError as e:
                # Use the shared retry logic handler from the base class but with async sleep
                delay = self._handle_retry_logic(e, attempt, ticker, "ticker info")
                await asyncio.sleep(delay)
        
        return result
    
    @async_rate_limited
    async def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            DataFrame containing historical data
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Create a temporary ticker object
        ticker_obj = None
        result = None
        
        try:
            # Get ticker object for this request only
            ticker_obj = await self._get_ticker_object(ticker)
            
            for attempt in range(self.max_retries):
                try:
                    # Use the shared extraction method but run in executor
                    history = await self._run_sync_in_executor(
                        lambda: self._extract_historical_data(ticker, ticker_obj, period, interval)
                    )
                    
                    # Create a deep copy of the DataFrame to avoid reference issues
                    if isinstance(history, pd.DataFrame):
                        # Use copy to ensure we don't maintain references to original data
                        result = history.copy(deep=True)
                        
                        # Clear any datetime64 objects with timezone info that might cause memory leaks
                        for col in result.select_dtypes(include=['datetime64[ns]']).columns:
                            result[col] = pd.to_datetime(result[col].dt.strftime('%Y-%m-%d %H:%M:%S'))
                            
                    else:
                        result = history
                        
                    break
                    
                except RateLimitError:
                    # Specific handling for rate limits - just re-raise with generic error
                    raise YFinanceError(ERROR_GENERIC)
                except YFinanceError as e:
                    # Use the shared retry logic handler from the base class but with async sleep
                    delay = self._handle_retry_logic(e, attempt, ticker, "historical data")
                    await asyncio.sleep(delay)
            
            return result
            
        finally:
            # Explicitly clean up and remove the reference to the ticker object
            if ticker_obj is not None:
                try:
                    # Use clean_ticker_object from yfinance_utils if available
                    from ...utils.yfinance_utils import clean_ticker_object
                    await self._run_sync_in_executor(clean_ticker_object, ticker_obj)
                except ImportError:
                    # Fall back to basic cleanup if the utility isn't available
                    pass
                    
                # Remove reference
                del ticker_obj
                
            # Force garbage collection to clean up any resources
            import gc
            gc.collect()
            
            # Clear DataFrame cache in pandas if possible
            try:
                pd.core.common._possibly_clean_cache()
            except (AttributeError, ImportError):
                pass
                
            # Use memory utilities for thorough cleanup
            try:
                from ...utils.memory_utils import clean_memory
                clean_memory()
            except ImportError:
                pass
    
    @async_rate_limited
    async def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the last two earnings dates for a stock asynchronously.
        
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
        ticker_obj = await self._get_ticker_object(ticker)
        
        for attempt in range(self.max_retries):
            try:
                # Get earnings data
                calendar = await self._run_sync_in_executor(lambda: ticker_obj.calendar)
                
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
                    
            except RateLimitError:
                # Specific handling for rate limits - just re-raise with generic error
                raise YFinanceError(ERROR_GENERIC)
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} earnings dates: {str(e)}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    raise e
    
    @async_rate_limited
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker asynchronously.
        
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
        
        ticker_obj = await self._get_ticker_object(ticker)
        
        for attempt in range(self.max_retries):
            try:
                # Get analyst consensus data using the enhanced base method
                # Run in executor since it takes a sync ticker_obj
                consensus = await self._run_sync_in_executor(
                    lambda: self._get_analyst_consensus(ticker_obj, ticker)
                )
                
                # Get the recommendations
                recommendations = await self._run_sync_in_executor(lambda: ticker_obj.recommendations)
                
                # Add symbol and date if available
                result = {"symbol": ticker}
                
                # Add date if we have recommendations
                if recommendations is not None and not recommendations.empty:
                    latest_date = recommendations.index.max()
                    result["date"] = self._format_date(latest_date)
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
                
            except RateLimitError:
                # Specific handling for rate limits - just re-raise with generic error
                raise YFinanceError(ERROR_GENERIC)
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} analyst ratings: {str(e)}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    raise e
    
    @async_rate_limited
    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of dicts containing insider transaction information
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        ticker_obj = await self._get_ticker_object(ticker)
        
        # Skip insider transactions for non-US tickers
        if not is_us_ticker(ticker):
            logger.debug(f"Skipping insider transactions for non-US ticker {ticker}")
            return []
        
        for attempt in range(self.max_retries):
            try:
                # Get insider transactions
                insiders = await self._run_sync_in_executor(lambda: ticker_obj.institutional_holders)
                
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
                        "date": self._format_date(row.get("Date Reported", None)),
                        "value": row.get("Value", 0),
                        "pct_out": row.get("% Out", 0) * 100 if row.get("% Out") else 0,
                    }
                    result.append(transaction)
                
                return result
                
            except RateLimitError:
                # Specific handling for rate limits - just re-raise with generic error
                raise YFinanceError(ERROR_GENERIC)
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} insider transactions: {str(e)}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    raise e
    
    @async_rate_limited
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query asynchronously.
        
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
                ticker_obj = await self._run_sync_in_executor(yf.Ticker, query)
                search_results = await self._run_sync_in_executor(lambda: ticker_obj.search())
                
                # Handle case where there are no search results
                if not search_results or 'quotes' not in search_results or not search_results['quotes']:
                    # If no results, it's not necessarily an error, just no match
                    return []
                
                # Return the search results
                return search_results.get('quotes', [])
            except Exception as e:
                # Log the error and retry if attempts remain
                self.logger.warning(f"Search attempt {attempt + 1} failed for query '{query}': {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (self.backoff_factor ** attempt))
                else:
                    # If last attempt failed, raise the exception
                    raise APIError(f"Failed to search for ticker '{query}' after {self.max_retries} attempts: {e}") from e

        # Should not reach here if max_retries > 0
        return []
            
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Search for tickers matching a query.
        
        Args:
            query: Search query
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
                ticker_obj = await self._run_sync_in_executor(yf.Ticker, query)
                search_results = await self._run_sync_in_executor(lambda obj=ticker_obj: obj.search())
                
                # Handle case where there are no search results
                if not search_results or 'quotes' not in search_results or not search_results['quotes']:
                    # If no results, it's not necessarily an error, just no match
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
            except RateLimitError as e:
                # Specific handling for rate limits
                self.logger.warning(f"Rate limit hit for search query '{query}': {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    self.logger.warning(f"Retrying search for '{query}' in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise APIError(f"Rate limit exceeded for search query '{query}' after {self.max_retries} attempts") from e
            except Exception as e:
                # Handle other exceptions
                self.logger.warning(f"Search attempt {attempt + 1} failed for query '{query}': {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    self.logger.warning(f"Retrying search for '{query}' in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise APIError(f"Failed to search for ticker '{query}' after {self.max_retries} attempts: {e}") from e

        # Should not reach here if max_retries > 0
        return []
    
    @async_rate_limited
    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get price data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing price data
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        logger.debug(f"Getting price data for {ticker}")
        info = await self.get_ticker_info(ticker)
        
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
        
    async def _get_single_ticker_info_async(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Helper to get comprehensive information for a single ticker asynchronously.
        This encapsulates the core logic used by get_ticker_info and batch_get_ticker_info.
        """
        logger.debug(f"Fetching single ticker info for {ticker}")
        ticker_obj = await self._get_ticker_object(ticker)

        result = {}
        for attempt in range(self.max_retries):
            try:
                info = await self._run_sync_in_executor(lambda: ticker_obj.info)
                if not info:
                    raise DataError(f"No information available for {ticker}")

                result = await self._run_sync_in_executor(
                    lambda data=info: self._extract_common_ticker_info(data)
                )
                result["symbol"] = ticker

                if is_us_ticker(ticker) and not skip_insider_metrics:
                    try:
                        insider_data = await self.get_insider_transactions(ticker)
                        if insider_data:
                            total_buys = sum(1 for tx in insider_data if tx.get("shares", 0) > 0)
                            total_sells = sum(1 for tx in insider_data if tx.get("shares", 0) < 0)
                            result["insider_transactions"] = len(insider_data)
                            result["insider_buys"] = total_buys
                            result["insider_sells"] = total_sells
                            result["insider_ratio"] = total_buys / (total_buys + total_sells) if (total_buys + total_sells) > 0 else 0
                    except YFinanceError as e:
                        logger.warning(f"Failed to get insider data for {ticker}: {str(e)}")

                break
            except RateLimitError:
                raise
            except YFinanceError as e:
                delay = self._handle_retry_logic(e, attempt, ticker, "single ticker info")
                await asyncio.sleep(delay)

        return result

    async def batch_get_ticker_info(self, tickers: List[str], skip_insider_metrics: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers in a batch asynchronously.
        
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
            
        logger.debug(f"Getting batch ticker info for {len(tickers)} tickers")
        
        # Create a list to store results
        results = {}
        
        try:
            # Process tickers in batches with controlled concurrency
            for ticker in tickers:
                try:
                    # Use the single ticker helper function
                    info = await self._get_single_ticker_info_async(ticker, skip_insider_metrics)
                    # Create a deep copy of the info to avoid reference issues
                    import copy
                    results[ticker] = copy.deepcopy(info)
                except YFinanceError as e:
                    results[ticker] = self._process_error_for_batch(ticker, e)
                
                # Force garbage collection after each ticker to prevent memory accumulation
                if len(results) % 5 == 0:  # Every 5 tickers
                    # Use memory utilities for thorough cleanup
                    try:
                        from ...utils.memory_utils import clean_memory
                        clean_memory()
                    except ImportError:
                        # Fall back to basic garbage collection
                        import gc
                        gc.collect()
            
            return results
        finally:
            # Explicitly trigger garbage collection to clean up resources
            # This helps prevent memory leaks when processing large batches
            import gc
            gc.collect()
            gc.collect()  # Run twice to handle objects with circular references
        
    
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
            "ticker_cache_keys": list(self._ticker_cache.keys()),
            "max_concurrency": self.max_concurrency
        }
        
    def __del__(self):
        """Cleanup resources when the object is garbage collected."""
        # Clear all caches
        if hasattr(self, '_ticker_cache'):
            self._ticker_cache.clear()
            
        # Shutdown the thread pool to prevent memory leaks
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
            
        # Use memory_utils if available for efficient cleanup
        try:
            from ...utils.memory_utils import clean_memory
            clean_memory()
        except ImportError:
            # Fallback to manual cleanup if memory_utils not available
            # Clear any module references that might cause memory leaks
            # Particularly ABC module and tzinfo from datetime
            import sys
            for module_name in ['abc', 'pandas', 'datetime']:
                if module_name in sys.modules:
                    try:
                        # Clear module-specific caches if needed
                        module = sys.modules[module_name]
                        if hasattr(module, '_abc_registry') and isinstance(module._abc_registry, dict):
                            module._abc_registry.clear()
                        if hasattr(module, '_abc_cache') and isinstance(module._abc_cache, dict):
                            module._abc_cache.clear()
                    except (AttributeError, KeyError):
                        pass
            
            # Force garbage collection
            import gc
            gc.collect()