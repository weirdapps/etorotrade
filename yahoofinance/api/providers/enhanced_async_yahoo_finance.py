"""
Enhanced asynchronous Yahoo Finance provider implementation.

This module implements an improved AsyncFinanceDataProvider for Yahoo Finance data
using true async I/O, circuit breaking, and enhanced resilience patterns.
It provides significantly improved performance and reliability compared to
the thread-pool based implementation.
"""

import logging
import asyncio
import time
import random
import aiohttp
from typing import Dict, Any, Optional, List, Tuple, cast, TypeVar, Callable, Awaitable, Union
import pandas as pd
from functools import wraps

from .base_provider import AsyncFinanceDataProvider
from ...core.errors import YFinanceError, APIError, ValidationError, RateLimitError, NetworkError
from ...utils.market.ticker_utils import validate_ticker, is_us_ticker

# Constants
EARNINGS_DATE_COL = "Earnings Date"
from ...utils.async_utils.enhanced import (
    AsyncRateLimiter, 
    process_batch_async, 
    enhanced_async_rate_limited,
    gather_with_concurrency
)
from ...utils.network.circuit_breaker import CircuitOpenError
from ...core.config import RATE_LIMIT, CIRCUIT_BREAKER, CACHE_CONFIG

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Return type for async functions

class EnhancedAsyncYahooFinanceProvider(AsyncFinanceDataProvider):
    """
    Enhanced asynchronous Yahoo Finance data provider implementation.
    
    This provider uses true async I/O with aiohttp for HTTP requests,
    along with advanced resilience patterns including circuit breaking,
    enhanced rate limiting, and intelligent retries.
    
    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
        max_concurrency: Maximum number of concurrent operations
        session: aiohttp ClientSession for HTTP requests
    """
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 1.0, 
                 max_concurrency: int = 5,
                 enable_circuit_breaker: bool = True):
        """
        Initialize the Enhanced Async Yahoo Finance provider.
        
        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries (exponential backoff applied)
            max_concurrency: Maximum number of concurrent operations
            enable_circuit_breaker: Whether to enable the circuit breaker pattern
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrency = max_concurrency
        self.enable_circuit_breaker = enable_circuit_breaker
        self._ticker_cache: Dict[str, Dict[str, Any]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = AsyncRateLimiter()
        
        # Circuit breaker configuration - used for all methods
        self._circuit_name = "yahoofinance_api"
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """
        Ensure that an aiohttp session exists, creating one if needed.
        
        Returns:
            aiohttp ClientSession instance
        """
        if self._session is None or self._session.closed:
            # Configure session with appropriate headers and timeouts
            timeout = aiohttp.ClientTimeout(
                total=RATE_LIMIT["API_TIMEOUT"],
                connect=10,
                sock_connect=10,
                sock_read=RATE_LIMIT["API_TIMEOUT"]
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Origin": "https://finance.yahoo.com",
                    "Referer": "https://finance.yahoo.com/"
                }
            )
        return self._session
    
    async def _fetch_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch JSON data from a URL with proper error handling.
        
        Args:
            url: URL to fetch
            params: Query parameters
            
        Returns:
            Parsed JSON data
            
        Raises:
            NetworkError: If there's a network error
            RateLimitError: If the request is rate limited
            APIError: For other API errors
        """
        # Decorator function for HTTP request with all protections
        @enhanced_async_rate_limited(
            circuit_name=self._circuit_name if self.enable_circuit_breaker else None,
            max_retries=self.max_retries,
            rate_limiter=self._rate_limiter
        )
        async def _do_fetch() -> Dict[str, Any]:
            session = await self._ensure_session()
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", "60"))
                        raise RateLimitError(
                            f"Yahoo Finance API rate limit exceeded. Retry after {retry_after} seconds",
                            retry_after=retry_after
                        )
                    elif response.status == 404:
                        raise ValidationError(f"Resource not found at {url}")
                    else:
                        text = await response.text()
                        details = {"status_code": response.status, "response_text": text[:100]}
                        raise APIError(
                            f"Yahoo Finance API error: {response.status} - {text[:100]}",
                            details=details
                        )
            except aiohttp.ClientError as e:
                raise NetworkError(f"Network error while fetching {url}: {str(e)}")
        
        try:
            return await _do_fetch()
        except CircuitOpenError as e:
            # Translate circuit breaker errors to more user-friendly messages
            retry_after = e.retry_after
            details = {"status_code": 503, "retry_after": retry_after}
            raise APIError(
                f"Yahoo Finance API is currently unavailable. Please try again in {retry_after} seconds",
                details=details
            )
    
    @enhanced_async_rate_limited(max_retries=0)
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
        # Validate the ticker format
        validate_ticker(ticker)
        
        logger.debug(f"Getting ticker info for {ticker}")
        
        # Check cache first
        if ticker in self._ticker_cache:
            logger.debug(f"Using cached data for {ticker}")
            return self._ticker_cache[ticker]
            
        try:
            # Use yfinance library directly for reliability
            import yfinance as yf
            
            # Create a new ticker object
            yticker = yf.Ticker(ticker)
            
            # Extract all needed data
            info: Dict[str, Any] = {
                "symbol": ticker
            }
            
            # Get info dictionary
            ticker_info = yticker.info
            
            # Extract key fields
            info["name"] = ticker_info.get("longName", ticker_info.get("shortName", ""))
            info["sector"] = ticker_info.get("sector", "")
            info["industry"] = ticker_info.get("industry", "")
            info["country"] = ticker_info.get("country", "")
            info["website"] = ticker_info.get("website", "")
            info["current_price"] = ticker_info.get("regularMarketPrice", None)
            info["currency"] = ticker_info.get("currency", "")
            info["market_cap"] = ticker_info.get("marketCap", None)
            info["exchange"] = ticker_info.get("exchange", "")
            info["quote_type"] = ticker_info.get("quoteType", "")
            info["pe_trailing"] = ticker_info.get("trailingPE", None)
            info["dividend_yield"] = ticker_info.get("dividendYield", None)
            if info["dividend_yield"] is not None:
                info["dividend_yield"] = info["dividend_yield"] * 100
            info["beta"] = ticker_info.get("beta", None)
            info["fifty_two_week_high"] = ticker_info.get("fiftyTwoWeekHigh", None)
            info["fifty_two_week_low"] = ticker_info.get("fiftyTwoWeekLow", None)
            info["pe_forward"] = ticker_info.get("forwardPE", None)
            info["peg_ratio"] = ticker_info.get("pegRatio", None)
            info["short_percent"] = ticker_info.get("shortPercentOfFloat", None)
            if info["short_percent"] is not None:
                info["short_percent"] = info["short_percent"] * 100
            
            # Get target price
            info["target_price"] = ticker_info.get("targetMeanPrice", None)
            info["recommendation"] = ticker_info.get("recommendationMean", None)
            
            # Get analyst ratings if available - using recommendationKey directly instead of intermediate map
            
            # Calculate buy percentage from recommendation
            if "numberOfAnalystOpinions" in ticker_info and ticker_info["numberOfAnalystOpinions"]:
                total_analysts = ticker_info["numberOfAnalystOpinions"]
                info["analyst_count"] = total_analysts
                
                # Determine buy percentage based on recommendation key
                rec_key = ticker_info.get("recommendationKey", "").lower()
                if rec_key in ("strong_buy", "buy"):
                    info["buy_percentage"] = 90  # Approximate buy percentage for buy recommendations
                elif rec_key == "hold":
                    info["buy_percentage"] = 65  # Approximate buy percentage for hold recommendations
                elif rec_key in ("sell", "strong_sell"):
                    info["buy_percentage"] = 30  # Approximate buy percentage for sell recommendations
                else:
                    info["buy_percentage"] = None
            else:
                info["analyst_count"] = 0
                info["buy_percentage"] = None
                
            # Format market cap
            if info.get("market_cap"):
                info["market_cap_fmt"] = self._format_market_cap(info["market_cap"])
                
            # Calculate upside potential
            price = info.get("current_price")
            target = info.get("target_price")
            info["upside"] = self._calculate_upside_potential(price, target)
            
            # Get earnings date
            try:
                calendar = yticker.calendar
                if calendar is not None and not calendar.empty and EARNINGS_DATE_COL in calendar.columns:
                    earnings_date = calendar[EARNINGS_DATE_COL].iloc[0]
                    if pd.notna(earnings_date):
                        info["last_earnings"] = earnings_date.strftime("%Y-%m-%d")
            except Exception as e:
                logger.warning(f"Failed to get earnings date for {ticker}: {str(e)}")
            
            # Add to cache before returning
            self._ticker_cache[ticker] = info
            return info
            
        except Exception as e:
            # Translate specific errors for better user experience
            if isinstance(e, (APIError, ValidationError, RateLimitError, NetworkError)):
                raise e
            else:
                raise APIError(f"Failed to get ticker info for {ticker}: {str(e)}")
    
    @enhanced_async_rate_limited(max_retries=0)
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
        # Validate the ticker format
        validate_ticker(ticker)
        
        try:
            # Use yfinance library directly for reliability
            import yfinance as yf
            
            # Map period and interval formats if needed
            period_map = {
                "1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo", 
                "6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "5y", 
                "10y": "10y", "ytd": "ytd", "max": "max"
            }
            api_period = period_map.get(period, period)
            
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", 
                "60m": "60m", "1h": "60m", "1d": "1d", "5d": "5d", 
                "1wk": "1wk", "1mo": "1mo", "3mo": "3mo"
            }
            api_interval = interval_map.get(interval, interval)
            
            # Create a ticker object
            yticker = yf.Ticker(ticker)
            
            # Get historical data directly
            df = yticker.history(period=api_period, interval=api_interval)
            
            # Rename columns to match expected format if needed
            if not df.empty and 'Open' not in df.columns and 'open' in df.columns:
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                }, inplace=True)
                
            return df
            
        except Exception as e:
            # Translate specific errors for better user experience
            if isinstance(e, (APIError, ValidationError, RateLimitError, NetworkError)):
                raise e
            else:
                raise APIError(f"Failed to get historical data for {ticker}: {str(e)}")
    
    @enhanced_async_rate_limited(max_retries=0)
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing earnings information
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Validate the ticker format
        validate_ticker(ticker)
        
        try:
            # Use yfinance library directly for reliability
            import yfinance as yf
            
            # Create a ticker object
            yticker = yf.Ticker(ticker)
            
            # Initialize earnings data
            earnings_data = {
                "symbol": ticker,
                "earnings_dates": [],
                "earnings_estimates": None,
                "revenue_estimates": None,
                "quarter_end": None,
                "earnings_average": None,
                "earnings_low": None,
                "earnings_high": None,
                "revenue_average": None,
                "revenue_low": None,
                "revenue_high": None,
                "earnings_history": []
            }
            
            # Get calendar for earnings dates
            try:
                calendar = yticker.calendar
                if calendar is not None and not calendar.empty and EARNINGS_DATE_COL in calendar.columns:
                    earnings_date = calendar[EARNINGS_DATE_COL].iloc[0]
                    if pd.notna(earnings_date):
                        earnings_data["earnings_dates"].append(earnings_date.strftime("%Y-%m-%d"))
            except Exception as e:
                logger.warning(f"Failed to get earnings calendar for {ticker}: {str(e)}")
            
            # Get earnings history
            try:
                earnings_hist = yticker.earnings_history
                if earnings_hist is not None and not earnings_hist.empty:
                    # Process earnings history
                    for idx, row in earnings_hist.iterrows():
                        quarter_data = {
                            "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                            "reported_eps": row.get("Reported EPS", None),
                            "estimated_eps": row.get("Estimated EPS", None),
                            "surprise": row.get("Surprise(%)", None)
                        }
                        earnings_data["earnings_history"].append(quarter_data)
            except Exception as e:
                logger.warning(f"Failed to get earnings history for {ticker}: {str(e)}")
            
            # Get earnings estimate
            try:
                ticker_info = yticker.info
                
                if "trailingEps" in ticker_info:
                    earnings_data["earnings_average"] = ticker_info["trailingEps"]
                
                if "forwardEps" in ticker_info:
                    earnings_data["earnings_estimates"] = ticker_info["forwardEps"]
                
                if "lastFiscalYearEnd" in ticker_info:
                    from datetime import datetime
                    last_fiscal = datetime.fromtimestamp(ticker_info["lastFiscalYearEnd"])
                    earnings_data["quarter_end"] = last_fiscal.strftime("%Y-%m-%d")
            except Exception as e:
                logger.warning(f"Failed to get earnings estimates for {ticker}: {str(e)}")
            
            return earnings_data
            
        except Exception as e:
            # Translate specific errors for better user experience
            if isinstance(e, (APIError, ValidationError, RateLimitError, NetworkError)):
                raise e
            else:
                raise APIError(f"Failed to get earnings data for {ticker}: {str(e)}")
    
    @enhanced_async_rate_limited(max_retries=0)
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
        # Validate the ticker format
        validate_ticker(ticker)
        
        # Skip analyst ratings for non-US tickers
        if not is_us_ticker(ticker):
            logger.debug(f"Skipping analyst ratings for non-US ticker {ticker}")
            return {
                "symbol": ticker,
                "recommendations": 0,
                "buy_percentage": None,
                "positive_percentage": None,
                "total_ratings": 0,
                "ratings_type": "buy_sell_hold",
                "strong_buy": 0,
                "buy": 0,
                "hold": 0,
                "sell": 0,
                "strong_sell": 0,
                "date": None
            }
        
        try:
            # Use yfinance library directly for reliability
            import yfinance as yf
            
            # Create a ticker object
            yticker = yf.Ticker(ticker)
            ticker_info = yticker.info
            
            # Default values
            ratings_data = {
                "symbol": ticker,
                "recommendations": 0,
                "buy_percentage": None,
                "positive_percentage": None,
                "total_ratings": 0,
                "ratings_type": "buy_sell_hold",
                "strong_buy": 0,
                "buy": 0,
                "hold": 0,
                "sell": 0,
                "strong_sell": 0,
                "date": None
            }
            
            # Get analyst opinions if available
            if "numberOfAnalystOpinions" in ticker_info and ticker_info["numberOfAnalystOpinions"]:
                total_analysts = ticker_info["numberOfAnalystOpinions"]
                ratings_data["recommendations"] = total_analysts
                ratings_data["total_ratings"] = total_analysts
                
                # Get recommendation key
                rec_key = ticker_info.get("recommendationKey", "").lower()
                
                # Map recommendation to buy percentage and individual ratings
                if rec_key == "strong_buy":
                    ratings_data["buy_percentage"] = 95
                    ratings_data["positive_percentage"] = 95
                    ratings_data["strong_buy"] = total_analysts
                elif rec_key == "buy":
                    ratings_data["buy_percentage"] = 85
                    ratings_data["positive_percentage"] = 85
                    ratings_data["buy"] = total_analysts
                elif rec_key == "hold":
                    ratings_data["buy_percentage"] = 50
                    ratings_data["positive_percentage"] = 50
                    ratings_data["hold"] = total_analysts
                elif rec_key == "sell":
                    ratings_data["buy_percentage"] = 20
                    ratings_data["positive_percentage"] = 20
                    ratings_data["sell"] = total_analysts
                elif rec_key == "strong_sell":
                    ratings_data["buy_percentage"] = 5
                    ratings_data["positive_percentage"] = 5
                    ratings_data["strong_sell"] = total_analysts
                
                # Get date (use current date if not available)
                from datetime import datetime
                ratings_data["date"] = datetime.now().strftime("%Y-%m")
            
            # Try to get recommendations dataframe if simple approach didn't work
            try:
                # Get recommendations
                recommendations = yticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    # Get the most recent recommendation
                    latest_rec = recommendations.iloc[-1]
                    
                    # Map the grade to our categories
                    grade = latest_rec.get('To Grade', '').lower()
                    
                    if 'buy' in grade or 'outperform' in grade or 'overweight' in grade:
                        ratings_data["buy"] = 1
                        ratings_data["buy_percentage"] = 100
                        ratings_data["positive_percentage"] = 100
                    elif 'neutral' in grade or 'hold' in grade or 'equal weight' in grade:
                        ratings_data["hold"] = 1
                        ratings_data["buy_percentage"] = 50
                        ratings_data["positive_percentage"] = 50
                    elif 'sell' in grade or 'underperform' in grade or 'underweight' in grade:
                        ratings_data["sell"] = 1
                        ratings_data["buy_percentage"] = 0
                        ratings_data["positive_percentage"] = 0
                    
                    ratings_data["total_ratings"] = 1
                    ratings_data["recommendations"] = 1
                    ratings_data["date"] = latest_rec.name.strftime("%Y-%m-%d") if hasattr(latest_rec.name, "strftime") else str(latest_rec.name)
            except Exception as e:
                logger.warning(f"Failed to get recommendations dataframe for {ticker}: {str(e)}")
            
            return ratings_data
            
        except Exception as e:
            # Translate specific errors for better user experience
            if isinstance(e, (APIError, ValidationError, RateLimitError, NetworkError)):
                raise e
            else:
                raise APIError(f"Failed to get analyst ratings for {ticker}: {str(e)}")
    
    @enhanced_async_rate_limited(max_retries=0)
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
        # Validate the ticker format
        validate_ticker(ticker)
        
        # Skip insider transactions for non-US tickers
        if not is_us_ticker(ticker):
            logger.debug(f"Skipping insider transactions for non-US ticker {ticker}")
            return []
        
        # Base URL for Yahoo Finance API
        base_url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"
        
        url = f"{base_url}/{ticker}"
        params = {
            "modules": "insiderTransactions,institutionOwnership",
            "formatted": "false"
        }
        
        try:
            data = await self._fetch_json(url, params)
            
            # Check for valid response
            if not data or "quoteSummary" not in data or "result" not in data["quoteSummary"] or not data["quoteSummary"]["result"]:
                raise APIError(f"Invalid response from Yahoo Finance API for {ticker} insider transactions")
            
            # Extract data from result
            result = data["quoteSummary"]["result"][0]
            
            transactions = []
            
            # Get insider transactions
            if "insiderTransactions" in result and "transactions" in result["insiderTransactions"]:
                for transaction in result["insiderTransactions"]["transactions"]:
                    # Format transaction data
                    tx = {
                        "name": transaction.get("filerName", ""),
                        "title": transaction.get("filerRelation", ""),
                        "date": self._format_date(pd.to_datetime(transaction["startDate"]["raw"], unit='s')) 
                            if "startDate" in transaction and "raw" in transaction["startDate"] else None,
                        "transaction": transaction.get("transactionText", ""),
                        "shares": transaction["shares"]["raw"] if "shares" in transaction and "raw" in transaction["shares"] else 0,
                        "value": transaction["value"]["raw"] if "value" in transaction and "raw" in transaction["value"] else 0,
                    }
                    transactions.append(tx)
            
            # Get institutional ownership
            if "institutionOwnership" in result and "ownershipList" in result["institutionOwnership"]:
                for institution in result["institutionOwnership"]["ownershipList"]:
                    # Format institution data
                    inst = {
                        "name": institution.get("organization", ""),
                        "type": "institution",
                        "date": self._format_date(pd.to_datetime(institution["reportDate"]["raw"], unit='s')) 
                            if "reportDate" in institution and "raw" in institution["reportDate"] else None,
                        "shares": institution["position"]["raw"] if "position" in institution and "raw" in institution["position"] else 0,
                        "value": institution["value"]["raw"] if "value" in institution and "raw" in institution["value"] else 0,
                        "pct_out": institution["pctHeld"]["raw"] * 100 if "pctHeld" in institution and "raw" in institution["pctHeld"] else 0,
                    }
                    transactions.append(inst)
            
            return transactions
            
        except Exception as e:
            # Translate specific errors for better user experience
            if isinstance(e, (APIError, ValidationError, RateLimitError, NetworkError)):
                raise e
            else:
                raise APIError(f"Failed to get insider transactions for {ticker}: {str(e)}")
    
    @enhanced_async_rate_limited(max_retries=0)
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
        
        # Base URL for Yahoo Finance API search
        base_url = "https://query1.finance.yahoo.com/v1/finance/search"
        
        params = {
            "q": query,
            "quotesCount": limit,
            "newsCount": 0,
            "enableFuzzyQuery": "true",
            "enableEnhancedTrivialQuery": "true"
        }
        
        try:
            data = await self._fetch_json(base_url, params)
            
            # Check for valid response
            if not data or "quotes" not in data:
                return []
            
            # Format results
            results = []
            for quote in data["quotes"][:limit]:
                result = {
                    "symbol": quote.get("symbol", ""),
                    "name": quote.get("longname", quote.get("shortname", "")),
                    "exchange": quote.get("exchange", ""),
                    "type": quote.get("quoteType", ""),
                    "score": quote.get("score", 0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            # Translate specific errors for better user experience
            if isinstance(e, (APIError, ValidationError, RateLimitError, NetworkError)):
                raise e
            else:
                raise APIError(f"Failed to search tickers for '{query}': {str(e)}")
    
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
        
        # Process batch with our enhanced batch processor
        async def get_info_for_ticker(ticker: str) -> Dict[str, Any]:
            try:
                return await self.get_ticker_info(ticker, skip_insider_metrics)
            except Exception as e:
                logger.warning(f"Error getting data for {ticker}: {str(e)}")
                return {"symbol": ticker, "error": str(e)}
        
        results = await process_batch_async(
            tickers,
            get_info_for_ticker,
            batch_size=RATE_LIMIT["BATCH_SIZE"],
            concurrency=self.max_concurrency
        )
        
        return results
    
    def _format_market_cap(self, value: Optional[float]) -> Optional[str]:
        """Format market cap value with appropriate suffix"""
        if value is None:
            return None
            
        if value >= 1e12:  # Trillion
            if value >= 10e12:
                return f"{value / 1e12:.1f}T"
            else:
                return f"{value / 1e12:.2f}T"
        elif value >= 1e9:  # Billion
            if value >= 100e9:
                return f"{int(value / 1e9)}B"
            elif value >= 10e9:
                return f"{value / 1e9:.1f}B"
            else:
                return f"{value / 1e9:.2f}B"
        elif value >= 1e6:  # Million
            if value >= 100e6:
                return f"{int(value / 1e6)}M"
            elif value >= 10e6:
                return f"{value / 1e6:.1f}M"
            else:
                return f"{value / 1e6:.2f}M"
        else:
            return f"{int(value):,}"
    
    def _calculate_upside_potential(self, current_price: Optional[float], target_price: Optional[float]) -> Optional[float]:
        """Calculate upside potential percentage"""
        if current_price is None or target_price is None or current_price <= 0:
            return None
            
        return ((target_price / current_price) - 1) * 100
    
    def _format_date(self, date: Any) -> Optional[str]:
        """Format date as YYYY-MM-DD string"""
        if date is None:
            return None
            
        try:
            if isinstance(date, (pd.Timestamp, pd.DatetimeIndex)):
                return date.strftime('%Y-%m-%d')
            elif isinstance(date, str):
                return pd.to_datetime(date).strftime('%Y-%m-%d')
            else:
                return str(date)
        except Exception:
            return str(date)
    
    @enhanced_async_rate_limited(max_retries=0)
    async def get_earnings_dates(self, ticker: str) -> List[str]:
        """
        Get upcoming earnings dates for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of earnings dates in YYYY-MM-DD format
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Try to get from earnings data, which already includes dates
        try:
            earnings_data = await self.get_earnings_data(ticker)
            if earnings_data and "earnings_dates" in earnings_data and earnings_data["earnings_dates"]:
                return earnings_data["earnings_dates"]
        except Exception as e:
            logger.warning(f"Error getting earnings dates from earnings data: {str(e)}")
        
        # Fallback to direct ticker info with calendar events
        try:
            # Validate the ticker format
            validate_ticker(ticker)
            
            # Base URL for Yahoo Finance API
            base_url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"
            
            url = f"{base_url}/{ticker}"
            params = {
                "modules": "calendarEvents",
                "formatted": "false"
            }
            
            data = await self._fetch_json(url, params)
            
            # Check for valid response
            if not data or "quoteSummary" not in data or "result" not in data["quoteSummary"] or not data["quoteSummary"]["result"]:
                return []
            
            # Extract data from result
            result = data["quoteSummary"]["result"][0]
            
            dates = []
            
            # Get upcoming earnings date
            if "calendarEvents" in result and "earnings" in result["calendarEvents"]:
                earnings = result["calendarEvents"]["earnings"]
                if "earningsDate" in earnings and earnings["earningsDate"]:
                    # Convert timestamp to dates
                    for date in earnings["earningsDate"]:
                        if "raw" in date:
                            timestamp = date["raw"]
                            dates.append(
                                self._format_date(pd.to_datetime(timestamp, unit='s'))
                            )
            
            return dates
            
        except Exception as e:
            # Translate specific errors for better user experience
            if isinstance(e, (APIError, ValidationError, RateLimitError, NetworkError)):
                raise e
            else:
                raise APIError(f"Failed to get earnings dates for {ticker}: {str(e)}")
                
    async def get_ticker_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing analysis data
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        try:
            # Get basic ticker info
            info = await self.get_ticker_info(ticker)
            
            # Get analyst ratings if not already included
            if "buy_percentage" not in info or info["buy_percentage"] is None:
                try:
                    ratings = await self.get_analyst_ratings(ticker)
                    if ratings:
                        # Update info with analyst ratings
                        info["buy_percentage"] = ratings.get("buy_percentage")
                        info["total_ratings"] = ratings.get("total_ratings")
                        info["analyst_count"] = ratings.get("total_ratings")
                except Exception as e:
                    logger.warning(f"Error getting analyst ratings for {ticker}: {str(e)}")
            
            # Get earnings dates if not already included
            if "last_earnings" not in info or info["last_earnings"] is None:
                try:
                    dates = await self.get_earnings_dates(ticker)
                    if dates:
                        # Use the first (most recent) date
                        info["last_earnings"] = dates[0]
                except Exception as e:
                    logger.warning(f"Error getting earnings dates for {ticker}: {str(e)}")
            
            # Format ticker and company name
            info["ticker"] = ticker
            if "name" in info and info["name"]:
                if len(info["name"]) > 14:
                    info["company"] = info["name"][:14]
                else:
                    info["company"] = info["name"]
                # Convert company name to uppercase
                info["company"] = info["company"].upper()
            else:
                info["company"] = ticker
            
            # Add expected return calculation if needed
            if "upside" in info and info["upside"] is not None and "buy_percentage" in info and info["buy_percentage"] is not None:
                info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
            
            return info
            
        except Exception as e:
            logger.error(f"Error in get_ticker_analysis for {ticker}: {str(e)}")
            # Return minimal info to avoid breaking the app
            return {
                "ticker": ticker,
                "company": ticker,
                "error": str(e)
            }

    async def close(self) -> None:
        """Close the aiohttp session"""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def clear_cache(self) -> None:
        """Clear the ticker cache"""
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
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()