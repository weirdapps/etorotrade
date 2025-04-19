"""
Enhanced asynchronous Yahoo Finance provider implementation.

This module implements an improved AsyncFinanceDataProvider for Yahoo Finance data
using true async I/O, circuit breaking, and enhanced resilience patterns.
It provides significantly improved performance and reliability compared to
the thread-pool based implementation.
"""

from ...core.logging_config import get_logger

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation
import asyncio
import time
import random
import aiohttp
from typing import Dict, Any, Optional, List, Tuple, cast, TypeVar, Callable, Awaitable, Union
import pandas as pd
from functools import wraps
import datetime # Added for date checks

# Use relative imports
from ...core.config import RATE_LIMIT, CIRCUIT_BREAKER, CACHE_CONFIG, COLUMN_NAMES
from .base_provider import AsyncFinanceDataProvider
from ...core.errors import YFinanceError, APIError, ValidationError, RateLimitError, NetworkError
from ...utils.market.ticker_utils import validate_ticker # Keep this import
from ...utils.network.circuit_breaker import CircuitOpenError
from ...utils.async_utils.enhanced import (
    AsyncRateLimiter,
    process_batch_async,
    enhanced_async_rate_limited,
    gather_with_concurrency
)

logger = get_logger(__name__)

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
        self._ticker_cache: Dict[str, Dict[str, Any]] = {} # Simple cache for now
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = AsyncRateLimiter()
        self._ratings_cache: Dict[str, Dict[str, Any]] = {} # Cache for post-earnings ratings
        self._stock_cache: Dict[str, Any] = {} # Cache for yf.Ticker objects
        self.POSITIVE_GRADES = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive", "Market Outperform", "Add", "Sector Outperform"]

        # Circuit breaker configuration - used for all methods
        self._circuit_name = "yahoofinance_api"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """
        Ensure that an aiohttp session exists, creating one if needed.
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=RATE_LIMIT["API_TIMEOUT"], connect=10,
                sock_connect=10, sock_read=RATE_LIMIT["API_TIMEOUT"]
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

    @with_retry
    async def _fetch_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch JSON data from a URL with proper error handling.
        """
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
                        raise RateLimitError(f"Yahoo Finance API rate limit exceeded. Retry after {retry_after} seconds", retry_after=retry_after)
                    elif response.status == 404:
                        raise YFinanceError("An error occurred")
                    else:
                        text = await response.text()
                        details = {"status_code": response.status, "response_text": text[:100]}
                        raise YFinanceError("An error occurred")
            except aiohttp.ClientError as e:
                raise NetworkError(f"Network error while fetching {url}: {str(e)}")

        try:
            return await _do_fetch()
        except CircuitOpenError as e:
            retry_after = e.retry_after
            details = {"status_code": 503, "retry_after": retry_after}
            raise e

    @enhanced_async_rate_limited(max_retries=0)
    async def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker asynchronously.
        """
        validate_ticker(ticker)
        logger.debug(f"Getting ticker info for {ticker}")

        if ticker in self._ticker_cache:
            logger.debug(f"Using cached data for {ticker}")
            return self._ticker_cache[ticker]

        try:
            import yfinance as yf
            yticker = yf.Ticker(ticker)
            
            # Handle potential NoneType errors with info
            try:
                ticker_info = yticker.info
                # Check if ticker_info returns None
                if ticker_info is None:
                    logger.warning(f"Received None response for ticker {ticker} info. Using fallback empty dict.")
                    ticker_info = {}
            except AttributeError as ae:
                logger.warning(f"AttributeError for ticker {ticker}: {str(ae)}. Using fallback empty dict.")
                ticker_info = {}
                
            info: Dict[str, Any] = {"symbol": ticker}

            # Extract key fields
            info["name"] = ticker_info.get("longName", ticker_info.get("shortName", ""))
            info["company"] = info["name"][:14].upper() if info["name"] else ""
            info["sector"] = ticker_info.get("sector", "")
            info["industry"] = ticker_info.get("industry", "")
            info["country"] = ticker_info.get("country", "")
            info["website"] = ticker_info.get("website", "")
            info["current_price"] = ticker_info.get("regularMarketPrice", None)
            info["price"] = info["current_price"]
            info["currency"] = ticker_info.get("currency", "")
            info["market_cap"] = ticker_info.get("marketCap", None)
            info["exchange"] = ticker_info.get("exchange", "")
            info["quote_type"] = ticker_info.get("quoteType", "")
            info["pe_trailing"] = ticker_info.get("trailingPE", None)
            # Keep dividend_yield in decimal format (0.XX rather than XX%)
            # This allows downstream processors to handle formatting consistently
            info["dividend_yield"] = ticker_info.get("dividendYield", None)
            info["beta"] = ticker_info.get("beta", None)
            info["fifty_two_week_high"] = ticker_info.get("fiftyTwoWeekHigh", None)
            info["fifty_two_week_low"] = ticker_info.get("fiftyTwoWeekLow", None)
            info["pe_forward"] = ticker_info.get("forwardPE", None)
            info["peg_ratio"] = ticker_info.get("pegRatio", None)
            info["short_percent"] = ticker_info.get("shortPercentOfFloat", None)
            if info["short_percent"] is not None:
                info["short_percent"] = info["short_percent"] * 100
            info["target_price"] = ticker_info.get("targetMeanPrice", None)
            info["recommendation"] = ticker_info.get("recommendationMean", None)
            
            # Add analyst data directly to reduce extra API calls
            number_of_analysts = ticker_info.get("numberOfAnalystOpinions", 0)
            info["analyst_count"] = number_of_analysts
            info["total_ratings"] = number_of_analysts
            
            # Get recommendations data
            try:
                # First check if we have any recommendation data
                if number_of_analysts > 0:
                    try:
                        # Try to directly get buy percentage from recommendationKey
                        rec_key = ticker_info.get("recommendationKey", "").lower()
                        if rec_key in ["buy", "strongbuy"]:
                            info["buy_percentage"] = 85.0
                        elif rec_key in ["outperform"]:
                            info["buy_percentage"] = 75.0
                        elif rec_key in ["hold"]:
                            info["buy_percentage"] = 50.0
                        elif rec_key in ["underperform"]:
                            info["buy_percentage"] = 25.0
                        elif rec_key in ["sell"]:
                            info["buy_percentage"] = 15.0
                        else:
                            # Use recommendationMean to estimate buy percentage if available
                            rec_mean = ticker_info.get("recommendationMean", None)
                            if rec_mean is not None:
                                # Convert 1-5 scale to percentage (1=Strong Buy, 5=Sell)
                                # 1 = 90%, 3 = 50%, 5 = 10%
                                info["buy_percentage"] = max(0, min(100, 110 - (rec_mean * 20)))
                            else:
                                info["buy_percentage"] = None
                    except Exception as e:
                        logger.warning(f"Error getting recommendation data for {ticker}: {e}")
                        info["buy_percentage"] = None
                else:
                    # No analysts covering
                    info["buy_percentage"] = None
            except Exception as e:
                logger.warning(f"Error processing recommendations for {ticker}: {e}")
                info["buy_percentage"] = None
            
            # Rating type - default to "A" for all-time
            info["A"] = "A"

            # --- Start: Improved Recommendations Logic ---
            # Ensure default values are set
            info["analyst_count"] = 0
            info["total_ratings"] = 0
            info["buy_percentage"] = None
            
            try:
                # First check if ticker_info has analyst data
                number_of_analysts = ticker_info.get("numberOfAnalystOpinions", 0)
                if number_of_analysts > 0:
                    info["analyst_count"] = number_of_analysts
                    info["total_ratings"] = number_of_analysts
                    
                    # Try to get recommendation key first
                    rec_key = ticker_info.get("recommendationKey", "").lower()
                    rec_mean = ticker_info.get("recommendationMean", None)
                    
                    # If we have a recommendation key, map it to a buy percentage
                    if rec_key:
                        if rec_key in ["buy", "strongbuy"]:
                            info["buy_percentage"] = 85.0
                            logger.debug(f"Using recommendationKey '{rec_key}' to set buy_percentage=85.0 for {ticker}")
                        elif rec_key in ["outperform"]:
                            info["buy_percentage"] = 75.0
                            logger.debug(f"Using recommendationKey '{rec_key}' to set buy_percentage=75.0 for {ticker}")
                        elif rec_key in ["hold"]:
                            info["buy_percentage"] = 50.0
                            logger.debug(f"Using recommendationKey '{rec_key}' to set buy_percentage=50.0 for {ticker}")
                        elif rec_key in ["underperform"]:
                            info["buy_percentage"] = 25.0
                            logger.debug(f"Using recommendationKey '{rec_key}' to set buy_percentage=25.0 for {ticker}")
                        elif rec_key in ["sell"]:
                            info["buy_percentage"] = 15.0
                            logger.debug(f"Using recommendationKey '{rec_key}' to set buy_percentage=15.0 for {ticker}")
                    # If no key but we have recommendation mean, use it to estimate buy percentage
                    elif rec_mean is not None:
                        # Convert 1-5 scale to percentage (1=Strong Buy, 5=Sell)
                        # 1 = 90%, 3 = 50%, 5 = 10%
                        info["buy_percentage"] = max(0, min(100, 110 - (rec_mean * 20)))
                        logger.debug(f"Using recommendationMean {rec_mean} to set buy_percentage={info['buy_percentage']} for {ticker}")
                
                # Try to get detailed recommendations from recommendations DataFrame
                recommendations_df = getattr(yticker, 'recommendations', None)
                if recommendations_df is not None and not recommendations_df.empty:
                    try:
                        # Get the most recent recommendations
                        latest_date = recommendations_df.index.max()
                        latest_recs = recommendations_df.loc[latest_date]
                        
                        # Extract counts
                        strong_buy = int(latest_recs.get('strongBuy', 0))
                        buy = int(latest_recs.get('buy', 0))
                        hold = int(latest_recs.get('hold', 0))
                        sell = int(latest_recs.get('sell', 0))
                        strong_sell = int(latest_recs.get('strongSell', 0))
                        
                        # Calculate the total
                        total = strong_buy + buy + hold + sell + strong_sell
                        
                        # Only update if we have valid data
                        if total > 0:
                            buy_count = strong_buy + buy
                            buy_percentage = (buy_count / total) * 100
                            
                            # This is the most accurate source of data, always update
                            info["buy_percentage"] = buy_percentage
                            info["total_ratings"] = total
                            info["analyst_count"] = total
                            
                            logger.debug(f"Using recommendations DataFrame to set analyst data for {ticker}: "
                                        f"buy_percentage={buy_percentage}, total_ratings={total}")
                    except (IndexError, KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Error extracting recommendations data for {ticker}: {e}. Using fallback values.")
                        
                # Fall back to analyzing upgrades_downgrades if needed
                if info["buy_percentage"] is None:
                    try:
                        upgrades_downgrades = getattr(yticker, 'upgrades_downgrades', None)
                        if upgrades_downgrades is not None and not upgrades_downgrades.empty:
                            # Count grades
                            positive_grades = ["Buy", "Overweight", "Outperform", "Strong Buy", 
                                            "Long-Term Buy", "Positive", "Market Outperform", "Add", 
                                            "Sector Outperform"]
                            
                            # Count positive grades in ToGrade column
                            if 'ToGrade' in upgrades_downgrades.columns:
                                total_grades = len(upgrades_downgrades)
                                positive_count = upgrades_downgrades[upgrades_downgrades['ToGrade'].isin(positive_grades)].shape[0]
                                
                                if total_grades > 0:
                                    buy_percentage = (positive_count / total_grades) * 100
                                    info["buy_percentage"] = buy_percentage
                                    info["total_ratings"] = total_grades
                                    info["analyst_count"] = total_grades
                                    
                                    logger.debug(f"Using upgrades_downgrades to set analyst data for {ticker}: "
                                                f"buy_percentage={buy_percentage}, total_ratings={total_grades}")
                    except Exception as e:
                        logger.warning(f"Error analyzing upgrades_downgrades for {ticker}: {e}. Using fallback values.")
                
                # If we still don't have a buy percentage but have analyst count, 
                # use recommendationMean as a last resort
                if info["analyst_count"] > 0 and info["buy_percentage"] is None and rec_mean is not None:
                    # Convert 1-5 scale to percentage (1=Strong Buy, 5=Sell)
                    info["buy_percentage"] = max(0, min(100, 110 - (rec_mean * 20)))
                    logger.debug(f"Final fallback: Using recommendationMean {rec_mean} to set buy_percentage={info['buy_percentage']} for {ticker}")
                
            except Exception as e:
                logger.warning(f"Error processing analyst data for {ticker}: {e}", exc_info=False)
                # Ensure default values if all attempts fail
                if info.get("analyst_count") is None:
                    info["analyst_count"] = 0
                if info.get("total_ratings") is None:
                    info["total_ratings"] = 0
                if info.get("buy_percentage") is None:
                    info["buy_percentage"] = None

            # Determine Rating Type ('A' or 'E')
            if self._is_us_ticker(ticker) and info.get("total_ratings", 0) > 0:
                 has_post_earnings = self._has_post_earnings_ratings(ticker, yticker)
                 if has_post_earnings and ticker in self._ratings_cache:
                     ratings_data = self._ratings_cache[ticker]
                     info["buy_percentage"] = ratings_data["buy_percentage"]
                     info["total_ratings"] = ratings_data["total_ratings"]
                     info["A"] = "E"
                     # Recalculate EXRET potentially needed here if buy_percentage changed
                     if info.get("upside") is not None:
                          try: info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
                          except TypeError: info["EXRET"] = None
                 else:
                     info["A"] = "A" # Default to All-time
            else:
                 info["A"] = "A" if info.get("total_ratings", 0) > 0 else "" # Set A based on whether ratings exist
            # --- End: Reinstated Recommendations Logic ---
            # --- End: Reinstated Recommendations Logic ---

            # --- Start: Reinstated Earnings Date Logic ---
            try:
                last_earnings_date = self._get_last_earnings_date(yticker)
                info["last_earnings"] = last_earnings_date
            except YFinanceError as e:
                logger.warning(f"Failed to get earnings date for {ticker}: {str(e)}", exc_info=False)
                info["last_earnings"] = None
            # --- End: Reinstated Earnings Date Logic ---

            # Re-add missing PEG and ensure Dividend Yield has the original raw value
            info["dividend_yield"] = ticker_info.get("dividendYield", None) # Get raw value - no modification
                
            info["peg_ratio"] = ticker_info.get("trailingPegRatio", None) # Use trailingPegRatio

            # Format market cap

            # --- Start: Reinstated Earnings Date Logic ---
            try:
                last_earnings_date = self._get_last_earnings_date(yticker)
                info["last_earnings"] = last_earnings_date
            except YFinanceError as e:
                logger.warning(f"Failed to get earnings date for {ticker}: {str(e)}", exc_info=False)
                info["last_earnings"] = None
            # --- End: Reinstated Earnings Date Logic ---

            # Format market cap
            if info.get("market_cap"):
                info["market_cap_fmt"] = self._format_market_cap(info["market_cap"])

            # Calculate upside potential
            price = info.get("current_price")
            target = info.get("target_price")
            info["upside"] = self._calculate_upside_potential(price, target)

            # Calculate EXRET (after potential ratings update)
            if info.get("upside") is not None and info.get("buy_percentage") is not None:
                 try: info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
                 except TypeError: info["EXRET"] = None
            else: info["EXRET"] = None

            # Add to simple cache
            self._ticker_cache[ticker] = info
            return info

        except (APIError, ValidationError, RateLimitError, NetworkError):
            raise YFinanceError("An error occurred")
        except YFinanceError as e:
            raise e

    @enhanced_async_rate_limited(max_retries=0)
    async def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a ticker asynchronously.
        """
        validate_ticker(ticker)
        try:
            import yfinance as yf
            period_map = {"1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "5y", "10y": "10y", "ytd": "ytd", "max": "max"}
            api_period = period_map.get(period, period)
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m", "1h": "60m", "1d": "1d", "5d": "5d", "1wk": "1wk", "1mo": "1mo", "3mo": "3mo"}
            api_interval = interval_map.get(interval, interval)
            yticker = yf.Ticker(ticker)
            df = yticker.history(period=api_period, interval=api_interval)
            if not df.empty and 'Open' not in df.columns and 'open' in df.columns:
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            return df
        except (ValueError, KeyError, TypeError, AttributeError, pd.errors.EmptyDataError, ImportError, ModuleNotFoundError, RuntimeError, MemoryError) as e:
            raise e
        except (IOError, ConnectionError, aiohttp.ClientError) as e:
            raise NetworkError(f"Network error when fetching historical data for {ticker}: {str(e)}")
        except (APIError, ValidationError, RateLimitError, NetworkError):
            raise e

    @enhanced_async_rate_limited(max_retries=0)
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a ticker asynchronously.
        """
        validate_ticker(ticker)
        try:
            import yfinance as yf
            yticker = yf.Ticker(ticker)
            earnings_data = {"symbol": ticker, "earnings_dates": [], "earnings_history": []}
            # Get calendar for earnings dates
            try:
                calendar = yticker.calendar
                if calendar is not None and not calendar.empty and EARNINGS_DATE_COL in calendar.columns:
                    earnings_date = calendar[EARNINGS_DATE_COL].iloc[0]
                    if pd.notna(earnings_date):
                        earnings_data["earnings_dates"].append(earnings_date.strftime("%Y-%m-%d"))
            except YFinanceError as e:
                logger.warning(f"Failed to get earnings calendar for {ticker}: {str(e)}")
            # Get earnings history
            try:
                earnings_hist = yticker.earnings_history
                if earnings_hist is not None and not earnings_hist.empty:
                    for idx, row in earnings_hist.iterrows():
                        quarter_data = {"date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx), "reported_eps": row.get("Reported EPS", None), "estimated_eps": row.get("Estimated EPS", None), "surprise": row.get("Surprise(%)", None)}
                        earnings_data["earnings_history"].append(quarter_data)
            except YFinanceError as e:
                logger.warning(f"Failed to get earnings history for {ticker}: {str(e)}")
            # Get earnings estimate from info
            try:
                ticker_info = yticker.info
                if "trailingEps" in ticker_info: earnings_data["earnings_average"] = ticker_info["trailingEps"]
                if "forwardEps" in ticker_info: earnings_data["earnings_estimates"] = ticker_info["forwardEps"]
                if "lastFiscalYearEnd" in ticker_info:
                    from datetime import datetime
                    last_fiscal = datetime.fromtimestamp(ticker_info["lastFiscalYearEnd"])
                    earnings_data["quarter_end"] = last_fiscal.strftime("%Y-%m-%d")
            except YFinanceError as e:
                logger.warning(f"Failed to get earnings estimates for {ticker}: {str(e)}")
            return earnings_data
        except (ValueError, KeyError, TypeError, AttributeError, pd.errors.EmptyDataError, ImportError, ModuleNotFoundError, RuntimeError, MemoryError) as e:
            raise e
        except (IOError, ConnectionError, aiohttp.ClientError) as e:
            raise NetworkError(f"Network error when fetching earnings data for {ticker}: {str(e)}")
        except (APIError, ValidationError, RateLimitError, NetworkError):
            raise e

    @enhanced_async_rate_limited(max_retries=0)
    async def get_earnings_dates(self, ticker: str) -> List[str]:
        """Get earnings dates (uses get_earnings_data)."""
        validate_ticker(ticker)
        try:
            earnings_data = await self.get_earnings_data(ticker)
            if "error" in earnings_data: # Propagate error if get_earnings_data failed
                logger.warning(f"Could not get earnings dates for {ticker} due to error: {earnings_data['error']}")
                return []
            # Ensure the key exists and the value is iterable before returning
            dates = earnings_data.get("earnings_dates")
            return dates if isinstance(dates, list) else []
        except YFinanceError as e:
            logger.error(f"Unexpected error getting earnings dates for {ticker}: {str(e)}")
            return []

    @enhanced_async_rate_limited(max_retries=0)
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker asynchronously.
        """
        validate_ticker(ticker)
        info = await self.get_ticker_info(ticker)
        if "error" in info: return {"symbol": ticker, "error": info["error"]}
        return {
            "symbol": ticker,
            "recommendations": info.get("total_ratings", 0),
            "buy_percentage": info.get("buy_percentage", None),
            "positive_percentage": info.get("buy_percentage", None),
            "total_ratings": info.get("total_ratings", 0),
            "ratings_type": info.get("A", "A"),
            "date": None
        }

    @enhanced_async_rate_limited(max_retries=0)
    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a ticker asynchronously.
        """
        validate_ticker(ticker)
        if not self._is_us_ticker(ticker):
            logger.debug(f"Skipping insider transactions for non-US ticker {ticker}")
            return []
        base_url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"
        url = f"{base_url}/{ticker}"
        params = {"modules": "insiderTransactions", "formatted": "false"}
        try:
            data = await self._fetch_json(url, params)
            if not data or "quoteSummary" not in data or "result" not in data["quoteSummary"] or not data["quoteSummary"]["result"]:
                raise YFinanceError("An error occurred")
            result = data["quoteSummary"]["result"][0]
            transactions = []
            if "insiderTransactions" in result and "transactions" in result["insiderTransactions"]:
                for transaction in result["insiderTransactions"]["transactions"]:
                    tx = {"name": transaction.get("filerName", ""), "title": transaction.get("filerRelation", ""), "date": self._format_date(pd.to_datetime(transaction["startDate"]["raw"], unit='s')) if "startDate" in transaction and "raw" in transaction["startDate"] else None, "transaction": transaction.get("transactionText", ""), "shares": transaction["shares"]["raw"] if "shares" in transaction and "raw" in transaction["shares"] else 0, "value": transaction["value"]["raw"] if "value" in transaction and "raw" in transaction["value"] else 0}
                    transactions.append(tx)
            return transactions
        except (ValueError, KeyError, TypeError, AttributeError, pd.errors.EmptyDataError, ModuleNotFoundError, RuntimeError, MemoryError) as e:
            raise e
        except (IOError, ConnectionError, aiohttp.ClientError) as e:
            raise NetworkError(f"Network error when fetching insider transactions for {ticker}: {str(e)}")
        except (APIError, ValidationError, RateLimitError, NetworkError):
            raise e

    @enhanced_async_rate_limited(max_retries=0)
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query asynchronously.
        """
        if not query or not query.strip():
            raise e
        base_url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": limit, "newsCount": 0, "enableFuzzyQuery": "true", "enableEnhancedTrivialQuery": "true"}
        try:
            data = await self._fetch_json(base_url, params)
            if not data or "quotes" not in data: return []
            results = []
            for quote in data["quotes"][:limit]:
                result = {"symbol": quote.get("symbol", ""), "name": quote.get("longname", quote.get("shortname", "")), "exchange": quote.get("exchange", ""), "type": quote.get("quoteType", ""), "score": quote.get("score", 0)}
                results.append(result)
            return results
        except (ValueError, KeyError, TypeError, AttributeError, ModuleNotFoundError, RuntimeError, MemoryError) as e:
            raise e
        except (IOError, ConnectionError, aiohttp.ClientError) as e:
            raise NetworkError(f"Network error when searching tickers for '{query}': {str(e)}")
        except (APIError, ValidationError, RateLimitError, NetworkError):
            raise e

    @enhanced_async_rate_limited(max_retries=0)
    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get price data for a ticker asynchronously."""
        logger.debug(f"Getting price data for {ticker} via Enhanced provider")
        info = await self.get_ticker_info(ticker)
        if "error" in info:
            return {"ticker": ticker, "error": info["error"]}
        return {
            "ticker": ticker,
            "current_price": info.get("current_price"),
            "price": info.get("price"),
            "target_price": info.get("target_price"),
            "upside": info.get("upside"),
            "fifty_two_week_high": info.get("fifty_two_week_high"),
            "fifty_two_week_low": info.get("fifty_two_week_low"),
            "fifty_day_avg": info.get("fifty_day_avg"),
            "two_hundred_day_avg": info.get("two_hundred_day_avg")
        }

    # --- Start: Added Helper Methods (at class level) ---
    def _has_post_earnings_ratings(self, ticker: str, yticker) -> bool:
        """Check if there are ratings available since the last earnings date."""
        try:
            is_us = self._is_us_ticker(ticker)
            if not is_us: return False
            last_earnings = self._get_last_earnings_date(yticker)
            if last_earnings is None: return False
            try:
                upgrades_downgrades = getattr(yticker, 'upgrades_downgrades', None)
                if upgrades_downgrades is None or upgrades_downgrades.empty: return False
                if hasattr(upgrades_downgrades, 'reset_index'):
                    df = upgrades_downgrades.reset_index()
                else: df = upgrades_downgrades
                if "GradeDate" not in df.columns:
                    if 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']):
                         df.rename(columns={'index': 'GradeDate'}, inplace=True)
                    elif hasattr(upgrades_downgrades, 'index') and isinstance(upgrades_downgrades.index, pd.DatetimeIndex):
                         df['GradeDate'] = upgrades_downgrades.index
                    else:
                         logger.warning(f"Could not find GradeDate column for {ticker}")
                         return False
                try:
                    df["GradeDate"] = pd.to_datetime(df["GradeDate"], errors='coerce')
                    df.dropna(subset=["GradeDate"], inplace=True)
                except YFinanceError as date_err:
                     logger.warning(f"Error converting GradeDate for {ticker}: {date_err}")
                     return False
                earnings_date = pd.to_datetime(last_earnings)
                post_earnings_df = df[df["GradeDate"] >= earnings_date]
                if not post_earnings_df.empty:
                    total_ratings = len(post_earnings_df)
                    positive_ratings = post_earnings_df[post_earnings_df["ToGrade"].isin(self.POSITIVE_GRADES)].shape[0]
                    self._ratings_cache[ticker] = {
                        "buy_percentage": (positive_ratings / total_ratings * 100) if total_ratings > 0 else 0,
                        "total_ratings": total_ratings,
                        "ratings_type": "E"
                    }
                    return True
                return False
            except YFinanceError as e:
                logger.warning(f"Error getting post-earnings ratings for {ticker}: {e}", exc_info=False)
            return False
        except YFinanceError as e:
            logger.warning(f"Error in _has_post_earnings_ratings for {ticker}: {e}", exc_info=False)
            return False

    def _get_last_earnings_date(self, yticker):
        """Get the last earnings date (kept identical to original)."""
        try: # Try calendar first
            calendar = getattr(yticker, 'calendar', None)
            if isinstance(calendar, dict) and COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                earnings_date_list = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                    today = pd.Timestamp.now().date()
                    past_earnings = [d for d in earnings_date_list if isinstance(d, datetime.date) and d < today]
                    if past_earnings: return max(past_earnings).strftime('%Y-%m-%d')
        except YFinanceError as e:
             logger.debug(f"Error getting earnings from calendar for {yticker.ticker}: {e}", exc_info=False)
        try: # Try earnings_dates attribute
            earnings_dates = getattr(yticker, 'earnings_dates', None)
            if earnings_dates is not None and not earnings_dates.empty:
                today = pd.Timestamp.now()
                tz = earnings_dates.index.tz
                if tz and today.tzinfo is None:
                    try: today = today.tz_localize('UTC').tz_convert(tz)
                    except YFinanceError as tz_err:
                         logger.warning(f"Could not localize 'today' timestamp for {yticker.ticker}: {tz_err}")
                         today = pd.Timestamp.now()
                elif tz is None and today.tzinfo is not None:
                     today = today.tz_localize(None)
                past_dates = [date for date in earnings_dates.index if date < today]
                if past_dates: return max(past_dates).strftime('%Y-%m-%d')
        except YFinanceError as e:
             logger.debug(f"Error getting earnings from earnings_dates for {yticker.ticker}: {e}", exc_info=False)
        return None

    def _is_us_ticker(self, ticker: str) -> bool: # Keep this helper
        """Check if a ticker is a US ticker based on suffix (kept identical)."""
        try:
            from ...utils.market.ticker_utils import is_us_ticker as util_is_us_ticker
            return util_is_us_ticker(ticker)
        except ImportError:
             logger.warning("Could not import is_us_ticker utility, using inline logic.")
             if ticker in ["BRK.A", "BRK.B", "BF.A", "BF.B"]: return True
             if "." not in ticker: return True
             if ticker.endswith(".US"): return True
             return False

    def _format_market_cap(self, value: Optional[float]) -> Optional[str]:
        """Format market cap value."""
        if value is None: return None
        try:
            val = float(value)
            if val >= 1e12: return f"{val / 1e12:.1f}T" if val >= 10e12 else f"{val / 1e12:.2f}T"
            elif val >= 1e9: return f"{int(val / 1e9)}B" if val >= 100e9 else (f"{val / 1e9:.1f}B" if val >= 10e9 else f"{val / 1e9:.2f}B")
            elif val >= 1e6: return f"{int(val / 1e6)}M" if val >= 100e6 else (f"{val / 1e6:.1f}M" if val >= 10e6 else f"{val / 1e6:.2f}M")
            else: return f"{int(val):,}"
        except (ValueError, TypeError): return str(value)

    def _calculate_upside_potential(self, current_price: Optional[float], target_price: Optional[float]) -> Optional[float]:
        """Calculate upside potential percentage."""
        if current_price is not None and target_price is not None and current_price > 0:
            try: return ((float(target_price) / float(current_price)) - 1) * 100
            except (ValueError, TypeError, ZeroDivisionError): pass
        return None

    def _format_date(self, date: Any) -> Optional[str]:
        """Format date object to YYYY-MM-DD string."""
        if date is None: return None
        if isinstance(date, str): return date[:10]
        if hasattr(date, 'strftime'): return date.strftime('%Y-%m-%d')
        try: return str(date)[:10]
        except Exception as e:
            # Translate standard exception to our error hierarchy
            error_context = {"location": __name__}
            custom_error = translate_error(e, context={"location": __name__})
            raise custom_error
        return None
    # --- End: Added Helper Methods ---

    async def batch_get_ticker_info(self, tickers: List[str], skip_insider_metrics: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers in a batch asynchronously.
        Uses optimized async processing with prioritization and proper error handling.
        """
        if not tickers: return {}

        # Identify high-priority tickers (important US stocks)
        from ...utils.market.ticker_utils import is_us_ticker
        high_priority_tickers = [ticker for ticker in tickers if is_us_ticker(ticker)]
        
        # Get known high-priority tickers from VIP_TICKERS in config
        from ...core.config import RATE_LIMIT
        vip_tickers = RATE_LIMIT.get("VIP_TICKERS", set())
        if vip_tickers:
            # Add any VIP tickers that are in our request list to the high priority list
            priority_set = set(high_priority_tickers)
            for ticker in tickers:
                if ticker in vip_tickers and ticker not in priority_set:
                    high_priority_tickers.append(ticker)

        async def get_info_for_ticker(ticker: str) -> Dict[str, Any]:
            """Wrapper to fetch info for a single ticker and handle errors."""
            try:
                return await self.get_ticker_info(ticker, skip_insider_metrics)
            except YFinanceError as e:
                logger.error(f"Error fetching batch data for {ticker}: {e}")
                return {"symbol": ticker, "ticker": ticker, "company": ticker, "error": str(e)}

        # Use the new process_batch_async with prioritization from async_utils.enhanced
        from ...utils.async_utils.enhanced import process_batch_async
        
        batch_results = await process_batch_async(
            items=tickers,
            processor=get_info_for_ticker,
            batch_size=min(15, len(tickers)),  # Reasonably sized batches
            concurrency=self.max_concurrency,
            priority_items=high_priority_tickers,
            description=f"Processing {len(tickers)} tickers"
        )

        # Ensure all results have the expected format
        processed_results = {}
        for ticker, result in batch_results.items():
            if isinstance(result, dict) and "symbol" in result:
                processed_results[result["symbol"]] = result
            elif isinstance(ticker, str):
                # Create a fallback result for unexpected return types
                logger.warning(f"Unexpected result type for {ticker}: {type(result)}")
                processed_results[ticker] = {
                    "symbol": ticker, 
                    "ticker": ticker, 
                    "company": ticker,
                    "error": "Invalid result format received"
                }
                
        return processed_results

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.debug("Enhanced provider session closed.")

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._ticker_cache.clear()
        self._stock_cache.clear()
        self._ratings_cache.clear()
        logger.info("Enhanced provider internal caches cleared.")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the internal caches."""
        # Note: This only reflects the simple instance cache, not CacheManager
        return {
            "ticker_cache_size": len(self._ticker_cache),
            "stock_object_cache_size": len(self._stock_cache),
            "ratings_cache_size": len(self._ratings_cache),
        }

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()