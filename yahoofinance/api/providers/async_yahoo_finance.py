"""
Asynchronous Yahoo Finance provider implementation.

This module implements an AsyncFinanceDataProvider for Yahoo Finance data
using true async I/O, circuit breaking, and enhanced resilience patterns.
It provides improved performance and reliability with advanced async features.
"""

import asyncio
import secrets
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import aiohttp
import pandas as pd

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ...utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ...core.logging import get_logger


# Define constants for repeated strings
DEFAULT_ERROR_MESSAGE = "An error occurred"
EARNINGS_DATE_COL = 'Earnings Date'
import datetime  # Added for date checks

# Use relative imports
from ...core.config import CACHE_CONFIG, CIRCUIT_BREAKER, COLUMN_NAMES, RATE_LIMIT
from ...core.errors import APIError, NetworkError, RateLimitError, ValidationError, YFinanceError
from ...utils.async_utils.enhanced import (
    AsyncRateLimiter,
    enhanced_async_rate_limited,
    gather_with_concurrency,
    process_batch_async,
)
from ...utils.market.ticker_utils import validate_ticker, is_stock_ticker  # Keep this import
from ...utils.network.circuit_breaker import CircuitOpenError
from .base_provider import AsyncFinanceDataProvider
from ...utils.network.session_manager import get_shared_session
from ...data.cache_compatibility import LRUCache

# Import from split modules
from .async_modules import (
    calculate_analyst_momentum,
    calculate_pe_vs_sector,
    calculate_earnings_growth,
    calculate_upside_potential,
    format_date,
    format_market_cap,
    get_last_earnings_date,
    has_post_earnings_ratings,
    is_us_ticker,
    parse_analyst_recommendations,
    POSITIVE_GRADES,
)


logger = get_logger(__name__)

T = TypeVar("T")  # Return type for async functions


class AsyncYahooFinanceProvider(AsyncFinanceDataProvider):
    """
    Asynchronous Yahoo Finance data provider implementation.

    This provider uses true async I/O with aiohttp for HTTP requests,
    along with advanced resilience patterns including circuit breaking,
    enhanced rate limiting, and intelligent retries.

    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
        max_concurrency: Maximum number of concurrent operations
        session: aiohttp ClientSession for HTTP requests
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrency: int = 5,
        enable_circuit_breaker: bool = True,
        **kwargs,
    ):
        """
        Initialize the Async Yahoo Finance provider.

        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries (exponential backoff applied)
            max_concurrency: Maximum number of concurrent operations
            enable_circuit_breaker: Whether to enable the circuit breaker pattern
            **kwargs: Additional keyword arguments (for factory compatibility)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrency = max_concurrency
        self.enable_circuit_breaker = enable_circuit_breaker
        # Smart cache with timestamps for TTL-based expiration
        # Using LRU cache to prevent unbounded memory growth (max 1000 tickers)
        self._ticker_cache: LRUCache[str, Dict[str, Any]] = LRUCache(max_size=1000)  # {ticker: {data: {...}, timestamp: float}}
        self._rate_limiter = AsyncRateLimiter()
        self._ratings_cache: LRUCache[str, Dict[str, Any]] = LRUCache(max_size=500)  # Cache for post-earnings ratings
        self._stock_cache: LRUCache[str, Any] = LRUCache(max_size=500)  # Cache for yf.Ticker objects

        # Backward compatibility: expose POSITIVE_GRADES as instance attribute
        self.POSITIVE_GRADES = POSITIVE_GRADES

        # Circuit breaker configuration - used for all methods
        self._circuit_name = "yahoofinance_api"

        # Price-sensitive fields that should NEVER be cached (always fresh)
        self._never_cache_fields = {
            "price", "current_price", "regularMarketPrice",
            "target_price", "targetMeanPrice",
            "upside", "pe_trailing", "trailingPE",
            "pe_forward", "forwardPE", "peg_ratio", "pegRatio",
            "price_performance", "twelve_month_performance",
            "expected_return", "EXRET",
            "pct_from_52w_high", "fiftyTwoWeekHighChangePercent",
            "above_200dma", "volume", "regularMarketVolume",
            "action", "signal",
        }

        # Cache TTL for non-price fields (4 hours = 14400 seconds)
        self._cache_ttl_seconds = 14400

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """
        Get the shared session from SharedSessionManager for connection pooling.
        """
        return await get_shared_session()

    @with_retry
    async def _fetch_json(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fetch JSON data from a URL with proper error handling.
        """

        @enhanced_async_rate_limited(
            circuit_name=self._circuit_name if self.enable_circuit_breaker else None,
            max_retries=self.max_retries,
            rate_limiter=self._rate_limiter,
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
                            retry_after=retry_after,
                        )
                    elif response.status == 404:
                        raise YFinanceError(DEFAULT_ERROR_MESSAGE)
                    else:
                        text = await response.text()
                        {"status_code": response.status, "response_text": text[:100]}
                        raise YFinanceError(DEFAULT_ERROR_MESSAGE)
            except aiohttp.ClientError as e:
                raise NetworkError(f"Network error while fetching {url}: {str(e)}")

        try:
            return await _do_fetch()
        except CircuitOpenError as e:
            retry_after = e.retry_after
            {"status_code": 503, "retry_after": retry_after}
            raise e

    @enhanced_async_rate_limited(max_retries=0)
    async def get_ticker_info(
        self, ticker: str, skip_insider_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker asynchronously.

        Uses smart caching:
        - Price-sensitive fields are ALWAYS fetched fresh (never cached)
        - Non-price fields use 4-hour TTL cache
        """
        validate_ticker(ticker)
        logger.debug(f"Getting ticker info for {ticker}")

        # Check if we have valid cached non-price data (within TTL)
        cached_entry = self._ticker_cache.get(ticker)
        use_cached_static = False
        cached_static_data = {}

        if cached_entry:
            cache_age = time.time() - cached_entry.get("timestamp", 0)
            if cache_age < self._cache_ttl_seconds:
                # Cache is valid - we can use non-price fields
                use_cached_static = True
                cached_static_data = cached_entry.get("data", {})
                logger.debug(f"Using cached static data for {ticker} (age: {cache_age:.0f}s)")
            else:
                logger.debug(f"Cache expired for {ticker} (age: {cache_age:.0f}s > {self._cache_ttl_seconds}s)")

        try:
            import yfinance as yf

            # Apply ticker mapping for data fetching
            from trade_modules.config_manager import get_config
            config = get_config()
            fetch_ticker = config.get_data_fetch_ticker(ticker)
            
            if fetch_ticker != ticker:
                logger.debug(f"Mapping ticker {ticker} to {fetch_ticker} for data fetching")
            
            # Create ticker object temporarily - don't store it in cache to prevent memory leaks
            try:
                # Use safe_create_ticker from yfinance_utils if available
                from ...utils.yfinance_utils import safe_create_ticker

                yticker = safe_create_ticker(fetch_ticker)
            except ImportError:
                # Fall back to regular creation if the utility isn't available
                yticker = yf.Ticker(fetch_ticker)

            # Handle potential NoneType errors with info
            try:
                ticker_info = yticker.info
                # Check if ticker_info returns None
                if ticker_info is None:
                    logger.warning(
                        f"Received None response for ticker {ticker} info. Using fallback empty dict."
                    )
                    ticker_info = {}
            except AttributeError as ae:
                logger.warning(
                    f"AttributeError for ticker {ticker}: {str(ae)}. Using fallback empty dict."
                )
                ticker_info = {}

            info: Dict[str, Any] = {"symbol": ticker, "ticker": ticker}

            # Extract key fields
            info["name"] = ticker_info.get("longName", ticker_info.get("shortName", ""))
            info["company"] = info["name"][:14].upper() if info["name"] else ""
            info["sector"] = ticker_info.get("sector", "")
            info["industry"] = ticker_info.get("industry", "")
            info["country"] = ticker_info.get("country", "")
            info["website"] = ticker_info.get("website", "")
            # Try multiple price fields in order of preference
            price = (
                ticker_info.get("regularMarketPrice") or
                ticker_info.get("currentPrice") or
                ticker_info.get("lastPrice") or
                ticker_info.get("previousClose")
            )
            info["current_price"] = price
            info["price"] = price
            info["currency"] = ticker_info.get("currency", "")
            info["market_cap"] = (
                ticker_info.get("marketCap") or  # Regular stocks
                ticker_info.get("totalAssets") or  # ETFs
                ticker_info.get("netAssets")  # Alternative ETF field
            )
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
            info["return_on_equity"] = ticker_info.get("returnOnEquity", None)
            info["debt_to_equity"] = ticker_info.get("debtToEquity", None)
            if info["short_percent"] is not None:
                info["short_percent"] = info["short_percent"] * 100
            # Convert ROE from decimal to percentage (1.09417 -> 109.417)
            if info["return_on_equity"] is not None:
                info["return_on_equity"] = info["return_on_equity"] * 100
            info["target_price"] = ticker_info.get("targetMeanPrice", None)
            info["recommendation"] = ticker_info.get("recommendationMean", None)

            # Free Cash Flow - for FCF Yield calculation (academically proven alpha factor)
            info["free_cash_flow"] = ticker_info.get("freeCashflow", None)

            # Calculate FCF Yield = FCF / Market Cap * 100
            if info["free_cash_flow"] and info["market_cap"] and info["market_cap"] > 0:
                info["fcf_yield"] = (info["free_cash_flow"] / info["market_cap"]) * 100
            else:
                info["fcf_yield"] = None

            # Revenue Growth (yfinance provides revenueGrowth as decimal, e.g., 0.15 = 15%)
            revenue_growth = ticker_info.get("revenueGrowth", None)
            if revenue_growth is not None:
                info["revenue_growth"] = revenue_growth * 100  # Convert to percentage
            else:
                info["revenue_growth"] = None

            # Price Momentum: 50/200-day moving averages
            info["fifty_day_average"] = ticker_info.get("fiftyDayAverage", None)
            info["two_hundred_day_average"] = ticker_info.get("twoHundredDayAverage", None)

            # Calculate momentum metrics
            if info["current_price"] and info["fifty_two_week_high"]:
                info["pct_from_52w_high"] = (info["current_price"] / info["fifty_two_week_high"]) * 100
            else:
                info["pct_from_52w_high"] = None

            if info["current_price"] and info["two_hundred_day_average"]:
                info["above_200dma"] = info["current_price"] > info["two_hundred_day_average"]
            else:
                info["above_200dma"] = None

            # Parse analyst recommendations using extracted module
            analyst_data = parse_analyst_recommendations(ticker_info, yticker)
            info["analyst_count"] = analyst_data["analyst_count"]
            info["total_ratings"] = analyst_data["total_ratings"]
            info["buy_percentage"] = analyst_data["buy_percentage"]

            # Calculate analyst momentum (3-month change in buy%)
            momentum_data = calculate_analyst_momentum(yticker)
            info["analyst_momentum"] = momentum_data["analyst_momentum"]
            info["analyst_count_trend"] = momentum_data["analyst_count_trend"]

            # Calculate PE vs sector median (sector-relative valuation)
            info["pe_vs_sector"] = calculate_pe_vs_sector(
                info.get("pe_forward"), info.get("sector")
            )

            # Determine Rating Type ('A' or 'E')
            # Only check post-earnings ratings for stock tickers, not ETFs/commodities/crypto
            if is_us_ticker(ticker) and is_stock_ticker(ticker) and info.get("total_ratings", 0) > 0:
                post_earnings_result = has_post_earnings_ratings(ticker, yticker, True, POSITIVE_GRADES)
                if post_earnings_result["has_ratings"]:
                    ratings_data = post_earnings_result["ratings_data"]
                    info["buy_percentage"] = ratings_data["buy_percentage"]
                    info["total_ratings"] = ratings_data["total_ratings"]
                    info["A"] = "E"
                    # Recalculate EXRET since buy_percentage changed
                    if info.get("upside") is not None:
                        try:
                            info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
                        except TypeError:
                            info["EXRET"] = None
                else:
                    info["A"] = "A"  # Default to All-time
            else:
                info["A"] = (
                    "A" if info.get("total_ratings", 0) > 0 else ""
                )  # Set A based on whether ratings exist

            # Get earnings date for stock tickers (not ETFs/commodities/crypto)
            if is_stock_ticker(ticker):
                try:
                    last_earnings_date = get_last_earnings_date(yticker)
                    info["last_earnings"] = last_earnings_date
                    info["earnings_date"] = last_earnings_date  # Also set earnings_date for console display
                    logger.debug(f"Set earnings_date for {ticker} to LAST earnings: {last_earnings_date}")
                except YFinanceError as e:
                    logger.warning(
                        f"Failed to get earnings date for {ticker}: {str(e)}", exc_info=False
                    )
                    info["last_earnings"] = None
                    info["earnings_date"] = None
            else:
                # Non-stock assets don't have earnings dates
                logger.debug(f"Skipping earnings date for non-stock asset {ticker}")
                info["last_earnings"] = None
                info["earnings_date"] = None

            # Calculate earnings growth from quarterly data
            info["earnings_growth"] = calculate_earnings_growth(ticker)

            # Re-add missing PEG and ensure Dividend Yield has the original raw value
            info["dividend_yield"] = ticker_info.get(
                "dividendYield", None
            )  # Get raw value - no modification

            info["peg_ratio"] = ticker_info.get("trailingPegRatio", None)  # Use trailingPegRatio

            # Format market cap
            if info.get("market_cap"):
                info["market_cap_fmt"] = format_market_cap(info["market_cap"])

            # Calculate upside potential
            price = info.get("current_price")
            target = info.get("target_price")
            info["upside"] = calculate_upside_potential(price, target)

            # Calculate EXRET (after potential ratings update)
            if info.get("upside") is not None and info.get("buy_percentage") is not None:
                try:
                    info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
                except TypeError:
                    info["EXRET"] = None
            else:
                info["EXRET"] = None

            # Smart caching: store static (non-price) fields with timestamp
            # Price fields are always fresh - we don't need to cache them
            static_data = {k: v for k, v in info.items() if k not in self._never_cache_fields}
            self._ticker_cache[ticker] = {
                "data": static_data,
                "timestamp": time.time()
            }
            logger.debug(f"Cached {len(static_data)} static fields for {ticker}")
            return info

        except (APIError, ValidationError, RateLimitError, NetworkError) as e:
            # Log the specific error but raise a generic one
            logger.warning(f"Error fetching ticker info for {ticker}: {e}")
            raise YFinanceError(DEFAULT_ERROR_MESSAGE) from e
        except YFinanceError as e:
            # Log and re-raise YFinanceError with additional context
            logger.error(f"YFinanceError encountered for {ticker}: {str(e)}")
            raise YFinanceError(f"Error processing ticker {ticker}: {str(e)}")

    @enhanced_async_rate_limited(max_retries=0)
    async def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data for a ticker asynchronously.
        """
        validate_ticker(ticker)
        try:
            import yfinance as yf

            period_map = {
                "1d": "1d",
                "5d": "5d",
                "1mo": "1mo",
                "3mo": "3mo",
                "6mo": "6mo",
                "1y": "1y",
                "2y": "2y",
                "5y": "5y",
                "10y": "10y",
                "ytd": "ytd",
                "max": "max",
            }
            api_period = period_map.get(period, period)
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "60m": "60m",
                "1h": "60m",
                "1d": "1d",
                "5d": "5d",
                "1wk": "1wk",
                "1mo": "1mo",
                "3mo": "3mo",
            }
            api_interval = interval_map.get(interval, interval)
            yticker = yf.Ticker(ticker)
            df = yticker.history(period=api_period, interval=api_interval)
            if not df.empty and "Open" not in df.columns and "open" in df.columns:
                df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    },
                    inplace=True,
                )
            return df
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            pd.errors.EmptyDataError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            MemoryError,
        ) as e:
            raise e
        except (IOError, ConnectionError, aiohttp.ClientError) as e:
            raise NetworkError(
                f"Network error when fetching historical data for {ticker}: {str(e)}"
            )
        except (APIError, ValidationError, RateLimitError, NetworkError) as exc:
            raise exc

    @enhanced_async_rate_limited(max_retries=0)
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a ticker asynchronously.
        """
        validate_ticker(ticker)
        try:
            import yfinance as yf

            yticker = yf.Ticker(ticker)
            earnings_data: Dict[str, Any] = {"symbol": ticker, "earnings_dates": [], "earnings_history": []}
            # Get calendar for earnings dates
            try:
                calendar = yticker.calendar
                if (
                    calendar is not None
                    and not calendar.empty
                    and EARNINGS_DATE_COL in calendar.columns
                ):
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
                        quarter_data = {
                            "date": (
                                idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                            ),
                            "reported_eps": row.get("Reported EPS", None),
                            "estimated_eps": row.get("Estimated EPS", None),
                            "surprise": row.get("Surprise(%)", None),
                        }
                        earnings_data["earnings_history"].append(quarter_data)
            except YFinanceError as e:
                logger.warning(f"Failed to get earnings history for {ticker}: {str(e)}")
            # Get earnings estimate from info
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
            except YFinanceError as e:
                logger.warning(f"Failed to get earnings estimates for {ticker}: {str(e)}")
            return earnings_data
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            pd.errors.EmptyDataError,
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            MemoryError,
        ):
            raise
        except (IOError, ConnectionError, aiohttp.ClientError) as e:
            raise NetworkError(f"Network error when fetching earnings data for {ticker}: {str(e)}")
        except (APIError, ValidationError, RateLimitError):
            raise YFinanceError(DEFAULT_ERROR_MESSAGE)
            raise

    @enhanced_async_rate_limited(max_retries=0)
    async def get_earnings_dates(self, ticker: str) -> List[str]:
        """Get earnings dates (uses get_earnings_data)."""
        validate_ticker(ticker)
        try:
            earnings_data = await self.get_earnings_data(ticker)
            if "error" in earnings_data:  # Propagate error if get_earnings_data failed
                logger.warning(
                    f"Could not get earnings dates for {ticker} due to error: {earnings_data['error']}"
                )
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
        if "error" in info:
            return {"symbol": ticker, "error": info["error"]}
        return {
            "symbol": ticker,
            "recommendations": info.get("total_ratings", 0),
            "buy_percentage": info.get("buy_percentage", None),
            "positive_percentage": info.get("buy_percentage", None),
            "total_ratings": info.get("total_ratings", 0),
            "ratings_type": info.get("A", "A"),
            "date": None,
        }

    @enhanced_async_rate_limited(max_retries=0)
    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a ticker asynchronously.
        """
        validate_ticker(ticker)
        if not is_us_ticker(ticker):
            logger.debug(f"Skipping insider transactions for non-US ticker {ticker}")
            return []

        # Skip insider transactions for non-stock assets (ETFs, commodities, crypto)
        if not is_stock_ticker(ticker):
            logger.debug(f"Skipping insider transactions for non-stock asset {ticker}")
            return []
        base_url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"
        url = f"{base_url}/{ticker}"
        params = {"modules": "insiderTransactions", "formatted": "false"}
        try:
            data = await self._fetch_json(url, params)
            if (
                not data
                or "quoteSummary" not in data
                or "result" not in data["quoteSummary"]
                or not data["quoteSummary"]["result"]
            ):
                raise YFinanceError(DEFAULT_ERROR_MESSAGE)
            result = data["quoteSummary"]["result"][0]
            transactions = []
            if "insiderTransactions" in result and "transactions" in result["insiderTransactions"]:
                for transaction in result["insiderTransactions"]["transactions"]:
                    tx = {
                        "name": transaction.get("filerName", ""),
                        "title": transaction.get("filerRelation", ""),
                        "date": (
                            format_date(
                                pd.to_datetime(transaction["startDate"]["raw"], unit="s")
                            )
                            if "startDate" in transaction and "raw" in transaction["startDate"]
                            else None
                        ),
                        "transaction": transaction.get("transactionText", ""),
                        "shares": (
                            transaction["shares"]["raw"]
                            if "shares" in transaction and "raw" in transaction["shares"]
                            else 0
                        ),
                        "value": (
                            transaction["value"]["raw"]
                            if "value" in transaction and "raw" in transaction["value"]
                            else 0
                        ),
                    }
                    transactions.append(tx)
            return transactions
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            pd.errors.EmptyDataError,
            ModuleNotFoundError,
            RuntimeError,
            MemoryError,
        ):
            raise
        except (IOError, ConnectionError, aiohttp.ClientError) as e:
            raise NetworkError(
                f"Network error when fetching insider transactions for {ticker}: {str(e)}"
            )
        except (APIError, ValidationError, RateLimitError):
            raise

    @enhanced_async_rate_limited(max_retries=0)
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query asynchronously.
        """
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        base_url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": limit,
            "newsCount": 0,
            "enableFuzzyQuery": "true",
            "enableEnhancedTrivialQuery": "true",
        }
        try:
            data = await self._fetch_json(base_url, params)
            if not data or "quotes" not in data:
                return []
            results = []
            for quote in data["quotes"][:limit]:
                result = {
                    "symbol": quote.get("symbol", ""),
                    "name": quote.get("longname", quote.get("shortname", "")),
                    "exchange": quote.get("exchange", ""),
                    "type": quote.get("quoteType", ""),
                    "score": quote.get("score", 0),
                }
                results.append(result)
            return results
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            ModuleNotFoundError,
            RuntimeError,
            MemoryError,
        ) as e:
            raise e
        except (IOError, ConnectionError, aiohttp.ClientError) as e:
            raise NetworkError(f"Network error when searching tickers for '{query}': {str(e)}")
        except (APIError, ValidationError, RateLimitError, NetworkError) as exc:
            raise exc

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
            "two_hundred_day_avg": info.get("two_hundred_day_avg"),
        }


    async def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers in a batch asynchronously.
        Uses optimized async processing with prioritization and proper error handling.
        """
        if not tickers:
            return {}

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
            description=f"Processing {len(tickers)} tickers",
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
                    "error": "Invalid result format received",
                }

        return processed_results

    async def close(self) -> None:
        """Close the provider and clean up resources."""
        # No longer need to close session directly since we use SharedSessionManager
        # Just clear internal caches and let SharedSessionManager handle session lifecycle
        
        # Clear all caches to prevent memory leaks
        self.clear_cache()

        # OPTIMIZED: Skip heavy memory cleanup for faster shutdown
        # Python's garbage collector will handle cleanup automatically
        # Heavy cleanup moved to background or eliminated for better UX
        
        logger.debug("Async provider resources cleaned up (optimized for speed).")

    def clear_cache(self) -> None:
        """Clear internal caches and free up memory."""
        # Clear all internal caches
        if hasattr(self, "_ticker_cache"):
            self._ticker_cache.clear()
        if hasattr(self, "_stock_cache"):
            self._stock_cache.clear()
        if hasattr(self, "_ratings_cache"):
            self._ratings_cache.clear()

        # Use memory utilities for thorough cleanup
        try:
            from ...utils.memory_utils import clean_memory

            cleanup_results = clean_memory()
            logger.debug(f"Memory cleanup results: {cleanup_results}")
        except ImportError:
            # Fall back to basic cleanup if the utility isn't available
            # Import datetime and tzinfo here to check for memory leaks
            import sys

            if "pandas" in sys.modules:
                # Clear pandas caches if pandas is loaded
                try:
                    import pandas as pd

                    pd.core.common._possibly_clean_cache()
                except (ImportError, AttributeError):
                    pass

        logger.info("Async provider internal caches cleared and memory freed.")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the internal caches including LRU stats."""
        # Note: This only reflects the simple instance cache, not CacheManager
        info = {
            "ticker_cache_size": len(self._ticker_cache),
            "stock_object_cache_size": len(self._stock_cache),
            "ratings_cache_size": len(self._ratings_cache),
        }
        # Add LRU cache stats if available
        if hasattr(self._ticker_cache, 'get_stats'):
            info["ticker_cache_stats"] = self._ticker_cache.get_stats()
        if hasattr(self._ratings_cache, 'get_stats'):
            info["ratings_cache_stats"] = self._ratings_cache.get_stats()
        if hasattr(self._stock_cache, 'get_stats'):
            info["stock_cache_stats"] = self._stock_cache.get_stats()
        return info

    async def __aenter__(self):
        # SharedSessionManager handles session lifecycle, no setup needed
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __del__(self):
        """Cleanup resources when the object is garbage collected."""
        # Check if Python is shutting down to avoid import errors
        import sys
        if sys.meta_path is None:
            # Python is shutting down, skip cleanup to avoid errors
            return
            
        try:
            # Clear all caches
            if hasattr(self, "_ticker_cache"):
                self._ticker_cache.clear()
            if hasattr(self, "_stock_cache"):
                self._stock_cache.clear()
            if hasattr(self, "_ratings_cache"):
                self._ratings_cache.clear()

            # No longer need to manage session directly since we use SharedSessionManager

            # Use memory_utils if available for efficient cleanup
            try:
                from ...utils.memory_utils import clean_memory
                clean_memory()
            except (ImportError, AttributeError):
                # Fallback to manual cleanup if memory_utils not available
                # Clear any module references that might cause memory leaks
                # Particularly ABC module and tzinfo from datetime
                if sys.modules:  # Check if modules dict still exists
                    for module_name in ["abc", "pandas", "datetime"]:
                        if module_name in sys.modules:
                            try:
                                # Clear module-specific caches if needed
                                module = sys.modules[module_name]
                                if hasattr(module, "_abc_registry") and isinstance(
                                    module._abc_registry, dict
                                ):
                                    module._abc_registry.clear()
                                if hasattr(module, "_abc_cache") and isinstance(module._abc_cache, dict):
                                    module._abc_cache.clear()
                            except (AttributeError, KeyError, TypeError):
                                pass

                # Force garbage collection
                try:
                    import gc
                    gc.collect()
                except (ImportError, AttributeError):
                    pass
        except (RuntimeError, ValueError, TypeError, OSError, IOError, AttributeError):
            # Silently ignore any errors during cleanup to prevent issues during shutdown
            pass

