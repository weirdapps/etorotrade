"""
Yahoo Finance provider implementation.

This module implements the FinanceDataProvider interface for Yahoo Finance data.
It provides a consistent API for retrieving financial information with
appropriate rate limiting, caching, and error handling.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import yfinance as yf

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ...utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ...core.config import CACHE_CONFIG, COLUMN_NAMES
from ...core.errors import APIError, RateLimitError, ValidationError, YFinanceError
from ...core.logging import get_logger
from ...data.cache import default_cache_manager
from ...utils.market.ticker_utils import is_us_ticker, is_stock_ticker
from ...utils.network.rate_limiter import rate_limited
from .base_provider import FinanceDataProvider
from .yahoo_finance_base import YahooFinanceBaseProvider


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

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, **kwargs):
        """
        Initialize the Yahoo Finance provider.

        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries
            **kwargs: Additional keyword arguments (for factory compatibility)
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
            "two_hundred_day_avg": info.get("two_hundred_day_avg"),
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
        # Fast path for performance-critical first check
        cache_key = f"ticker_info:{ticker}"
        logger.debug(f"Getting ticker info from cache with key: {cache_key}")
        try:
            cached_info = default_cache_manager.get(cache_key)
            logger.debug(f"Cache result: {'hit' if cached_info is not None else 'miss'}")
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            cached_info = None
        if cached_info is not None:
            # Add a flag to indicate this came from cache
            # This will be used by the rate_limited decorator to adjust delays
            if isinstance(cached_info, dict):
                cached_info = cached_info.copy()  # Make a copy to avoid modifying cached data
                cached_info["from_cache"] = True
            return cached_info

        # If not in cache, continue with normal processing
        logger.debug(f"Getting ticker info for {ticker}")

        # Determine if this is a US stock for regional caching
        is_us = is_us_ticker(ticker)

        # Pre-check for known missing data fields (very fast checks)
        si_missing = self.is_data_field_probably_missing(ticker, "short_interest")
        peg_missing = default_cache_manager.is_data_known_missing(ticker, "peg_ratio")

        # Get ticker object (which might be cached)
        ticker_obj = self._get_ticker_object(ticker)

        # Basic information with proper rate limiting
        result = {}

        for attempt in range(self.max_retries):
            try:
                # Get basic info - handle potential None result or AttributeError
                try:
                    info = ticker_obj.info
                    if not info:
                        raise DataError(f"No information found for ticker {ticker}")
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Error accessing ticker info for {ticker}: {str(e)}")
                    # Provide fallback minimal info
                    info = {"symbol": ticker}

                # Extract key metrics using the base class helper
                result = self._extract_common_ticker_info(info)
                result["symbol"] = ticker  # Ensure the symbol is set correctly

                # Track missing data for performance optimization

                # Check for missing PEG ratio
                if "peg_ratio" not in result or result["peg_ratio"] is None:
                    # If we didn't know it was missing before, mark it now
                    if not peg_missing:
                        default_cache_manager.set_missing_data(ticker, "peg_ratio", is_us)
                        logger.debug(f"Marked PEG ratio as missing for {ticker}")

                # Additional metrics for US stocks
                if is_us and not skip_insider_metrics:
                    # Get short interest data if it's not known to be missing
                    if not si_missing:
                        try:
                            # Check for short interest in info
                            si_value = info.get("shortPercentOfFloat")
                            if si_value is not None:
                                result["short_float_pct"] = (
                                    si_value * 100 if isinstance(si_value, float) else None
                                )
                            else:
                                # Mark short interest as missing for this ticker
                                default_cache_manager.set_missing_data(
                                    ticker, "short_interest", is_us
                                )
                                logger.debug(f"Marked short interest as missing for {ticker}")
                        except Exception as e:
                            logger.warning(
                                f"Error processing short interest for {ticker}: {str(e)}"
                            )
                            # Mark as missing on error
                            default_cache_manager.set_missing_data(ticker, "short_interest", is_us)

                    # Only try to get insider data for actual stocks, not ETFs/commodities/crypto
                    if is_stock_ticker(ticker):
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
                                result["insider_ratio"] = (
                                    total_buys / (total_buys + total_sells)
                                    if (total_buys + total_sells) > 0
                                    else 0
                                )
                        except YFinanceError as e:
                            logger.warning(f"Failed to get insider data for {ticker}: {str(e)}")
                    else:
                        logger.debug(f"Skipping insider data for non-stock asset {ticker}")
                else:
                    # For non-US stocks, we know short interest is generally not available
                    if not si_missing:
                        default_cache_manager.set_missing_data(ticker, "short_interest", is_us)
                        logger.debug(f"Marked short interest as missing for non-US ticker {ticker}")

                # Add analyst data for all tickers (both US and non-US)
                # However, some asset types like ETFs/commodities may not have analyst ratings
                try:
                    analyst_data = self.get_analyst_ratings(ticker)
                    if analyst_data:
                        # Transfer key analyst fields to ticker info
                        result["analyst_count"] = analyst_data.get("recommendations", 0)
                        result["total_ratings"] = analyst_data.get("recommendations", 0)
                        result["buy_percentage"] = analyst_data.get("buy_percentage")

                        # Also set the rating type if we have analyst data
                        if result["total_ratings"] > 0:
                            result["A"] = "A"  # Default to all-time ratings

                        logger.debug(
                            f"Added analyst data for {ticker}: {result['total_ratings']} ratings, {result['buy_percentage']}% buy"
                        )
                except YFinanceError as e:
                    logger.warning(f"Error fetching analyst data for {ticker}: {str(e)}")
                    # For non-stock assets, this is expected - don't treat as a critical error

                # Cache the result before returning
                default_cache_manager.set(
                    cache_key, result, data_type="ticker_info", is_us_stock=is_us
                )
                logger.debug(f"Cached ticker info for {ticker} (is_us={is_us})")

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
    def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
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
        # Check cache first
        cache_key = f"historical_data:{ticker}:{period}:{interval}"
        cached_data = default_cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(
                f"Using cached historical data for {ticker} (period={period}, interval={interval})"
            )
            # For DataFrames, we can't add the from_cache attribute directly
            # We'll use a special wrapper to indicate it's from cache
            # which will be detected by the rate limiter when called with cache_aware=True
            return cached_data

        # Get data from API if not in cache
        ticker_obj = self._get_ticker_object(ticker)

        for attempt in range(self.max_retries):
            try:
                # Use the shared _extract_historical_data method from the base class
                data = self._extract_historical_data(ticker, ticker_obj, period, interval)

                # Cache the result
                default_cache_manager.set(cache_key, data, data_type="historical_data")
                logger.debug(
                    f"Cached historical data for {ticker} (period={period}, interval={interval})"
                )

                return data
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
        # Check cache first
        cache_key = f"earnings_dates:{ticker}"
        cached_data = default_cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Using cached earnings dates for {ticker}")
            # Signal to rate limiter that this data came from cache
            self.record_cache_hit(ticker)
            return cached_data

        ticker_obj = self._get_ticker_object(ticker)

        for attempt in range(self.max_retries):
            try:
                # Get earnings data
                calendar = ticker_obj.calendar

                # Handle cases where calendar might be None or not have earnings date
                if calendar is None or COLUMN_NAMES["EARNINGS_DATE"] not in calendar:
                    logger.debug(f"No earnings dates found for {ticker} in calendar")
                    # Try to get earnings date from info
                    try:
                        info = ticker_obj.info
                        if "earningsDate" in info:
                            earnings_date = info["earningsDate"]
                            logger.debug(
                                f"Found earnings date in info for {ticker}: {earnings_date}"
                            )
                            # If it's just a single value, turn it into a list
                            if not isinstance(earnings_date, list):
                                earnings_date = [earnings_date]
                            # Format dates
                            formatted_dates = [
                                self._format_date(date)
                                for date in earnings_date
                                if date is not None
                            ]
                            # Sort dates in descending order
                            formatted_dates.sort(reverse=True)
                            # Return the dates we have
                            if len(formatted_dates) >= 2:
                                result = (formatted_dates[0], formatted_dates[1])
                                default_cache_manager.set(
                                    cache_key, result, data_type="earnings_data"
                                )
                                logger.debug(f"Cached earnings dates for {ticker}")
                                return result
                            elif len(formatted_dates) == 1:
                                result = (formatted_dates[0], None)
                                default_cache_manager.set(
                                    cache_key, result, data_type="earnings_data"
                                )
                                logger.debug(f"Cached earnings dates for {ticker}")
                                return result
                    except Exception as e:
                        logger.debug(
                            f"Error getting earnings date from info for {ticker}: {str(e)}"
                        )

                    result = (None, None)
                    default_cache_manager.set(cache_key, result, data_type="earnings_data")
                    logger.debug(f"Cached empty earnings dates for {ticker}")
                    return result

                earnings_date = calendar[COLUMN_NAMES["EARNINGS_DATE"]]

                # Convert to list even if there's only one date
                if not isinstance(earnings_date, list):
                    earnings_date = [earnings_date]

                # Format dates
                formatted_dates = [
                    self._format_date(date) for date in earnings_date if date is not None
                ]

                # Sort dates in descending order
                formatted_dates.sort(reverse=True)

                # Return the last two earnings dates
                if len(formatted_dates) >= 2:
                    result = (formatted_dates[0], formatted_dates[1])
                    default_cache_manager.set(cache_key, result, data_type="earnings_data")
                    logger.debug(f"Cached earnings dates for {ticker}")
                    return result
                elif len(formatted_dates) == 1:
                    result = (formatted_dates[0], None)
                    default_cache_manager.set(cache_key, result, data_type="earnings_data")
                    logger.debug(f"Cached earnings dates for {ticker}")
                    return result
                else:
                    result = (None, None)
                    default_cache_manager.set(cache_key, result, data_type="earnings_data")
                    logger.debug(f"Cached empty earnings dates for {ticker}")
                    return result

            except RateLimitError as rate_error:
                # Specific handling for rate limits - just re-raise YFinanceError("An error occurred")
                raise rate_error
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} earnings dates: {str(e)}. Retrying in {delay:.2f}s"
                    )
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
        # Check cache first
        cache_key = f"analyst_ratings:{ticker}"
        cached_data = default_cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Using cached analyst ratings for {ticker}")
            # Make a copy to avoid modifying cached data
            if isinstance(cached_data, dict):
                cached_data = cached_data.copy()
                cached_data["from_cache"] = True
            return cached_data

        # We used to skip analyst ratings for non-US tickers, but this is no longer necessary
        # Many international stocks have analyst coverage, so we'll try to get it for all tickers
        # We'll only return empty data if we can't find any

        ticker_obj = self._get_ticker_object(ticker)

        for attempt in range(self.max_retries):
            try:
                # Get analyst consensus data using the enhanced base method
                consensus = self._get_analyst_consensus(ticker_obj, ticker)

                # Get the recommendations - this can fail for ETFs/commodities/crypto
                recommendations = None
                try:
                    recommendations = ticker_obj.recommendations
                except Exception as e:
                    logger.debug(f"Could not get recommendations for {ticker}: {str(e)}")
                    # This is expected for non-stock assets like ETFs, commodities, crypto

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

                # Cache the result
                default_cache_manager.set(cache_key, result, data_type="analysis")
                logger.debug(f"Cached analyst ratings for {ticker}")

                return result

            except RateLimitError as rate_error:
                # Specific handling for rate limits - just re-raise YFinanceError("An error occurred")
                raise rate_error
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} analyst ratings: {str(e)}. Retrying in {delay:.2f}s"
                    )
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

        # Skip insider transactions for non-stock assets (ETFs, commodities, crypto)
        if not is_stock_ticker(ticker):
            logger.debug(f"Skipping insider transactions for non-stock asset {ticker}")
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
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} insider transactions: {str(e)}. Retrying in {delay:.2f}s"
                    )
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
                if (
                    not search_results
                    or "quotes" not in search_results
                    or not search_results["quotes"]
                ):
                    logger.debug(f"No search results found for query '{query}'")
                    return []

                # Format results
                results = []
                for quote in search_results["quotes"][:limit]:
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
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt+1}/{self.max_retries} failed for search query '{query}': {str(e)}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    raise YFinanceError(f"Failed to get data after {self.max_retries} attempts")

    @with_retry
    def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
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

        # Pre-check cache and known missing data to optimize batch processing
        results = {}
        tickers_to_fetch = []

        # Ultra-optimized first pass - bulk check cache with batch_get
        cache_keys = [f"ticker_info:{ticker}" for ticker in tickers]
        cached_results = default_cache_manager.batch_get(cache_keys, data_type="ticker_info")

        # Process cached results and identify tickers to fetch
        for ticker in tickers:
            # Determine if this is a US stock
            is_us = is_us_ticker(ticker)

            # Check if we got this ticker from cache
            cache_key = f"ticker_info:{ticker}"
            if cache_key in cached_results:
                cached_info = cached_results[cache_key]
                logger.debug(f"Using cached ticker info for {ticker} in batch")
                results[ticker] = cached_info
                # Record cache hit for rate limiting optimization
                self.record_cache_hit(ticker)
                continue

            # Pre-mark data that's likely to be missing based on ticker characteristics
            # This is especially useful for batch operations to avoid redundant API calls
            if not is_us:
                # For non-US stocks, SI data is almost always missing
                if not default_cache_manager.is_data_known_missing(ticker, "short_interest"):
                    default_cache_manager.set_missing_data(
                        ticker, "short_interest", is_us_stock=False
                    )
                    logger.debug(
                        f"Pre-emptively marked short interest as missing for non-US ticker {ticker}"
                    )

            # Add to list of tickers to fetch
            tickers_to_fetch.append(ticker)

        # Second pass - fetch data for tickers not in cache
        # Prepare for batch caching
        cache_batch_items = []

        for ticker in tickers_to_fetch:
            try:
                # Get info with robust error handling
                ticker_info = self.get_ticker_info(ticker, skip_insider_metrics)
                results[ticker] = ticker_info

                # Normally we'd cache in get_ticker_info, but we're collecting for batch set
                # Remove any 'from_cache' flag that get_ticker_info might have added
                if isinstance(ticker_info, dict) and "from_cache" in ticker_info:
                    ticker_info = ticker_info.copy()
                    del ticker_info["from_cache"]

                # Add to batch cache items
                is_us = is_us_ticker(ticker)
                cache_key = f"ticker_info:{ticker}"
                cache_batch_items.append((cache_key, ticker_info, "ticker_info", is_us, False))
            except YFinanceError as e:
                # Create error result with minimal data
                error_result = self._process_error_for_batch(ticker, e)
                # Ensure at least basic info is present
                if isinstance(error_result, dict):
                    error_result.update(
                        {
                            "symbol": ticker,
                            "ticker": ticker,
                            "company": ticker.upper(),
                            "error": str(e),
                        }
                    )
                    results[ticker] = error_result
                else:
                    # Fallback if _process_error_for_batch returned non-dict
                    results[ticker] = {
                        "symbol": ticker,
                        "ticker": ticker,
                        "company": ticker.upper(),
                        "error": str(e),
                    }
            except Exception as e:
                # Handle any unexpected exceptions
                logger.error(f"Unexpected error processing ticker {ticker}: {str(e)}")
                results[ticker] = {
                    "symbol": ticker,
                    "ticker": ticker,
                    "company": ticker.upper(),
                    "error": f"Unexpected error: {str(e)}",
                }

        # Batch update the cache for all successful fetches
        if cache_batch_items:
            # Use the optimized batch_set method for maximum performance
            default_cache_manager.batch_set(cache_batch_items)
            logger.debug(f"Batch cached {len(cache_batch_items)} ticker info items")

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
            "ticker_cache_keys": list(self._ticker_cache.keys()),
        }

    def record_cache_hit(self, ticker: str = None) -> None:
        """
        Record a cache hit for rate limiting purposes.

        Args:
            ticker: The ticker symbol that had a cache hit
        """
        # Use the global rate limiter to track cache hits
        from ...utils.network.rate_limiter import global_rate_limiter

        global_rate_limiter.record_cache_hit(ticker)

    def is_data_field_probably_missing(self, ticker: str, data_field: str) -> bool:
        """
        Determine if a data field is likely to be missing for a ticker based on heuristics.

        This is useful for pre-emptively marking data as missing to avoid unnecessary API calls.

        Args:
            ticker: Ticker symbol
            data_field: Data field name (e.g., "short_interest", "peg_ratio")

        Returns:
            True if the data field is likely to be missing, False otherwise
        """
        # First check if the cache manager already knows it's missing
        if default_cache_manager.is_data_known_missing(ticker, data_field):
            return True

        # For short interest, typically only US stocks have this data
        if data_field == "short_interest" and not is_us_ticker(ticker):
            # Mark it as missing for future reference
            default_cache_manager.set_missing_data(ticker, data_field, is_us_stock=False)
            logger.debug(f"Pre-emptively marked {data_field} as missing for non-US ticker {ticker}")
            return True

        # For PEG ratio, no reliable heuristic, we need to check actual data

        # Default to not missing
        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the provider.

        This method returns statistics about cache hits, misses, and known missing data
        which helps evaluate the effectiveness of the optimization strategies.

        Returns:
            Dict containing performance statistics
        """
        # Get cache manager stats
        cache_stats = default_cache_manager.get_stats()

        # Get known missing data count by type
        missing_si_count = 0
        missing_peg_count = 0
        us_tickers_count = 0
        non_us_tickers_count = 0

        # Count tickers in cache
        tickers_in_cache = set()
        for key in self._ticker_cache.keys():
            tickers_in_cache.add(key)

            # Count by region
            if is_us_ticker(key):
                us_tickers_count += 1
            else:
                non_us_tickers_count += 1

            # Count missing data by field
            if default_cache_manager.is_data_known_missing(key, "short_interest"):
                missing_si_count += 1
            if default_cache_manager.is_data_known_missing(key, "peg_ratio"):
                missing_peg_count += 1

        # Compile results
        return {
            "tickers_count": len(tickers_in_cache),
            "us_tickers": us_tickers_count,
            "non_us_tickers": non_us_tickers_count,
            "missing_short_interest_count": missing_si_count,
            "missing_peg_ratio_count": missing_peg_count,
            "memory_cache": cache_stats.get("memory_cache", {}),
            "disk_cache": cache_stats.get("disk_cache", {}),
            "api_calls_avoided_from_missing_data": missing_si_count + missing_peg_count,
        }

    def _get_last_earnings_date(self, ticker_obj):
        """Get the last (past) earnings date for a ticker."""
        ticker_symbol = getattr(ticker_obj, "ticker", "unknown")
        logger.debug(f"Getting last earnings date for {ticker_symbol}")
        
        try:
            # Try quarterly income statement first - this is the most reliable source
            quarterly_income = getattr(ticker_obj, "quarterly_income_stmt", None)
            if quarterly_income is not None and not quarterly_income.empty:
                # Get the most recent quarter date
                latest_date = quarterly_income.columns[0]  # Most recent is first column
                result = latest_date.strftime("%Y-%m-%d")
                logger.debug(f"Found last earnings date from quarterly_income_stmt for {ticker_symbol}: {result}")
                return result
        except Exception as e:
            logger.debug(f"Error getting earnings from quarterly_income_stmt: {str(e)}")
        
        try:
            # Try quarterly earnings second - this should contain historical earnings dates
            quarterly_earnings = getattr(ticker_obj, "quarterly_earnings", None)
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                # Get the most recent earnings date from quarterly earnings
                latest_date = quarterly_earnings.index.max()
                result = latest_date.strftime("%Y-%m-%d")
                logger.debug(f"Found last earnings date from quarterly_earnings for {ticker_symbol}: {result}")
                return result
        except Exception as e:
            logger.debug(f"Error getting last earnings date from quarterly_earnings: {str(e)}")
        
        try:
            # Try earnings_dates attribute
            earnings_dates = getattr(ticker_obj, "earnings_dates", None)
            if earnings_dates is not None and not earnings_dates.empty:
                import pandas as pd
                # Make timezone-aware comparison
                today = pd.Timestamp.now(tz=earnings_dates.index.tz)
                logger.debug(f"Found earnings_dates for {ticker_symbol}, filtering for past dates before {today}")
                
                # Filter for past dates
                past_dates = earnings_dates[earnings_dates.index < today]
                if not past_dates.empty:
                    # Get the most recent past date
                    latest_date = past_dates.index.max()
                    result = latest_date.strftime("%Y-%m-%d")
                    logger.debug(f"Found last earnings date from earnings_dates for {ticker_symbol}: {result}")
                    return result
                else:
                    logger.debug(f"No past earnings dates found in earnings_dates for {ticker_symbol}")
        except Exception as e:
            logger.debug(f"Error getting last earnings date from earnings_dates: {str(e)}")
        
        try:
            # Try calendar last - this might contain future dates  
            calendar = getattr(ticker_obj, "calendar", None)
            if calendar is not None and COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                earnings_date_list = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                logger.debug(f"Found calendar earnings dates for {ticker_symbol}: {earnings_date_list}")
                if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                    import pandas as pd
                    from datetime import datetime
                    today = pd.Timestamp.now().date()
                    logger.debug(f"Today's date: {today}")
                    
                    # Find the latest past earnings date
                    latest_date = None
                    for date in earnings_date_list:
                        logger.debug(f"Checking date {date} (type: {type(date)}) against today {today}")
                        # Handle both datetime.date and datetime.datetime objects
                        if hasattr(date, 'date'):
                            # It's a datetime object, get the date part
                            date_obj = date.date()
                        elif hasattr(date, 'strftime'):
                            # It's already a date object
                            date_obj = date
                        else:
                            # Try to convert to date
                            try:
                                date_obj = pd.to_datetime(date).date()
                            except:
                                continue
                        
                        if date_obj < today:
                            if latest_date is None or date_obj > latest_date:
                                latest_date = date_obj
                    
                    if latest_date is not None:
                        result = latest_date.strftime("%Y-%m-%d")
                        logger.debug(f"Found last earnings date for {ticker_symbol}: {result}")
                        return result
                    else:
                        logger.debug(f"No past earnings dates found in calendar for {ticker_symbol}")
        except Exception as e:
            logger.debug(f"Error getting last earnings date from calendar: {str(e)}")
        
        logger.debug(f"No last earnings date found for {ticker_symbol}")
        return None

    def _extract_last_earnings_date(self, info: Dict[str, Any]) -> Optional[str]:
        """
        Extract the last (past) earnings date from info dictionary.
        
        Args:
            info: Dictionary with ticker info
            
        Returns:
            Last earnings date in YYYY-MM-DD format or None
        """
        # Get the ticker symbol from info
        ticker_symbol = info.get("symbol")
        if not ticker_symbol:
            return None
        
        # Get the ticker object and use our _get_last_earnings_date method
        try:
            ticker_obj = self._get_ticker_object(ticker_symbol)
            return self._get_last_earnings_date(ticker_obj)
        except Exception as e:
            logger.debug(f"Error getting last earnings date for {ticker_symbol}: {str(e)}")
            return None
