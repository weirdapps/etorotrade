"""
Base Yahoo Finance provider implementation.

This module provides a base implementation for Yahoo Finance providers,
with common functionality shared by different provider types.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import yfinance as yf

from ...core.errors import (
    APIError,
    DataError,
    NetworkError,
    ResourceNotFoundError,
    ValidationError,
    YFinanceError,
)
from ...core.logging import get_logger
from ...data.cache import cached, default_cache_manager
from ...utils.error_handling import translate_error, with_retry
from ...utils.market.ticker_utils import is_us_ticker, validate_ticker


# Set up logging
logger = get_logger(__name__)


class YahooFinanceBaseProvider(ABC):
    """
    Base class for Yahoo Finance data providers.

    This class provides common functionality for both synchronous and
    asynchronous Yahoo Finance providers.

    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the base Yahoo Finance provider.

        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries (exponential backoff applied)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize cache
        self._ticker_cache = {}

        logger.debug(
            f"Initialized YahooFinanceBaseProvider with max_retries={max_retries}, retry_delay={retry_delay}"
        )

    def _handle_retry_logic(
        self, error: Exception, attempt: int, ticker: str, operation: str
    ) -> float:
        """
        Handle retry logic for failed API calls.

        Args:
            error: The error that occurred
            attempt: Current attempt number (0-based)
            ticker: Ticker symbol
            operation: Operation being performed

        Returns:
            Delay in seconds before next retry

        Raises:
            YFinanceError: When maximum retry attempts are reached
        """
        delay = self.retry_delay * (2**attempt)

        if attempt < self.max_retries - 1:
            logger.warning(
                f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} {operation}: {str(error)}. Retrying in {delay:.2f}s"
            )
            return delay
        else:
            logger.error(
                f"All {self.max_retries} attempts failed for {ticker} {operation}: {str(error)}"
            )
            raise YFinanceError(
                f"Failed to get {operation} for {ticker} after {self.max_retries} attempts",
                details={"ticker": ticker, "operation": operation, "attempts": self.max_retries},
            )

    def _format_date(self, dt) -> Optional[str]:
        """
        Format a date or datetime object to YYYY-MM-DD format.

        Args:
            dt: Date or datetime object, or None

        Returns:
            Formatted date string or None if dt is None
        """
        if dt is None:
            return None

        if isinstance(dt, str):
            # Try to parse string to datetime
            try:
                dt = pd.to_datetime(dt)
            except ValueError:
                return dt  # Return original string if parsing fails

        # Handle timestamps (sometimes Yahoo Finance returns Unix timestamps)
        if (
            isinstance(dt, (int, float)) and dt > 1000000000
        ):  # Check if it could be a timestamp (over ~1973)
            try:
                dt = datetime.fromtimestamp(dt)
            except Exception:
                pass

        if isinstance(dt, (datetime, pd.Timestamp)):
            return dt.strftime("%Y-%m-%d")
        elif isinstance(dt, date):
            return dt.strftime("%Y-%m-%d")

        # Return string representation as a fallback
        return str(dt)

    def _extract_earnings_date(self, info: Dict[str, Any]) -> Optional[str]:
        """
        Extract earnings date from info dictionary handling different formats.

        Args:
            info: Dictionary with ticker info

        Returns:
            Earnings date in YYYY-MM-DD format or None
        """
        # Try different possible field names
        for field in ["earningsDate", "earnings_date", "next_earnings_date"]:
            value = safe_extract_value(info, field)
            if value is not None:
                logger.debug(
                    f"Found earnings date in field '{field}': {value} (type: {type(value)})"
                )

                # Handle different formats
                if isinstance(value, list):
                    # Some providers return a list of dates - use the first one (newest)
                    # Filter out any None values
                    valid_dates = [date for date in value if date is not None]
                    if valid_dates:
                        # Sort by date (assuming they're all timestamps or date objects)
                        try:
                            # Convert all to datetime objects for comparison
                            dates = []
                            for d in valid_dates:
                                if isinstance(d, (int, float)):
                                    dates.append(datetime.fromtimestamp(d))
                                else:
                                    parsed_date = pd.to_datetime(d, errors="coerce")
                                    if pd.notna(parsed_date):
                                        dates.append(parsed_date)

                            # Sort and take the latest
                            if dates:
                                dates.sort(reverse=True)  # Latest first
                                return self._format_date(dates[0])
                        except Exception as e:
                            logger.debug(f"Error sorting earnings dates: {str(e)}")

                        # If sorting fails, just use the first one
                        return self._format_date(valid_dates[0])
                else:
                    # Single value
                    return self._format_date(value)

        # None of the fields found or had valid data
        return None

    def _extract_last_earnings_date(self, info: Dict[str, Any]) -> Optional[str]:
        """
        Extract the last (past) earnings date from info dictionary.
        
        Args:
            info: Dictionary with ticker info
            
        Returns:
            Last earnings date in YYYY-MM-DD format or None
        """
        # This is a placeholder - the actual implementation should be in the provider
        # For now, return None and let the provider handle it
        return None

    def _extract_common_ticker_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common ticker information fields from the raw info dictionary.

        Args:
            info: Raw ticker information dictionary

        Returns:
            Dictionary containing extracted ticker information
        """
        # Check if info is a valid dictionary - if not, provide minimal result
        if not isinstance(info, dict):
            logger.warning(f"Invalid info object type: {type(info)}. Using empty info.")
            info = {"symbol": info.get("symbol", "unknown") if hasattr(info, "get") else "unknown"}

        # Helper function to safely extract values with type checking
        global safe_extract_value

        def safe_extract_value(data, key, default=None):
            try:
                if isinstance(data, dict) and key in data and data[key] is not None:
                    return data[key]
            except (TypeError, AttributeError, KeyError) as e:
                logger.debug(f"Error extracting {key}: {str(e)}")
            return default

        # Get ticker symbol for logging
        symbol = safe_extract_value(info, "symbol", safe_extract_value(info, "ticker", "unknown"))

        try:
            # Get name safely (multiple fallbacks)
            name = safe_extract_value(
                info, "shortName", safe_extract_value(info, "longName", symbol)
            )
            if name:
                try:
                    name = name.upper()
                except (AttributeError, TypeError):
                    name = str(name).upper()
            else:
                name = symbol.upper()

            # Extract the common fields
            result = {
                "symbol": symbol,
                "name": name,
                "company": name,  # Match both formats
                # Try multiple price fields in order of preference
                "price": (
                    safe_extract_value(info, "regularMarketPrice") or
                    safe_extract_value(info, "currentPrice") or
                    safe_extract_value(info, "lastPrice") or
                    safe_extract_value(info, "previousClose")
                ),
                "current_price": (
                    safe_extract_value(info, "regularMarketPrice") or
                    safe_extract_value(info, "currentPrice") or
                    safe_extract_value(info, "lastPrice") or
                    safe_extract_value(info, "previousClose")
                ),  # Match both formats
                "change": safe_extract_value(info, "regularMarketChange"),
                "change_percent": safe_extract_value(info, "regularMarketChangePercent"),
                "market_cap": safe_extract_value(info, "marketCap"),
                "volume": safe_extract_value(info, "regularMarketVolume"),
                "avg_volume": safe_extract_value(info, "averageVolume"),
                "pe_trailing": safe_extract_value(info, "trailingPE"),  # Match both formats
                "pe_ratio": safe_extract_value(info, "trailingPE"),
                "pe_forward": safe_extract_value(info, "forwardPE"),  # Match both formats
                "forward_pe": safe_extract_value(info, "forwardPE"),
                "dividend_yield": safe_extract_value(info, "dividendYield"),
                # Price target fields - use median as primary, extract all for validation
                "target_price": safe_extract_value(info, "targetMedianPrice"),  # Changed to median
                "target_price_mean": safe_extract_value(info, "targetMeanPrice"),
                "target_price_median": safe_extract_value(info, "targetMedianPrice"),
                "target_price_high": safe_extract_value(info, "targetHighPrice"),
                "target_price_low": safe_extract_value(info, "targetLowPrice"),
                "price_target_analyst_count": safe_extract_value(info, "numberOfAnalystOpinions"),
                "beta": safe_extract_value(info, "beta"),
                "eps": safe_extract_value(info, "trailingEps"),
                "forward_eps": safe_extract_value(info, "forwardEps"),
                "peg_ratio": safe_extract_value(info, "pegRatio"),
                "sector": safe_extract_value(info, "sector"),
                "industry": safe_extract_value(info, "industry"),
                "fifty_two_week_high": safe_extract_value(info, "fiftyTwoWeekHigh"),
                "fifty_two_week_low": safe_extract_value(info, "fiftyTwoWeekLow"),
                "fifty_day_avg": safe_extract_value(info, "fiftyDayAverage"),
                "two_hundred_day_avg": safe_extract_value(info, "twoHundredDayAverage"),
                "exchange": safe_extract_value(info, "exchange"),
                "country": safe_extract_value(info, "country"),
                "short_percent": safe_extract_value(info, "shortPercentOfFloat"),  # Short interest
                # Extract and format earnings date - handle different possible field names and formats
                # Set both earnings_date and last_earnings for consistency with async provider
                "earnings_date": self._extract_last_earnings_date(info),  # Last earnings date
                "last_earnings": self._extract_last_earnings_date(info),  # Last earnings date (for data processor)
                # Calculate earnings growth from quarterly data
                "earnings_growth": self._calculate_earnings_growth(ticker),
                "data_source": "yfinance",
            }

            # Process special fields

            # Add fallback logic for missing PE ratios
            self._add_fallback_pe_ratios(result, info)

            # Process short interest - convert to percentage
            if result.get("short_percent") is not None:
                try:
                    result["short_float_pct"] = float(result["short_percent"]) * 100
                except (ValueError, TypeError):
                    result["short_float_pct"] = None

            # Validate PEG ratio - sometimes YFinance returns extreme values that aren't usable
            if result.get("peg_ratio") is not None:
                try:
                    peg = float(result["peg_ratio"])
                    # Check if the PEG is in a reasonable range
                    if peg < 0 or peg > 50:
                        logger.debug(f"Invalid PEG ratio for {symbol}: {peg} - setting to None")
                        result["peg_ratio"] = None
                except (ValueError, TypeError):
                    result["peg_ratio"] = None

            # Yahoo Finance returns dividend yield numbers correctly
            # We don't modify them as the display formatter will handle properly
            if result["dividend_yield"] is not None:
                try:
                    # Just make sure it's a float
                    result["dividend_yield"] = float(result["dividend_yield"])
                except (ValueError, TypeError):
                    pass

            # Note: upside will be calculated dynamically from price and target_price
            # No longer storing upside as a field to ensure consistency

            return result

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            # Handle data processing errors
            logger.warning(f"Data processing error for {symbol}: {str(e)}. Returning minimal info.")
            return {
                "symbol": symbol,
                "name": symbol.upper(),
                "company": symbol.upper(),
                "data_source": "yfinance",
                "error": f"Data processing error: {str(e)}",
            }
        except (APIError, NetworkError) as e:
            # Handle API-related errors
            logger.warning(f"API error for {symbol}: {str(e)}. Returning minimal info.")
            return {
                "symbol": symbol,
                "name": symbol.upper(),
                "company": symbol.upper(),
                "data_source": "yfinance",
                "error": f"API error: {str(e)}",
            }
        except YFinanceError as e:
            # Handle other YFinance-specific errors
            logger.warning(f"YFinance error for {symbol}: {str(e)}. Returning minimal info.")
            return {
                "symbol": symbol,
                "name": symbol.upper(),
                "company": symbol.upper(),
                "data_source": "yfinance",
                "error": f"YFinance error: {str(e)}",
            }
        except Exception as e:
            # Handle truly unexpected errors
            logger.error(f"Unexpected error extracting info for {symbol}: {str(e)}. Returning minimal info.")
            return {
                "symbol": symbol,
                "name": symbol.upper(),
                "company": symbol.upper(),
                "data_source": "yfinance",
                "error": f"Unexpected error: {str(e)}",
            }

    def _add_fallback_pe_ratios(self, result, info):
        """
        Add fallback logic for missing PE ratios by calculating from price and EPS.
        
        Args:
            result: The result dictionary to update
            info: The raw info data from Yahoo Finance
        """
        try:
            # If trailing PE is missing, try to calculate from price and EPS
            if result.get("pe_trailing") is None or result.get("pe_ratio") is None:
                price = result.get("price")
                eps = result.get("eps")
                
                if price is not None and eps is not None and eps > 0:
                    calculated_pe = price / eps
                    # Sanity check: PE should be positive and reasonable (0.1 to 1000)
                    if 0.1 <= calculated_pe <= 1000:
                        if result.get("pe_trailing") is None:
                            result["pe_trailing"] = calculated_pe
                        if result.get("pe_ratio") is None:
                            result["pe_ratio"] = calculated_pe
                        logger.debug(f"Calculated trailing PE for {result.get('symbol')}: {calculated_pe:.2f}")
            
            # If forward PE is missing, try to calculate from price and forward EPS
            if result.get("pe_forward") is None or result.get("forward_pe") is None:
                price = result.get("price")
                forward_eps = result.get("forward_eps")
                
                if price is not None and forward_eps is not None and forward_eps > 0:
                    calculated_forward_pe = price / forward_eps
                    # Sanity check: PE should be positive and reasonable (0.1 to 1000)
                    if 0.1 <= calculated_forward_pe <= 1000:
                        if result.get("pe_forward") is None:
                            result["pe_forward"] = calculated_forward_pe
                        if result.get("forward_pe") is None:
                            result["forward_pe"] = calculated_forward_pe
                        logger.debug(f"Calculated forward PE for {result.get('symbol')}: {calculated_forward_pe:.2f}")
                        
        except Exception as e:
            logger.debug(f"Error in fallback PE calculation for {result.get('symbol', 'unknown')}: {e}")

    def _calculate_upside_potential(self, current_price, target_price):
        """
        Calculate upside potential as a percentage.
        
        Note: This method is deprecated. Upside is now calculated dynamically
        from price and target_price to ensure consistency.

        Args:
            current_price: Current stock price
            target_price: Target price

        Returns:
            Upside potential as a percentage, or None if either price is None
        """
        if current_price is None or target_price is None or current_price == 0:
            return None

        return ((target_price - current_price) / current_price) * 100

    @abstractmethod
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker.

        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict containing stock information
        """
        pass

    @abstractmethod
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
            DataFrame with historical price data
        """
        pass

    @with_retry
    def _get_ticker_object(self, ticker: str) -> yf.Ticker:
        """
        Get a yfinance Ticker object for the given symbol with caching.

        Args:
            ticker: Stock ticker symbol

        Returns:
            yf.Ticker: Ticker object for the given symbol

        Raises:
            ValidationError: If the ticker is invalid
        """
        # Validate the ticker format
        validate_ticker(ticker)

        # Use cache key based on ticker
        cache_key = f"ticker_obj:{ticker}"

        # Check in global cache first
        cached_obj = default_cache_manager.get(cache_key)
        if cached_obj is not None:
            logger.debug(f"Using cached ticker object for {ticker} from global cache")
            return cached_obj

        # Check in local cache next
        if ticker in self._ticker_cache:
            logger.debug(f"Using cached ticker object for {ticker} from local cache")
            return self._ticker_cache[ticker]

        # Create new ticker object
        try:
            ticker_obj = yf.Ticker(ticker)

            # Store in both caches
            self._ticker_cache[ticker] = ticker_obj
            default_cache_manager.set(cache_key, ticker_obj, data_type="ticker_info")

            return ticker_obj
        except YFinanceError as e:
            raise e

    def calculate_upside_potential(self, current_price, target_price):
        """
        Calculate upside potential as a percentage.

        Args:
            current_price: Current stock price
            target_price: Target price

        Returns:
            Upside potential as a percentage, or None if either price is None
        """
        if current_price is None or target_price is None or current_price == 0:
            return None

        return ((target_price - current_price) / current_price) * 100

    def _extract_historical_data(self, ticker, ticker_obj, period, interval):
        """
        Extract historical price data.

        Args:
            ticker: Stock ticker symbol
            ticker_obj: yfinance Ticker object
            period: Time period
            interval: Data interval

        Returns:
            DataFrame containing historical data
        """
        # Validate period and interval
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        valid_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]

        if period not in valid_periods:
            raise ValidationError(
                f"Invalid period: {period}. Valid periods are: {', '.join(valid_periods)}"
            )

        if interval not in valid_intervals:
            raise ValidationError(
                f"Invalid interval: {interval}. Valid intervals are: {', '.join(valid_intervals)}"
            )

        # Get historical data
        try:
            history = ticker_obj.history(period=period, interval=interval)

            # Check if data is empty
            if history.empty:
                logger.warning(
                    f"No historical data available for {ticker} with period={period}, interval={interval}"
                )
                return pd.DataFrame()

            # Reset index to have Date as a column
            history = history.reset_index()

            # Rename columns to match our schema
            history.columns = [c if c != "Date" else "date" for c in history.columns]

            # Format date column
            if "date" in history.columns:
                history["date"] = history["date"].apply(self._format_date)

            return history
        except YFinanceError as e:
            logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
            raise e

    def _get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get price data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing price data
        """
        ticker_obj = self._get_ticker_object(ticker)

        try:
            # Get info to access price data
            info = ticker_obj.info

            # Extract relevant price information
            def safe_extract_value(data, key, default=None):
                if key in data and data[key] is not None:
                    return data[key]
                return default

            return {
                "ticker": ticker,
                "price": safe_extract_value(info, "regularMarketPrice"),
                "previous_close": safe_extract_value(info, "regularMarketPreviousClose"),
                "open": safe_extract_value(info, "regularMarketOpen"),
                "day_high": safe_extract_value(info, "regularMarketDayHigh"),
                "day_low": safe_extract_value(info, "regularMarketDayLow"),
                "volume": safe_extract_value(info, "regularMarketVolume"),
                "price_change": safe_extract_value(info, "regularMarketDayHigh", 0)
                - safe_extract_value(info, "regularMarketDayLow", 0),
                "price_change_percentage": safe_extract_value(
                    info, "regularMarketDayChangePercent", 0
                ),
            }
        except YFinanceError as e:
            logger.error(f"Error extracting price data for {ticker}: {str(e)}")
            raise e

    def _process_ticker_info(
        self, ticker: str, ticker_obj: yf.Ticker, skip_insider_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Process ticker information to extract relevant fields.

        Args:
            ticker: Stock ticker symbol
            ticker_obj: yfinance Ticker object
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict containing stock information
        """
        # Get ticker info
        try:
            info = ticker_obj.info
            if not info:
                raise APIError(f"Failed to retrieve info for {ticker}")

            # Use shared extraction logic
            result = self._extract_common_ticker_info(info)
            result["symbol"] = ticker  # Ensure the symbol is set correctly

            # Add additional metrics for US stocks - skip insider metrics if requested
            if is_us_ticker(ticker) and not skip_insider_metrics:
                try:
                    # Get insider data
                    insider_data = ticker_obj.institutional_holders

                    if insider_data is not None and not insider_data.empty:
                        # Calculate insider metrics
                        total_buys = sum(
                            1 for _, row in insider_data.iterrows() if row.get("Shares", 0) > 0
                        )
                        total_sells = sum(
                            1 for _, row in insider_data.iterrows() if row.get("Shares", 0) < 0
                        )

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

            return result
        except YFinanceError as e:
            logger.error(f"Error processing ticker info for {ticker}: {str(e)}")
            raise e

    def _get_analyst_consensus(self, ticker_obj: yf.Ticker, ticker: str) -> Dict[str, Any]:
        """
        Get analyst consensus data.

        Args:
            ticker_obj: yfinance Ticker object
            ticker: Ticker symbol for logging

        Returns:
            Dict containing analyst consensus information
        """
        # Default values
        result = {
            "total_ratings": 0,
            "buy_percentage": None,
            "analyst_count": 0,
            "recommendations": None,
        }

        try:
            # Try to get data directly from info first (faster and more reliable)
            try:
                ticker_info = ticker_obj.info
                if ticker_info and isinstance(ticker_info, dict):
                    # Get number of analysts
                    number_of_analysts = ticker_info.get("numberOfAnalystOpinions", 0)
                    if number_of_analysts > 0:
                        result["total_ratings"] = number_of_analysts
                        result["analyst_count"] = number_of_analysts

                        # Try to get recommendation key
                        rec_key = ticker_info.get("recommendationKey", "").lower()
                        if rec_key:
                            # Map recommendation key to buy percentage
                            if rec_key in ["buy", "strongbuy"]:
                                result["buy_percentage"] = 85.0
                                logger.debug(
                                    f"Using recommendationKey '{rec_key}' for {ticker}: 85%"
                                )
                            elif rec_key in ["outperform"]:
                                result["buy_percentage"] = 75.0
                                logger.debug(
                                    f"Using recommendationKey '{rec_key}' for {ticker}: 75%"
                                )
                            elif rec_key in ["hold"]:
                                result["buy_percentage"] = 50.0
                                logger.debug(
                                    f"Using recommendationKey '{rec_key}' for {ticker}: 50%"
                                )
                            elif rec_key in ["underperform"]:
                                result["buy_percentage"] = 25.0
                                logger.debug(
                                    f"Using recommendationKey '{rec_key}' for {ticker}: 25%"
                                )
                            elif rec_key in ["sell"]:
                                result["buy_percentage"] = 15.0
                                logger.debug(
                                    f"Using recommendationKey '{rec_key}' for {ticker}: 15%"
                                )

                        # If no key but we have mean, estimate percentage that way
                        if result["buy_percentage"] is None:
                            rec_mean = ticker_info.get("recommendationMean", None)
                            if rec_mean is not None:
                                # Convert 1-5 scale to percentage (1=Strong Buy, 5=Sell)
                                # 1 = 90%, 3 = 50%, 5 = 10%
                                result["buy_percentage"] = max(0, min(100, 110 - (rec_mean * 20)))
                                logger.debug(
                                    f"Used recommendationMean ({rec_mean}) to estimate buy percentage for {ticker}: {result['buy_percentage']:.1f}%"
                                )

                # If we got analyst count but no buy percentage, try to get it from recommendations
                if result["analyst_count"] > 0 and result["buy_percentage"] is None:
                    logger.debug(
                        f"Got analyst count but no buy percentage from info for {ticker}, trying recommendations"
                    )
            except Exception as e:
                logger.warning(f"Error extracting analyst data from info for {ticker}: {str(e)}")

            # Try to get data from recommendations attribute
            # This can fail for ETFs, commodities, and cryptocurrencies
            try:
                recommendations = None
                try:
                    recommendations = ticker_obj.recommendations
                except Exception as rec_error:
                    logger.debug(f"Could not get recommendations for {ticker}: {str(rec_error)}")
                    # This is expected for non-stock assets
                    recommendations = None

                # Skip if no data
                if recommendations is None or recommendations.empty:
                    if result["total_ratings"] == 0:
                        logger.debug(f"No analyst ratings available for {ticker}")
                    return result

                # Get the most recent date
                latest_date = recommendations.index.max()

                # Filter for the most recent recommendations
                latest_recommendations = recommendations[recommendations.index == latest_date]

                # Count the recommendations by grade
                recommendation_counts = {}

                for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
                    if col in latest_recommendations.columns:
                        value = latest_recommendations[col].sum()
                        if not pd.isna(value) and value > 0:
                            recommendation_counts[col.lower()] = int(value)

                total_ratings = sum(recommendation_counts.values())

                # If we have recommendation counts, use them preferentially
                if total_ratings > 0:
                    # Calculate buy percentage
                    buy_ratings = recommendation_counts.get(
                        "strongbuy", 0
                    ) + recommendation_counts.get("buy", 0)
                    buy_percentage = buy_ratings / total_ratings * 100

                    # Update result
                    result["total_ratings"] = total_ratings
                    result["analyst_count"] = total_ratings
                    result["buy_percentage"] = buy_percentage
                    result["recommendations"] = recommendation_counts

                    logger.debug(
                        f"Used recommendations data for {ticker}: {buy_ratings}/{total_ratings} analysts recommend buy ({buy_percentage:.1f}%)"
                    )
            except Exception as e:
                logger.warning(f"Error getting recommendations for {ticker}: {str(e)}")

            # If we still don't have buy percentage but have total ratings, try one more approach
            if result["total_ratings"] > 0 and result["buy_percentage"] is None:
                try:
                    # Try to get upgrades_downgrades as a last resort
                    upgrades_downgrades = getattr(ticker_obj, "upgrades_downgrades", None)
                    if upgrades_downgrades is not None and not upgrades_downgrades.empty:
                        # Count positive grades
                        positive_grades = [
                            "Buy",
                            "Overweight",
                            "Outperform",
                            "Strong Buy",
                            "Long-Term Buy",
                            "Positive",
                            "Market Outperform",
                            "Add",
                            "Sector Outperform",
                        ]

                        # Count positive grades in ToGrade column
                        if "ToGrade" in upgrades_downgrades.columns:
                            total_grades = len(upgrades_downgrades)
                            positive_count = upgrades_downgrades[
                                upgrades_downgrades["ToGrade"].isin(positive_grades)
                            ].shape[0]

                            if total_grades > 0:
                                buy_percentage = (positive_count / total_grades) * 100
                                result["buy_percentage"] = buy_percentage
                                logger.debug(
                                    f"Used upgrades_downgrades for {ticker}: {positive_count}/{total_grades} positive grades ({buy_percentage:.1f}%)"
                                )
                except Exception as e:
                    logger.warning(f"Error analyzing upgrades_downgrades for {ticker}: {str(e)}")

            return result
        except Exception as e:
            logger.warning(f"Error getting analyst consensus for {ticker}: {str(e)}")
            # Return what we have so far rather than raising
            return result

    def _get_empty_analyst_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get empty analyst data for tickers without analyst coverage.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing empty analyst data
        """
        logger.debug(f"Using empty analyst data for {ticker}")
        return {
            "symbol": ticker,
            "date": None,
            "recommendations": 0,
            "buy_percentage": 0,
            "strong_buy": 0,
            "buy": 0,
            "hold": 0,
            "sell": 0,
            "strong_sell": 0,
        }

    def _process_error_for_batch(self, ticker: str, error: Exception) -> Dict[str, Any]:
        """
        Process an error that occurred during batch processing.

        This method is used to provide a partial result for a ticker that
        encountered an error during batch processing, allowing the batch
        operation to continue for other tickers.

        Args:
            ticker: Ticker symbol
            error: Exception that occurred

        Returns:
            Dict with error information
        """
        logger.warning(f"Error processing {ticker} in batch: {str(error)}")

        # Return a minimal result with error information
        return {
            "symbol": ticker,
            "error": str(error),
            "error_type": type(error).__name__,
            "success": False,
            "data_source": "error",
        }

    def _calculate_earnings_growth(self, ticker: str) -> Optional[float]:
        """
        Calculate earnings growth by comparing recent quarters.
        
        Uses quarterly income statement data since quarterly_earnings is deprecated.
        Tries year-over-year growth first, falls back to quarter-over-quarter if needed.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Earnings growth as percentage, or None if unable to calculate
        """
        try:
            import yfinance as yf
            import pandas as pd
            
            yticker = yf.Ticker(ticker)
            
            # Use quarterly income statement (quarterly_earnings is deprecated)
            quarterly_income = yticker.quarterly_income_stmt
            if quarterly_income is None or quarterly_income.empty:
                logger.debug(f"No quarterly income statement data for {ticker}")
                return None
            
            # Look for net income fields in order of preference
            earnings_row = None
            potential_keys = [
                'Net Income From Continuing Operation Net Minority Interest',
                'Net Income Common Stockholders', 
                'Net Income',
                'Net Income Including Noncontrolling Interests',
                'Net Income Continuous Operations'
            ]
            
            for potential_key in potential_keys:
                if potential_key in quarterly_income.index:
                    earnings_row = quarterly_income.loc[potential_key]
                    logger.debug(f"Using earnings field '{potential_key}' for {ticker}")
                    break
            
            if earnings_row is None:
                logger.debug(f"No earnings row found in quarterly income statement for {ticker}")
                return None
            
            # Remove NaN values and sort by date descending (most recent first)
            earnings_data = earnings_row.dropna().sort_index(ascending=False)
            
            if len(earnings_data) < 2:
                logger.debug(f"Insufficient income statement data for {ticker} (only {len(earnings_data)} quarters)")
                return None
            
            # Convert to numeric and handle any string/object types
            earnings_data = pd.to_numeric(earnings_data, errors='coerce').dropna()
            
            if len(earnings_data) < 2:
                logger.debug(f"Insufficient numeric earnings data for {ticker}")
                return None
            
            # Try year-over-year calculation first (preferred)
            if len(earnings_data) >= 4:
                current_quarter = float(earnings_data.iloc[0])  # Most recent
                year_ago_quarter = float(earnings_data.iloc[3])  # 4 quarters ago
                
                if year_ago_quarter != 0 and abs(year_ago_quarter) > 1000:  # Avoid division by small numbers
                    # Year-over-year growth
                    yoy_growth = ((current_quarter - year_ago_quarter) / abs(year_ago_quarter)) * 100
                    logger.debug(f"Calculated YoY earnings growth for {ticker}: {yoy_growth:.1f}% (current: {current_quarter:,.0f}, year ago: {year_ago_quarter:,.0f})")
                    return round(yoy_growth, 1)
            
            # Fall back to quarter-over-quarter calculation
            current_quarter = float(earnings_data.iloc[0])  # Most recent
            previous_quarter = float(earnings_data.iloc[1])  # Previous quarter
            
            if previous_quarter != 0 and abs(previous_quarter) > 1000:  # Avoid division by small numbers
                # Quarter-over-quarter growth (not annualized for display)
                qoq_growth = ((current_quarter - previous_quarter) / abs(previous_quarter)) * 100
                logger.debug(f"Calculated QoQ earnings growth for {ticker}: {qoq_growth:.1f}% (current: {current_quarter:,.0f}, previous: {previous_quarter:,.0f})")
                return round(qoq_growth, 1)
            
            logger.debug(f"Unable to calculate earnings growth for {ticker} - zero or insufficient base earnings")
            return None
                
        except Exception as e:
            logger.debug(f"Error calculating earnings growth for {ticker}: {e}")
            return None
