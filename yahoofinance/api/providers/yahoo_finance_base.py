"""
Base functionality for Yahoo Finance providers.

This module provides common functionality that is shared between synchronous and 
asynchronous Yahoo Finance provider implementations to reduce code duplication.
"""

import logging
import time
import abc
from typing import Dict, Any, Optional, List, Tuple, Union, Set, TypeVar, Generic
import pandas as pd
import yfinance as yf
from datetime import datetime

from ...core.errors import YFinanceError, APIError, ValidationError, RateLimitError
from ...utils.market.ticker_utils import validate_ticker, is_us_ticker
from ...core.config import COLUMN_NAMES
from ...utils.data.format_utils import format_market_cap
from ...utils.network.provider_utils import is_rate_limit_error, safe_extract_value

logger = logging.getLogger(__name__)

# Type variables for provider implementations
T = TypeVar('T')  # Return type
U = TypeVar('U')  # Input type

class YahooFinanceBaseProvider(abc.ABC):
    """
    Base provider with shared functionality for Yahoo Finance data access.
    
    This class is not meant to be instantiated directly. Instead, use one of the
    concrete provider implementations:
    - YahooFinanceProvider for synchronous access
    - AsyncYahooFinanceProvider for asynchronous access
    
    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Yahoo Finance base provider.
        
        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries (exponential backoff applied)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def _calculate_upside_potential(self, current_price, target_price):
        """
        Calculate the upside potential percentage between current price and target price.
        
        Args:
            current_price: Current stock price
            target_price: Analyst target price
            
        Returns:
            float: Upside potential percentage or None if either price is None
        """
        try:
            if current_price is not None and target_price is not None and current_price > 0:
                return ((target_price / current_price) - 1) * 100
            return None
        except (TypeError, ZeroDivisionError):
            return None
        
    @abc.abstractmethod
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Get ticker information for a single symbol.
        Must be implemented by concrete provider classes.
        
        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: Whether to skip fetching insider trading metrics
            
        Returns:
            Dictionary with ticker information
        """
        pass
        
    @abc.abstractmethod
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker.
        Must be implemented by concrete provider classes.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with price data
        """
        pass
        
    @abc.abstractmethod
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a ticker.
        Must be implemented by concrete provider classes.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical price data
        """
        pass

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
        
        # Return cached ticker object if available
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]
        
        # Create new ticker object
        try:
            ticker_obj = yf.Ticker(ticker)
            self._ticker_cache[ticker] = ticker_obj
            return ticker_obj
        except Exception as e:
            raise ValidationError(f"Failed to create ticker object for {ticker}: {str(e)}")
            
    def calculate_upside_potential(self, current_price, target_price):
        """
        Calculate upside potential as a percentage.
        
        Args:
            current_price: Current stock price
            target_price: Target stock price
            
        Returns:
            Upside potential as a percentage or None if not calculable
        """
        if not current_price or not target_price or current_price <= 0:
            return None
        return ((target_price - current_price) / current_price) * 100
        
    def format_date(self, date_obj):
        """
        Format a date object to a string.
        
        Args:
            date_obj: Date object or string to format
            
        Returns:
            Formatted date string or None if input is None
        """
        if not date_obj:
            return None
            
        try:
            # If it's already a string in the right format, return it
            if isinstance(date_obj, str):
                # Try to parse and format to ensure consistency
                try:
                    date = datetime.strptime(date_obj, "%Y-%m-%d")
                    return date.strftime("%Y-%m-%d")
                except ValueError:
                    # If it's not in the expected format, just return it as is
                    return date_obj
                    
            # If it's a datetime, format it
            elif isinstance(date_obj, datetime):
                return date_obj.strftime("%Y-%m-%d")
                
            # Try to convert from pandas Timestamp or other date-like object
            return pd.Timestamp(date_obj).strftime("%Y-%m-%d")
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
            # These are common errors when parsing dates
            logger.debug(f"Error formatting date: {str(e)}")
            # If we can't parse it, return as string
            return str(date_obj) if date_obj else None
            
    def _format_market_cap(self, market_cap: Optional[Union[int, float]]) -> Optional[str]:
        """
        Format market cap value to a readable string with B/T suffix.
        
        Args:
            market_cap: Market cap value in numeric format
            
        Returns:
            Formatted market cap string with appropriate suffix
        """
        # Use the canonical format_market_cap function from format_utils
        return format_market_cap(market_cap)
            
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            Dict containing cache information
        """
        return {
            "cached_tickers": len(self._ticker_cache),
            "cached_ticker_symbols": list(self._ticker_cache.keys())
        }
        
    def clear_cache(self) -> None:
        """
        Clear any cached data.
        """
        self._ticker_cache.clear()
        
    def _extract_price_data(self, ticker: str, ticker_obj: yf.Ticker) -> Dict[str, Any]:
        """
        Extract price data from ticker object.
        
        This is a helper method to be used by all provider implementations.
        
        Args:
            ticker: Stock ticker symbol
            ticker_obj: yfinance Ticker object
            
        Returns:
            Dictionary with price data
            
        Raises:
            APIError: When price data can't be retrieved
        """
        try:
            # Get fast info (contains price data)
            info = ticker_obj.fast_info
            if not info:
                raise APIError(f"No price data found for {ticker}")
                
            # Extract current price
            current_price = info.get('lastPrice', None)
            if current_price is None:
                current_price = info.get('regularMarketPrice', None)
                
            if current_price is None:
                raise APIError(f"No price data available for {ticker}")
                
            # Get price targets
            detailed_info = ticker_obj.info
            target_price = detailed_info.get('targetMeanPrice', None)
            
            # Calculate upside potential
            upside_potential = self.calculate_upside_potential(current_price, target_price)
            
            # Use safe_extract_value for numerical values with proper defaults
            return {
                'ticker': ticker,
                'current_price': current_price,
                'target_price': target_price,
                'upside_potential': upside_potential,
                'price_change': safe_extract_value(info, 'regularMarketDayHigh', 0) - 
                               safe_extract_value(info, 'regularMarketDayLow', 0),
                'price_change_percentage': safe_extract_value(info, 'regularMarketDayChangePercent', 0)
            }
        except Exception as e:
            logger.error(f"Error extracting price data for {ticker}: {str(e)}")
            raise APIError(f"Failed to extract price data: {str(e)}")
    
    def _process_ticker_info(self, ticker: str, ticker_obj: yf.Ticker, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Process ticker information from a yfinance Ticker object.
        
        This method extracts and formats relevant information from the 
        ticker object for consistent presentation.
        
        Args:
            ticker: Stock ticker symbol
            ticker_obj: yfinance Ticker object
            skip_insider_metrics: If True, skip fetching insider trading metrics.
                                  Note: This parameter is passed through from provider implementations
                                  but not used directly in the base class. Provider subclasses
                                  should implement insider metrics handling as needed.
            
        Returns:
            Dict containing processed stock information
            
        Raises:
            APIError: When data can't be retrieved from the ticker object
        """
        try:
            # Get basic price data
            price_data = self._extract_price_data(ticker, ticker_obj)
            
            # Get detailed info (slower)
            detailed_info = ticker_obj.info
            
            # Get price targets and analyst consensus
            analyst_data = self._get_analyst_consensus(ticker_obj, ticker)
            
            # Build result using safe_extract_value for numerical values to avoid errors
            result = {
                'ticker': ticker,
                'name': detailed_info.get('shortName', detailed_info.get('longName', ticker)),
                'sector': detailed_info.get('sector', 'Unknown'),
                'market_cap': safe_extract_value(detailed_info, 'marketCap', 
                              safe_extract_value(ticker_obj.fast_info, 'marketCap')),
                'beta': safe_extract_value(detailed_info, 'beta', None),
                'pe_trailing': safe_extract_value(detailed_info, 'trailingPE', None),
                'pe_forward': safe_extract_value(detailed_info, 'forwardPE', None),
                'dividend_yield': safe_extract_value(detailed_info, 'dividendYield', None),
                'current_price': price_data['current_price'],
                'price_change': price_data['price_change'],
                'price_change_percentage': price_data['price_change_percentage'],
                'target_price': price_data['target_price'],
                'upside_potential': price_data['upside_potential'],
                'analyst_count': analyst_data.get('analyst_count', 0),
                'buy_percentage': analyst_data.get('buy_percentage', None),
                'total_ratings': analyst_data.get('total_ratings', 0),
                'peg_ratio': safe_extract_value(detailed_info, 'pegRatio', None),
                'short_float_pct': safe_extract_value(detailed_info, 'shortPercentOfFloat', None) * 100 
                                 if detailed_info.get('shortPercentOfFloat') else None,
            }
            
            # Add earnings dates if available
            earnings_dates = self._get_earnings_dates_from_obj(ticker_obj)
            if earnings_dates:
                last_date, previous_date = earnings_dates
                result['last_earnings'] = last_date
                result['previous_earnings'] = previous_date
            
            return result
        except Exception as e:
            logger.error(f"Error processing ticker info for {ticker}: {str(e)}")
            raise APIError(f"Failed to process ticker data: {str(e)}")
    
    def _get_analyst_consensus(self, ticker_obj: yf.Ticker, ticker: str = None) -> Dict[str, Any]:
        """
        Extract analyst consensus and price target from a ticker object.
        
        Args:
            ticker_obj: yfinance Ticker object
            ticker: Optional ticker symbol for logging
            
        Returns:
            Dict with analyst consensus data
        """
        # Skip analyst ratings for non-US tickers if ticker is provided
        if ticker and not is_us_ticker(ticker):
            logger.debug(f"Skipping analyst ratings for non-US ticker {ticker}")
            return self._get_empty_analyst_data(ticker)
            
        result = {
            'buy_percentage': None,
            'analyst_count': 0,
            'total_ratings': 0,
            'target_price': None,
            'recommendations': {}
        }
        
        try:
            # Get analyst recommendations
            recommendations = ticker_obj.recommendations
            if recommendations is not None and not recommendations.empty:
                # Get most recent recommendation summary
                recent = recommendations.iloc[-1]
                
                # Extract recommendation counts safely using imported safe_extract_value
                strong_buy = safe_extract_value(recent, 'strongBuy')
                buy = safe_extract_value(recent, 'buy')
                hold = safe_extract_value(recent, 'hold')
                sell = safe_extract_value(recent, 'sell')
                strong_sell = safe_extract_value(recent, 'strongSell')
                
                # Calculate total
                total = strong_buy + buy + hold + sell + strong_sell
                
                if total > 0:
                    buys = strong_buy + buy
                    result['buy_percentage'] = (buys / total) * 100
                    result['total_ratings'] = total
                    
                    # Store individual counts
                    result['recommendations'] = {
                        'strong_buy': strong_buy,
                        'buy': buy,
                        'hold': hold,
                        'sell': sell,
                        'strong_sell': strong_sell
                    }
            
            # Get price targets
            price_targets = ticker_obj.info.get('targetMeanPrice')
            if price_targets:
                result['target_price'] = price_targets
                result['analyst_count'] = ticker_obj.info.get('numberOfAnalystOpinions', 0)
            
            return result
        except Exception as e:
            logger.warning(f"Error getting analyst consensus: {str(e)}")
            return result
        
    # Removed _safe_extract_value method since we're now using the imported utility function
            
    def _get_empty_analyst_data(self, ticker: str = None) -> Dict[str, Any]:
        """
        Get empty analyst data structure for cases where data isn't available.
        
        Args:
            ticker: Optional ticker symbol to include
            
        Returns:
            Dict with empty analyst data
        """
        result = {
            "recommendations": 0,
            "buy_percentage": None,
            "strong_buy": 0,
            "buy": 0,
            "hold": 0,
            "sell": 0,
            "strong_sell": 0,
            "date": None,
            "total_ratings": 0,
            "analyst_count": 0
        }
        
        # Add symbol if ticker is provided
        if ticker:
            result["symbol"] = ticker
            
        return result
            
    def _process_recommendation_key(self, ticker_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process recommendation key as fallback for buy percentage
        
        Args:
            ticker_info: Ticker info dictionary
            
        Returns:
            Dict with buy percentage and total ratings
        """
        result = {
            'buy_percentage': 50,  # Default value
            'total_ratings': ticker_info.get("numberOfAnalystOpinions", 0)
        }
        
        rec_key = ticker_info.get("recommendationKey", "").lower()
        if rec_key == "strong_buy":
            result["buy_percentage"] = 95
        elif rec_key == "buy":
            result["buy_percentage"] = 85
        elif rec_key == "hold":
            result["buy_percentage"] = 65
        elif rec_key == "sell":
            result["buy_percentage"] = 30
        elif rec_key == "strong_sell":
            result["buy_percentage"] = 10
            
        return result
    
    def _get_earnings_dates_from_obj(self, ticker_obj: yf.Ticker) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the last two earnings dates from a ticker object.
        
        Args:
            ticker_obj: yfinance Ticker object
            
        Returns:
            Tuple of (most_recent_date, previous_date) as strings
        """
        try:
            calendar = ticker_obj.calendar
            if calendar is not None and COLUMN_NAMES['EARNINGS_DATE'] in calendar:
                earnings_date = calendar[COLUMN_NAMES['EARNINGS_DATE']]
                if isinstance(earnings_date, pd.Timestamp):
                    return self.format_date(earnings_date), None
                return None, None
                
            # Try earnings history
            earnings = ticker_obj.earnings_dates
            if earnings is not None and not earnings.empty:
                # Get the two most recent dates
                sorted_dates = sorted(earnings.index, reverse=True)
                if len(sorted_dates) >= 2:
                    return (
                        self.format_date(sorted_dates[0]),
                        self.format_date(sorted_dates[1])
                    )
                elif len(sorted_dates) == 1:
                    return self.format_date(sorted_dates[0]), None
                    
            return None, None
        except Exception as e:
            logger.warning(f"Error getting earnings dates: {str(e)}")
            return None, None
            
    def _get_last_earnings_date(self, ticker_obj: yf.Ticker) -> Optional[str]:
        """
        Get the most recent past earnings date.
        
        Args:
            ticker_obj: yfinance Ticker object
            
        Returns:
            str: Formatted earnings date or None if not available
        """
        try:
            # Try calendar approach first - it usually has the most recent past earnings
            calendar = ticker_obj.calendar
            if isinstance(calendar, dict) and COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                earnings_date_list = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                    # Look for the most recent PAST earnings date, not future ones
                    today = pd.Timestamp.now().date()
                    past_earnings = [date for date in earnings_date_list if date < today]
                    
                    if past_earnings:
                        return self.format_date(max(past_earnings))
                        
            # Try earnings_dates approach if we didn't get a past earnings date
            earnings_dates = ticker_obj.earnings_dates if hasattr(ticker_obj, 'earnings_dates') else None
            if earnings_dates is not None and not earnings_dates.empty:
                # Handle timezone-aware dates
                today = pd.Timestamp.now()
                if hasattr(earnings_dates.index, 'tz') and earnings_dates.index.tz is not None:
                    today = pd.Timestamp.now(tz=earnings_dates.index.tz)
                
                # Find past dates for last earnings
                past_dates = [date for date in earnings_dates.index if date < today]
                if past_dates:
                    return self.format_date(max(past_dates))
                    
            return None
        except Exception as e:
            logger.warning(f"Error getting last earnings date: {str(e)}")
            return None
            
    def _handle_ticker_info_error(self, ticker: str, error: Exception) -> Dict[str, Any]:
        """
        Handle errors that occur during ticker info retrieval.
        
        Args:
            ticker: Stock ticker symbol
            error: Exception that occurred
            
        Returns:
            Error information dictionary
        """
        logger.warning(f"Error getting data for {ticker}: {str(error)}")
        return {"symbol": ticker, "error": str(error)}
        
    def _is_us_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker is a US ticker based on suffix.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            bool: True if it's a US ticker, False otherwise
        """
        # Some special cases of US stocks with dots in the ticker
        if ticker in ["BRK.A", "BRK.B", "BF.A", "BF.B"]:
            return True
            
        # Most US tickers don't have a suffix
        if "." not in ticker:
            return True
            
        # Handle .US suffix
        if ticker.endswith(".US"):
            return True
            
        return False
        
    def _handle_retry_logic(self, exception: Exception, attempt: int, ticker: str, retry_msg: str) -> float:
        """
        Handle retry logic for API calls, with consistent pattern.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (0-based)
            ticker: Ticker symbol for logging
            retry_msg: Type of operation being retried (for logging)
            
        Returns:
            float: Delay to wait before next attempt
            
        Raises:
            RateLimitError: If the exception is a rate limit error
            APIError: If this was the last attempt
        """
        if isinstance(exception, RateLimitError) or is_rate_limit_error(exception):
            # Special handling for rate limits - always re-raise
            raise RateLimitError(f"Rate limit exceeded for {ticker}")
    
        if attempt < self.max_retries - 1:
            # Calculate delay with exponential backoff
            delay = self.retry_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} {retry_msg}: {str(exception)}. Retrying in {delay:.2f}s")
            return delay
        else:
            # Last attempt failed - raise an error
            raise APIError(f"Failed to get {retry_msg} for {ticker} after {self.max_retries} attempts: {str(exception)}")

    # Helper function for rate limit detection that uses the imported function
    def _check_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if an error is related to rate limiting.
        Uses the imported is_rate_limit_error utility function.
        
        Args:
            error: Exception to check
            
        Returns:
            bool: True if it's a rate limit error
        """
        return is_rate_limit_error(error)
        
    def _calculate_peg_ratio(self, ticker_info: Dict[str, Any]) -> Optional[float]:
        """
        Calculate PEG ratio from available financial metrics.
        
        Args:
            ticker_info: Ticker info dictionary
            
        Returns:
            float: PEG ratio or None if not available
        """
        # Get the trailingPegRatio directly from Yahoo Finance's API using safe_extract_value
        peg_ratio = safe_extract_value(ticker_info, 'trailingPegRatio', None)
        
        # Format PEG ratio to ensure consistent precision (one decimal place)
        if peg_ratio is not None:
            try:
                # Round to 1 decimal place for consistency
                peg_ratio = round(peg_ratio, 1)
            except (ValueError, TypeError):
                # Keep original value if conversion fails
                pass
                
        return peg_ratio
    
    # Common helper methods used by both sync and async providers
    def _extract_common_ticker_info(self, ticker_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common ticker information from Yahoo Finance data.
        
        Args:
            ticker_info: Raw ticker info dictionary from Yahoo Finance API
            
        Returns:
            Dict with standardized ticker info
        """
        # Extract relevant fields with proper fallbacks using safe_extract_value for numerical fields
        result = {
            "symbol": ticker_info.get("symbol", ""),
            "name": ticker_info.get("longName", ticker_info.get("shortName", "")),
            "sector": ticker_info.get("sector", ""),
            "industry": ticker_info.get("industry", ""),
            "price": safe_extract_value(ticker_info, "currentPrice", 
                     safe_extract_value(ticker_info, "regularMarketPrice")),
            "currency": ticker_info.get("currency", "USD"),
            "market_cap": safe_extract_value(ticker_info, "marketCap"),
            "pe_ratio": safe_extract_value(ticker_info, "trailingPE"),
            "forward_pe": safe_extract_value(ticker_info, "forwardPE"),
            "peg_ratio": safe_extract_value(ticker_info, "pegRatio"),
            "beta": safe_extract_value(ticker_info, "beta"),
            "fifty_day_avg": safe_extract_value(ticker_info, "fiftyDayAverage"),
            "two_hundred_day_avg": safe_extract_value(ticker_info, "twoHundredDayAverage"),
            "fifty_two_week_high": safe_extract_value(ticker_info, "fiftyTwoWeekHigh"),
            "fifty_two_week_low": safe_extract_value(ticker_info, "fiftyTwoWeekLow"),
            "target_price": safe_extract_value(ticker_info, "targetMeanPrice"),
            "dividend_yield": safe_extract_value(ticker_info, "dividendYield", 0) * 100 
                             if ticker_info.get("dividendYield") else None,
            "short_percent": safe_extract_value(ticker_info, "shortPercentOfFloat", 0) * 100 
                           if ticker_info.get("shortPercentOfFloat") else None,
            "country": ticker_info.get("country", ""),
        }
        
        # Add market cap formatted
        result["market_cap_fmt"] = self._format_market_cap(result["market_cap"])
        
        # Calculate upside potential if possible
        price = result.get("price")
        target = result.get("target_price")
        result["upside"] = self._calculate_upside_potential(price, target)
        
        return result
    
    def _process_error_for_batch(self, ticker: str, error: Exception) -> Dict[str, Any]:
        """
        Process an error for batch operations, providing a standardized error response.
        
        Args:
            ticker: The ticker symbol that caused the error
            error: The exception that occurred
            
        Returns:
            Dict with error information
        """
        logger.warning(f"Error getting data for {ticker}: {str(error)}")
        return {
            "symbol": ticker,
            "error": str(error),
            "ticker": ticker,
            "name": ticker
        }
        
    def _extract_historical_data(self, ticker: str, ticker_obj: yf.Ticker, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Extract historical price data from ticker object.
        This is a common implementation that can be used by both sync and async providers.
        
        Args:
            ticker: Stock ticker symbol
            ticker_obj: yfinance Ticker object
            period: Time period to fetch
            interval: Data interval
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            APIError: When historical data can't be retrieved
        """
        try:
            # Validate ticker
            validate_ticker(ticker)
            
            # Get history data
            history = ticker_obj.history(period=period, interval=interval)
            
            # Process the result
            if history.empty:
                logger.warning(f"No historical data found for {ticker}")
                return pd.DataFrame()
                
            # Return the data
            return history
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            if "Rate limit" in str(e) or "429" in str(e):
                raise RateLimitError(f"Rate limit exceeded while fetching historical data for {ticker}")
            raise APIError(f"Failed to get historical data: {str(e)}")
            
    def _get_price_data_from_obj(self, ticker: str, ticker_obj: yf.Ticker) -> Dict[str, Any]:
        """
        Get price data using a ticker object.
        
        Args:
            ticker: Stock ticker symbol
            ticker_obj: yfinance Ticker object
            
        Returns:
            Dictionary with price data
            
        Raises:
            APIError: When price data can't be retrieved
        """
        return self._extract_price_data(ticker, ticker_obj)