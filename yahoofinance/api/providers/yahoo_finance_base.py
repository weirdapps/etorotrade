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
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'target_price': target_price,
                'upside_potential': upside_potential,
                'price_change': info.get('regularMarketDayHigh', 0) - info.get('regularMarketDayLow', 0),
                'price_change_percentage': info.get('regularMarketDayChangePercent', 0)
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
            analyst_data = self._get_analyst_consensus(ticker_obj)
            
            # Build result
            result = {
                'ticker': ticker,
                'name': detailed_info.get('shortName', detailed_info.get('longName', ticker)),
                'sector': detailed_info.get('sector', 'Unknown'),
                'market_cap': detailed_info.get('marketCap', ticker_obj.fast_info.get('marketCap')),
                'beta': detailed_info.get('beta', None),
                'pe_trailing': detailed_info.get('trailingPE', None),
                'pe_forward': detailed_info.get('forwardPE', None),
                'dividend_yield': detailed_info.get('dividendYield', None),
                'current_price': price_data['current_price'],
                'price_change': price_data['price_change'],
                'price_change_percentage': price_data['price_change_percentage'],
                'target_price': price_data['target_price'],
                'upside_potential': price_data['upside_potential'],
                'analyst_count': analyst_data.get('analyst_count', 0),
                'buy_percentage': analyst_data.get('buy_percentage', None),
                'total_ratings': analyst_data.get('total_ratings', 0),
                'peg_ratio': detailed_info.get('pegRatio', None),
                'short_float_pct': detailed_info.get('shortPercentOfFloat', None) * 100 if detailed_info.get('shortPercentOfFloat') else None,
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
    
    def _get_analyst_consensus(self, ticker_obj: yf.Ticker) -> Dict[str, Any]:
        """
        Extract analyst consensus and price target from a ticker object.
        
        Args:
            ticker_obj: yfinance Ticker object
            
        Returns:
            Dict with analyst consensus data
        """
        result = {
            'buy_percentage': None,
            'analyst_count': 0,
            'total_ratings': 0,
            'target_price': None
        }
        
        try:
            # Get analyst recommendations
            recommendations = ticker_obj.recommendations
            if recommendations is not None and not recommendations.empty:
                # Get most recent recommendation summary
                recent = recommendations.iloc[-1]
                
                # Calculate buy percentage
                total = sum([
                    recent.get('strongBuy', 0),
                    recent.get('buy', 0),
                    recent.get('hold', 0),
                    recent.get('sell', 0),
                    recent.get('strongSell', 0)
                ])
                
                if total > 0:
                    buys = recent.get('strongBuy', 0) + recent.get('buy', 0)
                    result['buy_percentage'] = (buys / total) * 100
                    result['total_ratings'] = total
            
            # Get price targets
            price_targets = ticker_obj.info.get('targetMeanPrice')
            if price_targets:
                result['target_price'] = price_targets
                result['analyst_count'] = ticker_obj.info.get('numberOfAnalystOpinions', 0)
            
            return result
        except Exception as e:
            logger.warning(f"Error getting analyst consensus: {str(e)}")
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
        
    def _calculate_peg_ratio(self, ticker_info: Dict[str, Any]) -> Optional[float]:
        """
        Calculate PEG ratio from available financial metrics.
        
        Args:
            ticker_info: Ticker info dictionary
            
        Returns:
            float: PEG ratio or None if not available
        """
        # Get the trailingPegRatio directly from Yahoo Finance's API
        peg_ratio = ticker_info.get('trailingPegRatio')
        
        # Format PEG ratio to ensure consistent precision (one decimal place)
        if peg_ratio is not None:
            try:
                # Round to 1 decimal place for consistency
                peg_ratio = round(float(peg_ratio), 1)
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
        # Extract relevant fields with proper fallbacks
        result = {
            "symbol": ticker_info.get("symbol", ""),
            "name": ticker_info.get("longName", ticker_info.get("shortName", "")),
            "sector": ticker_info.get("sector", ""),
            "industry": ticker_info.get("industry", ""),
            "price": ticker_info.get("currentPrice", ticker_info.get("regularMarketPrice")),
            "currency": ticker_info.get("currency", "USD"),
            "market_cap": ticker_info.get("marketCap"),
            "pe_ratio": ticker_info.get("trailingPE"),
            "forward_pe": ticker_info.get("forwardPE"),
            "peg_ratio": ticker_info.get("pegRatio"),
            "beta": ticker_info.get("beta"),
            "fifty_day_avg": ticker_info.get("fiftyDayAverage"),
            "two_hundred_day_avg": ticker_info.get("twoHundredDayAverage"),
            "fifty_two_week_high": ticker_info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": ticker_info.get("fiftyTwoWeekLow"),
            "target_price": ticker_info.get("targetMeanPrice"),
            "dividend_yield": ticker_info.get("dividendYield", 0) * 100 if ticker_info.get("dividendYield") else None,
            "short_percent": ticker_info.get("shortPercentOfFloat", 0) * 100 if ticker_info.get("shortPercentOfFloat") else None,
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