"""
Base functionality for Yahoo Finance providers.

This module provides common functionality that is shared between synchronous and 
asynchronous Yahoo Finance provider implementations to reduce code duplication.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import yfinance as yf
from datetime import datetime

from ...core.errors import YFinanceError, APIError, ValidationError, RateLimitError
from ...utils.market.ticker_utils import validate_ticker, is_us_ticker
from ...core.config import COLUMN_NAMES

logger = logging.getLogger(__name__)

class YahooFinanceBaseProvider:
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
        self._ticker_cache = {}

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
        if not market_cap:
            return None
            
        # Convert to billions or trillions
        if market_cap >= 1_000_000_000_000:  # Trillion
            value = market_cap / 1_000_000_000_000
            if value >= 10:
                return f"{value:.1f}T"
            else:
                return f"{value:.2f}T"
        elif market_cap >= 1_000_000_000:  # Billion
            value = market_cap / 1_000_000_000
            if value >= 100:
                return f"{value:.0f}B"
            elif value >= 10:
                return f"{value:.1f}B"
            else:
                return f"{value:.2f}B"
        elif market_cap >= 1_000_000:  # Million
            value = market_cap / 1_000_000
            return f"{value:.2f}M"
        else:
            return f"{market_cap:,.0f}"
            
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
            # Get basic info
            info = ticker_obj.fast_info
            if not info:
                raise APIError(f"No info data found for {ticker}")
                
            # Extract current price
            current_price = info.get('lastPrice', None)
            if current_price is None:
                current_price = info.get('regularMarketPrice', None)
                
            # Get detailed info (slower)
            detailed_info = ticker_obj.info
            
            # Get price targets and analyst consensus
            analyst_data = self._get_analyst_consensus(ticker_obj)
            
            # Calculate upside
            target_price = analyst_data.get('target_price')
            upside = self.calculate_upside_potential(current_price, target_price)
            
            # Build result
            result = {
                'ticker': ticker,
                'name': detailed_info.get('shortName', detailed_info.get('longName', ticker)),
                'sector': detailed_info.get('sector', 'Unknown'),
                'market_cap': detailed_info.get('marketCap', info.get('marketCap')),
                'beta': detailed_info.get('beta', None),
                'pe_trailing': detailed_info.get('trailingPE', None),
                'pe_forward': detailed_info.get('forwardPE', None),
                'dividend_yield': detailed_info.get('dividendYield', None),
                'current_price': current_price,
                'price_change': info.get('regularMarketDayHigh', 0) - info.get('regularMarketDayLow', 0),
                'price_change_percentage': info.get('regularMarketDayChangePercent', 0),
                'target_price': target_price,
                'upside_potential': upside,
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