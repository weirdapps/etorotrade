"""
Base Yahoo Finance provider implementation.

This module provides a base implementation for Yahoo Finance providers,
with common functionality shared by different provider types.
"""

from ...core.logging_config import get_logger
import logging
import time
import re
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List, Tuple, Union, cast
import pandas as pd
import yfinance as yf
import numpy as np
from abc import ABC, abstractmethod

from ...core.errors import YFinanceError, ValidationError, APIError, DataError, ResourceNotFoundError
from ...utils.market.ticker_utils import validate_ticker, is_us_ticker
from ...utils.error_handling import with_retry, translate_error

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
        
        logger.debug(f"Initialized YahooFinanceBaseProvider with max_retries={max_retries}, retry_delay={retry_delay}")
    
    def _handle_retry_logic(self, error: Exception, attempt: int, ticker: str, operation: str) -> float:
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
        delay = self.retry_delay * (2 ** attempt)
        
        if attempt < self.max_retries - 1:
            logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} {operation}: {str(error)}. Retrying in {delay:.2f}s")
            return delay
        else:
            logger.error(f"All {self.max_retries} attempts failed for {ticker} {operation}: {str(error)}")
            raise YFinanceError(f"Failed to get {operation} for {ticker} after {self.max_retries} attempts", 
                               details={"ticker": ticker, "operation": operation, "attempts": self.max_retries})
    
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
                
        if isinstance(dt, (datetime, pd.Timestamp)):
            return dt.strftime('%Y-%m-%d')
        elif isinstance(dt, date):
            return dt.strftime('%Y-%m-%d')
            
        # Return string representation as a fallback
        return str(dt)
    
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
        def safe_extract_value(data, key, default=None):
            try:
                if isinstance(data, dict) and key in data and data[key] is not None:
                    return data[key]
            except (TypeError, AttributeError, KeyError) as e:
                logger.debug(f"Error extracting {key}: {str(e)}")
            return default
        
        # Get ticker symbol for logging
        symbol = safe_extract_value(info, 'symbol', safe_extract_value(info, 'ticker', 'unknown'))
        
        try:
            # Get name safely (multiple fallbacks)
            name = safe_extract_value(info, 'shortName', safe_extract_value(info, 'longName', symbol))
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
                "price": safe_extract_value(info, 'regularMarketPrice'),
                "current_price": safe_extract_value(info, 'regularMarketPrice'),  # Match both formats
                "change": safe_extract_value(info, 'regularMarketChange'),
                "change_percent": safe_extract_value(info, 'regularMarketChangePercent'),
                "market_cap": safe_extract_value(info, 'marketCap'),
                "volume": safe_extract_value(info, 'regularMarketVolume'),
                "avg_volume": safe_extract_value(info, 'averageVolume'),
                "pe_trailing": safe_extract_value(info, 'trailingPE'),  # Match both formats
                "pe_ratio": safe_extract_value(info, 'trailingPE'),
                "pe_forward": safe_extract_value(info, 'forwardPE'),  # Match both formats
                "forward_pe": safe_extract_value(info, 'forwardPE'),
                "dividend_yield": safe_extract_value(info, 'dividendYield'),
                "target_price": safe_extract_value(info, 'targetMeanPrice'),
                "beta": safe_extract_value(info, 'beta'),
                "eps": safe_extract_value(info, 'trailingEps'),
                "forward_eps": safe_extract_value(info, 'forwardEps'),
                "peg_ratio": safe_extract_value(info, 'pegRatio'),
                "sector": safe_extract_value(info, 'sector'),
                "industry": safe_extract_value(info, 'industry'),
                "fifty_two_week_high": safe_extract_value(info, 'fiftyTwoWeekHigh'),
                "fifty_two_week_low": safe_extract_value(info, 'fiftyTwoWeekLow'),
                "fifty_day_avg": safe_extract_value(info, 'fiftyDayAverage'),
                "two_hundred_day_avg": safe_extract_value(info, 'twoHundredDayAverage'),
                "exchange": safe_extract_value(info, 'exchange'),
                "country": safe_extract_value(info, 'country'),
                "data_source": "yfinance",
            }
            
            # Convert dividend yield to percentage if available
            if result["dividend_yield"] is not None:
                try:
                    result["dividend_yield"] = float(result["dividend_yield"]) * 100
                except (ValueError, TypeError):
                    pass
            
            # Add calculated fields
            result['upside'] = self._calculate_upside_potential(result['price'], result['target_price'])
            # Also add original field name for compatibility
            result['upside_potential'] = result['upside']
                
            return result
            
        except Exception as e:
            # Provide minimal info in case of unexpected error
            logger.warning(f"Error extracting info for {symbol}: {str(e)}. Returning minimal info.")
            return {
                "symbol": symbol,
                "name": symbol.upper(),
                "company": symbol.upper(),
                "data_source": "yfinance",
                "error": f"Error extracting data: {str(e)}"
            }
        
    def _calculate_upside_potential(self, current_price, target_price):
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
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
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
        
        # Return cached ticker object if available
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]
        
        # Create new ticker object
        try:
            ticker_obj = yf.Ticker(ticker)
            self._ticker_cache[ticker] = ticker_obj
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
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        
        if period not in valid_periods:
            raise ValidationError(f"Invalid period: {period}. Valid periods are: {', '.join(valid_periods)}")
            
        if interval not in valid_intervals:
            raise ValidationError(f"Invalid interval: {interval}. Valid intervals are: {', '.join(valid_intervals)}")
        
        # Get historical data
        try:
            history = ticker_obj.history(period=period, interval=interval)
            
            # Check if data is empty
            if history.empty:
                logger.warning(f"No historical data available for {ticker} with period={period}, interval={interval}")
                return pd.DataFrame()
                
            # Reset index to have Date as a column
            history = history.reset_index()
            
            # Rename columns to match our schema
            history.columns = [
                c if c != 'Date' else 'date' for c in history.columns
            ]
            
            # Format date column
            if 'date' in history.columns:
                history['date'] = history['date'].apply(self._format_date)
                
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
                'ticker': ticker,
                'price': safe_extract_value(info, 'regularMarketPrice'),
                'previous_close': safe_extract_value(info, 'regularMarketPreviousClose'),
                'open': safe_extract_value(info, 'regularMarketOpen'),
                'day_high': safe_extract_value(info, 'regularMarketDayHigh'),
                'day_low': safe_extract_value(info, 'regularMarketDayLow'),
                'volume': safe_extract_value(info, 'regularMarketVolume'),
                'price_change': safe_extract_value(info, 'regularMarketDayHigh', 0) - 
                               safe_extract_value(info, 'regularMarketDayLow', 0),
                'price_change_percentage': safe_extract_value(info, 'regularMarketDayChangePercent', 0)
            }
        except YFinanceError as e:
            logger.error(f"Error extracting price data for {ticker}: {str(e)}")
            raise e
    
    def _process_ticker_info(self, ticker: str, ticker_obj: yf.Ticker, skip_insider_metrics: bool = False) -> Dict[str, Any]:
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
                        total_buys = sum(1 for _, row in insider_data.iterrows() if row.get('Shares', 0) > 0)
                        total_sells = sum(1 for _, row in insider_data.iterrows() if row.get('Shares', 0) < 0)
                        
                        result["insider_transactions"] = len(insider_data)
                        result["insider_buys"] = total_buys
                        result["insider_sells"] = total_sells
                        result["insider_ratio"] = total_buys / (total_buys + total_sells) if (total_buys + total_sells) > 0 else 0
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
        try:
            recommendations = ticker_obj.recommendations
            
            # Default values
            result = {
                "total_ratings": 0,
                "buy_percentage": 0.0,
                "recommendations": None
            }
            
            # Check if we have data
            if recommendations is None or recommendations.empty:
                logger.debug(f"No analyst ratings available for {ticker}")
                return result
                
            # Get the most recent date
            latest_date = recommendations.index.max()
            
            # Filter for the most recent recommendations
            latest_recommendations = recommendations[recommendations.index == latest_date]
            
            # Count the recommendations by grade
            recommendation_counts = {}
            
            for col in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']:
                if col in latest_recommendations.columns:
                    value = latest_recommendations[col].sum()
                    if not pd.isna(value) and value > 0:
                        recommendation_counts[col.lower()] = int(value)
            
            total_ratings = sum(recommendation_counts.values())
            
            # Calculate buy percentage
            buy_ratings = recommendation_counts.get('strongbuy', 0) + recommendation_counts.get('buy', 0)
            buy_percentage = (buy_ratings / total_ratings * 100) if total_ratings > 0 else 0
            
            return {
                "total_ratings": total_ratings,
                "buy_percentage": buy_percentage,
                "recommendations": recommendation_counts
            }
        except YFinanceError as e:
            logger.warning(f"Error getting analyst consensus for {ticker}: {str(e)}")
            raise e
    
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
            "strong_sell": 0
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
            "data_source": "error"
        }