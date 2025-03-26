"""
Yahoo Finance provider implementation.

This module implements the FinanceDataProvider interface for Yahoo Finance data.
It provides a consistent API for retrieving financial information with 
appropriate rate limiting, caching, and error handling.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, cast
import pandas as pd
import yfinance as yf
from functools import lru_cache

from .base_provider import FinanceDataProvider
from ...core.errors import YFinanceError, APIError, ValidationError, RateLimitError
from ...utils.market.ticker_utils import validate_ticker, is_us_ticker
from ...utils.network.rate_limiter import rate_limited
from ...core.config import CACHE_CONFIG

logger = logging.getLogger(__name__)

class YahooFinanceProvider(FinanceDataProvider):
    """
    Yahoo Finance data provider implementation.
    
    This provider wraps the yfinance library with proper rate limiting,
    error handling, and caching to provide reliable access to financial data.
    
    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Yahoo Finance provider.
        
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
            
        if isinstance(date_obj, str):
            try:
                # Try to parse as ISO format
                from datetime import datetime
                date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                # Return as is if parsing fails
                return date_obj
                
        try:
            return date_obj.strftime('%Y-%m-%d')
        except (AttributeError, TypeError):
            return str(date_obj)
    
    def format_market_cap(self, market_cap):
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
        logger.debug(f"Getting ticker info for {ticker}")
        ticker_obj = self._get_ticker_object(ticker)
        
        # Basic information with proper rate limiting
        result = {}
        
        for attempt in range(self.max_retries):
            try:
                # Get basic info
                info = ticker_obj.info
                if not info:
                    raise APIError(f"Failed to retrieve info for {ticker}")
                
                # Extract key metrics
                result = {
                    "symbol": ticker,
                    "name": info.get("longName", info.get("shortName", "")),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "price": info.get("currentPrice", info.get("regularMarketPrice")),
                    "currency": info.get("currency", "USD"),
                    "market_cap": info.get("marketCap"),
                    "market_cap_fmt": self.format_market_cap(info.get("marketCap")),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "beta": info.get("beta"),
                    "fifty_day_avg": info.get("fiftyDayAverage"),
                    "two_hundred_day_avg": info.get("twoHundredDayAverage"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "target_price": info.get("targetMeanPrice"),
                    "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
                    "short_percent": info.get("shortPercentOfFloat", 0) * 100 if info.get("shortPercentOfFloat") else None,
                    "country": info.get("country", ""),
                }
                
                # Calculate upside potential if possible
                price = result.get("price")
                target = result.get("target_price")
                result["upside"] = self.calculate_upside_potential(price, target)
                
                # Additional metrics for US stocks
                if is_us_ticker(ticker) and not skip_insider_metrics:
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
                            result["insider_ratio"] = total_buys / (total_buys + total_sells) if (total_buys + total_sells) > 0 else 0
                    except Exception as e:
                        logger.warning(f"Failed to get insider data for {ticker}: {str(e)}")
                
                break
            except RateLimitError:
                # Specific handling for rate limits - just re-raise
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker}: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise APIError(f"Failed to get ticker info for {ticker} after {self.max_retries} attempts: {str(e)}")
        
        return result
    
    @rate_limited
    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
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
        ticker_obj = self._get_ticker_object(ticker)
        
        for attempt in range(self.max_retries):
            try:
                history = ticker_obj.history(period=period, interval=interval)
                if history.empty:
                    raise APIError(f"No historical data returned for {ticker}")
                return history
            except RateLimitError:
                # Specific handling for rate limits - just re-raise
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} historical data: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise APIError(f"Failed to get historical data for {ticker} after {self.max_retries} attempts: {str(e)}")
    
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
        ticker_obj = self._get_ticker_object(ticker)
        
        for attempt in range(self.max_retries):
            try:
                # Get earnings data
                calendar = ticker_obj.calendar
                
                # Handle cases where calendar might be None or not have earnings date
                if calendar is None or 'Earnings Date' not in calendar:
                    logger.debug(f"No earnings dates found for {ticker}")
                    return None, None
                    
                earnings_date = calendar['Earnings Date']
                
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
                # Specific handling for rate limits - just re-raise
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} earnings dates: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise APIError(f"Failed to get earnings dates for {ticker} after {self.max_retries} attempts: {str(e)}")
    
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
        ticker_obj = self._get_ticker_object(ticker)
        
        # Skip analyst ratings for non-US tickers
        if not is_us_ticker(ticker):
            logger.debug(f"Skipping analyst ratings for non-US ticker {ticker}")
            return {
                "symbol": ticker,
                "recommendations": 0,
                "buy_percentage": None,
                "strong_buy": 0,
                "buy": 0,
                "hold": 0,
                "sell": 0,
                "strong_sell": 0,
                "date": None
            }
        
        for attempt in range(self.max_retries):
            try:
                # Get recommendations
                recommendations = ticker_obj.recommendations
                
                # Handle cases where recommendations might be None or empty
                if recommendations is None or recommendations.empty:
                    logger.debug(f"No analyst ratings found for {ticker}")
                    return {
                        "symbol": ticker,
                        "recommendations": 0,
                        "buy_percentage": None,
                        "strong_buy": 0,
                        "buy": 0,
                        "hold": 0,
                        "sell": 0,
                        "strong_sell": 0,
                        "date": None
                    }
                
                # Get the most recent recommendations
                latest_date = recommendations.index.max()
                latest_recs = recommendations.loc[latest_date]
                
                # Calculate buy percentage
                total_recs = latest_recs.sum()
                strong_buy = latest_recs.get('strongBuy', 0)
                buy = latest_recs.get('buy', 0)
                hold = latest_recs.get('hold', 0)
                sell = latest_recs.get('sell', 0)
                strong_sell = latest_recs.get('strongSell', 0)
                
                buy_percentage = ((strong_buy + buy) / total_recs * 100) if total_recs > 0 else 0
                
                return {
                    "symbol": ticker,
                    "recommendations": total_recs,
                    "buy_percentage": buy_percentage,
                    "strong_buy": strong_buy,
                    "buy": buy,
                    "hold": hold,
                    "sell": sell,
                    "strong_sell": strong_sell,
                    "date": self.format_date(latest_date)
                }
                
            except RateLimitError:
                # Specific handling for rate limits - just re-raise
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} analyst ratings: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise APIError(f"Failed to get analyst ratings for {ticker} after {self.max_retries} attempts: {str(e)}")
    
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
                        "date": self.format_date(row.get("Date Reported", None)),
                        "value": row.get("Value", 0),
                        "pct_out": row.get("% Out", 0) * 100 if row.get("% Out") else 0,
                    }
                    result.append(transaction)
                
                return result
                
            except RateLimitError:
                # Specific handling for rate limits - just re-raise
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {ticker} insider transactions: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise APIError(f"Failed to get insider transactions for {ticker} after {self.max_retries} attempts: {str(e)}")
    
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
                if not search_results or 'quotes' not in search_results or not search_results['quotes']:
                    logger.debug(f"No search results found for query '{query}'")
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
                
            except RateLimitError:
                # Specific handling for rate limits - just re-raise
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for search query '{query}': {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise APIError(f"Failed to search tickers for '{query}' after {self.max_retries} attempts: {str(e)}")
    
    def batch_get_ticker_info(self, tickers: List[str], skip_insider_metrics: bool = False) -> Dict[str, Dict[str, Any]]:
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
        
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_ticker_info(ticker, skip_insider_metrics)
            except Exception as e:
                logger.warning(f"Error getting data for {ticker}: {str(e)}")
                results[ticker] = {"symbol": ticker, "error": str(e)}
        
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
            "ticker_cache_keys": list(self._ticker_cache.keys())
        }