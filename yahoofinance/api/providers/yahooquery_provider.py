"""
Yahoo Query provider implementation.

This module implements the FinanceDataProvider interface using the yahooquery library.
It provides a consistent API for retrieving financial information with
appropriate rate limiting, caching, and error handling.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, cast
import pandas as pd
from yahooquery import Ticker

from .base_provider import FinanceDataProvider
from .yahoo_finance_base import YahooFinanceBaseProvider
from ...core.errors import YFinanceError, APIError, ValidationError, RateLimitError
from ...utils.market.ticker_utils import is_us_ticker
from ...utils.network.rate_limiter import rate_limited
from ...core.config import CACHE_CONFIG, COLUMN_NAMES, POSITIVE_GRADES

logger = logging.getLogger(__name__)

class YahooQueryProvider(YahooFinanceBaseProvider, FinanceDataProvider):
    """
    Yahoo Query data provider implementation.
    
    This provider wraps the yahooquery library with proper rate limiting,
    error handling, and caching to provide reliable access to financial data.
    
    Attributes:
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
        _ticker_cache: Cache of ticker information to avoid repeated fetches
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Yahoo Query provider.
        
        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries
        """
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        
    def _handle_delay(self, delay: float):
        """
        Handle delaying execution for retry logic using synchronous time.sleep().
        
        Args:
            delay: Time in seconds to delay
        """
        time.sleep(delay)
    
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
        try:
            logger.debug(f"Fetching ticker info for {ticker}")
            start_time = time.time()
            
            # Create a yahooquery Ticker instance
            yq_ticker = Ticker(ticker)
            
            # Get basic summary information
            summary_data = yq_ticker.summary_detail.get(ticker, {})
            price_data = yq_ticker.price.get(ticker, {})
            quote_type = yq_ticker.quote_type.get(ticker, {})
            financial_data = yq_ticker.financial_data.get(ticker, {})
            key_stats = yq_ticker.key_stats.get(ticker, {})
            
            # Check if any data was returned - if all are empty, raise error
            if not any([summary_data, price_data, quote_type]):
                raise APIError(f"No data returned for ticker {ticker}")
            
            # Get company name
            company_name = price_data.get('shortName') or quote_type.get('shortName')
            
            # Create base info dict
            info = {
                'ticker': ticker,
                'company_name': company_name,
                'sector': price_data.get('sector', ''),
                'industry': price_data.get('industry', ''),
                'price': price_data.get('regularMarketPrice', None),
                'target_price': financial_data.get('targetMeanPrice', None),
                'market_cap': price_data.get('marketCap', None),
                'beta': key_stats.get('beta', None),
                'pe_trailing': key_stats.get('trailingPE', None),
                'pe_forward': key_stats.get('forwardPE', None),
                'peg_ratio': key_stats.get('pegRatio', None),
                'short_percent': key_stats.get('shortPercentOfFloat', None),
                'dividend_yield': summary_data.get('dividendYield', None) * 100 if summary_data.get('dividendYield') else None,
                'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'from_cache': False,
            }
            
            # Get analyst ratings and recommendations
            try:
                recommendations = yq_ticker.recommendation_trend.get(ticker, {})
                if recommendations and isinstance(recommendations, List) and recommendations:
                    latest_rec = recommendations[0]  # Most recent period's data
                    
                    # Calculate buy percentage
                    buy_count = latest_rec.get('strongBuy', 0) + latest_rec.get('buy', 0)
                    total_ratings = (buy_count + latest_rec.get('hold', 0) + 
                                    latest_rec.get('sell', 0) + latest_rec.get('strongSell', 0))
                    
                    buy_percentage = (buy_count / total_ratings * 100) if total_ratings > 0 else None
                    
                    info.update({
                        'buy_percentage': buy_percentage,
                        'total_ratings': total_ratings,
                        'analyst_count': total_ratings,  # Use total ratings as analyst count
                    })
            except Exception as e:
                logger.debug(f"Error fetching recommendations for {ticker}: {str(e)}")
                # Set defaults for these fields
                info.update({
                    'buy_percentage': None,
                    'total_ratings': 0,
                    'analyst_count': 0,
                })
            
            # Get earnings dates if available
            try:
                calendar = yq_ticker.calendar_events.get(ticker, {})
                earnings_dates = calendar.get('earnings_dates', [])
                if earnings_dates and isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                    most_recent = earnings_dates[0].get('earnings_date')
                    if most_recent:
                        info['earnings_date'] = pd.Timestamp(most_recent).strftime('%Y-%m-%d')
            except Exception as e:
                logger.debug(f"Error fetching earnings dates for {ticker}: {str(e)}")
            
            # Get insider transactions unless skipped
            if not skip_insider_metrics:
                try:
                    insider_data = yq_ticker.insider_transactions.get(ticker, [])
                    if isinstance(insider_data, list) and insider_data:
                        # Process insider transactions
                        info['insider_transactions'] = self._process_insider_transactions(insider_data)
                except Exception as e:
                    logger.debug(f"Error fetching insider data for {ticker}: {str(e)}")
                    info['insider_transactions'] = []
            
            # Add a processing time field
            processing_time = time.time() - start_time
            info['processing_time'] = processing_time
            
            # Calculate upside if price and target are available
            if info.get('price') and info.get('target_price'):
                try:
                    upside = ((info['target_price'] / info['price']) - 1) * 100
                    info['upside'] = upside
                except (TypeError, ZeroDivisionError):
                    info['upside'] = None
            
            return info
        
        except Exception as e:
            # Check for rate limit related errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded for ticker {ticker}: {str(e)}")
            
            # Check for ticker not found errors
            if "no data found" in str(e).lower() or "not found" in str(e).lower():
                raise ValidationError(f"Ticker {ticker} not found or invalid")
            
            # Generic error handler
            raise YFinanceError(f"Error fetching data for ticker {ticker}: {str(e)}")
    
    def _process_insider_transactions(self, insider_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process insider transactions data from yahooquery format.
        
        Args:
            insider_data: List of insider transactions from yahooquery
            
        Returns:
            List of processed insider transaction dicts
        """
        processed_transactions = []
        
        for transaction in insider_data:
            # Convert to our standardized format
            processed_transaction = {
                'name': transaction.get('filerName', ''),
                'title': transaction.get('filerRelation', ''),
                'date': pd.Timestamp(transaction.get('startDate', '')).strftime('%Y-%m-%d') if transaction.get('startDate') else '',
                'transaction': transaction.get('transactionDescription', ''),
                'shares': transaction.get('shares', 0),
                'value': transaction.get('value', 0),
            }
            processed_transactions.append(processed_transaction)
        
        return processed_transactions
    
    @rate_limited
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing price data
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        try:
            logger.debug(f"Fetching price data for {ticker}")
            
            # Create a yahooquery Ticker instance
            yq_ticker = Ticker(ticker)
            
            # Get price data
            price_data = yq_ticker.price.get(ticker, {})
            financial_data = yq_ticker.financial_data.get(ticker, {})
            
            if not price_data:
                raise APIError(f"No price data returned for ticker {ticker}")
            
            # Create price info dict
            price_info = {
                'ticker': ticker,
                'price': price_data.get('regularMarketPrice', None),
                'change': price_data.get('regularMarketChange', None),
                'change_percent': price_data.get('regularMarketChangePercent', None),
                'volume': price_data.get('regularMarketVolume', None),
                'target_price': financial_data.get('targetMeanPrice', None),
                'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            # Calculate upside if price and target are available
            if price_info.get('price') and price_info.get('target_price'):
                try:
                    upside = ((price_info['target_price'] / price_info['price']) - 1) * 100
                    price_info['upside'] = upside
                except (TypeError, ZeroDivisionError):
                    price_info['upside'] = None
            
            return price_info
        
        except Exception as e:
            # Check for rate limit related errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded for ticker {ticker}: {str(e)}")
            
            # Generic error handler
            raise YFinanceError(f"Error fetching price data for {ticker}: {str(e)}")
    
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
        try:
            logger.debug(f"Fetching historical data for {ticker}, period={period}, interval={interval}")
            
            # Create a yahooquery Ticker instance
            yq_ticker = Ticker(ticker)
            
            # Map period and interval to yahooquery format if needed
            yq_period = period
            yq_interval = interval
            
            # Get historical data
            hist_data = yq_ticker.history(period=yq_period, interval=yq_interval)
            
            # If hist_data is a multi-index DataFrame with ticker as the first level, get the data for this ticker
            if isinstance(hist_data.index, pd.MultiIndex) and ticker in hist_data.index.levels[0]:
                hist_data = hist_data.loc[ticker]
            
            if hist_data.empty:
                logger.warning(f"No historical data returned for ticker {ticker}")
                return pd.DataFrame()
            
            # Rename columns to match our standard format
            column_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjclose': 'Adj Close',
            }
            hist_data = hist_data.rename(columns={k: v for k, v in column_map.items() if k in hist_data.columns})
            
            return hist_data
        
        except Exception as e:
            # Check for rate limit related errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded for ticker {ticker}: {str(e)}")
            
            # Generic error handler
            raise YFinanceError(f"Error fetching historical data for {ticker}: {str(e)}")
    
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
        try:
            logger.debug(f"Fetching earnings dates for {ticker}")
            
            # Create a yahooquery Ticker instance
            yq_ticker = Ticker(ticker)
            
            # Get calendar events
            calendar = yq_ticker.calendar_events.get(ticker, {})
            earnings_dates = calendar.get('earnings_dates', [])
            
            if not earnings_dates or not isinstance(earnings_dates, list):
                logger.debug(f"No earnings dates found for {ticker}")
                return None, None
            
            # Extract dates
            most_recent_date = None
            previous_date = None
            
            if len(earnings_dates) > 0:
                most_recent = earnings_dates[0].get('earnings_date')
                if most_recent:
                    most_recent_date = pd.Timestamp(most_recent).strftime('%Y-%m-%d')
            
            if len(earnings_dates) > 1:
                previous = earnings_dates[1].get('earnings_date')
                if previous:
                    previous_date = pd.Timestamp(previous).strftime('%Y-%m-%d')
            
            return most_recent_date, previous_date
        
        except Exception as e:
            # Check for rate limit related errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded for ticker {ticker}: {str(e)}")
            
            # Generic error handler
            raise YFinanceError(f"Error fetching earnings dates for {ticker}: {str(e)}")
    
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
        try:
            logger.debug(f"Fetching analyst ratings for {ticker}")
            
            # Create a yahooquery Ticker instance
            yq_ticker = Ticker(ticker)
            
            # Get recommendations
            recommendations = yq_ticker.recommendation_trend.get(ticker, {})
            
            # Initialize ratings dict with defaults
            ratings = {
                'ticker': ticker,
                'buy_percentage': None,
                'total_ratings': 0,
                'analyst_count': 0,
                'rating_details': {},
                'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            if not recommendations or not isinstance(recommendations, list) or not recommendations:
                logger.debug(f"No analyst ratings found for {ticker}")
                return ratings
            
            # Most recent period's data
            latest_rec = recommendations[0]
            
            # Calculate buy percentage
            buy_count = latest_rec.get('strongBuy', 0) + latest_rec.get('buy', 0)
            total_ratings = (buy_count + latest_rec.get('hold', 0) + 
                           latest_rec.get('sell', 0) + latest_rec.get('strongSell', 0))
            
            buy_percentage = (buy_count / total_ratings * 100) if total_ratings > 0 else None
            
            # Update ratings with actual data
            ratings.update({
                'buy_percentage': buy_percentage,
                'total_ratings': total_ratings,
                'analyst_count': total_ratings,  # Use total ratings as analyst count
                'rating_details': {
                    'strongBuy': latest_rec.get('strongBuy', 0),
                    'buy': latest_rec.get('buy', 0),
                    'hold': latest_rec.get('hold', 0),
                    'sell': latest_rec.get('sell', 0),
                    'strongSell': latest_rec.get('strongSell', 0),
                }
            })
            
            return ratings
        
        except Exception as e:
            # Check for rate limit related errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded for ticker {ticker}: {str(e)}")
            
            # Generic error handler
            raise YFinanceError(f"Error fetching analyst ratings for {ticker}: {str(e)}")
    
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
        try:
            logger.debug(f"Fetching insider transactions for {ticker}")
            
            # Create a yahooquery Ticker instance
            yq_ticker = Ticker(ticker)
            
            # Get insider transactions
            insider_data = yq_ticker.insider_transactions.get(ticker, [])
            
            if not insider_data or not isinstance(insider_data, list):
                logger.debug(f"No insider transactions found for {ticker}")
                return []
            
            return self._process_insider_transactions(insider_data)
        
        except Exception as e:
            # Check for rate limit related errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded for ticker {ticker}: {str(e)}")
            
            # Generic error handler
            raise YFinanceError(f"Error fetching insider transactions for {ticker}: {str(e)}")
    
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
        try:
            logger.debug(f"Searching for tickers matching query: {query}")
            
            # Unfortunately, yahooquery doesn't have a built-in search function
            # This would require additional implementation using requests to Yahoo Finance search API
            
            # For now, return an empty list with a warning
            logger.warning("Ticker search not implemented in yahooquery provider")
            return []
            
        except Exception as e:
            # Check for rate limit related errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded for ticker search: {str(e)}")
            
            # Generic error handler
            raise YFinanceError(f"Error searching for tickers: {str(e)}")
    
    @rate_limited
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
        try:
            logger.debug(f"Fetching batch ticker info for {len(tickers)} tickers")
            start_time = time.time()
            
            # Create a yahooquery Ticker instance with multiple tickers
            yq_ticker = Ticker(tickers)
            
            # Get batch data
            summary_detail = yq_ticker.summary_detail
            price_data = yq_ticker.price
            quote_type = yq_ticker.quote_type
            financial_data = yq_ticker.financial_data
            key_stats = yq_ticker.key_stats
            
            # Get recommendation trend for all tickers
            recommendations = yq_ticker.recommendation_trend
            
            # Get calendar events for earnings dates
            calendar_events = yq_ticker.calendar_events
            
            # Prepare results dictionary
            results = {}
            
            # Process each ticker
            for ticker in tickers:
                ticker_summary = summary_detail.get(ticker, {})
                ticker_price = price_data.get(ticker, {})
                ticker_quote = quote_type.get(ticker, {})
                ticker_financial = financial_data.get(ticker, {})
                ticker_stats = key_stats.get(ticker, {})
                
                # Skip this ticker if no data was returned
                if not any([ticker_summary, ticker_price, ticker_quote]):
                    logger.debug(f"No data returned for ticker {ticker} in batch")
                    continue
                
                # Get company name
                company_name = ticker_price.get('shortName') or ticker_quote.get('shortName')
                
                # Create base info dict
                info = {
                    'ticker': ticker,
                    'company_name': company_name,
                    'sector': ticker_price.get('sector', ''),
                    'industry': ticker_price.get('industry', ''),
                    'price': ticker_price.get('regularMarketPrice', None),
                    'target_price': ticker_financial.get('targetMeanPrice', None),
                    'market_cap': ticker_price.get('marketCap', None),
                    'beta': ticker_stats.get('beta', None),
                    'pe_trailing': ticker_stats.get('trailingPE', None),
                    'pe_forward': ticker_stats.get('forwardPE', None),
                    'peg_ratio': ticker_stats.get('pegRatio', None),
                    'short_percent': ticker_stats.get('shortPercentOfFloat', None),
                    'dividend_yield': ticker_summary.get('dividendYield', None) * 100 if ticker_summary.get('dividendYield') else None,
                    'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'from_cache': False,
                }
                
                # Get analyst ratings and recommendations
                ticker_recommendations = recommendations.get(ticker, [])
                if ticker_recommendations and isinstance(ticker_recommendations, list) and ticker_recommendations:
                    latest_rec = ticker_recommendations[0]  # Most recent period's data
                    
                    # Calculate buy percentage
                    buy_count = latest_rec.get('strongBuy', 0) + latest_rec.get('buy', 0)
                    total_ratings = (buy_count + latest_rec.get('hold', 0) + 
                                    latest_rec.get('sell', 0) + latest_rec.get('strongSell', 0))
                    
                    buy_percentage = (buy_count / total_ratings * 100) if total_ratings > 0 else None
                    
                    info.update({
                        'buy_percentage': buy_percentage,
                        'total_ratings': total_ratings,
                        'analyst_count': total_ratings,  # Use total ratings as analyst count
                    })
                else:
                    # Set defaults for these fields
                    info.update({
                        'buy_percentage': None,
                        'total_ratings': 0,
                        'analyst_count': 0,
                    })
                
                # Get earnings dates if available
                ticker_calendar = calendar_events.get(ticker, {})
                earnings_dates = ticker_calendar.get('earnings_dates', [])
                if earnings_dates and isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                    most_recent = earnings_dates[0].get('earnings_date')
                    if most_recent:
                        info['earnings_date'] = pd.Timestamp(most_recent).strftime('%Y-%m-%d')
                
                # Calculate upside if price and target are available
                if info.get('price') and info.get('target_price'):
                    try:
                        upside = ((info['target_price'] / info['price']) - 1) * 100
                        info['upside'] = upside
                    except (TypeError, ZeroDivisionError):
                        info['upside'] = None
                
                # Add to results
                results[ticker] = info
            
            # Add insider transactions if not skipped (needs separate API calls)
            if not skip_insider_metrics:
                for ticker in tickers:
                    if ticker in results:
                        try:
                            insider_data = yq_ticker.insider_transactions.get(ticker, [])
                            if isinstance(insider_data, list) and insider_data:
                                # Process insider transactions
                                results[ticker]['insider_transactions'] = self._process_insider_transactions(insider_data)
                            else:
                                results[ticker]['insider_transactions'] = []
                        except Exception as e:
                            logger.debug(f"Error fetching insider data for {ticker}: {str(e)}")
                            results[ticker]['insider_transactions'] = []
            
            # Add total processing time
            processing_time = time.time() - start_time
            logger.debug(f"Batch processing completed in {processing_time:.2f} seconds")
            
            return results
        
        except Exception as e:
            # Check for rate limit related errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded for batch ticker info: {str(e)}")
            
            # Generic error handler
            raise YFinanceError(f"Error fetching batch ticker info: {str(e)}")