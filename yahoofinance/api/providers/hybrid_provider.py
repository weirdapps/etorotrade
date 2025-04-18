"""
Hybrid provider implementation combining YFinance and YahooQuery.

This module implements a hybrid provider that uses YFinance as the primary
data source and supplements any missing data with YahooQuery to provide
the most complete and accurate data possible.
"""

from ...core.logging_config import get_logger

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation
import time
from typing import Dict, Any, Optional, List, Tuple, cast
import pandas as pd

from .base_provider import FinanceDataProvider
from .yahoo_finance_base import YahooFinanceBaseProvider
from .yahoo_finance import YahooFinanceProvider
from .yahooquery_provider import YahooQueryProvider
from ...core.errors import YFinanceError, APIError, ValidationError, RateLimitError
from ...utils.network.rate_limiter import rate_limited

logger = get_logger(__name__)

class HybridProvider(YahooFinanceBaseProvider, FinanceDataProvider):
    """
    Hybrid data provider implementation combining YFinance and YahooQuery.
    
    This provider first attempts to fetch data using YFinance, then supplements
    any missing data using YahooQuery to provide the most complete and accurate
    data possible.
    
    Attributes:
        yf_provider: The YFinance provider instance
        yq_provider: The YahooQuery provider instance
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Hybrid provider.
        
        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries
        """
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        
        # Import config to check yahooquery status
        from ...core.config import PROVIDER_CONFIG
        
        # Check if yahooquery is enabled
        self.enable_yahooquery = PROVIDER_CONFIG.get("ENABLE_YAHOOQUERY", False)
        
        # Initialize underlying providers
        self.yf_provider = YahooFinanceProvider(max_retries=max_retries, retry_delay=retry_delay)
        
        # Log whether yahooquery is enabled or disabled
        if self.enable_yahooquery:
            logger.info("HybridProvider initialized with yahooquery supplementation ENABLED")
            self.yq_provider = YahooQueryProvider(max_retries=max_retries, retry_delay=retry_delay)
        else:
            logger.info("HybridProvider initialized with yahooquery supplementation DISABLED")
            # Still create the provider instance but we won't use it unless config changes at runtime
            self.yq_provider = YahooQueryProvider(max_retries=max_retries, retry_delay=retry_delay)
    
    def _handle_delay(self, delay: float):
        """
        Handle delaying execution for retry logic using synchronous time.sleep().
        
        Args:
            delay: Time in seconds to delay
        """
        time.sleep(delay)
    
    def _supplement_with_yahooquery(self, ticker: str, yf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supplement YFinance data with YahooQuery data for any missing fields.
        
        Args:
            ticker: Stock ticker symbol
            yf_data: Data from YFinance
            
        Returns:
            Dict containing the combined data
        """
        # First check if yahooquery is enabled
        if not self.enable_yahooquery:
            logger.debug(f"Skipping yahooquery supplement for {ticker}, yahooquery is disabled")
            return yf_data
            
        # Skip if YFinance data is already complete
        needs_supplement = False
        supplemental_fields = ['pe_forward', 'peg_ratio', 'beta', 'short_percent', 'dividend_yield']
        
        # Check if we need to supplement
        for field in supplemental_fields:
            if field not in yf_data or yf_data[field] is None:
                needs_supplement = True
                break
        
        # Skip supplement if data is complete
        if not needs_supplement:
            logger.debug(f"No supplement needed for {ticker}, YFinance data is complete")
            return yf_data
        
        try:
            # Get data from YahooQuery
            logger.debug(f"Supplementing {ticker} data with YahooQuery")
            start_time = time.time()
            yq_data = self.yq_provider.get_ticker_info(ticker, skip_insider_metrics=True)
            processing_time = time.time() - start_time
            
            # Mark as hybrid data source
            yf_data['hybrid_source'] = 'YFinance+YahooQuery'
            yf_data['yq_processing_time'] = processing_time
            
            # Transfer any missing fields from YahooQuery to YFinance data
            for field in supplemental_fields:
                if (field not in yf_data or yf_data[field] is None) and field in yq_data and yq_data[field] is not None:
                    yf_data[field] = yq_data[field]
                    logger.debug(f"Supplemented {field} for {ticker} from YahooQuery")
            
            # Calculate upside if needed
            if 'upside' not in yf_data or yf_data['upside'] is None:
                if yf_data.get('price') and yf_data.get('target_price'):
                    try:
                        upside = ((yf_data['target_price'] / yf_data['price']) - 1) * 100
                        yf_data['upside'] = upside
                    except (TypeError, ZeroDivisionError):
                        pass
            
            return yf_data
        except YFinanceError as e:
            # Log the error but don't fail - return the original YFinance data
            logger.warning(f"Error supplementing {ticker} with YahooQuery: {str(e)}")
            return yf_data
    
    @rate_limited
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker using both YFinance and YahooQuery.
        
        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics
            
        Returns:
            Dict containing combined stock information
            
        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        try:
            logger.debug(f"Fetching ticker info for {ticker} using Hybrid provider")
            start_time = time.time()
            
            # First try with YFinance
            yf_data = self.yf_provider.get_ticker_info(ticker, skip_insider_metrics)
            
            # Mark data source
            yf_data['data_source'] = 'YFinance'
            
            # Supplement with YahooQuery if needed
            combined_data = self._supplement_with_yahooquery(ticker, yf_data)
            
            # Record total processing time
            combined_data['processing_time'] = time.time() - start_time
            
            return combined_data
        except YFinanceError as e:
            # If YFinance fails completely, try YahooQuery as a fallback
            logger.warning(f"YFinance failed for {ticker}, trying YahooQuery as fallback: {str(e)}")
            
            try:
                yq_data = self.yq_provider.get_ticker_info(ticker, skip_insider_metrics)
                yq_data['data_source'] = 'YahooQuery'
                yq_data['processing_time'] = time.time() - start_time
                return yq_data
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for {ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching data for ticker {ticker}: {str(e)}")
    
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
            # First try with YFinance
            price_data = self.yf_provider.get_price_data(ticker)
            price_data['data_source'] = 'YFinance'
            return price_data
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(f"YFinance failed for price data of {ticker}, trying YahooQuery: {str(e)}")
            
            try:
                price_data = self.yq_provider.get_price_data(ticker)
                price_data['data_source'] = 'YahooQuery'
                return price_data
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for price data of {ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching price data for ticker {ticker}: {str(e)}")
    
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
            # First try with YFinance
            hist_data = self.yf_provider.get_historical_data(ticker, period, interval)
            return hist_data
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(f"YFinance failed for historical data of {ticker}, trying YahooQuery: {str(e)}")
            
            try:
                hist_data = self.yq_provider.get_historical_data(ticker, period, interval)
                return hist_data
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for historical data of {ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching historical data for ticker {ticker}: {str(e)}")
    
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
            # First try with YFinance
            most_recent, previous = self.yf_provider.get_earnings_dates(ticker)
            
            # If we got data from YFinance, return it
            if most_recent or previous:
                return most_recent, previous
            
            # Otherwise try YahooQuery
            logger.debug(f"No earnings dates from YFinance for {ticker}, trying YahooQuery")
            return self.yq_provider.get_earnings_dates(ticker)
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(f"YFinance failed for earnings dates of {ticker}, trying YahooQuery: {str(e)}")
            
            try:
                return self.yq_provider.get_earnings_dates(ticker)
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for earnings dates of {ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching earnings dates for ticker {ticker}: {str(e)}")
    
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
            # First try with YFinance
            ratings = self.yf_provider.get_analyst_ratings(ticker)
            ratings['data_source'] = 'YFinance'
            
            # If buy_percentage is missing, try to get it from YahooQuery
            if 'buy_percentage' not in ratings or ratings['buy_percentage'] is None:
                try:
                    logger.debug(f"Missing buy_percentage for {ticker}, trying YahooQuery")
                    yq_ratings = self.yq_provider.get_analyst_ratings(ticker)
                    
                    # Update with YahooQuery data if available
                    if 'buy_percentage' in yq_ratings and yq_ratings['buy_percentage'] is not None:
                        ratings['buy_percentage'] = yq_ratings['buy_percentage']
                        ratings['data_source'] = 'YFinance+YahooQuery'
                    
                    # Update other fields if missing
                    for field in ['total_ratings', 'analyst_count']:
                        if field not in ratings or ratings[field] is None or ratings[field] == 0:
                            if field in yq_ratings and yq_ratings[field] is not None and yq_ratings[field] > 0:
                                ratings[field] = yq_ratings[field]
                except YFinanceError as yq_error:
                    logger.debug(f"YahooQuery supplement failed for {ticker} ratings: {str(yq_error)}")
            
            return ratings
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(f"YFinance failed for ratings of {ticker}, trying YahooQuery: {str(e)}")
            
            try:
                ratings = self.yq_provider.get_analyst_ratings(ticker)
                ratings['data_source'] = 'YahooQuery'
                return ratings
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for ratings of {ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching analyst ratings for ticker {ticker}: {str(e)}")
    
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
            # First try with YFinance
            transactions = self.yf_provider.get_insider_transactions(ticker)
            
            # If we got data from YFinance, return it
            if transactions:
                return transactions
            
            # Otherwise try YahooQuery
            logger.debug(f"No insider transactions from YFinance for {ticker}, trying YahooQuery")
            return self.yq_provider.get_insider_transactions(ticker)
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(f"YFinance failed for insider transactions of {ticker}, trying YahooQuery: {str(e)}")
            
            try:
                return self.yq_provider.get_insider_transactions(ticker)
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for insider transactions of {ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching insider transactions for ticker {ticker}: {str(e)}")
    
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
        # Just use YFinance for search - YahooQuery doesn't have a search function
        return self.yf_provider.search_tickers(query, limit)
    
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
            # First try with YFinance
            logger.debug(f"Fetching batch ticker info for {len(tickers)} tickers using Hybrid provider")
            start_time = time.time()
            
            # Get data from YFinance
            yf_results = self.yf_provider.batch_get_ticker_info(tickers, skip_insider_metrics)
            
            # Mark data source for each result
            for ticker, data in yf_results.items():
                data['data_source'] = 'YFinance'
            
            # Track tickers that need supplementing
            supplemented_count = 0
            
            # Check if yahooquery supplementation is enabled
            tickers_to_supplement = []
            
            if self.enable_yahooquery:
                # Supplement with YahooQuery if needed
                # YahooQuery's batch is more efficient, so we'll gather all tickers needing supplements
                for ticker, data in yf_results.items():
                    # Check if we need to supplement this ticker
                    needs_supplement = False
                    key_fields = ['pe_forward', 'peg_ratio', 'beta', 'short_percent', 'buy_percentage']
                    
                    for field in key_fields:
                        if field not in data or data[field] is None:
                            needs_supplement = True
                            break
                    
                    if needs_supplement:
                        tickers_to_supplement.append(ticker)
            else:
                logger.debug("Skipping yahooquery supplementation for batch (disabled in config)")
            
            # If we have tickers to supplement and yahooquery is enabled, get them all at once
            if tickers_to_supplement:
                logger.debug(f"Supplementing {len(tickers_to_supplement)} tickers with YahooQuery")
                
                try:
                    # Get batch data from YahooQuery
                    yq_results = self.yq_provider.batch_get_ticker_info(tickers_to_supplement, skip_insider_metrics)
                    
                    # Update the results with YahooQuery data
                    for ticker in tickers_to_supplement:
                        if ticker in yq_results and ticker in yf_results:
                            yq_data = yq_results[ticker]
                            
                            # Supplement missing fields
                            for field in key_fields:
                                if (field not in yf_results[ticker] or yf_results[ticker][field] is None) and field in yq_data and yq_data[field] is not None:
                                    yf_results[ticker][field] = yq_data[field]
                            
                            # Mark as hybrid data source
                            yf_results[ticker]['data_source'] = 'YFinance+YahooQuery'
                            supplemented_count += 1
                except YFinanceError as e:
                    logger.warning(f"YahooQuery batch supplement failed: {str(e)}")
            
            # Add processing time and stats
            processing_time = time.time() - start_time
            logger.debug(f"Hybrid batch completed in {processing_time:.2f}s, supplemented {supplemented_count}/{len(tickers)} tickers")
            
            return yf_results
        except YFinanceError as e:
            # If YFinance batch fails completely, try YahooQuery as a fallback
            logger.warning(f"YFinance batch failed, trying YahooQuery as fallback: {str(e)}")
            
            try:
                yq_results = self.yq_provider.batch_get_ticker_info(tickers, skip_insider_metrics)
                
                # Mark data source for each result
                for ticker, data in yq_results.items():
                    data['data_source'] = 'YahooQuery'
                
                return yq_results
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for batch ticker info: {str(yq_error)}")
                raise YFinanceError(f"Error fetching batch ticker info: {str(e)}")
    
    def clear_cache(self) -> None:
        """
        Clear any cached data in both providers.
        """
        self.yf_provider.clear_cache()
        self.yq_provider.clear_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state from both providers.
        
        Returns:
            Dict containing cache information
        """
        yf_cache_info = self.yf_provider.get_cache_info()
        yq_cache_info = self.yq_provider.get_cache_info()
        
        return {
            'yfinance': yf_cache_info,
            'yahooquery': yq_cache_info
        }