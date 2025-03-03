from typing import Dict, List, Optional, Any, Tuple
import yfinance as yf
from functools import lru_cache
import time
import logging
import pandas as pd
import requests
from .insiders import InsiderAnalyzer
from .types import (
    YFinanceError, APIError, ValidationError, StockData,
    RateLimitError, ConnectionError, TimeoutError, ResourceNotFoundError
)
from .utils.market_utils import is_us_ticker
from .config import RATE_LIMIT, RISK_METRICS
from .errors import format_error_details, classify_api_error

class YFinanceClient:
    """Base client for interacting with Yahoo Finance API"""
    
    def _get_backoff_time(self, attempt: int, base: float = 1.0, max_time: float = 10.0) -> float:
        """
        Calculate exponential backoff time.
        
        Args:
            attempt: Current attempt number (1-based)
            base: Base time in seconds
            max_time: Maximum backoff time in seconds
            
        Returns:
            Time to wait in seconds
        """
        backoff = min(base * (2 ** (attempt - 1)), max_time)
        return backoff
    
    def __init__(self, retry_attempts: int = None, timeout: int = None):
        """
        Initialize YFinanceClient.

        Args:
            retry_attempts: Number of retry attempts for API calls (defaults to config value)
            timeout: Timeout in seconds for API calls (defaults to config value)
        """
        self.retry_attempts = retry_attempts if retry_attempts is not None else RATE_LIMIT["MAX_RETRY_ATTEMPTS"]
        self.timeout = timeout if timeout is not None else RATE_LIMIT["API_TIMEOUT"]
        self.logger = logging.getLogger(__name__)
        self.insider_analyzer = InsiderAnalyzer(self)

    def _validate_ticker(self, ticker) -> None:
        """
        Validate ticker format.
        
        Args:
            ticker: Ticker symbol to validate
            
        Raises:
            ValidationError: When ticker validation fails
        """
        # Check that ticker is a non-empty string
        if not isinstance(ticker, str):
            raise ValidationError("Ticker must be a string")
        if not ticker.strip():
            raise ValidationError("Ticker must be a non-empty string")
        
        # Check for numeric input as test expects this to fail
        if ticker.isdigit():
            raise ValidationError("Ticker cannot be a numeric string")
        
        # Check if ticker has exchange suffix that might make it longer
        has_exchange_suffix = '.' in ticker
        max_length = 20 if has_exchange_suffix else 10  # Allow longer tickers for exchange-specific symbols
        
        if len(ticker) > max_length:
            raise ValidationError("Ticker length exceeds maximum allowed")

    def _fetch_and_process_earnings(self, ticker: str) -> List[pd.Timestamp]:
        """
        Fetch and process earnings dates for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of processed past earnings dates
            
        Raises:
            Various exceptions that will be caught by the calling method
        """
        stock = yf.Ticker(ticker)
        earnings_dates = stock.get_earnings_dates()

        if earnings_dates is None or earnings_dates.empty:
            return []

        # Convert index to datetime and filter only past earnings dates
        current_time = pd.Timestamp.now(tz=earnings_dates.index.tz)
        past_earnings = earnings_dates[earnings_dates.index < current_time]

        # Return sorted dates (most recent first)
        return sorted(past_earnings.index, reverse=True)
    
    def _handle_api_error(self, e, ticker: str, attempts: int) -> Optional[Exception]:
        """
        Handle API errors and create appropriate exception objects.
        
        Args:
            e: Exception object
            ticker: Stock ticker symbol
            attempts: Current attempt number
            
        Returns:
            Exception object or None
            
        Raises:
            ResourceNotFoundError: For 404 errors that should not be retried
        """
        if isinstance(e, requests.exceptions.Timeout):
            error = TimeoutError(f"Request timed out for {ticker}", {"original_error": str(e)})
            self.logger.debug(f"Timeout on attempt {attempts + 1} for {ticker}: {str(e)}")
            return error
            
        if isinstance(e, requests.exceptions.ConnectionError):
            error = ConnectionError(f"Connection error for {ticker}", {"original_error": str(e)})
            self.logger.debug(f"Connection error on attempt {attempts + 1} for {ticker}: {str(e)}")
            return error
            
        if isinstance(e, requests.exceptions.HTTPError):
            if e.response.status_code == 429:
                error = RateLimitError(f"Rate limit exceeded for {ticker}", None, {"original_error": str(e)})
                self.logger.warning(f"Rate limit hit on attempt {attempts + 1} for {ticker}")
                return error
            elif e.response.status_code == 404:
                # No need to retry for not found
                raise ResourceNotFoundError(f"Ticker {ticker} not found", {"status_code": 404, "original_error": str(e)})
            else:
                error = classify_api_error(e.response.status_code, e.response.text)
                self.logger.debug(f"HTTP error on attempt {attempts + 1} for {ticker}: {str(e)}")
                return error
                
        # Generic exception handling
        error = APIError(f"Error fetching earnings dates for {ticker}", {"original_error": str(e)})
        self.logger.debug(f"Attempt {attempts + 1} failed for {ticker}: {str(e)}")
        return error
    
    def get_past_earnings_dates(self, ticker: str) -> List[pd.Timestamp]:
        """
        Retrieve past earnings dates sorted in descending order.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of past earnings dates as pandas Timestamps, sorted most recent first

        Raises:
            ValidationError: When ticker validation fails
            ResourceNotFoundError: When ticker is not found
            RateLimitError: When API rate limit is reached
            ConnectionError: When network connection fails
            TimeoutError: When request times out
            APIError: For other API-related errors
        """
        self._validate_ticker(ticker)
        
        attempts = 0
        last_error = None
        
        while attempts < self.retry_attempts:
            try:
                return self._fetch_and_process_earnings(ticker)
                
            except Exception as e:
                # Handle the error and get appropriate exception object
                last_error = self._handle_api_error(e, ticker, attempts)
            
            # Increment attempts counter
            attempts += 1
            
            # If we've exhausted all retries, raise the last error
            if attempts == self.retry_attempts:
                if not last_error:
                    last_error = APIError(f"Failed to fetch earnings dates for {ticker} after {self.retry_attempts} attempts")
                self.logger.error(format_error_details(last_error))
                raise last_error
                
            # Exponential backoff
            backoff_time = self._get_backoff_time(attempts)
            self.logger.debug(f"Retrying in {backoff_time:.2f} seconds (attempt {attempts}/{self.retry_attempts})")
            time.sleep(backoff_time)
        
    def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the last two earnings dates for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing:
                - most_recent_date: The most recent earnings date in YYYY-MM-DD format
                - previous_date: The second most recent earnings date in YYYY-MM-DD format
                Both values will be None if no earnings dates are found

        Raises:
            ValidationError: When ticker validation fails
            APIError: When API call fails after retries
        """
        self._validate_ticker(ticker)
        try:
            past_dates = self.get_past_earnings_dates(ticker)
            
            if not past_dates:
                return None, None
                
            most_recent = past_dates[0].strftime('%Y-%m-%d') if past_dates else None
            previous = past_dates[1].strftime('%Y-%m-%d') if len(past_dates) >= 2 else None
            
            return most_recent, previous
        except APIError:
            # Re-raise API errors
            raise
        except Exception as e:
            raise APIError(f"Failed to process earnings dates for {ticker}: {str(e)}")
        
    def _calculate_price_changes(self, hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate various price change metrics from historical data."""
        price_change = mtd_change = ytd_change = two_year_change = None
        
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            if len(hist) > 1:
                prev_day = hist['Close'].iloc[-2]
                price_change = ((current - prev_day) / prev_day) * 100
            
            if len(hist) >= 22:  # Approx. one month of trading days
                prev_month = hist['Close'].iloc[-22]
                mtd_change = ((current - prev_month) / prev_month) * 100
            
            if len(hist) > 0:
                start_year = hist['Close'].iloc[0]
                ytd_change = ((current - start_year) / start_year) * 100
                two_year_change = ytd_change  # Same as YTD for now
        
        return price_change, mtd_change, ytd_change, two_year_change

    def _calculate_risk_metrics(self, hist: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate risk metrics from historical data."""
        alpha = sharpe = sortino = None
        
        if not hist.empty:
            returns = hist['Close'].pct_change().dropna()
            if len(returns) > 0:
                # Use values from configuration
                risk_free_rate = RISK_METRICS["RISK_FREE_RATE"]
                trading_days = RISK_METRICS["TRADING_DAYS_PER_YEAR"]
                daily_rf = risk_free_rate / trading_days
                
                # Calculate alpha (excess return)
                excess_returns = returns - daily_rf
                alpha = excess_returns.mean() * trading_days  # Annualized
                
                # Calculate Sharpe ratio
                returns_std = returns.std() * (trading_days ** 0.5)  # Annualized
                if returns_std > 0:
                    sharpe = (returns.mean() * trading_days - risk_free_rate) / returns_std
                
                # Calculate Sortino ratio
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std() * (trading_days ** 0.5)
                    if downside_std > 0:
                        sortino = (returns.mean() * trading_days - risk_free_rate) / downside_std
        
        return alpha, sharpe, sortino

    # Using the centralized utility for US ticker detection
    def _is_us_ticker(self, ticker: str) -> bool:
        """
        Determine if ticker is for US market.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            bool: True if US ticker, False otherwise
        """
        return is_us_ticker(ticker)
        
    def _get_ticker_basic_info(self, ticker: str):
        """Get basic ticker info from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            return stock, info
        except Exception as e:
            self.logger.error(f"Failed to get ticker info: {str(e)}")
            raise YFinanceError(f"Failed to get ticker info: {str(e)}")
            
    def _get_historical_data(self, stock):
        """Get historical price data for a ticker"""
        try:
            return stock.history(period="2y")
        except Exception as e:
            self.logger.warning(f"Failed to get historical data: {str(e)}")
            return pd.DataFrame()
            
    def _get_short_interest_data(self, info, is_us_ticker):
        """Get short interest data for US tickers"""
        short_float_pct = None
        short_ratio = None
        if is_us_ticker:
            short_float_pct = info.get("shortPercentOfFloat", 0) * 100 if info.get("shortPercentOfFloat") is not None else None
            short_ratio = info.get("shortRatio")
        return short_float_pct, short_ratio
        
    def _create_stock_data_object(self, stock, info, price_metrics, risk_metrics, 
                                 earnings_dates, insider_metrics, short_interest):
        """Create StockData object from collected data"""
        price_change, mtd_change, ytd_change, two_year_change = price_metrics
        alpha, sharpe, sortino = risk_metrics
        last_earnings, previous_earnings = earnings_dates
        short_float_pct, short_ratio = short_interest
        
        return StockData(
            # Basic Info
            name=info.get("longName", "N/A"),
            sector=info.get("sector", "N/A"),
            market_cap=info.get("marketCap"),
            
            # Price Data
            current_price=info.get("currentPrice"),
            target_price=info.get("targetMeanPrice"),
            price_change_percentage=price_change,
            mtd_change=mtd_change,
            ytd_change=ytd_change,
            two_year_change=two_year_change,
            
            # Analyst Coverage
            recommendation_mean=info.get("recommendationMean"),
            recommendation_key=info.get("recommendationKey", "N/A"),
            analyst_count=info.get("numberOfAnalystOpinions"),
            
            # Valuation Metrics
            pe_trailing=info.get("trailingPE"),
            pe_forward=info.get("forwardPE"),
            peg_ratio=info.get("trailingPegRatio"),
            
            # Financial Health
            quick_ratio=info.get("quickRatio"),
            current_ratio=info.get("currentRatio"),
            debt_to_equity=info.get("debtToEquity"),
            
            # Risk Metrics
            short_float_pct=short_float_pct,
            short_ratio=short_ratio,
            beta=info.get("beta"),
            alpha=alpha,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            cash_percentage=info.get("cashToDebt"),
            
            # Dividends
            dividend_yield=info.get("dividendYield"),
            
            # Events
            last_earnings=last_earnings,
            previous_earnings=previous_earnings,
            
            # Insider Activity
            insider_buy_pct=insider_metrics.get("insider_buy_pct"),
            insider_transactions=insider_metrics.get("transaction_count"),
            
            # Internal
            ticker_object=stock
        )
    
    @lru_cache(maxsize=50)  # Limit cache size to prevent memory issues
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> StockData:
        """
        Get stock information with retry mechanism and caching.
        
        The data is cached for a maximum of 50 entries to prevent memory issues.
        Cache is automatically cleared when full.
        
        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics
            
        Returns:
            StockData: Object containing comprehensive stock information
            
        Raises:
            ValidationError: When ticker validation fails
            APIError: When API call fails after retries
        """
        self._validate_ticker(ticker)
        
        # Check if it's a US ticker to determine which metrics to fetch
        is_us_ticker_flag = is_us_ticker(ticker)
        
        attempts = 0
        while attempts < self.retry_attempts:
            try:
                # Get ticker basic info
                stock, info = self._get_ticker_basic_info(ticker)
                
                # Get historical data for price changes
                hist = self._get_historical_data(stock)
                
                # Calculate metrics
                price_metrics = self._calculate_price_changes(hist)
                risk_metrics = self._calculate_risk_metrics(hist)
                
                # Get earnings dates
                earnings_dates = self.get_earnings_dates(ticker)
                
                # Get insider metrics if not skipped and this is a US ticker
                insider_metrics = (
                    {"insider_buy_pct": None, "transaction_count": None}
                    if skip_insider_metrics or not is_us_ticker_flag
                    else self.insider_analyzer.get_insider_metrics(ticker)
                )
                
                # Get short interest data
                short_interest = self._get_short_interest_data(info, is_us_ticker_flag)
                
                # Create and return StockData object
                return self._create_stock_data_object(
                    stock, info, price_metrics, risk_metrics, 
                    earnings_dates, insider_metrics, short_interest
                )
                
            except Exception as e:
                self.logger.debug(f"Attempt {attempts + 1} failed for {ticker}: {str(e)}")
                attempts += 1
                if attempts == self.retry_attempts:
                    raise APIError(f"Failed to fetch data for {ticker} after {self.retry_attempts} attempts: {str(e)}")
                time.sleep(self._get_backoff_time(attempts))  # Exponential backoff
                
    def clear_cache(self) -> None:
        """
        Clear the internal cache for ticker information.
        This will force the next get_ticker_info call to fetch fresh data.
        """
        self.get_ticker_info.cache_clear()
        
    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about the current cache state.
        
        Returns:
            Dict[str, int]: Dictionary containing:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - maxsize: Maximum cache size
                - currsize: Current cache size
        """
        info = self.get_ticker_info.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'maxsize': info.maxsize,
            'currsize': info.currsize
        }