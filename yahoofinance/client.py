from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import yfinance as yf
from functools import lru_cache
import time
import logging
import pandas as pd
from .insiders import InsiderAnalyzer

class YFinanceError(Exception):
    """Base exception for YFinance client errors"""
    pass

class APIError(YFinanceError):
    """Raised when API calls fails"""
    pass

class ValidationError(YFinanceError):
    """Raised when data validation fails"""
    pass

@dataclass
class StockData:
    """
    Comprehensive stock information data class.
    
    Contains fundamental data, technical indicators, analyst ratings,
    and market metrics for a given stock. All numeric fields are
    optional as they may not be available for all stocks.
    
    Fields are grouped by category:
    - Basic Info: name, sector
    - Market Data: market_cap, current_price, target_price
    - Analyst Coverage: recommendation_mean, recommendation_key, analyst_count
    - Valuation Metrics: pe_trailing, pe_forward, peg_ratio
    - Financial Health: quick_ratio, current_ratio, debt_to_equity
    - Risk Metrics: short_float_pct, short_ratio, beta
    - Dividends: dividend_yield
    - Events: last_earnings, previous_earnings
    - Insider Activity: insider_buy_pct, insider_transactions
    """
    name: str
    sector: str
    market_cap: Optional[float]
    current_price: Optional[float]
    target_price: Optional[float]
    recommendation_mean: Optional[float]
    recommendation_key: str
    analyst_count: Optional[int]
    pe_trailing: Optional[float]
    pe_forward: Optional[float]
    peg_ratio: Optional[float]
    quick_ratio: Optional[float]
    current_ratio: Optional[float]
    debt_to_equity: Optional[float]
    short_float_pct: Optional[float]
    short_ratio: Optional[float]
    beta: Optional[float]
    dividend_yield: Optional[float]
    last_earnings: Optional[str]
    previous_earnings: Optional[str]
    insider_buy_pct: Optional[float]
    insider_transactions: Optional[int]
    ticker_object: Optional[yf.Ticker] = field(default=None)

    @property
    def _stock(self) -> yf.Ticker:
        """
        Access the underlying yfinance Ticker object.
        
        This property provides access to the raw yfinance Ticker object,
        which can be used for additional API calls not covered by the
        standard properties.
        
        Returns:
            yfinance.Ticker object for additional API access
            
        Raises:
            AttributeError: If ticker_object is None
        """
        if self.ticker_object is None:
            raise AttributeError("No ticker object available")
        return self.ticker_object

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
    
    def __init__(self, retry_attempts: int = 3, timeout: int = 10):
        """
        Initialize YFinanceClient.

        Args:
            retry_attempts: Number of retry attempts for API calls
            timeout: Timeout in seconds for API calls
        """
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.insider_analyzer = InsiderAnalyzer(self)

    def _validate_ticker(self, ticker: str) -> None:
        """Validate ticker format"""
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValidationError("Ticker must be a non-empty string")
        if len(ticker) > 10:  # Most tickers are 1-5 characters
            raise ValidationError("Ticker length exceeds maximum allowed")

    def get_past_earnings_dates(self, ticker: str) -> List[pd.Timestamp]:
        """
        Retrieve past earnings dates sorted in descending order.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of past earnings dates as pandas Timestamps, sorted most recent first

        Raises:
            ValidationError: When ticker validation fails
            APIError: When API call fails after retries
        """
        self._validate_ticker(ticker)
        
        attempts = 0
        while attempts < self.retry_attempts:
            try:
                stock = yf.Ticker(ticker)
                earnings_dates = stock.get_earnings_dates()

                if earnings_dates is None or earnings_dates.empty:
                    return []

                # Convert index to datetime and filter only past earnings dates
                current_time = pd.Timestamp.now(tz=earnings_dates.index.tz)
                past_earnings = earnings_dates[earnings_dates.index < current_time]

                # Return sorted dates (most recent first)
                return sorted(past_earnings.index, reverse=True)
            except Exception as e:
                self.logger.warning(f"Attempt {attempts + 1} failed for {ticker}: {str(e)}")
                attempts += 1
                if attempts == self.retry_attempts:
                    raise APIError(f"Failed to fetch earnings dates for {ticker} after {self.retry_attempts} attempts: {str(e)}")
                time.sleep(self._get_backoff_time(attempts))  # Exponential backoff
        
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
            StockData: Object containing comprehensive stock information including:
                - Basic info (name, sector, market cap)
                - Price data (current, target)
                - Analyst recommendations
                - Financial ratios (PE, PEG, etc.)
                - Risk metrics (beta, short interest)
                - Earnings dates
                - Insider trading metrics (unless skipped)
            
        Raises:
            ValidationError: When ticker validation fails
            APIError: When API call fails after retries
        """
        self._validate_ticker(ticker)
        
        attempts = 0
        while attempts < self.retry_attempts:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                
                # Get earnings dates
                last_earnings, previous_earnings = self.get_earnings_dates(ticker)
                
                # Get insider metrics if not skipped
                insider_metrics = (
                    {"insider_buy_pct": None, "transaction_count": None}
                    if skip_insider_metrics
                    else self.insider_analyzer.get_insider_metrics(ticker)
                )
                
                return StockData(
                    name=info.get("longName", "N/A"),
                    sector=info.get("sector", "N/A"),
                    market_cap=info.get("marketCap"),
                    current_price=info.get("currentPrice"),
                    target_price=info.get("targetMeanPrice"),
                    recommendation_mean=info.get("recommendationMean"),
                    recommendation_key=info.get("recommendationKey", "N/A"),
                    analyst_count=info.get("numberOfAnalystOpinions"),
                    pe_trailing=info.get("trailingPE"),
                    pe_forward=info.get("forwardPE"),
                    peg_ratio=info.get("trailingPegRatio"),
                    quick_ratio=info.get("quickRatio"),
                    current_ratio=info.get("currentRatio"),
                    debt_to_equity=info.get("debtToEquity"),
                    short_float_pct=info.get("shortPercentOfFloat", 0) * 100 if info.get("shortPercentOfFloat") is not None else None,
                    short_ratio=info.get("shortRatio"),
                    beta=info.get("beta"),
                    dividend_yield=info.get("dividendYield"),
                    last_earnings=last_earnings,
                    previous_earnings=previous_earnings,
                    insider_buy_pct=insider_metrics.get("insider_buy_pct"),
                    insider_transactions=insider_metrics.get("transaction_count"),
                    ticker_object=stock
                )
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempts + 1} failed for {ticker}: {str(e)}")
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