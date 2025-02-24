from typing import Dict, List, Optional, Any, Tuple
import yfinance as yf
from functools import lru_cache
import time
import logging
import pandas as pd
from .insiders import InsiderAnalyzer
from .types import YFinanceError, APIError, ValidationError, StockData

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
                self.logger.debug(f"Attempt {attempts + 1} failed for {ticker}: {str(e)}")
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
                risk_free_rate = 0.05  # 5% annual risk-free rate
                daily_rf = risk_free_rate / 252  # Daily risk-free rate
                
                # Calculate alpha (excess return)
                excess_returns = returns - daily_rf
                alpha = excess_returns.mean() * 252  # Annualized
                
                # Calculate Sharpe ratio
                returns_std = returns.std() * (252 ** 0.5)  # Annualized
                if returns_std > 0:
                    sharpe = (returns.mean() * 252 - risk_free_rate) / returns_std
                
                # Calculate Sortino ratio
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std() * (252 ** 0.5)
                    if downside_std > 0:
                        sortino = (returns.mean() * 252 - risk_free_rate) / downside_std
        
        return alpha, sharpe, sortino

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
        
        attempts = 0
        while attempts < self.retry_attempts:
            try:
                # Get ticker info
                stock = None
                info = {}
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info or {}
                except Exception as e:
                    self.logger.error(f"Failed to get ticker info: {str(e)}")
                    raise YFinanceError(f"Failed to get ticker info: {str(e)}")
                
                # Get historical data for price changes
                hist = pd.DataFrame()
                try:
                    hist = stock.history(period="2y")
                except Exception as e:
                    self.logger.warning(f"Failed to get historical data: {str(e)}")
                
                # Calculate metrics
                price_change, mtd_change, ytd_change, two_year_change = self._calculate_price_changes(hist)
                alpha, sharpe, sortino = self._calculate_risk_metrics(hist)
                
                # Get earnings dates
                last_earnings, previous_earnings = self.get_earnings_dates(ticker)
                
                # Get insider metrics if not skipped
                insider_metrics = (
                    {"insider_buy_pct": None, "transaction_count": None}
                    if skip_insider_metrics
                    else self.insider_analyzer.get_insider_metrics(ticker)
                )
                
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
                    short_float_pct=info.get("shortPercentOfFloat", 0) * 100 if info.get("shortPercentOfFloat") is not None else None,
                    short_ratio=info.get("shortRatio"),
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