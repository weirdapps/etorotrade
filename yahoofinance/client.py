from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import yfinance as yf
from functools import lru_cache
import time
import logging
from datetime import datetime
import pandas as pd
import time
from requests.exceptions import HTTPError
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
    """Data class for stock information"""
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
    previous_earnings: Optional[str]  # Second most recent earnings date
    insider_buy_pct: Optional[float]  # Percentage of insider buy transactions
    insider_transactions: Optional[int]  # Total number of insider transactions
    ticker_object: Any = field(default=None)  # Store the yfinance Ticker object

    @property
    def _stock(self) -> yf.Ticker:
        """Access the underlying yfinance Ticker object"""
        return self.ticker_object

class YFinanceClient:
    """Base client for interacting with Yahoo Finance API"""
    
    def __init__(self, retry_attempts: int = 3, timeout: int = 10, cache_ttl: int = 300):
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
        """Retrieve only past earnings dates sorted in descending order."""
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
            self.logger.info(f"Could not fetch earnings dates for {ticker}: {str(e)}")
            return []
        
    def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the last two earnings dates for a stock.
        Returns tuple of (most_recent_date, previous_date)
        """
        past_dates = self.get_past_earnings_dates(ticker)
        
        if not past_dates:
            return None, None
            
        most_recent = past_dates[0].strftime('%Y-%m-%d') if past_dates else None
        previous = past_dates[1].strftime('%Y-%m-%d') if len(past_dates) >= 2 else None
        
        return most_recent, previous
        
    @lru_cache(maxsize=100)
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> StockData:
        """
        Get stock information with retry mechanism and caching.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            StockData object containing stock information
            
        Raises:
            YFinanceError: When API call fails after retries
            ValidationError: When ticker validation fails
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
                time.sleep(1 * attempts)  # Exponential backoff
                
    def clear_cache(self):
        """Clear the internal cache"""
        self.get_ticker_info.cache_clear()