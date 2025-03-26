from typing import Dict, Optional, List, NamedTuple, Union, Any
from datetime import datetime, timedelta
import pandas as pd
from .core.client import YFinanceClient
from .core.errors import YFinanceError, ValidationError
import logging

logger = logging.getLogger(__name__)

class PriceTarget(NamedTuple):
    """Price target data structure"""
    mean: Optional[float]
    high: Optional[float]
    low: Optional[float]
    num_analysts: int

class PriceData(NamedTuple):
    """Historical price data structure"""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float

class PricingAnalyzer:
    """Class to handle stock pricing and target analysis"""
    
    def __init__(self, client: YFinanceClient):
        self.client = client

    def _safe_float_conversion(self, value: Any) -> Optional[float]:
        """Safely convert a value to float"""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                value = value.replace(',', '')
            return float(value)
        except (ValueError, TypeError):
            return None

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current stock price with validation.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current stock price or None if unavailable
            
        Raises:
            ValidationError: When ticker validation fails
            YFinanceError: When price data is unavailable
        """
        self.client._validate_ticker(ticker)
        
        try:
            stock_data = self.client.get_ticker_info(ticker)
            return self._safe_float_conversion(stock_data.current_price)
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to fetch current price: {str(e)}")

    def get_historical_prices(self, 
                            ticker: str, 
                            period: str = "1mo") -> List[PriceData]:
        """
        Get historical price data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            List of PriceData objects containing historical prices
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        self.client._validate_ticker(ticker)
        valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
        
        if period not in valid_periods:
            raise ValidationError(f"Invalid period. Must be one of: {', '.join(valid_periods)}")

        try:
            stock = self.client.get_ticker_info(ticker)._stock
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No historical data available for {ticker}")
                return []

            return [
                PriceData(
                    date=index,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                    adjusted_close=float(row["Close"])
                )
                for index, row in hist.iterrows()
            ]

        except Exception as e:
            logger.error(f"Error fetching historical prices for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to fetch historical prices: {str(e)}")

    def get_price_targets(self, ticker: str) -> PriceTarget:
        """
        Get analyst price targets.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            PriceTarget object containing target information
            
        Raises:
            ValidationError: When ticker validation fails
            YFinanceError: When target data is unavailable
        """
        self.client._validate_ticker(ticker)
        
        try:
            stock_data = self.client.get_ticker_info(ticker)
            
            target_price = self._safe_float_conversion(stock_data.target_price)
            analyst_count = stock_data.analyst_count
            
            if target_price is None or analyst_count is None:
                logger.warning(f"No price target data available for {ticker}")
                return PriceTarget(
                    mean=None,
                    high=None,
                    low=None,
                    num_analysts=0
                )

            return PriceTarget(
                mean=target_price,
                high=None,  # Could be added if available
                low=None,   # Could be added if available
                num_analysts=analyst_count
            )

        except Exception as e:
            logger.error(f"Error fetching price targets for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to fetch price targets: {str(e)}")

    def calculate_price_metrics(self, 
                              ticker: str) -> Dict[str, Optional[float]]:
        """
        Calculate various price-related metrics.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing calculated metrics
            
        Raises:
            ValidationError: When ticker validation fails
            YFinanceError: When calculation fails
        """
        try:
            current_price = self.get_current_price(ticker)
            targets = self.get_price_targets(ticker)
            
            metrics = {
                "current_price": current_price,
                "target_price": targets.mean,
                "upside_potential": None
            }
            
            if current_price and targets.mean:
                try:
                    metrics["upside_potential"] = ((targets.mean / current_price) - 1) * 100
                except (ZeroDivisionError, TypeError):
                    metrics["upside_potential"] = None
                    
            return metrics

        except Exception as e:
            logger.error(f"Error calculating price metrics for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to calculate price metrics: {str(e)}")