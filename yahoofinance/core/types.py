"""
Core data types for Yahoo Finance data.

This module defines the core data structures used throughout the package.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StockData:
    """
    Comprehensive stock data container.

    This class represents a comprehensive set of stock data,
    including price data, analyst coverage, valuation metrics,
    risk metrics, and more.

    Attributes:
        name: Company name
        sector: Company sector
        market_cap: Market capitalization
        current_price: Current stock price
        target_price: Average analyst target price
        price_change_percentage: Percentage price change (daily)
        mtd_change: Month-to-date price change percentage
        ytd_change: Year-to-date price change percentage
        two_year_change: Two-year price change percentage
        recommendation_mean: Average analyst recommendation (1-5 scale)
        recommendation_key: Recommendation key (buy, sell, hold, etc.)
        analyst_count: Number of analysts covering the stock
        pe_trailing: Trailing P/E ratio
        pe_forward: Forward P/E ratio
        peg_ratio: PEG ratio
        quick_ratio: Quick ratio
        current_ratio: Current ratio
        debt_to_equity: Debt-to-equity ratio
        short_float_pct: Short float percentage
        short_ratio: Short ratio
        beta: Beta value
        alpha: Alpha value
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        cash_percentage: Cash-to-debt ratio
        ma50: 50-day moving average
        ma200: 200-day moving average
        dividend_yield: Dividend yield
        last_earnings: Last earnings date (YYYY-MM-DD format)
        previous_earnings: Previous earnings date (YYYY-MM-DD format)
        insider_buy_pct: Insider buy percentage
        insider_transactions: Number of insider transactions
        ticker_object: Underlying ticker object (for internal use)
    """

    # Basic Info
    name: str = "N/A"
    sector: str = "N/A"
    market_cap: Optional[float] = None

    # Price Data
    current_price: Optional[float] = None
    target_price: Optional[float] = None
    price_change_percentage: Optional[float] = None
    mtd_change: Optional[float] = None
    ytd_change: Optional[float] = None
    two_year_change: Optional[float] = None

    # Analyst Coverage
    recommendation_mean: Optional[float] = None
    recommendation_key: str = "N/A"
    analyst_count: Optional[int] = None

    # Valuation Metrics
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    peg_ratio: Optional[float] = None

    # Financial Health
    quick_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None

    # Risk Metrics
    short_float_pct: Optional[float] = None
    short_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    cash_percentage: Optional[float] = None

    # Technical Indicators
    ma50: Optional[float] = None
    ma200: Optional[float] = None

    # Dividends
    dividend_yield: Optional[float] = None

    # Events
    last_earnings: Optional[str] = None
    previous_earnings: Optional[str] = None

    # Insider Activity
    insider_buy_pct: Optional[float] = None
    insider_transactions: Optional[int] = None

    # Internal
    ticker_object: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation of the stock data
        """
        result = {}
        for key, value in self.__dict__.items():
            # Skip internal ticker object
            if key == "ticker_object":
                continue
            result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockData":
        """
        Create StockData from dictionary.

        Args:
            data: Dictionary containing stock data

        Returns:
            StockData instance
        """
        # Filter out unknown keys
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
