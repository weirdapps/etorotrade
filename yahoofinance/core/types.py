"""
Core data types for Yahoo Finance data.

This module defines the core data structures used throughout the package.
"""

from dataclasses import dataclass
from typing import Any


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
    market_cap: float | None = None

    # Price Data
    current_price: float | None = None
    target_price: float | None = None
    price_change_percentage: float | None = None
    mtd_change: float | None = None
    ytd_change: float | None = None
    two_year_change: float | None = None

    # Analyst Coverage
    recommendation_mean: float | None = None
    recommendation_key: str = "N/A"
    analyst_count: int | None = None

    # Valuation Metrics
    pe_trailing: float | None = None
    pe_forward: float | None = None
    peg_ratio: float | None = None

    # Financial Health
    quick_ratio: float | None = None
    current_ratio: float | None = None
    debt_to_equity: float | None = None

    # Risk Metrics
    short_float_pct: float | None = None
    short_ratio: float | None = None
    beta: float | None = None
    alpha: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    cash_percentage: float | None = None

    # Technical Indicators
    ma50: float | None = None
    ma200: float | None = None

    # Dividends
    dividend_yield: float | None = None

    # Events
    last_earnings: str | None = None
    previous_earnings: str | None = None

    # Insider Activity
    insider_buy_pct: float | None = None
    insider_transactions: int | None = None

    # Internal
    ticker_object: Any = None

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "StockData":
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
