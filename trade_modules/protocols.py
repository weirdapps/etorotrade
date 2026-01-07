"""
Protocol definitions for trade modules.

This module defines Protocol classes for type hints that don't require
runtime imports, helping to break circular dependencies while maintaining
type safety.

Protocols are structural subtyping (duck typing with static type checking).
They define interfaces without requiring inheritance.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import pandas as pd


@runtime_checkable
class FinanceDataProviderProtocol(Protocol):
    """
    Protocol for finance data providers.

    This protocol defines the interface that all finance data providers must implement.
    Using Protocol instead of ABC allows for structural subtyping and avoids
    circular imports when used for type hints.
    """

    async def get_ticker_info(
        self, ticker: str, skip_insider_metrics: bool = False
    ) -> Dict[str, Any]:
        """Get comprehensive information for a ticker."""
        ...

    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get current price data for a ticker."""
        ...

    async def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical price data for a ticker."""
        ...

    async def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Get ticker information for multiple tickers in a batch."""
        ...

    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings for a ticker."""
        ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logger objects."""

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        ...

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        ...


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol for configuration objects."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        ...


@runtime_checkable
class TradingCriteriaProtocol(Protocol):
    """Protocol for trading criteria configuration."""

    UNIVERSAL_THRESHOLDS: Dict[str, Any]

    def get_tier_from_market_cap(self, market_cap: float) -> str:
        """Determine the market cap tier for a given value."""
        ...

    def get_region_from_ticker(self, ticker: str) -> str:
        """Determine the region for a ticker based on suffix."""
        ...

    def get_tier_thresholds(self, tier: str, action_type: str) -> Dict[str, Any]:
        """Get thresholds for a specific tier and action type."""
        ...

    def get_sector_adjusted_thresholds(
        self, ticker: str, action_type: str, thresholds: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply sector-specific adjustments to thresholds."""
        ...


@runtime_checkable
class AnalysisServiceProtocol(Protocol):
    """Protocol for analysis service."""

    def calculate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals for a DataFrame."""
        ...

    def calculate_confidence_score(self, row: pd.Series) -> float:
        """Calculate confidence score for a single row."""
        ...


@runtime_checkable
class FilterServiceProtocol(Protocol):
    """Protocol for filter service."""

    def filter_buy_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame for buy opportunities."""
        ...

    def filter_sell_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame for sell opportunities."""
        ...

    def filter_hold_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame for hold opportunities."""
        ...

    def filter_notrade_tickers(self, df: pd.DataFrame, notrade_path: str) -> pd.DataFrame:
        """Filter out notrade tickers."""
        ...


@runtime_checkable
class PortfolioServiceProtocol(Protocol):
    """Protocol for portfolio service."""

    def apply_portfolio_filter(
        self, market_df: pd.DataFrame, portfolio_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply portfolio filter to market data."""
        ...

    def apply_portfolio_filters(
        self, results: Dict[str, pd.DataFrame], portfolio_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Apply portfolio filters to results dictionary."""
        ...


@runtime_checkable
class DataProcessingServiceProtocol(Protocol):
    """Protocol for data processing service."""

    async def process_ticker_batch(
        self, tickers: List[str], batch_size: int = 50
    ) -> pd.DataFrame:
        """Process a batch of tickers for market data."""
        ...


# Type aliases for common types
TickerInfo = Dict[str, Any]
MarketData = pd.DataFrame
TradingSignal = str  # "B", "S", "H", "I"
