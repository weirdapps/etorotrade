"""
Trade modules repository pattern implementation.

This package provides repository interfaces and implementations for abstracting
data access operations while maintaining backward compatibility with existing
CSV and data access patterns.
"""

from .csv_repository import CsvRepository
from .interfaces import ICsvRepository, IDataRepository, IMarketDataRepository, IPortfolioRepository
from .market_data_repository import MarketDataRepository
from .portfolio_repository import PortfolioRepository

__all__ = [
    "IDataRepository",
    "IPortfolioRepository",
    "IMarketDataRepository",
    "ICsvRepository",
    "CsvRepository",
    "PortfolioRepository",
    "MarketDataRepository",
]
