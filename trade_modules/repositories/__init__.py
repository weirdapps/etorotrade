"""
Trade modules repository pattern implementation.

This package provides repository interfaces and implementations for abstracting
data access operations while maintaining backward compatibility with existing
CSV and data access patterns.
"""

from .interfaces import (
    IDataRepository,
    IPortfolioRepository,
    IMarketDataRepository,
    ICsvRepository
)
from .csv_repository import CsvRepository
from .portfolio_repository import PortfolioRepository
from .market_data_repository import MarketDataRepository

__all__ = [
    'IDataRepository',
    'IPortfolioRepository', 
    'IMarketDataRepository',
    'ICsvRepository',
    'CsvRepository',
    'PortfolioRepository',
    'MarketDataRepository'
]