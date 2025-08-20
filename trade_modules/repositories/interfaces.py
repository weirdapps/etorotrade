"""
Repository interfaces for data access abstraction.

These interfaces define contracts for data access operations, allowing
for different implementations (CSV, database, API) while maintaining
the same interface across the trade modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd


class IDataRepository(ABC):
    """Base interface for data repository operations."""
    
    @abstractmethod
    def read(self, identifier: str, **kwargs) -> pd.DataFrame:
        """Read data by identifier."""
        pass
    
    @abstractmethod
    def write(self, identifier: str, data: pd.DataFrame, **kwargs) -> bool:
        """Write data to storage."""
        pass
    
    @abstractmethod
    def exists(self, identifier: str) -> bool:
        """Check if data exists."""
        pass
    
    @abstractmethod
    def delete(self, identifier: str) -> bool:
        """Delete data."""
        pass


class ICsvRepository(IDataRepository):
    """Interface for CSV-based data operations."""
    
    @abstractmethod
    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read CSV file into DataFrame."""
        pass
    
    @abstractmethod
    def write_csv(self, file_path: Union[str, Path], data: pd.DataFrame, **kwargs) -> bool:
        """Write DataFrame to CSV file."""
        pass
    
    @abstractmethod
    def get_file_path(self, identifier: str) -> Path:
        """Get full file path for identifier."""
        pass


class IPortfolioRepository(IDataRepository):
    """Interface for portfolio data operations."""
    
    @abstractmethod
    def get_portfolio(self) -> pd.DataFrame:
        """Get current portfolio holdings."""
        pass
    
    @abstractmethod
    def update_portfolio(self, data: pd.DataFrame) -> bool:
        """Update portfolio holdings."""
        pass
    
    @abstractmethod
    def get_holdings_for_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get holdings information for specific ticker."""
        pass
    
    @abstractmethod
    def add_holding(self, ticker: str, quantity: float, **kwargs) -> bool:
        """Add new holding to portfolio."""
        pass
    
    @abstractmethod
    def remove_holding(self, ticker: str) -> bool:
        """Remove holding from portfolio."""
        pass


class IMarketDataRepository(IDataRepository):
    """Interface for market data operations."""
    
    @abstractmethod
    def get_market_data(self, data_type: str) -> pd.DataFrame:
        """Get market data by type (buy, sell, hold, market)."""
        pass
    
    @abstractmethod
    def save_market_data(self, data_type: str, data: pd.DataFrame) -> bool:
        """Save market data by type."""
        pass
    
    @abstractmethod
    def get_available_data_types(self) -> List[str]:
        """Get list of available market data types."""
        pass
    
    @abstractmethod
    def clear_market_data(self, data_type: str) -> bool:
        """Clear market data for specific type."""
        pass
    
    @abstractmethod
    def backup_market_data(self, backup_suffix: str = None) -> bool:
        """Create backup of market data files."""
        pass