"""
Market data repository implementation for market data operations.

This implementation provides market data-specific access operations
while maintaining backward compatibility with existing market CSV files
(buy.csv, sell.csv, hold.csv, market.csv).
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

from .interfaces import IMarketDataRepository
from .csv_repository import CsvRepository
from ..errors import DataProcessingError


logger = logging.getLogger(__name__)


class MarketDataRepository(IMarketDataRepository):
    """
    Market data-specific repository implementation.
    
    This implementation provides market data access operations
    while maintaining backward compatibility with existing market CSV files
    (buy.csv, sell.csv, hold.csv, market.csv) throughout the trade modules.
    """
    
    # Standard market data types used throughout the application
    STANDARD_DATA_TYPES = ['buy', 'sell', 'hold', 'market']
    
    def __init__(self, market_data_directory: Union[str, Path]):
        """
        Initialize market data repository.
        
        Args:
            market_data_directory: Directory containing market data CSV files
        """
        self.market_data_directory = Path(market_data_directory)
        self.csv_repo = CsvRepository(self.market_data_directory)
        self.logger = logger
    
    def read(self, identifier: str, **kwargs) -> pd.DataFrame:
        """
        Read data by identifier.
        
        Args:
            identifier: Data identifier (market data type)
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with data
        """
        return self.get_market_data(identifier, **kwargs)
    
    def write(self, identifier: str, data: pd.DataFrame, **kwargs) -> bool:
        """
        Write data to storage.
        
        Args:
            identifier: Data identifier (market data type)
            data: DataFrame to write
            **kwargs: Additional arguments
            
        Returns:
            True if write successful, False otherwise
        """
        return self.save_market_data(identifier, data, **kwargs)
    
    def exists(self, identifier: str) -> bool:
        """
        Check if data exists.
        
        Args:
            identifier: Data identifier (market data type)
            
        Returns:
            True if data exists, False otherwise
        """
        return self.csv_repo.exists(identifier)
    
    def delete(self, identifier: str) -> bool:
        """
        Delete data.
        
        Args:
            identifier: Data identifier (market data type)
            
        Returns:
            True if deletion successful, False otherwise
        """
        return self.clear_market_data(identifier)
    
    def get_market_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """
        Get market data by type.
        
        Args:
            data_type: Market data type (buy, sell, hold, market)
            **kwargs: Additional arguments for CSV reading
            
        Returns:
            DataFrame with market data
        """
        try:
            if not self._is_valid_data_type(data_type):
                self.logger.warning(f"Unknown market data type: {data_type}")
            
            df = self.csv_repo.read(data_type, **kwargs)
            self.logger.debug(f"Retrieved {data_type} market data with {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading {data_type} market data: {e}")
            # Return empty DataFrame with consistent structure for backward compatibility
            return pd.DataFrame()
    
    def save_market_data(self, data_type: str, data: pd.DataFrame, **kwargs) -> bool:
        """
        Save market data by type.
        
        Args:
            data_type: Market data type (buy, sell, hold, market)
            data: DataFrame with market data
            **kwargs: Additional arguments for CSV writing
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            if not self._is_valid_data_type(data_type):
                self.logger.warning(f"Unknown market data type: {data_type}")
            
            # Validate market data structure if needed
            if not self._validate_market_data(data, data_type):
                self.logger.error(f"Invalid {data_type} market data structure")
                return False
            
            success = self.csv_repo.write(data_type, data, **kwargs)
            if success:
                self.logger.info(f"Saved {data_type} market data with {len(data)} records")
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving {data_type} market data: {e}")
            return False
    
    def get_available_data_types(self) -> List[str]:
        """
        Get list of available market data types.
        
        Returns:
            List of available data types based on existing CSV files
        """
        try:
            csv_files = self.csv_repo.list_files("*.csv")
            data_types = [f.stem for f in csv_files]
            
            # Filter to known data types and add any that exist
            available_types = []
            
            # Add standard types if they exist
            for data_type in self.STANDARD_DATA_TYPES:
                if data_type in data_types:
                    available_types.append(data_type)
            
            # Add any additional types found
            for data_type in data_types:
                if data_type not in available_types:
                    available_types.append(data_type)
            
            self.logger.debug(f"Available market data types: {available_types}")
            return available_types
            
        except Exception as e:
            self.logger.error(f"Error getting available data types: {e}")
            return self.STANDARD_DATA_TYPES.copy()
    
    def clear_market_data(self, data_type: str) -> bool:
        """
        Clear market data for specific type.
        
        Args:
            data_type: Market data type to clear
            
        Returns:
            True if clear successful, False otherwise
        """
        try:
            success = self.csv_repo.delete(data_type)
            if success:
                self.logger.info(f"Cleared {data_type} market data")
            return success
            
        except Exception as e:
            self.logger.error(f"Error clearing {data_type} market data: {e}")
            return False
    
    def backup_market_data(self, backup_suffix: str = None) -> bool:
        """
        Create backup of all market data files.
        
        Args:
            backup_suffix: Optional suffix for backup files
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            available_types = self.get_available_data_types()
            all_success = True
            
            for data_type in available_types:
                if self.exists(data_type):
                    success = self.csv_repo.backup_file(data_type, backup_suffix)
                    if not success:
                        all_success = False
                        self.logger.error(f"Failed to backup {data_type} market data")
            
            if all_success:
                self.logger.info(f"Successfully backed up all market data files")
            
            return all_success
            
        except Exception as e:
            self.logger.error(f"Error backing up market data: {e}")
            return False
    
    def get_market_data_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of all market data files.
        
        Returns:
            Dictionary with summary information for each data type
        """
        try:
            summary = {}
            available_types = self.get_available_data_types()
            
            for data_type in available_types:
                try:
                    df = self.get_market_data(data_type)
                    file_info = self.csv_repo.get_file_info(data_type)
                    
                    summary[data_type] = {
                        'record_count': len(df),
                        'columns': list(df.columns) if not df.empty else [],
                        'file_size': file_info.get('size', 0),
                        'last_modified': file_info.get('modified', 0),
                        'exists': file_info.get('exists', False)
                    }
                    
                    # Add basic statistics if data exists
                    if not df.empty:
                        summary[data_type]['has_data'] = True
                        if 'price' in df.columns:
                            summary[data_type]['price_range'] = {
                                'min': df['price'].min(),
                                'max': df['price'].max(),
                                'mean': df['price'].mean()
                            }
                    else:
                        summary[data_type]['has_data'] = False
                        
                except Exception as e:
                    self.logger.error(f"Error summarizing {data_type}: {e}")
                    summary[data_type] = {'error': str(e)}
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting market data summary: {e}")
            return {}
    
    def consolidate_market_data(self, target_data_type: str = 'consolidated') -> bool:
        """
        Consolidate all market data into a single file.
        
        Args:
            target_data_type: Name for consolidated data file
            
        Returns:
            True if consolidation successful, False otherwise
        """
        try:
            available_types = [dt for dt in self.get_available_data_types() 
                             if dt != target_data_type]
            
            if not available_types:
                self.logger.warning("No market data to consolidate")
                return False
            
            consolidated_data = []
            
            for data_type in available_types:
                df = self.get_market_data(data_type)
                if not df.empty:
                    # Add source column to track origin
                    df = df.copy()
                    df['source_type'] = data_type
                    consolidated_data.append(df)
            
            if not consolidated_data:
                self.logger.warning("No data found to consolidate")
                return False
            
            # Combine all data
            consolidated_df = pd.concat(consolidated_data, ignore_index=True, sort=False)
            
            # Save consolidated data
            success = self.save_market_data(target_data_type, consolidated_df)
            if success:
                self.logger.info(f"Consolidated {len(available_types)} data types into {target_data_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error consolidating market data: {e}")
            return False
    
    def _is_valid_data_type(self, data_type: str) -> bool:
        """
        Check if data type is valid/recognized.
        
        Args:
            data_type: Data type to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Allow standard types and any existing files
        return (data_type in self.STANDARD_DATA_TYPES or 
                self.csv_repo.exists(data_type) or
                len(data_type) > 0)  # Allow any non-empty string for flexibility
    
    def _validate_market_data(self, data: pd.DataFrame, data_type: str) -> bool:
        """
        Validate market data structure.
        
        Args:
            data: Market data DataFrame to validate
            data_type: Type of market data
            
        Returns:
            True if valid, False otherwise
        """
        if data.empty:
            return True
        
        # Basic validation - ensure DataFrame has reasonable structure
        try:
            # Check that we have some data
            if len(data) == 0:
                return True  # Empty is valid
            
            # For trade modules, we expect ticker-based data
            # Check if we have ticker information (index or column)
            has_ticker_info = (
                data.index.name in ['ticker', 'Ticker', 'symbol', 'Symbol'] or
                any(col in data.columns for col in ['ticker', 'Ticker', 'symbol', 'Symbol']) or
                len(data.index) > 0  # Basic structure check
            )
            
            if not has_ticker_info:
                self.logger.warning(f"Market data for {data_type} may be missing ticker information")
                # Don't fail validation - just warn for backward compatibility
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating market data: {e}")
            return False