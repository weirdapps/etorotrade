"""
CSV repository implementation for file-based data operations.

This implementation maintains backward compatibility with existing CSV
file operations while providing a clean repository interface.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

from .interfaces import ICsvRepository
from ..errors import DataProcessingError


logger = logging.getLogger(__name__)


class CsvRepository(ICsvRepository):
    """
    CSV-based repository implementation.
    
    This implementation provides file-based data access operations
    while maintaining backward compatibility with existing CSV usage
    patterns throughout the trade modules.
    """
    
    def __init__(self, base_directory: Union[str, Path]):
        """
        Initialize CSV repository.
        
        Args:
            base_directory: Base directory for CSV files
        """
        self.base_directory = Path(base_directory)
        self.logger = logger
        
        # Ensure base directory exists
        self.base_directory.mkdir(parents=True, exist_ok=True)
    
    def read(self, identifier: str, **kwargs) -> pd.DataFrame:
        """
        Read data by identifier.
        
        Args:
            identifier: File identifier (without extension)
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            DataFrame with file contents
        """
        file_path = self.get_file_path(identifier)
        return self.read_csv(file_path, **kwargs)
    
    def write(self, identifier: str, data: pd.DataFrame, **kwargs) -> bool:
        """
        Write data to storage.
        
        Args:
            identifier: File identifier (without extension)  
            data: DataFrame to write
            **kwargs: Additional arguments for pandas.to_csv
            
        Returns:
            True if write successful, False otherwise
        """
        file_path = self.get_file_path(identifier)
        return self.write_csv(file_path, data, **kwargs)
    
    def exists(self, identifier: str) -> bool:
        """
        Check if data exists.
        
        Args:
            identifier: File identifier
            
        Returns:
            True if file exists, False otherwise
        """
        file_path = self.get_file_path(identifier)
        return file_path.exists()
    
    def delete(self, identifier: str) -> bool:
        """
        Delete data.
        
        Args:
            identifier: File identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            file_path = self.get_file_path(identifier)
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except (OSError, IOError, PermissionError, FileNotFoundError) as e:
            self.logger.error(f"Error deleting file {identifier}: {e}")
            return False
    
    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Read CSV file into DataFrame.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            DataFrame with CSV contents
            
        Raises:
            DataProcessingError: If file cannot be read
        """
        try:
            file_path = Path(file_path)
            
            # Set default CSV reading parameters that match existing behavior
            csv_kwargs = {
                'index_col': 0,  # Most existing CSV files use first column as index
                'encoding': 'utf-8',
                **kwargs  # Allow override of defaults
            }
            
            if not file_path.exists():
                self.logger.warning(f"CSV file not found: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(file_path, **csv_kwargs)
            self.logger.debug(f"Read CSV file: {file_path} ({len(df)} rows)")
            return df
            
        except (FileNotFoundError, pd.errors.EmptyDataError, OSError, IOError, ValueError, KeyError) as e:
            self.logger.error(f"Error reading CSV file {file_path}: {e}")
            raise DataProcessingError(f"Failed to read CSV file {file_path}") from e
    
    def write_csv(self, file_path: Union[str, Path], data: pd.DataFrame, **kwargs) -> bool:
        """
        Write DataFrame to CSV file.
        
        Args:
            file_path: Path to CSV file
            data: DataFrame to write
            **kwargs: Additional arguments for pandas.to_csv
            
        Returns:
            True if write successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set default CSV writing parameters that match existing behavior
            csv_kwargs = {
                'index': True,  # Preserve index in CSV files
                'encoding': 'utf-8',
                'float_format': '%.6f',  # Consistent float formatting
                **kwargs  # Allow override of defaults
            }
            
            data.to_csv(file_path, **csv_kwargs)
            self.logger.info(f"Wrote CSV file: {file_path} ({len(data)} rows)")
            return True
            
        except (OSError, IOError, PermissionError, ValueError) as e:
            self.logger.error(f"Error writing CSV file {file_path}: {e}")
            return False
    
    def get_file_path(self, identifier: str) -> Path:
        """
        Get full file path for identifier.
        
        Args:
            identifier: File identifier
            
        Returns:
            Full path to CSV file
        """
        # Add .csv extension if not present
        if not identifier.endswith('.csv'):
            identifier = f"{identifier}.csv"
        
        return self.base_directory / identifier
    
    def list_files(self, pattern: str = "*.csv") -> List[Path]:
        """
        List CSV files matching pattern.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of matching file paths
        """
        try:
            return list(self.base_directory.glob(pattern))
        except (OSError, IOError, ValueError) as e:
            self.logger.error(f"Error listing files with pattern {pattern}: {e}")
            return []
    
    def backup_file(self, identifier: str, backup_suffix: Optional[str] = None) -> bool:
        """
        Create backup of CSV file.
        
        Args:
            identifier: File identifier
            backup_suffix: Optional suffix for backup file
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            file_path = self.get_file_path(identifier)
            if not file_path.exists():
                return False
            
            if backup_suffix is None:
                from datetime import datetime
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            backup_name = f"{file_path.stem}_backup_{backup_suffix}.csv"
            backup_path = file_path.parent / backup_name
            
            # Copy file contents
            backup_path.write_bytes(file_path.read_bytes())
            self.logger.info(f"Created backup: {backup_path}")
            return True
            
        except (OSError, IOError, PermissionError, FileNotFoundError) as e:
            self.logger.error(f"Error creating backup for {identifier}: {e}")
            return False
    
    def get_file_info(self, identifier: str) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            identifier: File identifier
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = self.get_file_path(identifier)
            if not file_path.exists():
                return {}
            
            stat = file_path.stat()
            return {
                'path': str(file_path),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'exists': True
            }
        except (OSError, IOError, FileNotFoundError) as e:
            self.logger.error(f"Error getting file info for {identifier}: {e}")
            return {'exists': False}