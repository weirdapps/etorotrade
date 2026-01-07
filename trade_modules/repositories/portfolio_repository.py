"""
Portfolio repository implementation for portfolio data operations.

This implementation provides portfolio-specific data access operations
while maintaining backward compatibility with existing portfolio CSV usage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging

from .interfaces import IPortfolioRepository
from .csv_repository import CsvRepository
from ..errors import DataProcessingError


logger = logging.getLogger(__name__)


class PortfolioRepository(IPortfolioRepository):
    """
    Portfolio-specific repository implementation.
    
    This implementation provides portfolio data access operations
    while maintaining backward compatibility with existing portfolio.csv
    usage patterns throughout the trade modules.
    """
    
    def __init__(self, portfolio_file_path: Union[str, Path]):
        """
        Initialize portfolio repository.
        
        Args:
            portfolio_file_path: Path to portfolio CSV file
        """
        self.portfolio_file_path = Path(portfolio_file_path)
        self.csv_repo = CsvRepository(self.portfolio_file_path.parent)
        self.logger = logger
        
        # Portfolio file identifier for CSV repo
        self.portfolio_identifier = self.portfolio_file_path.stem
    
    def read(self, identifier: str, **kwargs) -> pd.DataFrame:
        """
        Read data by identifier (portfolio-specific).
        
        Args:
            identifier: Data identifier (defaults to portfolio)
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with data
        """
        if identifier == 'portfolio' or identifier == self.portfolio_identifier:
            return self.get_portfolio(**kwargs)
        else:
            return self.csv_repo.read(identifier, **kwargs)
    
    def write(self, identifier: str, data: pd.DataFrame, **kwargs) -> bool:
        """
        Write data to storage (portfolio-specific).
        
        Args:
            identifier: Data identifier
            data: DataFrame to write
            **kwargs: Additional arguments
            
        Returns:
            True if write successful, False otherwise
        """
        if identifier == 'portfolio' or identifier == self.portfolio_identifier:
            return self.update_portfolio(data, **kwargs)
        else:
            return self.csv_repo.write(identifier, data, **kwargs)
    
    def exists(self, identifier: str) -> bool:
        """
        Check if data exists.
        
        Args:
            identifier: Data identifier
            
        Returns:
            True if data exists, False otherwise
        """
        if identifier == 'portfolio' or identifier == self.portfolio_identifier:
            return self.portfolio_file_path.exists()
        else:
            return self.csv_repo.exists(identifier)
    
    def delete(self, identifier: str) -> bool:
        """
        Delete data (portfolio-specific).
        
        Args:
            identifier: Data identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        if identifier == 'portfolio' or identifier == self.portfolio_identifier:
            return self.csv_repo.delete(self.portfolio_identifier)
        else:
            return self.csv_repo.delete(identifier)
    
    def get_portfolio(self, **kwargs) -> pd.DataFrame:
        """
        Get current portfolio holdings.
        
        Args:
            **kwargs: Additional arguments for CSV reading
            
        Returns:
            DataFrame with portfolio holdings
        """
        try:
            if not self.portfolio_file_path.exists():
                self.logger.warning(f"Portfolio file not found: {self.portfolio_file_path}")
                return pd.DataFrame()
            
            df = self.csv_repo.read_csv(self.portfolio_file_path, **kwargs)
            self.logger.debug(f"Retrieved portfolio with {len(df)} holdings")
            return df
            
        except (FileNotFoundError, pd.errors.EmptyDataError, OSError, IOError, KeyError, ValueError) as e:
            self.logger.error(f"Error reading portfolio: {e}")
            raise DataProcessingError("Failed to read portfolio data") from e
    
    def update_portfolio(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        Update portfolio holdings.
        
        Args:
            data: DataFrame with portfolio data
            **kwargs: Additional arguments for CSV writing
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate portfolio data structure
            if not self._validate_portfolio_data(data):
                self.logger.error("Invalid portfolio data structure")
                return False
            
            success = self.csv_repo.write_csv(self.portfolio_file_path, data, **kwargs)
            if success:
                self.logger.info(f"Updated portfolio with {len(data)} holdings")
            return success
            
        except (OSError, IOError, PermissionError, KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error updating portfolio: {e}")
            return False
    
    def get_holdings_for_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get holdings information for specific ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with holding information or None if not found
        """
        try:
            portfolio_df = self.get_portfolio()
            if portfolio_df.empty:
                return None
            
            ticker_upper = ticker.upper()
            
            # First check if ticker is in the index
            if ticker_upper in portfolio_df.index:
                holding = portfolio_df.loc[ticker_upper].to_dict()
                self.logger.debug(f"Found holding for {ticker} in index: {holding}")
                return holding
            
            # Check different possible ticker column names
            ticker_column = None
            for col in ['Ticker', 'ticker', 'Symbol', 'symbol']:
                if col in portfolio_df.columns:
                    ticker_column = col
                    break
            
            if ticker_column is not None:
                # Find holding for ticker (case-insensitive)
                mask = portfolio_df[ticker_column].str.upper() == ticker_upper
                holdings = portfolio_df[mask]
                
                if not holdings.empty:
                    # Return first matching holding as dictionary
                    holding = holdings.iloc[0].to_dict()
                    self.logger.debug(f"Found holding for {ticker} in column {ticker_column}: {holding}")
                    return holding
            
            # If no exact match, try case-insensitive index search
            for idx in portfolio_df.index:
                if str(idx).upper() == ticker_upper:
                    holding = portfolio_df.loc[idx].to_dict()
                    self.logger.debug(f"Found holding for {ticker} via case-insensitive index search: {holding}")
                    return holding
            
            return None
            
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"Error getting holdings for ticker {ticker}: {e}")
            return None
    
    def add_holding(self, ticker: str, quantity: float, **kwargs) -> bool:
        """
        Add new holding to portfolio.
        
        Args:
            ticker: Ticker symbol
            quantity: Number of shares
            **kwargs: Additional holding data (price, value, etc.)
            
        Returns:
            True if addition successful, False otherwise
        """
        try:
            portfolio_df = self.get_portfolio()
            
            # Create new holding record
            new_holding = {
                'Ticker': ticker.upper(),
                'Quantity': quantity,
                **kwargs
            }
            
            # Add to portfolio DataFrame
            if portfolio_df.empty:
                # Create new portfolio
                portfolio_df = pd.DataFrame([new_holding])
                portfolio_df.set_index('Ticker', inplace=True)
            else:
                # Add to existing portfolio
                ticker_upper = ticker.upper()
                if ticker_upper in portfolio_df.index:
                    # Update existing holding
                    for key, value in new_holding.items():
                        if key != 'Ticker':  # Skip ticker since it's the index
                            portfolio_df.loc[ticker_upper, key] = value
                else:
                    # Add new holding
                    new_df = pd.DataFrame([new_holding]).set_index('Ticker')
                    portfolio_df = pd.concat([portfolio_df, new_df])
            
            success = self.update_portfolio(portfolio_df)
            if success:
                self.logger.info(f"Added holding: {ticker} ({quantity} shares)")
            return success
            
        except (KeyError, ValueError, TypeError, AttributeError, OSError, IOError) as e:
            self.logger.error(f"Error adding holding {ticker}: {e}")
            return False
    
    def remove_holding(self, ticker: str) -> bool:
        """
        Remove holding from portfolio.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            True if removal successful, False otherwise
        """
        try:
            portfolio_df = self.get_portfolio()
            if portfolio_df.empty:
                return False
            
            ticker_upper = ticker.upper()
            initial_len = len(portfolio_df)
            
            # First try to remove by index (exact match)
            if ticker_upper in portfolio_df.index:
                portfolio_df = portfolio_df.drop(ticker_upper)
            else:
                # Try case-insensitive index removal
                indices_to_remove = []
                for idx in portfolio_df.index:
                    if str(idx).upper() == ticker_upper:
                        indices_to_remove.append(idx)
                
                if indices_to_remove:
                    portfolio_df = portfolio_df.drop(indices_to_remove)
                else:
                    # Check if ticker exists in a column
                    ticker_column = None
                    for col in ['Ticker', 'ticker', 'Symbol', 'symbol']:
                        if col in portfolio_df.columns:
                            ticker_column = col
                            break
                    
                    if ticker_column:
                        # Remove by column value
                        mask = portfolio_df[ticker_column].str.upper() != ticker_upper
                        portfolio_df = portfolio_df[mask]
                    else:
                        self.logger.warning(f"Ticker {ticker} not found in portfolio")
                        return False
            
            # Check if anything was actually removed
            if len(portfolio_df) == initial_len:
                self.logger.warning(f"Ticker {ticker} not found in portfolio")
                return False
            
            success = self.update_portfolio(portfolio_df)
            if success:
                self.logger.info(f"Removed holding: {ticker}")
            return success
            
        except (KeyError, ValueError, TypeError, AttributeError, OSError, IOError) as e:
            self.logger.error(f"Error removing holding {ticker}: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary with portfolio summary
        """
        try:
            portfolio_df = self.get_portfolio()
            if portfolio_df.empty:
                return {
                    'total_holdings': 0,
                    'total_value': 0.0,
                    'total_quantity': 0.0
                }
            
            summary = {
                'total_holdings': len(portfolio_df),
                'holdings': portfolio_df.index.tolist() if hasattr(portfolio_df.index, 'tolist') else []
            }
            
            # Add value-based metrics if available
            if 'Value' in portfolio_df.columns:
                summary['total_value'] = portfolio_df['Value'].sum()
                summary['avg_holding_value'] = portfolio_df['Value'].mean()
                summary['largest_holding_value'] = portfolio_df['Value'].max()
                summary['smallest_holding_value'] = portfolio_df['Value'].min()
            
            # Add quantity-based metrics if available
            if 'Quantity' in portfolio_df.columns:
                summary['total_quantity'] = portfolio_df['Quantity'].sum()
                summary['avg_quantity'] = portfolio_df['Quantity'].mean()
            
            return summary
            
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def backup_portfolio(self, backup_suffix: Optional[str] = None) -> bool:
        """
        Create backup of portfolio file.
        
        Args:
            backup_suffix: Optional suffix for backup file
            
        Returns:
            True if backup successful, False otherwise
        """
        return self.csv_repo.backup_file(self.portfolio_identifier, backup_suffix)
    
    def _validate_portfolio_data(self, data: pd.DataFrame) -> bool:
        """
        Validate portfolio data structure.
        
        Args:
            data: Portfolio DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        if data.empty:
            return True
        
        # Check for required columns or index structure
        has_ticker_column = any(col in data.columns for col in ['Ticker', 'ticker', 'Symbol', 'symbol'])
        has_ticker_index = data.index.name in ['Ticker', 'ticker', 'Symbol', 'symbol'] if data.index.name else False
        has_meaningful_index = len(data.index) > 0 and not data.index.equals(pd.RangeIndex(len(data)))
        
        if not (has_ticker_column or has_ticker_index or has_meaningful_index):
            self.logger.warning("Portfolio data missing ticker information")
            return False
        
        return True