#!/usr/bin/env python3
"""
Data filtering and selection module extracted from trade.py.
Contains logic for filtering market data, applying trading criteria, and data selection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable

import pandas as pd
import numpy as np

from yahoofinance.core.logging import get_logger
from yahoofinance.core.errors import ValidationError, YFinanceError

from .utils import (
    clean_ticker_symbol,
    safe_float_conversion,
    validate_dataframe,
)

logger = get_logger(__name__)


class TradingCriteriaFilter:
    """Applies trading criteria filters to market data."""
    
    def __init__(self, criteria_config: Optional[Dict] = None):
        """
        Initialize trading criteria filter.
        
        Args:
            criteria_config: Dictionary containing trading criteria configuration
        """
        self.criteria = criteria_config or self._get_default_criteria()
        self.logger = logger
    
    def _get_default_criteria(self) -> Dict:
        """Get default trading criteria."""
        return {
            'min_market_cap': 1e9,      # $1B minimum market cap
            'max_pe_ratio': 25,         # Maximum P/E ratio
            'min_volume': 100000,       # Minimum daily volume
            'max_beta': 2.0,            # Maximum beta (volatility)
            'min_price': 5.0,           # Minimum stock price
            'max_price': 1000.0,        # Maximum stock price
            'min_expected_return': 0.05, # Minimum 5% expected return
            'min_confidence': 0.6,      # Minimum confidence score
            'sectors_exclude': [],      # Sectors to exclude
            'regions_include': ['US'],  # Regions to include
        }
    
    def apply_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all trading criteria filters to DataFrame.
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            Filtered DataFrame meeting all criteria
        """
        if df.empty:
            return df
        
        try:
            initial_count = len(df)
            filtered_df = df.copy()
            
            # Apply each filter
            filtered_df = self._filter_by_market_cap(filtered_df)
            filtered_df = self._filter_by_pe_ratio(filtered_df)
            filtered_df = self._filter_by_volume(filtered_df)
            filtered_df = self._filter_by_beta(filtered_df)
            filtered_df = self._filter_by_price_range(filtered_df)
            filtered_df = self._filter_by_expected_return(filtered_df)
            filtered_df = self._filter_by_confidence(filtered_df)
            filtered_df = self._filter_by_sectors(filtered_df)
            filtered_df = self._filter_by_regions(filtered_df)
            
            final_count = len(filtered_df)
            filtered_count = initial_count - final_count
            
            self.logger.info(f"Applied trading criteria: {initial_count} → {final_count} "
                           f"({filtered_count} filtered out)")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error applying trading criteria: {str(e)}")
            raise TradingFilterError(f"Criteria filtering failed: {str(e)}") from e
    
    def _filter_by_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by minimum market cap."""
        if 'market_cap' not in df.columns:
            return df
        
        min_cap = self.criteria.get('min_market_cap', 0)
        if min_cap > 0:
            initial_count = len(df)
            mask = (df['market_cap'] >= min_cap) | df['market_cap'].isna()
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Market cap filter: removed {filtered_count} tickers")
        
        return df
    
    def _filter_by_pe_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by maximum P/E ratio."""
        if 'pe_ratio' not in df.columns:
            return df
        
        max_pe = self.criteria.get('max_pe_ratio', float('inf'))
        if max_pe < float('inf'):
            initial_count = len(df)
            mask = (df['pe_ratio'] <= max_pe) | df['pe_ratio'].isna()
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"P/E ratio filter: removed {filtered_count} tickers")
        
        return df
    
    def _filter_by_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by minimum volume."""
        if 'volume' not in df.columns:
            return df
        
        min_vol = self.criteria.get('min_volume', 0)
        if min_vol > 0:
            initial_count = len(df)
            mask = (df['volume'] >= min_vol) | df['volume'].isna()
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Volume filter: removed {filtered_count} tickers")
        
        return df
    
    def _filter_by_beta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by maximum beta."""
        if 'beta' not in df.columns:
            return df
        
        max_beta = self.criteria.get('max_beta', float('inf'))
        if max_beta < float('inf'):
            initial_count = len(df)
            mask = (df['beta'] <= max_beta) | df['beta'].isna()
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Beta filter: removed {filtered_count} tickers")
        
        return df
    
    def _filter_by_price_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by price range."""
        if 'price' not in df.columns:
            return df
        
        min_price = self.criteria.get('min_price', 0)
        max_price = self.criteria.get('max_price', float('inf'))
        
        if min_price > 0 or max_price < float('inf'):
            initial_count = len(df)
            mask = ((df['price'] >= min_price) & (df['price'] <= max_price)) | df['price'].isna()
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Price range filter: removed {filtered_count} tickers")
        
        return df
    
    def _filter_by_expected_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by minimum expected return."""
        if 'expected_return' not in df.columns:
            return df
        
        min_return = self.criteria.get('min_expected_return', 0)
        if min_return > 0:
            initial_count = len(df)
            mask = (df['expected_return'] >= min_return) | df['expected_return'].isna()
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Expected return filter: removed {filtered_count} tickers")
        
        return df
    
    def _filter_by_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by minimum confidence score."""
        if 'confidence_score' not in df.columns:
            return df
        
        min_confidence = self.criteria.get('min_confidence', 0)
        if min_confidence > 0:
            initial_count = len(df)
            mask = (df['confidence_score'] >= min_confidence) | df['confidence_score'].isna()
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Confidence filter: removed {filtered_count} tickers")
        
        return df
    
    def _filter_by_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by excluded sectors."""
        if 'sector' not in df.columns:
            return df
        
        excluded_sectors = self.criteria.get('sectors_exclude', [])
        if excluded_sectors:
            initial_count = len(df)
            mask = ~df['sector'].isin(excluded_sectors)
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Sector filter: removed {filtered_count} tickers")
        
        return df
    
    def _filter_by_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by included regions."""
        if 'region' not in df.columns:
            return df
        
        included_regions = self.criteria.get('regions_include', [])
        if included_regions:
            initial_count = len(df)
            mask = df['region'].isin(included_regions)
            df = df[mask]
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Region filter: removed {filtered_count} tickers")
        
        return df


class PortfolioFilter:
    """Filter data based on portfolio holdings and constraints."""
    
    def __init__(self, portfolio_df: Optional[pd.DataFrame] = None):
        """
        Initialize portfolio filter.
        
        Args:
            portfolio_df: DataFrame containing current portfolio holdings
        """
        self.portfolio_df = portfolio_df
        self.portfolio_tickers = self._extract_portfolio_tickers()
        self.logger = logger
    
    def _extract_portfolio_tickers(self) -> Set[str]:
        """Extract ticker symbols from portfolio DataFrame."""
        tickers = set()
        
        if self.portfolio_df is not None and not self.portfolio_df.empty:
            # Try different possible column names for tickers
            for col in ['Ticker', 'ticker', 'Symbol', 'symbol']:
                if col in self.portfolio_df.columns:
                    tickers.update(self.portfolio_df[col].str.upper().tolist())
                    break
        
        return tickers
    
    def filter_new_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for new investment opportunities (not in portfolio).
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame containing only tickers not in portfolio
        """
        if df.empty or not self.portfolio_tickers:
            return df
        
        try:
            initial_count = len(df)
            
            # Filter out tickers already in portfolio
            ticker_index = df.index.str.upper() if hasattr(df.index, 'str') else df.index
            mask = ~ticker_index.isin(self.portfolio_tickers)
            filtered_df = df[mask]
            
            filtered_count = initial_count - len(filtered_df)
            self.logger.info(f"Portfolio filter (new): {initial_count} → {len(filtered_df)} "
                           f"({filtered_count} already in portfolio)")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error filtering new opportunities: {str(e)}")
            return df
    
    def filter_existing_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for existing portfolio holdings only.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame containing only tickers in portfolio
        """
        if df.empty or not self.portfolio_tickers:
            return pd.DataFrame()
        
        try:
            initial_count = len(df)
            
            # Filter for tickers in portfolio
            ticker_index = df.index.str.upper() if hasattr(df.index, 'str') else df.index
            mask = ticker_index.isin(self.portfolio_tickers)
            filtered_df = df[mask]
            
            self.logger.info(f"Portfolio filter (existing): {initial_count} → {len(filtered_df)} "
                           f"(portfolio holdings only)")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error filtering existing holdings: {str(e)}")
            return df
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio metrics and statistics."""
        if self.portfolio_df is None or self.portfolio_df.empty:
            return {}
        
        try:
            metrics = {
                'total_holdings': len(self.portfolio_tickers),
                'unique_tickers': list(self.portfolio_tickers),
            }
            
            # Add value-based metrics if available
            if 'value' in self.portfolio_df.columns:
                metrics['total_value'] = self.portfolio_df['value'].sum()
                metrics['avg_position_size'] = self.portfolio_df['value'].mean()
                metrics['largest_position'] = self.portfolio_df['value'].max()
                metrics['smallest_position'] = self.portfolio_df['value'].min()
            
            # Add quantity-based metrics if available
            if 'quantity' in self.portfolio_df.columns:
                metrics['total_shares'] = self.portfolio_df['quantity'].sum()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}


class DataQualityFilter:
    """Filter data based on quality and completeness criteria."""
    
    def __init__(self, min_completeness: float = 0.7):
        """
        Initialize data quality filter.
        
        Args:
            min_completeness: Minimum data completeness ratio (0.0 to 1.0)
        """
        self.min_completeness = min_completeness
        self.logger = logger
    
    def filter_by_data_quality(self, df: pd.DataFrame, 
                             required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter DataFrame based on data quality criteria.
        
        Args:
            df: Input DataFrame
            required_columns: List of columns that must have data
            
        Returns:
            Filtered DataFrame with good data quality
        """
        if df.empty:
            return df
        
        try:
            initial_count = len(df)
            filtered_df = df.copy()
            
            # Filter by required columns
            if required_columns:
                filtered_df = self._filter_by_required_columns(filtered_df, required_columns)
            
            # Filter by overall completeness
            filtered_df = self._filter_by_completeness(filtered_df)
            
            # Filter out obvious data errors
            filtered_df = self._filter_data_errors(filtered_df)
            
            final_count = len(filtered_df)
            filtered_count = initial_count - final_count
            
            self.logger.info(f"Data quality filter: {initial_count} → {final_count} "
                           f"({filtered_count} removed for poor quality)")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error filtering by data quality: {str(e)}")
            return df
    
    def _filter_by_required_columns(self, df: pd.DataFrame, 
                                  required_columns: List[str]) -> pd.DataFrame:
        """Filter rows that have data in all required columns."""
        for col in required_columns:
            if col in df.columns:
                initial_count = len(df)
                df = df[df[col].notna()]
                filtered_count = initial_count - len(df)
                self.logger.debug(f"Required column '{col}': removed {filtered_count} rows")
        
        return df
    
    def _filter_by_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter rows with insufficient data completeness."""
        if self.min_completeness > 0:
            initial_count = len(df)
            
            # Calculate completeness ratio for each row
            completeness = df.notna().sum(axis=1) / len(df.columns)
            mask = completeness >= self.min_completeness
            df = df[mask]
            
            filtered_count = initial_count - len(df)
            self.logger.debug(f"Completeness filter: removed {filtered_count} rows")
        
        return df
    
    def _filter_data_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out obvious data errors."""
        initial_count = len(df)
        
        # Filter negative prices
        if 'price' in df.columns:
            df = df[(df['price'] > 0) | df['price'].isna()]
        
        # Filter unrealistic P/E ratios
        if 'pe_ratio' in df.columns:
            df = df[(df['pe_ratio'] > 0) & (df['pe_ratio'] < 1000) | df['pe_ratio'].isna()]
        
        # Filter negative market cap
        if 'market_cap' in df.columns:
            df = df[(df['market_cap'] > 0) | df['market_cap'].isna()]
        
        # Filter unrealistic beta values
        if 'beta' in df.columns:
            df = df[(df['beta'] > -5) & (df['beta'] < 5) | df['beta'].isna()]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            self.logger.debug(f"Data error filter: removed {filtered_count} rows")
        
        return df


class CustomFilter:
    """Custom filter class for user-defined filtering logic."""
    
    def __init__(self):
        """Initialize custom filter."""
        self.filters = []
        self.logger = logger
    
    def add_filter(self, filter_func: Callable[[pd.DataFrame], pd.DataFrame], 
                  name: str = None) -> None:
        """
        Add a custom filter function.
        
        Args:
            filter_func: Function that takes DataFrame and returns filtered DataFrame
            name: Optional name for the filter
        """
        filter_info = {
            'function': filter_func,
            'name': name or f"custom_filter_{len(self.filters) + 1}"
        }
        self.filters.append(filter_info)
        self.logger.debug(f"Added custom filter: {filter_info['name']}")
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all custom filters in sequence."""
        if df.empty or not self.filters:
            return df
        
        try:
            filtered_df = df.copy()
            initial_count = len(filtered_df)
            
            for filter_info in self.filters:
                try:
                    before_count = len(filtered_df)
                    filtered_df = filter_info['function'](filtered_df)
                    after_count = len(filtered_df)
                    removed_count = before_count - after_count
                    
                    self.logger.debug(f"Custom filter '{filter_info['name']}': "
                                    f"removed {removed_count} rows")
                    
                except Exception as e:
                    self.logger.error(f"Error in custom filter '{filter_info['name']}': {str(e)}")
                    continue
            
            final_count = len(filtered_df)
            total_removed = initial_count - final_count
            
            self.logger.info(f"Custom filters: {initial_count} → {final_count} "
                           f"({total_removed} total removed)")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error applying custom filters: {str(e)}")
            return df


class TradingFilterError(YFinanceError):
    """Custom exception for trading filter errors."""
    pass


def create_criteria_filter(criteria_config: Optional[Dict] = None) -> TradingCriteriaFilter:
    """Factory function to create a trading criteria filter."""
    return TradingCriteriaFilter(criteria_config)


def create_portfolio_filter(portfolio_df: Optional[pd.DataFrame] = None) -> PortfolioFilter:
    """Factory function to create a portfolio filter."""
    return PortfolioFilter(portfolio_df)


def create_quality_filter(min_completeness: float = 0.7) -> DataQualityFilter:
    """Factory function to create a data quality filter."""
    return DataQualityFilter(min_completeness)


def create_custom_filter() -> CustomFilter:
    """Factory function to create a custom filter."""
    return CustomFilter()