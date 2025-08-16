"""
Unified ticker service for consolidated ticker operations.

This service provides a single point of access for all ticker-related
operations while maintaining backward compatibility with existing
ticker utilities throughout the codebase.
"""

from typing import List, Dict, Any, Optional, Set
import pandas as pd
import logging

# Import existing ticker utilities - maintain all existing functionality
from yahoofinance.utils.data.ticker_utils import (
    normalize_ticker,
    process_ticker_input,
    get_ticker_for_display,
    standardize_ticker_format,
    validate_ticker_format,
    get_ticker_exchange_suffix,
    get_all_ticker_variants,
    get_ticker_info_summary,
    check_equivalent_tickers,
    get_ticker_equivalents,
    is_ticker_dual_listed,
    get_geographic_region,
    get_ticker_for_data_fetch
)

# Import error types
from ..errors import DataProcessingError

logger = logging.getLogger(__name__)


class TickerService:
    """
    Unified service for ticker operations across trade modules.
    
    This service consolidates ticker-related operations while maintaining
    full backward compatibility with existing ticker utilities. It acts
    as a facade over the existing ticker_utils system.
    """
    
    def __init__(self):
        """Initialize the ticker service."""
        pass
    
    def normalize(self, ticker: str) -> str:
        """
        Normalize a ticker symbol to its canonical form.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Normalized ticker symbol
        """
        try:
            return normalize_ticker(ticker)
        except Exception as e:
            logger.error(f"Error normalizing ticker {ticker}: {e}")
            raise DataProcessingError(f"Failed to normalize ticker {ticker}") from e
    
    def process_input(self, ticker: str) -> str:
        """
        Process ticker input through complete normalization pipeline.
        
        Args:
            ticker: Raw ticker input
            
        Returns:
            Processed and normalized ticker symbol
        """
        try:
            return process_ticker_input(ticker)
        except Exception as e:
            logger.error(f"Error processing ticker input {ticker}: {e}")
            raise DataProcessingError(f"Failed to process ticker input {ticker}") from e
    
    def get_display_format(self, ticker: str) -> str:
        """
        Get ticker formatted for display purposes.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Display-formatted ticker symbol
        """
        try:
            return get_ticker_for_display(ticker)
        except Exception as e:
            logger.error(f"Error formatting ticker for display {ticker}: {e}")
            raise DataProcessingError(f"Failed to format ticker for display {ticker}") from e
    
    def validate_format(self, ticker: str) -> bool:
        """
        Validate ticker format.
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        try:
            return validate_ticker_format(ticker)
        except Exception as e:
            logger.error(f"Error validating ticker format {ticker}: {e}")
            return False
    
    def normalize_list(self, tickers: List[str]) -> List[str]:
        """
        Normalize a list of ticker symbols.
        
        Args:
            tickers: List of ticker symbols to normalize
            
        Returns:
            List of normalized ticker symbols
        """
        normalized = []
        for ticker in tickers:
            if ticker and ticker.strip():
                try:
                    normalized_ticker = self.normalize(ticker)
                    if self.validate_format(normalized_ticker):
                        normalized.append(normalized_ticker)
                    else:
                        logger.warning(f"Invalid ticker format after normalization: {ticker}")
                except Exception as e:
                    logger.warning(f"Failed to normalize ticker {ticker}: {e}")
        return normalized
    
    def normalize_dataframe_column(self, df: pd.DataFrame, ticker_column: str = 'ticker') -> pd.DataFrame:
        """
        Normalize ticker symbols in a DataFrame column.
        
        Args:
            df: DataFrame containing ticker symbols
            ticker_column: Name of the ticker column
            
        Returns:
            DataFrame with normalized ticker symbols
        """
        if df is None or df.empty or ticker_column not in df.columns:
            return df
        
        try:
            df = df.copy()
            df[ticker_column] = df[ticker_column].apply(
                lambda x: self.process_input(x) if pd.notna(x) and x else x
            )
            return df
        except Exception as e:
            logger.error(f"Error normalizing DataFrame ticker column {ticker_column}: {e}")
            raise DataProcessingError(f"Failed to normalize DataFrame ticker column {ticker_column}") from e
    
    def get_equivalents(self, ticker: str) -> Set[str]:
        """
        Get all equivalent ticker variants for the same underlying asset.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Set of equivalent ticker symbols
        """
        try:
            return get_ticker_equivalents(ticker)
        except Exception as e:
            logger.error(f"Error getting ticker equivalents for {ticker}: {e}")
            raise DataProcessingError(f"Failed to get ticker equivalents for {ticker}") from e
    
    def are_equivalent(self, ticker1: str, ticker2: str) -> bool:
        """
        Check if two tickers represent the same underlying asset.
        
        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            
        Returns:
            True if tickers are equivalent, False otherwise
        """
        try:
            return check_equivalent_tickers(ticker1, ticker2)
        except Exception as e:
            logger.error(f"Error checking ticker equivalence {ticker1} vs {ticker2}: {e}")
            return False
    
    def get_info_summary(self, ticker: str) -> Dict[str, str]:
        """
        Get comprehensive summary information about a ticker.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Dictionary with ticker information
        """
        try:
            return get_ticker_info_summary(ticker)
        except Exception as e:
            logger.error(f"Error getting ticker info summary for {ticker}: {e}")
            raise DataProcessingError(f"Failed to get ticker info summary for {ticker}") from e
    
    def get_geographic_region(self, ticker: str) -> str:
        """
        Get the geographic region for a ticker symbol.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Geographic region code
        """
        try:
            return get_geographic_region(ticker)
        except Exception as e:
            logger.error(f"Error getting geographic region for {ticker}: {e}")
            raise DataProcessingError(f"Failed to get geographic region for {ticker}") from e
    
    def is_dual_listed(self, ticker: str) -> bool:
        """
        Check if a ticker has dual listings on multiple exchanges.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            True if ticker has dual listings, False otherwise
        """
        try:
            return is_ticker_dual_listed(ticker)
        except Exception as e:
            logger.error(f"Error checking dual listing status for {ticker}: {e}")
            return False


# Create a default instance for convenience
default_ticker_service = TickerService()

# Convenience functions that use the default service instance
def normalize_ticker_safe(ticker: str) -> str:
    """Safely normalize a ticker using the default service."""
    return default_ticker_service.normalize(ticker)

def normalize_ticker_list_safe(tickers: List[str]) -> List[str]:
    """Safely normalize a list of tickers using the default service."""
    return default_ticker_service.normalize_list(tickers)

def check_ticker_equivalence_safe(ticker1: str, ticker2: str) -> bool:
    """Safely check ticker equivalence using the default service."""
    return default_ticker_service.are_equivalent(ticker1, ticker2)