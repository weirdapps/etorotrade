"""
Yahoo Finance boundary interface for trade modules.

This module provides a stable interface for trade modules to access
yahoofinance functionality without direct coupling.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging
from abc import ABC, abstractmethod

from ..errors import DataProcessingError


logger = logging.getLogger(__name__)


class IYahooFinanceBoundary(ABC):
    """Interface for Yahoo Finance operations accessed by trade modules."""
    
    @abstractmethod
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance."""
        pass
    
    @abstractmethod
    def get_base_config(self) -> Dict[str, Any]:
        """Get base configuration."""
        pass
    
    @abstractmethod
    def get_ticker_utils(self) -> Dict[str, Any]:
        """Get ticker utility functions."""
        pass
    
    @abstractmethod
    def get_error_classes(self) -> Dict[str, Any]:
        """Get error classes."""
        pass
    
    @abstractmethod
    def get_data_utilities(self) -> Dict[str, Any]:
        """Get data processing utilities."""
        pass


class YahooFinanceBoundary(IYahooFinanceBoundary):
    """
    Concrete implementation of Yahoo Finance boundary.
    
    This class provides controlled access to yahoofinance functionality
    while maintaining clean module boundaries.
    """
    
    def __init__(self):
        """Initialize the boundary."""
        self._logger_cache = {}
        self._config_cache = None
        self._ticker_utils_cache = None
        self._error_classes_cache = None
        self._data_utils_cache = None
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in self._logger_cache:
            try:
                from yahoofinance.core.logging import get_logger
                self._logger_cache[name] = get_logger(name)
            except ImportError:
                # Fallback to standard logging
                self._logger_cache[name] = logging.getLogger(name)
        
        return self._logger_cache[name]
    
    def get_base_config(self) -> Dict[str, Any]:
        """
        Get base configuration.
        
        Returns:
            Dictionary with base configuration
        """
        if self._config_cache is None:
            try:
                from yahoofinance.core.config import get_base_config  # type: ignore[attr-defined]
                self._config_cache = get_base_config()
            except ImportError:
                logger.warning("Could not import yahoofinance config, using defaults")
                self._config_cache = self._get_default_config()
        
        return self._config_cache
    
    def get_ticker_utils(self) -> Dict[str, Any]:
        """
        Get ticker utility functions.
        
        Returns:
            Dictionary with ticker utility functions
        """
        if self._ticker_utils_cache is None:
            try:
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
                    get_ticker_for_data_fetch,
                    are_equivalent_tickers
                )
                
                self._ticker_utils_cache = {
                    'normalize_ticker': normalize_ticker,
                    'process_ticker_input': process_ticker_input,
                    'get_ticker_for_display': get_ticker_for_display,
                    'standardize_ticker_format': standardize_ticker_format,
                    'validate_ticker_format': validate_ticker_format,
                    'get_ticker_exchange_suffix': get_ticker_exchange_suffix,
                    'get_all_ticker_variants': get_all_ticker_variants,
                    'get_ticker_info_summary': get_ticker_info_summary,
                    'check_equivalent_tickers': check_equivalent_tickers,
                    'get_ticker_equivalents': get_ticker_equivalents,
                    'is_ticker_dual_listed': is_ticker_dual_listed,
                    'get_geographic_region': get_geographic_region,
                    'get_ticker_for_data_fetch': get_ticker_for_data_fetch,
                    'are_equivalent_tickers': are_equivalent_tickers
                }
            except ImportError as e:
                logger.warning(f"Could not import yahoofinance ticker utils: {e}")
                self._ticker_utils_cache = self._get_fallback_ticker_utils()
        
        return self._ticker_utils_cache
    
    def get_error_classes(self) -> Dict[str, Any]:
        """
        Get error classes.
        
        Returns:
            Dictionary with error classes
        """
        if self._error_classes_cache is None:
            try:
                from yahoofinance.core.errors import (  # type: ignore[attr-defined]
                    YFinanceError,
                    ValidationError,
                    DataError,
                    ConfigurationError as YFConfigurationError
                )
                
                self._error_classes_cache = {
                    'YFinanceError': YFinanceError,
                    'ValidationError': ValidationError,
                    'DataError': DataError,
                    'ConfigurationError': YFConfigurationError
                }
            except ImportError:
                logger.warning("Could not import yahoofinance errors, using fallbacks")
                self._error_classes_cache = self._get_fallback_error_classes()
        
        return self._error_classes_cache
    
    def get_data_utilities(self) -> Dict[str, Any]:
        """
        Get data processing utilities.
        
        Returns:
            Dictionary with data utility functions
        """
        if self._data_utils_cache is None:
            try:
                from yahoofinance.utils.data.format_utils import (  # type: ignore[attr-defined]
                    safe_float_conversion,
                    format_currency,
                    format_percentage,
                    format_large_number
                )
                
                self._data_utils_cache = {
                    'safe_float_conversion': safe_float_conversion,
                    'format_currency': format_currency,
                    'format_percentage': format_percentage,
                    'format_large_number': format_large_number
                }
            except ImportError:
                logger.warning("Could not import yahoofinance data utils, using fallbacks")
                self._data_utils_cache = self._get_fallback_data_utils()
        
        return self._data_utils_cache
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when yahoofinance config is unavailable."""
        return {
            'logging_level': 'INFO',
            'cache_enabled': True,
            'timeout': 30,
            'max_retries': 3
        }
    
    def _get_fallback_ticker_utils(self) -> Dict[str, Any]:
        """Get fallback ticker utilities when yahoofinance utils are unavailable."""
        def normalize_ticker_fallback(ticker: str) -> str:
            """Fallback ticker normalization."""
            if not ticker:
                return ticker
            return ticker.upper().strip()
        
        def process_ticker_input_fallback(ticker: str) -> str:
            """Fallback ticker input processing."""
            return normalize_ticker_fallback(ticker)
        
        def get_ticker_for_display_fallback(ticker: str) -> str:
            """Fallback ticker display formatting."""
            return normalize_ticker_fallback(ticker)
        
        def are_equivalent_tickers_fallback(ticker1: str, ticker2: str) -> bool:
            """Fallback ticker equivalence check."""
            return normalize_ticker_fallback(ticker1) == normalize_ticker_fallback(ticker2)
        
        return {
            'normalize_ticker': normalize_ticker_fallback,
            'process_ticker_input': process_ticker_input_fallback,
            'get_ticker_for_display': get_ticker_for_display_fallback,
            'are_equivalent_tickers': are_equivalent_tickers_fallback,
            'check_equivalent_tickers': are_equivalent_tickers_fallback,
            # Provide minimal implementations for other functions
            'standardize_ticker_format': normalize_ticker_fallback,
            'validate_ticker_format': lambda x: bool(x and x.strip()),
            'get_ticker_exchange_suffix': lambda x: '',
            'get_all_ticker_variants': lambda x: {x},
            'get_ticker_info_summary': lambda x: {'ticker': x},
            'get_ticker_equivalents': lambda x: {x},
            'is_ticker_dual_listed': lambda x: False,
            'get_geographic_region': lambda x: 'Unknown',
            'get_ticker_for_data_fetch': normalize_ticker_fallback
        }
    
    def _get_fallback_error_classes(self) -> Dict[str, Any]:
        """Get fallback error classes when yahoofinance errors are unavailable."""
        return {
            'YFinanceError': Exception,
            'ValidationError': ValueError,
            'DataError': DataProcessingError,
            'ConfigurationError': RuntimeError
        }
    
    def _get_fallback_data_utils(self) -> Dict[str, Any]:
        """Get fallback data utilities when yahoofinance utils are unavailable."""
        def safe_float_conversion_fallback(value: Any) -> float:
            """Fallback safe float conversion."""
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        def format_currency_fallback(value: float) -> str:
            """Fallback currency formatting."""
            return f"${value:,.2f}"
        
        def format_percentage_fallback(value: float) -> str:
            """Fallback percentage formatting."""
            return f"{value:.2f}%"
        
        def format_large_number_fallback(value: float) -> str:
            """Fallback large number formatting."""
            if value >= 1e9:
                return f"{value/1e9:.1f}B"
            elif value >= 1e6:
                return f"{value/1e6:.1f}M"
            elif value >= 1e3:
                return f"{value/1e3:.1f}K"
            else:
                return f"{value:.0f}"
        
        return {
            'safe_float_conversion': safe_float_conversion_fallback,
            'format_currency': format_currency_fallback,
            'format_percentage': format_percentage_fallback,
            'format_large_number': format_large_number_fallback
        }


# Create default boundary instance
default_yahoo_finance_boundary = YahooFinanceBoundary()