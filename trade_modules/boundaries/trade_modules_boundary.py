"""
Trade modules boundary interface for yahoofinance access.

This module provides a stable interface for yahoofinance to access
trade modules functionality without direct coupling.
"""

from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
import logging
from abc import ABC, abstractmethod

from ..errors import TradeModuleError


logger = logging.getLogger(__name__)


class ITradeModulesBoundary(ABC):
    """Interface for trade modules operations accessed by yahoofinance."""
    
    @abstractmethod
    def get_trading_filters(self) -> Dict[str, Any]:
        """Get trading filter classes and functions."""
        pass
    
    @abstractmethod
    def get_trade_criteria(self) -> Dict[str, Any]:
        """Get trade criteria functions."""
        pass
    
    @abstractmethod
    def get_utils(self) -> Dict[str, Any]:
        """Get utility functions."""
        pass
    
    @abstractmethod
    def get_error_classes(self) -> Dict[str, Any]:
        """Get error classes."""
        pass


class TradeModulesBoundary(ITradeModulesBoundary):
    """
    Concrete implementation of trade modules boundary.
    
    This class provides controlled access to trade modules functionality
    while maintaining clean module boundaries.
    """
    
    def __init__(self):
        """Initialize the boundary."""
        self._filters_cache = None
        self._criteria_cache = None
        self._utils_cache = None
        self._error_classes_cache = None
    
    def get_trading_filters(self) -> Dict[str, Any]:
        """
        Get trading filter classes and functions.
        
        Returns:
            Dictionary with filter classes and functions
        """
        if self._filters_cache is None:
            try:
                from ..trade_filters import (
                    TradingCriteriaFilter,
                    PortfolioFilter,
                    DataQualityFilter,
                    CustomFilter,
                    create_criteria_filter,
                    create_portfolio_filter,
                    create_quality_filter,
                    create_custom_filter
                )
                
                self._filters_cache = {
                    'TradingCriteriaFilter': TradingCriteriaFilter,
                    'PortfolioFilter': PortfolioFilter,
                    'DataQualityFilter': DataQualityFilter,
                    'CustomFilter': CustomFilter,
                    'create_criteria_filter': create_criteria_filter,
                    'create_portfolio_filter': create_portfolio_filter,
                    'create_quality_filter': create_quality_filter,
                    'create_custom_filter': create_custom_filter
                }
            except ImportError as e:
                logger.warning(f"Could not import trade filters: {e}")
                self._filters_cache = {}
        
        return self._filters_cache
    
    def get_trade_criteria(self) -> Dict[str, Any]:
        """
        Get trade criteria functions.
        
        Returns:
            Dictionary with trade criteria functions
        """
        if self._criteria_cache is None:
            try:
                from ..utils import (  # type: ignore[attr-defined]
                    apply_buy_criteria,
                    apply_sell_criteria,
                    calculate_position_size,
                    evaluate_risk_metrics
                )
                
                self._criteria_cache = {
                    'apply_buy_criteria': apply_buy_criteria,
                    'apply_sell_criteria': apply_sell_criteria,
                    'calculate_position_size': calculate_position_size,
                    'evaluate_risk_metrics': evaluate_risk_metrics
                }
            except ImportError as e:
                logger.warning(f"Could not import trade criteria: {e}")
                self._criteria_cache = self._get_fallback_criteria()
        
        return self._criteria_cache
    
    def get_utils(self) -> Dict[str, Any]:
        """
        Get utility functions.
        
        Returns:
            Dictionary with utility functions
        """
        if self._utils_cache is None:
            try:
                from ..utils import (  # type: ignore[attr-defined]
                    clean_ticker_symbol,
                    safe_float_conversion,
                    validate_dataframe,
                    normalize_ticker_for_display,
                    calculate_percentage_change,
                    format_number_display,
                    sanitize_filename
                )
                
                self._utils_cache = {
                    'clean_ticker_symbol': clean_ticker_symbol,
                    'safe_float_conversion': safe_float_conversion,
                    'validate_dataframe': validate_dataframe,
                    'normalize_ticker_for_display': normalize_ticker_for_display,
                    'calculate_percentage_change': calculate_percentage_change,
                    'format_number_display': format_number_display,
                    'sanitize_filename': sanitize_filename
                }
            except ImportError as e:
                logger.warning(f"Could not import trade utils: {e}")
                self._utils_cache = self._get_fallback_utils()
        
        return self._utils_cache
    
    def get_error_classes(self) -> Dict[str, Any]:
        """
        Get error classes.
        
        Returns:
            Dictionary with error classes
        """
        if self._error_classes_cache is None:
            try:
                from ..errors import (
                    TradeModuleError,
                    TradingEngineError,
                    TradingFilterError,
                    DataProcessingError,
                    ConfigurationError
                )
                
                self._error_classes_cache = {
                    'TradeModuleError': TradeModuleError,
                    'TradingEngineError': TradingEngineError,
                    'TradingFilterError': TradingFilterError,
                    'DataProcessingError': DataProcessingError,
                    'ConfigurationError': ConfigurationError
                }
            except ImportError:
                logger.warning("Could not import trade errors, using fallbacks")
                self._error_classes_cache = self._get_fallback_error_classes()
        
        return self._error_classes_cache
    
    def _get_fallback_criteria(self) -> Dict[str, Any]:
        """Get fallback criteria functions when trade criteria are unavailable."""
        def apply_buy_criteria_fallback(df: pd.DataFrame, criteria: Dict) -> pd.DataFrame:
            """Fallback buy criteria application."""
            logger.warning("Using fallback buy criteria")
            return df
        
        def apply_sell_criteria_fallback(df: pd.DataFrame, criteria: Dict) -> pd.DataFrame:
            """Fallback sell criteria application."""
            logger.warning("Using fallback sell criteria")
            return df
        
        def calculate_position_size_fallback(price: float, risk: float) -> float:
            """Fallback position size calculation."""
            return min(1000.0, 10000.0 / price) if price > 0 else 0.0
        
        def evaluate_risk_metrics_fallback(df: pd.DataFrame) -> Dict[str, float]:
            """Fallback risk metrics evaluation."""
            return {'risk_score': 0.5, 'volatility': 0.2}
        
        return {
            'apply_buy_criteria': apply_buy_criteria_fallback,
            'apply_sell_criteria': apply_sell_criteria_fallback,
            'calculate_position_size': calculate_position_size_fallback,
            'evaluate_risk_metrics': evaluate_risk_metrics_fallback
        }
    
    def _get_fallback_utils(self) -> Dict[str, Any]:
        """Get fallback utilities when trade utils are unavailable."""
        def clean_ticker_symbol_fallback(ticker: str) -> str:
            """Fallback ticker cleaning."""
            return ticker.upper().strip() if ticker else ''
        
        def safe_float_conversion_fallback(value: Any) -> float:
            """Fallback safe float conversion."""
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        def validate_dataframe_fallback(df: pd.DataFrame) -> bool:
            """Fallback DataFrame validation."""
            return isinstance(df, pd.DataFrame) and not df.empty
        
        def normalize_ticker_for_display_fallback(ticker: str) -> str:
            """Fallback ticker display normalization."""
            return clean_ticker_symbol_fallback(ticker)
        
        def calculate_percentage_change_fallback(old: float, new: float) -> float:
            """Fallback percentage change calculation."""
            if old == 0:
                return 0.0
            return ((new - old) / old) * 100
        
        def format_number_display_fallback(value: float) -> str:
            """Fallback number display formatting."""
            return f"{value:,.2f}"
        
        def sanitize_filename_fallback(filename: str) -> str:
            """Fallback filename sanitization."""
            import re
            return re.sub(r'[^\w\-_\.]', '_', filename)
        
        return {
            'clean_ticker_symbol': clean_ticker_symbol_fallback,
            'safe_float_conversion': safe_float_conversion_fallback,
            'validate_dataframe': validate_dataframe_fallback,
            'normalize_ticker_for_display': normalize_ticker_for_display_fallback,
            'calculate_percentage_change': calculate_percentage_change_fallback,
            'format_number_display': format_number_display_fallback,
            'sanitize_filename': sanitize_filename_fallback
        }
    
    def _get_fallback_error_classes(self) -> Dict[str, Any]:
        """Get fallback error classes when trade errors are unavailable."""
        return {
            'TradeModuleError': Exception,
            'TradingEngineError': RuntimeError,
            'TradingFilterError': ValueError,
            'DataProcessingError': Exception,
            'ConfigurationError': RuntimeError
        }


# Create default boundary instance
default_trade_modules_boundary = TradeModulesBoundary()