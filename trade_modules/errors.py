"""
Trade modules error hierarchy for consolidated error handling.

This module consolidates trade-specific exceptions while maintaining
backward compatibility with existing error handling patterns.
"""

from yahoofinance.core.errors import YFinanceError


class TradeModuleError(YFinanceError):
    """
    Base class for all trade module errors.
    
    This provides a unified error hierarchy for all trade-specific
    operations while maintaining compatibility with the existing
    YFinanceError system.
    """
    pass


class TradingEngineError(TradeModuleError):
    """
    Error specific to trading engine operations.
    
    This maintains compatibility with existing TradingEngineError
    usage while consolidating into the unified error hierarchy.
    """
    pass


class TradingFilterError(TradeModuleError):
    """
    Error specific to trading filter operations.
    
    This maintains compatibility with existing TradingFilterError
    usage while consolidating into the unified error hierarchy.
    """
    pass


class DataProcessingError(TradeModuleError):
    """
    Error specific to data processing operations.
    
    New unified error type for data processing issues across
    all trade modules.
    """
    pass


class ConfigurationError(TradeModuleError):
    """
    Error specific to trade configuration issues.
    
    New unified error type for configuration problems across
    all trade modules.
    """
    pass


# Backward compatibility aliases - these ensure existing code continues to work
TradingError = TradingEngineError  # Legacy alias
FilterError = TradingFilterError   # Legacy alias