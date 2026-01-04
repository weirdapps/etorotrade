"""
Configuration interfaces for dependency injection pattern.

This module defines abstract interfaces to break circular dependencies
between trade_modules and yahoofinance configurations.

SAFETY: Creates new interfaces without changing existing code.
ZERO RISK: All existing imports continue to work unchanged.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum


class TradeAction(Enum):
    """Trading actions."""
    BUY = "B"
    SELL = "S"
    HOLD = "H"
    INCONCLUSIVE = "I"


class IConfigProvider(ABC):
    """Interface for configuration providers."""
    
    @abstractmethod
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting."""
        pass
    
    @abstractmethod
    def get_file_path(self, file_key: str) -> str:
        """Get a file path from configuration."""
        pass


class ITradingCriteriaProvider(ABC):
    """Interface for trading criteria providers."""
    
    @abstractmethod
    def get_thresholds(self, option: str, action: str, tier: Optional[str] = None) -> Dict[str, Any]:
        """Get trading thresholds for specific option and action."""
        pass
    
    @abstractmethod
    def get_universal_thresholds(self) -> Dict[str, Any]:
        """Get universal thresholds applied to all options."""
        pass
    
    @abstractmethod
    def get_tier_thresholds(self, tier: str, action: str) -> Dict[str, Any]:
        """Get tier-specific thresholds."""
        pass


class IDisplayProvider(ABC):
    """Interface for display configuration providers."""
    
    @abstractmethod
    def get_display_columns(self, option: str, sub_option: Optional[str] = None,
                          output_type: str = "console") -> List[str]:
        """Get display columns for specific option and output type."""
        pass

    @abstractmethod
    def get_sort_config(self, option: str, sub_option: Optional[str] = None) -> Dict[str, str]:
        """Get sorting configuration."""
        pass
    
    @abstractmethod
    def get_format_rule(self, column: str) -> Dict[str, Any]:
        """Get formatting rule for a column."""
        pass


class IRateLimitProvider(ABC):
    """Interface for rate limiting configuration providers."""
    
    @abstractmethod
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        pass
    
    @abstractmethod
    def get_concurrent_limits(self) -> Dict[str, int]:
        """Get concurrent request limits."""
        pass


class IPathProvider(ABC):
    """Interface for path configuration providers."""
    
    @abstractmethod
    def get_input_dir(self) -> str:
        """Get input directory path."""
        pass
    
    @abstractmethod
    def get_output_dir(self) -> str:
        """Get output directory path."""
        pass
    
    @abstractmethod
    def get_log_dir(self) -> str:
        """Get log directory path."""
        pass


class ConfigurationContext:
    """
    Dependency injection context for configuration providers.
    
    This class uses dependency injection to provide configuration
    without creating circular imports. Existing code continues
    to work unchanged.
    """
    
    def __init__(self):
        self._config_provider: Optional[IConfigProvider] = None
        self._trading_criteria_provider: Optional[ITradingCriteriaProvider] = None
        self._display_provider: Optional[IDisplayProvider] = None
        self._rate_limit_provider: Optional[IRateLimitProvider] = None
        self._path_provider: Optional[IPathProvider] = None
    
    def set_config_provider(self, provider: IConfigProvider) -> None:
        """Set the configuration provider."""
        self._config_provider = provider
    
    def set_trading_criteria_provider(self, provider: ITradingCriteriaProvider) -> None:
        """Set the trading criteria provider."""
        self._trading_criteria_provider = provider
    
    def set_display_provider(self, provider: IDisplayProvider) -> None:
        """Set the display provider."""
        self._display_provider = provider
    
    def set_rate_limit_provider(self, provider: IRateLimitProvider) -> None:
        """Set the rate limit provider."""
        self._rate_limit_provider = provider
    
    def set_path_provider(self, provider: IPathProvider) -> None:
        """Set the path provider."""
        self._path_provider = provider
    
    @property
    def config(self) -> Optional[IConfigProvider]:
        """Get the configuration provider."""
        return self._config_provider
    
    @property
    def trading_criteria(self) -> Optional[ITradingCriteriaProvider]:
        """Get the trading criteria provider."""
        return self._trading_criteria_provider
    
    @property
    def display(self) -> Optional[IDisplayProvider]:
        """Get the display provider."""
        return self._display_provider
    
    @property
    def rate_limit(self) -> Optional[IRateLimitProvider]:
        """Get the rate limit provider."""
        return self._rate_limit_provider
    
    @property
    def paths(self) -> Optional[IPathProvider]:
        """Get the path provider."""
        return self._path_provider


# Global context instance for dependency injection
# This enables gradual migration without breaking existing code
global_config_context = ConfigurationContext()


def get_config_context() -> ConfigurationContext:
    """Get the global configuration context."""
    return global_config_context


# Export key components
__all__ = [
    'TradeAction',
    'IConfigProvider',
    'ITradingCriteriaProvider', 
    'IDisplayProvider',
    'IRateLimitProvider',
    'IPathProvider',
    'ConfigurationContext',
    'global_config_context',
    'get_config_context',
]