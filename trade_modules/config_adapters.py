"""
Configuration adapters that implement the interfaces using existing configurations.

These adapters wrap existing configuration classes to implement the new interfaces,
enabling dependency injection without changing existing behavior.

SAFETY: Zero behavioral changes - all existing functionality preserved.
ZERO RISK: Uses adapter pattern to wrap existing code unchanged.
"""

from typing import Dict, List, Any, Optional
from .config_interfaces import (
    IConfigProvider,
    ITradingCriteriaProvider,
    IDisplayProvider,
    IRateLimitProvider,
    IPathProvider,
)


class TradeConfigAdapter(ITradingCriteriaProvider, IDisplayProvider):
    """
    Adapter for trade_modules.trade_config.TradeConfig.
    
    Wraps existing TradeConfig to implement new interfaces
    without changing existing behavior.
    """
    
    def __init__(self, trade_config=None):
        # Import here to avoid circular imports
        if trade_config is None:
            try:
                from .trade_config import TradeConfig
                self._trade_config = TradeConfig
            except ImportError:
                # Fallback if trade_config not available
                self._trade_config = None
        else:
            self._trade_config = trade_config
    
    def get_thresholds(self, option: str, action: str, tier: str = None) -> Dict[str, Any]:
        """Get trading thresholds for specific option and action."""
        if self._trade_config is None:
            return {}
        return self._trade_config.get_thresholds(option, action, tier)
    
    def get_universal_thresholds(self) -> Dict[str, Any]:
        """Get universal thresholds applied to all options."""
        if self._trade_config is None:
            return {}
        return getattr(self._trade_config, 'UNIVERSAL_THRESHOLDS', {})
    
    def get_tier_thresholds(self, tier: str, action: str) -> Dict[str, Any]:
        """Get tier-specific thresholds."""
        if self._trade_config is None:
            return {}
        return self._trade_config.get_tier_thresholds(tier, action)
    
    def get_display_columns(self, option: str, sub_option: str = None, 
                          output_type: str = "console") -> List[str]:
        """Get display columns for specific option and output type."""
        if self._trade_config is None:
            return []
        return self._trade_config.get_display_columns(option, sub_option, output_type)
    
    def get_sort_config(self, option: str, sub_option: str = None) -> Dict[str, str]:
        """Get sorting configuration."""
        if self._trade_config is None:
            return {}
        return self._trade_config.get_sort_config(option, sub_option)
    
    def get_format_rule(self, column: str) -> Dict[str, Any]:
        """Get formatting rule for a column."""
        if self._trade_config is None:
            return {"type": "text"}
        return self._trade_config.get_format_rule(column)


class YahooFinanceConfigAdapter(IConfigProvider, IRateLimitProvider, IPathProvider):
    """
    Adapter for yahoofinance.core.config configuration.
    
    Wraps existing yahoofinance configuration to implement new interfaces
    without changing existing behavior.
    """
    
    def __init__(self, yahoo_config=None):
        # Import here to avoid circular imports
        if yahoo_config is None:
            try:
                from yahoofinance.core.config import get_setting, get_config
                self._get_setting = get_setting
                self._config = get_config()
            except ImportError:
                # Fallback if yahoofinance config not available
                self._get_setting = lambda key, default=None: default
                self._config = None
        else:
            self._get_setting = yahoo_config.get
            self._config = yahoo_config
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting."""
        return self._get_setting(key, default)
    
    def get_file_path(self, file_key: str) -> str:
        """Get a file path from configuration."""
        # Try to get from FILE_PATHS configuration
        try:
            from yahoofinance.core.config import FILE_PATHS
            return FILE_PATHS.get(file_key, "")
        except ImportError:
            return ""
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        try:
            from yahoofinance.core.config import RATE_LIMIT
            return RATE_LIMIT
        except ImportError:
            return {}
    
    def get_concurrent_limits(self) -> Dict[str, int]:
        """Get concurrent request limits."""
        rate_config = self.get_rate_limit_config()
        return {
            'max_concurrent_calls': rate_config.get('MAX_CONCURRENT_CALLS', 10),
            'batch_size': rate_config.get('BATCH_SIZE', 10),
            'max_total_connections': rate_config.get('MAX_TOTAL_CONNECTIONS', 50),
            'max_connections_per_host': rate_config.get('MAX_CONNECTIONS_PER_HOST', 20),
        }
    
    def get_input_dir(self) -> str:
        """Get input directory path."""
        try:
            from yahoofinance.core.config import PATHS
            return PATHS.get('INPUT_DIR', 'yahoofinance/input')
        except ImportError:
            return 'yahoofinance/input'
    
    def get_output_dir(self) -> str:
        """Get output directory path."""
        try:
            from yahoofinance.core.config import PATHS
            return PATHS.get('OUTPUT_DIR', 'yahoofinance/output')
        except ImportError:
            return 'yahoofinance/output'
    
    def get_log_dir(self) -> str:
        """Get log directory path."""
        try:
            from yahoofinance.core.config import PATHS
            return PATHS.get('LOG_DIR', 'logs')
        except ImportError:
            return 'logs'


def initialize_default_adapters():
    """
    Initialize the global configuration context with default adapters.
    
    This function sets up the dependency injection context using existing
    configurations, enabling gradual migration without breaking changes.
    
    SAFETY: Only adds new functionality, doesn't modify existing behavior.
    """
    from .config_interfaces import get_config_context
    
    context = get_config_context()
    
    # Create adapters for existing configurations
    trade_adapter = TradeConfigAdapter()
    yahoo_adapter = YahooFinanceConfigAdapter()
    
    # Set up dependency injection
    context.set_config_provider(yahoo_adapter)
    context.set_trading_criteria_provider(trade_adapter)
    context.set_display_provider(trade_adapter)
    context.set_rate_limit_provider(yahoo_adapter)
    context.set_path_provider(yahoo_adapter)
    
    return context


# Export key components
__all__ = [
    'TradeConfigAdapter',
    'YahooFinanceConfigAdapter',
    'initialize_default_adapters',
]