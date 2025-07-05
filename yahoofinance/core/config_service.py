"""
Configuration service for dependency injection.

This module provides a configuration service that can be injected into components,
allowing for better testability and isolation of configuration concerns.
"""

import threading
from typing import Any, Dict, Optional

try:
    from .config import (
        CACHE_CONFIG,
        CIRCUIT_BREAKER,
        COLUMN_NAMES,
        FILE_PATHS,
        MESSAGES,
        PATHS,
        PORTFOLIO_CONFIG,
        PROVIDER_CONFIG,
        RATE_LIMIT,
        STANDARD_DISPLAY_COLUMNS,
        TRADING_CRITERIA,
    )
except ImportError:
    # Import from the config/__init__.py fallback if main config not available
    from . import (
        CACHE_CONFIG,
        CIRCUIT_BREAKER,
        COLUMN_NAMES,
        FILE_PATHS,
        MESSAGES,
        PATHS,
        PORTFOLIO_CONFIG,
        PROVIDER_CONFIG,
        RATE_LIMIT,
        STANDARD_DISPLAY_COLUMNS,
        TRADING_CRITERIA,
    )
from .logging import get_logger


logger = get_logger(__name__)


class ConfigurationService:
    """
    Configuration service for dependency injection.
    
    This service provides centralized access to application configuration
    while allowing for easy testing and configuration overrides.
    """
    
    def __init__(self):
        """Initialize the configuration service."""
        self._lock = threading.RLock()
        self._overrides: Dict[str, Any] = {}
        self._config_cache: Dict[str, Any] = {}
        
        # Load default configurations
        self._default_configs = {
            "rate_limit": RATE_LIMIT,
            "cache": CACHE_CONFIG,
            "circuit_breaker": CIRCUIT_BREAKER,
            "column_names": COLUMN_NAMES,
            "file_paths": FILE_PATHS,
            "messages": MESSAGES,
            "paths": PATHS,
            "portfolio": PORTFOLIO_CONFIG,
            "provider": PROVIDER_CONFIG,
            "standard_display_columns": STANDARD_DISPLAY_COLUMNS,
            "trading_criteria": TRADING_CRITERIA,
        }
        
        logger.debug("Configuration service initialized")
    
    def get_config(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section name
            key: Optional key within the section
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        with self._lock:
            # Check for override first
            override_key = f"{section}.{key}" if key else section
            if override_key in self._overrides:
                return self._overrides[override_key]
            
            # Get from default config
            if section in self._default_configs:
                config = self._default_configs[section]
                if key is None:
                    return config
                return config.get(key, default) if isinstance(config, dict) else default
            
            return default
    
    def set_override(self, section: str, key: Optional[str] = None, value: Any = None) -> None:
        """
        Set configuration override.
        
        Args:
            section: Configuration section name
            key: Optional key within the section
            value: Value to set
        """
        with self._lock:
            override_key = f"{section}.{key}" if key else section
            self._overrides[override_key] = value
            # Clear cache for this section
            self._config_cache.pop(section, None)
            logger.debug(f"Set configuration override: {override_key} = {value}")
    
    def clear_overrides(self) -> None:
        """Clear all configuration overrides."""
        with self._lock:
            self._overrides.clear()
            self._config_cache.clear()
            logger.debug("Cleared all configuration overrides")
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return self.get_config("rate_limit")
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return self.get_config("cache")
    
    def get_circuit_breaker_config(self) -> Dict[str, Any]:
        """Get circuit breaker configuration."""
        return self.get_config("circuit_breaker")
    
    def get_provider_config(self) -> Dict[str, Any]:
        """Get provider configuration."""
        return self.get_config("provider")
    
    def get_trading_criteria_config(self) -> Dict[str, Any]:
        """Get trading criteria configuration."""
        return self.get_config("trading_criteria")
    
    def get_column_names(self) -> Dict[str, str]:
        """Get column names configuration."""
        return self.get_config("column_names")
    
    def get_file_paths(self) -> Dict[str, str]:
        """Get file paths configuration."""
        return self.get_config("file_paths")
    
    def get_messages(self) -> Dict[str, str]:
        """Get messages configuration."""
        return self.get_config("messages")
    
    def get_standard_display_columns(self) -> list:
        """Get standard display columns configuration."""
        return self.get_config("standard_display_columns")


class ConfigurationContext:
    """
    Context manager for temporary configuration overrides.
    
    This allows for scoped configuration changes that are automatically
    reverted when the context exits.
    """
    
    def __init__(self, config_service: ConfigurationService, **overrides):
        """
        Initialize configuration context.
        
        Args:
            config_service: Configuration service instance
            **overrides: Configuration overrides as section__key=value
        """
        self.config_service = config_service
        self.overrides = overrides
        self.previous_values = {}
    
    def __enter__(self):
        """Enter the configuration context."""
        # Save previous values and set overrides
        for key, value in self.overrides.items():
            if "__" in key:
                section, config_key = key.split("__", 1)
                current_value = self.config_service.get_config(section, config_key)
                self.previous_values[key] = current_value
                self.config_service.set_override(section, config_key, value)
            else:
                current_value = self.config_service.get_config(key)
                self.previous_values[key] = current_value
                self.config_service.set_override(key, value=value)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the configuration context."""
        # Restore previous values
        for key, previous_value in self.previous_values.items():
            if "__" in key:
                section, config_key = key.split("__", 1)
                if previous_value is not None:
                    self.config_service.set_override(section, config_key, previous_value)
            else:
                if previous_value is not None:
                    self.config_service.set_override(key, value=previous_value)


# Create a default configuration service instance
_default_config_service = ConfigurationService()


def get_config_service() -> ConfigurationService:
    """
    Get the default configuration service instance.
    
    Returns:
        ConfigurationService instance
    """
    return _default_config_service


def with_config(**overrides):
    """
    Create a configuration context with overrides.
    
    Args:
        **overrides: Configuration overrides
        
    Returns:
        ConfigurationContext instance
        
    Example:
        with with_config(rate_limit__base_delay=0.5):
            # Code that uses the overridden configuration
            pass
    """
    return ConfigurationContext(_default_config_service, **overrides)