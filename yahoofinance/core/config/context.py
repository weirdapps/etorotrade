"""
Configuration context management for safe runtime modifications.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from . import config, get_setting, update_setting


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


@contextmanager
def config_override(**overrides) -> Generator[None, None, None]:
    """Context manager for temporarily overriding configuration.
    
    This should only be used in tests or benchmarking scenarios.
    Production code should not use this.
    
    Args:
        **overrides: Configuration keys and values to override
        
    Yields:
        None
        
    Raises:
        ConfigurationError: If used in production environment
        
    Example:
        with config_override(rate_limit_base_delay=0.1, rate_limit_batch_size=5):
            # Configuration is temporarily modified
            provider = get_provider()
            # ... test code ...
        # Configuration is restored
    """
    env = os.getenv("ETOROTRADE_ENV", "development").lower()
    if env == "production":
        raise ConfigurationError("Configuration overrides not allowed in production")
    
    # Store original values
    original_values = {}
    
    try:
        # Apply overrides and store originals
        for key, value in overrides.items():
            # Convert underscore notation to dot notation
            config_key = key.replace('_', '.')
            original_values[config_key] = get_setting(config_key)
            update_setting(config_key, value)
        
        yield
        
    finally:
        # Restore original values
        for config_key, original_value in original_values.items():
            if original_value is not None:
                update_setting(config_key, original_value)


@contextmanager  
def rate_limit_override(**overrides) -> Generator[None, None, None]:
    """Context manager for temporarily overriding rate limit configuration.
    
    Args:
        **overrides: Rate limit configuration keys and values to override
        
    Example:
        with rate_limit_override(batch_size=5, base_delay=0.1):
            # Rate limiting is temporarily modified
            provider = get_provider()
            # ... test code ...
        # Rate limiting configuration is restored
    """
    prefixed_overrides = {f"rate_limit.{key}": value for key, value in overrides.items()}
    with config_override(**prefixed_overrides):
        yield


@contextmanager
def trading_criteria_override(**overrides) -> Generator[None, None, None]:
    """Context manager for temporarily overriding trading criteria configuration.
    
    Args:
        **overrides: Trading criteria configuration keys and values to override
        
    Example:
        with trading_criteria_override(buy_min_upside=25.0, sell_max_upside=3.0):
            # Trading criteria are temporarily modified
            result = filter_buy_opportunities(df)
            # ... test code ...
        # Trading criteria configuration is restored
    """
    prefixed_overrides = {f"trading_criteria.{key}": value for key, value in overrides.items()}
    with config_override(**prefixed_overrides):
        yield


@contextmanager
def provider_override(**overrides) -> Generator[None, None, None]:
    """Context manager for temporarily overriding provider configuration.
    
    Args:
        **overrides: Provider configuration keys and values to override
        
    Example:
        with provider_override(enable_yahooquery=True):
            # Provider configuration is temporarily modified
            provider = get_provider()
            # ... test code ...
        # Provider configuration is restored
    """
    prefixed_overrides = {f"providers.{key}": value for key, value in overrides.items()}
    with config_override(**prefixed_overrides):
        yield


def create_test_config(**overrides) -> None:
    """Create a test configuration with overrides.
    
    This is a convenience function for test setup.
    Only works in non-production environments.
    
    Args:
        **overrides: Configuration keys and values to override
    """
    env = os.getenv("ETOROTRADE_ENV", "development").lower()
    if env == "production":
        raise ConfigurationError("Test configuration not allowed in production")
    
    for key, value in overrides.items():
        config_key = key.replace('_', '.')
        update_setting(config_key, value)


def reset_to_defaults() -> None:
    """Reset configuration to defaults.
    
    Only works in non-production environments.
    """
    env = os.getenv("ETOROTRADE_ENV", "development").lower()
    if env == "production":
        raise ConfigurationError("Configuration reset not allowed in production")
    
    # Force recreation of configuration
    global config
    from . import get_config
    config = get_config()


__all__ = [
    'config_override',
    'rate_limit_override', 
    'trading_criteria_override',
    'provider_override',
    'create_test_config',
    'reset_to_defaults',
    'ConfigurationError',
]