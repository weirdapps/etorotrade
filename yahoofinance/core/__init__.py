"""
Core functionality for Yahoo Finance data access.

This module contains the foundational components of the package:
- Client: The base client for interacting with Yahoo Finance data
- Config: Centralized configuration settings
- Errors: Error hierarchy and handling utilities
- Types: Core data structures and type definitions
- Logging: Logging configuration and utilities
- Field Cache: Intelligent field-level caching system with user-defined TTL
"""

from .client import YFinanceClient
from .errors import (
    APIError,
    CacheError,
    ConfigError,
    ConnectionError,
    DataError,
    DataQualityError,
    MissingDataError,
    RateLimitError,
    ResourceNotFoundError,
    TimeoutError,
    ValidationError,
    YFinanceError,
    classify_api_error,
    format_error_details,
)
from .logging import get_logger, get_ticker_logger, setup_logging
from .types import StockData

# Field-level caching system (optional import - disabled by default)
try:
    from .field_cache import (
        FieldLevelCache,
        get_field_cache,
        enable_field_cache,
        CacheStats
    )
    from .field_cache_config import (
        FieldCacheSettings,
        load_cache_settings_from_env,
        validate_field_cache_config
    )
    from .cache_wrapper import (
        wrap_provider_with_cache,
        enable_global_field_cache,
        disable_global_field_cache,
        is_global_cache_enabled
    )
    _FIELD_CACHE_AVAILABLE = True
except ImportError:
    # Field cache dependencies not available
    _FIELD_CACHE_AVAILABLE = False


# Build __all__ list dynamically based on available features
__all__ = [
    # Client
    "YFinanceClient",
    # Error types
    "YFinanceError",
    "APIError",
    "ValidationError",
    "RateLimitError",
    "ConnectionError",
    "TimeoutError",
    "ResourceNotFoundError",
    "DataError",
    "DataQualityError",
    "MissingDataError",
    "CacheError",
    "ConfigError",
    # Error utilities
    "format_error_details",
    "classify_api_error",
    # Data types
    "StockData",
    # Logging utilities
    "setup_logging",
    "get_logger",
    "get_ticker_logger",
]

# Add field cache exports if available
if _FIELD_CACHE_AVAILABLE:
    __all__.extend([
        # Field cache core
        "FieldLevelCache",
        "get_field_cache", 
        "enable_field_cache",
        "CacheStats",
        # Field cache configuration
        "FieldCacheSettings",
        "load_cache_settings_from_env",
        "validate_field_cache_config",
        # Field cache wrappers
        "wrap_provider_with_cache",
        "enable_global_field_cache",
        "disable_global_field_cache",
        "is_global_cache_enabled",
    ])
