"""
Core functionality for the Yahoo Finance API client.

This module contains the foundational components of the package:
- Client: Interface to the Yahoo Finance API
- Cache: Caching system for improved performance
- Config: Centralized configuration
- Errors: Error hierarchy and handling
- Types: Common data structures
- Logging: Logging configuration and utilities
"""

from .client import YFinanceClient
from .cache import Cache, market_cache, news_cache, earnings_cache
from .errors import (
    YFinanceError, APIError, ValidationError, RateLimitError,
    ConnectionError, TimeoutError, ResourceNotFoundError,
    DataError, DataQualityError, MissingDataError,
    CacheError, ConfigError, format_error_details, classify_api_error
)
from .types import StockData
from .logging import setup_logging, get_logger, get_ticker_logger

__all__ = [
    # Client and base classes
    'YFinanceClient',
    'Cache',
    
    # Cached instances
    'market_cache',
    'news_cache',
    'earnings_cache',
    
    # Error types
    'YFinanceError',
    'APIError',
    'ValidationError',
    'RateLimitError',
    'ConnectionError',
    'TimeoutError',
    'ResourceNotFoundError',
    'DataError',
    'DataQualityError',
    'MissingDataError',
    'CacheError',
    'ConfigError',
    
    # Data types
    'StockData',
    
    # Utility functions
    'format_error_details',
    'classify_api_error',
    
    # Logging utilities
    'setup_logging',
    'get_logger',
    'get_ticker_logger',
]