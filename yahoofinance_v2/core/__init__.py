"""
Core functionality for Yahoo Finance data access.

This module contains the foundational components of the package:
- Client: The base client for interacting with Yahoo Finance data
- Config: Centralized configuration settings
- Errors: Error hierarchy and handling utilities
- Types: Core data structures and type definitions
- Logging: Logging configuration and utilities
"""

from .client import YFinanceClient
from .errors import (
    YFinanceError, APIError, ValidationError, RateLimitError,
    ConnectionError, TimeoutError, ResourceNotFoundError,
    DataError, DataQualityError, MissingDataError,
    CacheError, ConfigError, format_error_details, classify_api_error
)
from .types import StockData
from .logging import setup_logging, get_logger, get_ticker_logger

__all__ = [
    # Client
    'YFinanceClient',
    
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
    
    # Error utilities
    'format_error_details',
    'classify_api_error',
    
    # Data types
    'StockData',
    
    # Logging utilities
    'setup_logging',
    'get_logger',
    'get_ticker_logger',
]
