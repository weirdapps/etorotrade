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
