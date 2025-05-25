"""
Yahoo Finance Market Analysis Package

A robust Python-based market analysis system that leverages Yahoo Finance data
to provide comprehensive stock analysis, portfolio management, and market intelligence.

This package features:
- Enhanced async architecture with true async I/O
- Circuit breaker pattern for improved reliability
- Disk-based caching for better performance
- Provider pattern for data access abstraction
"""

import logging

# Import standard libraries
import os
import sys

# Backward compatibility for existing code
from .core.logging import setup_logging as old_setup_logging

# Import our new standardized logging configuration
from .core.logging import (
    configure_logging,
    enable_debug_for_module,
    get_logger,
    get_ticker_logger,
    set_log_level,
)


# Set up default logging if not already configured
if not logging.root.handlers:
    # Determine log level from environment or use INFO as default
    log_level = os.environ.get("YAHOOFINANCE_LOG_LEVEL", "INFO")

    # Determine log file path from config or use default
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "yahoofinance.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Configure logging using our new standardized configuration
    configure_logging(
        level=log_level,
        log_file=log_path,
        console=False,  # Default to no console output for library usage
        debug=os.environ.get("YAHOOFINANCE_DEBUG", "").lower() == "true",
    )

# Import and re-export key analysis components
from .analysis import AnalysisResults, StockAnalyzer

# Import and re-export the main API components
from .api import (
    AsyncFinanceDataProvider,
    FinanceDataProvider,
    get_all_providers,
    get_default_provider,
    get_provider,
)

# Import and re-export error types
from .core.errors import (
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
)

# Import and re-export core data types
from .core.types import StockData

# Import and re-export key utility functions
from .utils.market import is_us_ticker, normalize_hk_ticker


# The provider_registry already defaults to hybrid provider


__version__ = "2.0.0"  # Updated version to indicate breaking change from compat removal
__author__ = "Roo"

__all__ = [
    # Provider API
    "get_provider",
    "get_default_provider",
    "get_all_providers",
    "FinanceDataProvider",
    "AsyncFinanceDataProvider",
    # Analysis components
    "StockAnalyzer",
    "AnalysisResults",
    # Core data types
    "StockData",
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
    # Utility functions
    "is_us_ticker",
    "normalize_hk_ticker",
    # Logging
    "configure_logging",
    "get_logger",
    "set_log_level",
    "enable_debug_for_module",
    "get_ticker_logger",
    # For backward compatibility
    "old_setup_logging",  # Imported from .core.logging
]
