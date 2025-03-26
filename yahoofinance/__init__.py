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

# Import only the core components needed for the package interface
from .core.logging import setup_logging, get_logger
import os
import logging

# Set up default logging if not already configured
if not logging.root.handlers:
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "yahoofinance.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logging(log_level=logging.INFO, log_file=log_path)
    
    # Reduce noise from third-party libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

# Import and re-export the main API components
from .api import get_provider, FinanceDataProvider, AsyncFinanceDataProvider

# Import and re-export key analysis components
from .analysis import StockAnalyzer, AnalysisResults

# Import and re-export core data types
from .core.types import StockData

# Import and re-export error types
from .core.errors import (
    YFinanceError,
    APIError,
    ValidationError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    ResourceNotFoundError,
    DataError,
    DataQualityError,
    MissingDataError,
    CacheError,
    ConfigError
)

# Import and re-export key utility functions
from .utils.market import is_us_ticker, normalize_hk_ticker

__version__ = "1.0.0"
__author__ = "Roo"

__all__ = [
    # Provider API
    'get_provider',
    'FinanceDataProvider',
    'AsyncFinanceDataProvider',
    
    # Analysis components
    'StockAnalyzer',
    'AnalysisResults',
    
    # Core data types
    'StockData',
    
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
    
    # Utility functions
    'is_us_ticker',
    'normalize_hk_ticker',
    
    # Logging
    'setup_logging',
    'get_logger'
]
