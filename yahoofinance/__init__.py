"""
Yahoo Finance Market Analysis Package

A robust Python-based market analysis system that leverages Yahoo Finance data
to provide comprehensive stock analysis, portfolio management, and market intelligence.
The system features advanced rate limiting, intelligent caching, and multiple output formats.

The package handles logging, error management, and performance optimization automatically,
allowing you to focus on analyzing financial data rather than handling infrastructure concerns.

Example usage:
    
    # Basic usage - Get ticker information
    from yahoofinance import YFinanceClient
    
    client = YFinanceClient()
    stock_data = client.get_ticker_info("AAPL")
    print(f"Current price: ${stock_data.current_price}")
    print(f"Recommendation: {stock_data.recommendation_key}")
    
    # Get analyst ratings
    from yahoofinance import AnalystData
    
    analyst = AnalystData(client)
    ratings = analyst.get_ratings_summary("MSFT")
    print(f"Buy percentage: {ratings['positive_percentage']}%")
    print(f"Total ratings: {ratings['total_ratings']}")
    
    # Display market data with formatting
    from yahoofinance import MarketDisplay
    
    display = MarketDisplay()
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    display.display_report(tickers)
"""

# Setup logging
import os
import logging
from .core.logging import setup_logging, get_logger, get_ticker_logger

# Set up default logging if not already configured
if not logging.root.handlers:
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "yahoofinance.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logging(log_level=logging.INFO, log_file=log_path)
    
    # Reduce noise from third-party libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)

# Client and data types from core
from .core.client import YFinanceClient
from .core.types import StockData

# Analysis modules
from .analyst import AnalystData
from .pricing import PricingAnalyzer, PriceTarget, PriceData

# Display and formatting
from .formatting import DisplayFormatter, DisplayConfig, Color
from .display import MarketDisplay

# Error types
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

# Utilities
from .utils.market_utils import is_us_ticker, normalize_hk_ticker
from .core.cache import market_cache, news_cache, earnings_cache

__version__ = "0.3.0"  # Updated version for organization improvements
__author__ = "Roo"

__all__ = [
    # Client and core classes
    'YFinanceClient',
    'StockData',
    'AnalystData',
    'PricingAnalyzer',
    'PriceTarget',
    'PriceData',
    'DisplayFormatter',
    'DisplayConfig',
    'Color',
    'MarketDisplay',
    
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
    
    # Utilities
    'is_us_ticker',
    'normalize_hk_ticker',
    'market_cache',
    'news_cache',
    'earnings_cache',
    
    # Logging
    'setup_logging',
    'get_logger',
    'get_ticker_logger'
]