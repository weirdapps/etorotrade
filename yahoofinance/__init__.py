"""
Yahoo Finance Market Analysis Package

A robust Python-based market analysis system that leverages Yahoo Finance data
to provide comprehensive stock analysis, portfolio management, and market intelligence.
The system features advanced rate limiting, intelligent caching, and multiple output formats.

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

# Client and data types
from .client import YFinanceClient
from .types import StockData
from .analyst import AnalystData
from .pricing import PricingAnalyzer, PriceTarget, PriceData
from .formatting import DisplayFormatter, DisplayConfig, Color
from .display import MarketDisplay

# Error types
from .errors import (
    YFinanceError,
    APIError,
    ValidationError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    ResourceNotFoundError,
    DataError,
    CacheError,
    ConfigError
)

# Utilities
from .utils.market_utils import is_us_ticker, normalize_hk_ticker
from .cache import market_cache, news_cache, earnings_cache

__version__ = "0.2.0"  # Updated version for improvements
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
    'CacheError',
    'ConfigError',
    
    # Utilities
    'is_us_ticker',
    'normalize_hk_ticker',
    'market_cache',
    'news_cache',
    'earnings_cache'
]