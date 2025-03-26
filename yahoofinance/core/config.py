"""
Configuration settings for Yahoo Finance data access.

This module defines configuration settings for rate limiting, caching,
API timeouts, and more. It provides a central location for all configuration
values used throughout the package.
"""

import os
from typing import Dict, Any, List, Set

# Rate limiting configuration
RATE_LIMIT = {
    # Time window for rate limiting in seconds
    "WINDOW_SIZE": 60,
    
    # Maximum API calls per window
    "MAX_CALLS": 60,
    
    # Base delay between calls in seconds
    "BASE_DELAY": 1.0,
    
    # Minimum delay after many successful calls in seconds
    "MIN_DELAY": 0.5,
    
    # Maximum delay after errors in seconds
    "MAX_DELAY": 30.0,
    
    # Number of items per batch
    "BATCH_SIZE": 15,
    
    # Delay between batches in seconds
    "BATCH_DELAY": 15.0,
    
    # Maximum retry attempts for API calls
    "MAX_RETRY_ATTEMPTS": 3,
    
    # API request timeout in seconds
    "API_TIMEOUT": 30,
    
    # Maximum concurrent API calls (for async)
    "MAX_CONCURRENT_CALLS": 5,
    
    # Problematic tickers that should use longer delays
    "SLOW_TICKERS": set(),
}

# Circuit breaker configuration
CIRCUIT_BREAKER = {
    # Failure threshold to trip the circuit breaker
    "FAILURE_THRESHOLD": 5,
    
    # Time window in seconds to count failures
    "FAILURE_WINDOW": 60,
    
    # Recovery timeout in seconds before circuit half-opens
    "RECOVERY_TIMEOUT": 300,
    
    # Maximum consecutive successes required to close circuit
    "SUCCESS_THRESHOLD": 3,
    
    # Percentage of requests to allow through in half-open state
    "HALF_OPEN_ALLOW_PERCENTAGE": 10,
    
    # Maximum time in seconds a circuit can stay open
    "MAX_OPEN_TIMEOUT": 1800,  # 30 minutes
    
    # Enable circuit breaker by default
    "ENABLED": True,
    
    # Path to persistent circuit state file
    "STATE_FILE": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "circuit_state.json"
    ),
}

# Caching configuration
CACHE_CONFIG = {
    # Enable memory cache
    "ENABLE_MEMORY_CACHE": True,
    
    # Enable disk cache
    "ENABLE_DISK_CACHE": True,
    
    # Memory cache size (items)
    "MEMORY_CACHE_SIZE": 1000,
    
    # Default memory cache TTL (seconds)
    "MEMORY_CACHE_TTL": 300,  # 5 minutes
    
    # Disk cache size (MB)
    "DISK_CACHE_SIZE_MB": 100,
    
    # Default disk cache TTL (seconds)
    "DISK_CACHE_TTL": 3600,  # 1 hour
    
    # Disk cache directory
    "DISK_CACHE_DIR": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "cache"
    ),
    
    # TTL settings by data type (seconds)
    "TICKER_INFO_MEMORY_TTL": 300,      # 5 minutes
    "TICKER_INFO_DISK_TTL": 3600,       # 1 hour
    "MARKET_DATA_MEMORY_TTL": 60,       # 1 minute
    "MARKET_DATA_DISK_TTL": 1800,       # 30 minutes
    "FUNDAMENTALS_MEMORY_TTL": 3600,    # 1 hour
    "FUNDAMENTALS_DISK_TTL": 86400,     # 1 day
    "NEWS_MEMORY_TTL": 900,             # 15 minutes
    "NEWS_DISK_TTL": 7200,              # 2 hours
    "ANALYSIS_MEMORY_TTL": 1800,        # 30 minutes
    "ANALYSIS_DISK_TTL": 14400,         # 4 hours
}

# Risk metrics configuration
RISK_METRICS = {
    # Risk-free rate (annual)
    "RISK_FREE_RATE": 0.03,
    
    # Trading days per year
    "TRADING_DAYS_PER_YEAR": 252,
}

# Trading criteria configuration
TRADING_CRITERIA = {
    "CONFIDENCE": {
        # Minimum number of analysts covering the stock
        "MIN_ANALYST_COUNT": 5,
        
        # Minimum number of price targets
        "MIN_PRICE_TARGETS": 5,
    },
    "SELL": {
        # Maximum upside potential for sell recommendation
        "MAX_UPSIDE": 5.0,
        
        # Minimum buy percentage for sell recommendation
        "MIN_BUY_PERCENTAGE": 65.0,
        
        # Maximum forward P/E for sell recommendation
        "MAX_FORWARD_PE": 45.0,
        
        # Maximum PEG ratio for sell recommendation
        "MAX_PEG": 3.0,
        
        # Maximum short interest for sell recommendation
        "MAX_SHORT_INTEREST": 4.0,
        
        # Maximum beta for sell recommendation
        "MAX_BETA": 3.0,
        
        # Minimum expected return for sell recommendation
        "MIN_EXRET": 10.0,
    },
    "BUY": {
        # Minimum upside potential for buy recommendation
        "MIN_UPSIDE": 20.0,
        
        # Minimum buy percentage for buy recommendation
        "MIN_BUY_PERCENTAGE": 82.0,
        
        # Minimum beta for buy recommendation
        "MIN_BETA": 0.2,
        
        # Maximum beta for buy recommendation
        "MAX_BETA": 3.0,
        
        # Minimum forward P/E for buy recommendation
        "MIN_FORWARD_PE": 0.5,
        
        # Maximum forward P/E for buy recommendation
        "MAX_FORWARD_PE": 45.0,
        
        # Maximum PEG ratio for buy recommendation
        "MAX_PEG": 3.0,
        
        # Maximum short interest for buy recommendation
        "MAX_SHORT_INTEREST": 3.0,
    },
}

# Display configuration
DISPLAY = {
    # Maximum company name length
    "MAX_COMPANY_NAME_LENGTH": 14,
    
    # Default display columns
    "DEFAULT_COLUMNS": [
        "ticker", 
        "company", 
        "market_cap", 
        "price", 
        "target_price", 
        "upside", 
        "analyst_count",
        "buy_percentage", 
        "total_ratings", 
        "beta",
        "pe_trailing", 
        "pe_forward", 
        "peg_ratio", 
        "dividend_yield",
        "short_float_pct"
    ],
    
    # Column formatters
    "FORMATTERS": {
        "price": {"precision": 2},
        "target_price": {"precision": 2},
        "upside": {"precision": 1, "as_percentage": True},
        "buy_percentage": {"precision": 0, "as_percentage": True},
        "beta": {"precision": 2},
        "pe_trailing": {"precision": 1},
        "pe_forward": {"precision": 1},
        "peg_ratio": {"precision": 1},
        "dividend_yield": {"precision": 2, "as_percentage": True},
        "short_float_pct": {"precision": 1, "as_percentage": True},
    },
}

# Paths configuration
PATHS = {
    # Input directory
    "INPUT_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "input"),
    
    # Output directory
    "OUTPUT_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
    
    # Log directory
    "LOG_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"),
    
    # Default log file
    "DEFAULT_LOG_FILE": os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "logs", 
        "yahoofinance.log"
    ),
}

# File paths for data files
FILE_PATHS = {
    # Input files
    "MARKET_FILE": os.path.join(PATHS["INPUT_DIR"], "market.csv"),
    "PORTFOLIO_FILE": os.path.join(PATHS["INPUT_DIR"], "portfolio.csv"),
    "ETORO_FILE": os.path.join(PATHS["INPUT_DIR"], "etoro.csv"),
    "YFINANCE_FILE": os.path.join(PATHS["INPUT_DIR"], "yfinance.csv"),
    "NOTRADE_FILE": os.path.join(PATHS["INPUT_DIR"], "notrade.csv"),
    "CONS_FILE": os.path.join(PATHS["INPUT_DIR"], "cons.csv"),
    "US_TICKERS_FILE": os.path.join(PATHS["INPUT_DIR"], "us_tickers.csv"),
    
    # Output files
    "MARKET_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "market.csv"),
    "PORTFOLIO_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "portfolio.csv"),
    "BUY_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "buy.csv"),
    "SELL_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "sell.csv"),
    "HOLD_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "hold.csv"),
    "HTML_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "index.html"),
    "PORTFOLIO_HTML": os.path.join(PATHS["OUTPUT_DIR"], "portfolio.html"),
    "CSS_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "styles.css"),
    "JS_OUTPUT": os.path.join(PATHS["OUTPUT_DIR"], "script.js"),
}

# Special tickers configuration
SPECIAL_TICKERS = {
    # US stocks with dots in their symbols
    "US_SPECIAL_CASES": {
        'BRK.A', 'BRK.B',  # Berkshire Hathaway
        'BF.A', 'BF.B',    # Brown-Forman
    },
}

# Load environment variables if needed
def load_env_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dictionary containing configuration values from environment variables
    """
    config = {}
    
    # Rate limit settings
    if 'YFINANCE_MAX_CALLS' in os.environ:
        config['RATE_LIMIT.MAX_CALLS'] = int(os.environ['YFINANCE_MAX_CALLS'])
    
    # Cache settings
    if 'YFINANCE_CACHE_TTL' in os.environ:
        config['CACHE_CONFIG.MEMORY_CACHE_TTL'] = int(os.environ['YFINANCE_CACHE_TTL'])
    
    # API settings
    if 'YFINANCE_API_TIMEOUT' in os.environ:
        config['RATE_LIMIT.API_TIMEOUT'] = int(os.environ['YFINANCE_API_TIMEOUT'])
    
    # Circuit breaker settings
    if 'YFINANCE_CIRCUIT_BREAKER_ENABLED' in os.environ:
        config['CIRCUIT_BREAKER.ENABLED'] = os.environ['YFINANCE_CIRCUIT_BREAKER_ENABLED'].lower() == 'true'
    
    return config

# Apply environment variable configuration
ENV_CONFIG = load_env_config()

# Update configuration with environment variables
def apply_env_config(env_config: Dict[str, Any]) -> None:
    """
    Apply environment variable configuration.
    
    Args:
        env_config: Dictionary containing configuration values from environment variables
    """
    for key, value in env_config.items():
        parts = key.split('.')
        if len(parts) == 2:
            module_name, setting_name = parts
            if module_name in globals() and setting_name in globals()[module_name]:
                globals()[module_name][setting_name] = value

# Apply environment configuration
apply_env_config(ENV_CONFIG)