"""
Centralized configuration management system.

This module provides a centralized approach to configuration management,
supporting environment-specific settings and preventing runtime modifications.
"""

import os
from typing import Any, Dict

from .base import BaseConfig
from .development import DevelopmentConfig
from .production import ProductionConfig


def get_config() -> BaseConfig:
    """Get the appropriate configuration based on environment.
    
    Returns:
        BaseConfig: Configuration instance for current environment
    """
    env = os.getenv("ETOROTRADE_ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "development":
        return DevelopmentConfig()
    else:
        # Default to development
        return DevelopmentConfig()


# Global configuration instance
config = get_config()


def get_setting(key: str, default: Any = None) -> Any:
    """Get a configuration setting.
    
    Args:
        key: Configuration key (supports dot notation like 'rate_limit.base_delay')
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    try:
        # Support dot notation for nested keys
        keys = key.split('.')
        value = config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    except (AttributeError, KeyError, TypeError):
        return default


def update_setting(key: str, value: Any) -> None:
    """Update a configuration setting (for testing purposes only).
    
    WARNING: This should only be used in tests or development.
    Production code should not modify configuration at runtime.
    
    Args:
        key: Configuration key
        value: New value
    """
    if os.getenv("ETOROTRADE_ENV", "development").lower() == "production":
        raise RuntimeError("Configuration modification not allowed in production")
    
    # Allow modification only in non-production environments
    keys = key.split('.')
    obj = config
    
    for k in keys[:-1]:
        if hasattr(obj, k):
            obj = getattr(obj, k)
        elif isinstance(obj, dict):
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
        else:
            raise ValueError(f"Cannot update nested key: {key}")
    
    final_key = keys[-1]
    if hasattr(obj, final_key):
        setattr(obj, final_key, value)
    elif isinstance(obj, dict):
        obj[final_key] = value
    else:
        raise ValueError(f"Cannot update key: {key}")


# Re-export key configuration components for backward compatibility
from .rate_limiting import RateLimitConfig
from .trading_criteria import TradingCriteriaConfig
from .providers import ProviderConfig
from .context import (
    config_override,
    rate_limit_override,
    trading_criteria_override,
    provider_override,
    create_test_config,
    reset_to_defaults,
    ConfigurationError,
)

# Export legacy config dictionary for backward compatibility
# This will be gradually phased out
# Import conditionally to avoid circular imports
try:
    from ..config import (
        FILE_PATHS,
        PATHS,
        RATE_LIMIT,
        TRADING_CRITERIA,
        PORTFOLIO_CONFIG,
        PROVIDER_CONFIG,
        POSITIVE_GRADES,
        SPECIAL_TICKERS,
        COLUMN_NAMES,
        US_SPECIAL_CASES,
        CIRCUIT_BREAKER,
        PAGINATION,
        CACHE_CONFIG,
        MESSAGES,
        STANDARD_DISPLAY_COLUMNS,
        RISK_METRICS,
        PERFORMANCE_CONFIG,
        DISPLAY,
    )
except ImportError:
    # Define minimal fallbacks to prevent circular import issues
    FILE_PATHS = {
        # Minimal fallback file paths
        "BUY_OUTPUT": "yahoofinance/output/buy.csv",
        "SELL_OUTPUT": "yahoofinance/output/sell.csv",
        "MARKET_OUTPUT": "yahoofinance/output/market.csv",
        "PORTFOLIO_OUTPUT": "yahoofinance/output/portfolio.csv",
        "MANUAL_OUTPUT": "yahoofinance/output/manual.csv",
        "MARKET_FILE": "yahoofinance/input/market.csv",
        "PORTFOLIO_FILE": "yahoofinance/input/portfolio.csv",
        "HOLD_OUTPUT": "yahoofinance/output/hold.csv",
        "NOTRADE_FILE": "yahoofinance/input/notrade.csv",
        "ETORO_FILE": "yahoofinance/input/etoro.csv",
        "YFINANCE_FILE": "yahoofinance/input/yfinance.csv",
        "CONS_FILE": "yahoofinance/input/cons.csv",
        "US_TICKERS_FILE": "yahoofinance/input/us_tickers.csv",
    }
    PATHS = {
        # Minimal fallback paths
        "INPUT_DIR": "yahoofinance/input",
        "OUTPUT_DIR": "yahoofinance/output", 
        "LOG_DIR": "logs",
        "DEFAULT_LOG_FILE": "logs/yahoofinance.log",
    }
    RATE_LIMIT = {
        # Optimized fallback rate limit configuration for performance
        "WINDOW_SIZE": 60,
        "MAX_CALLS": 75,
        "BASE_DELAY": 0.15,  # Optimized: reduced from 0.3
        "MIN_DELAY": 0.1,
        "MAX_DELAY": 2.0,    # Optimized: reduced from 30.0
        "SUCCESS_THRESHOLD": 5,
        "SUCCESS_DELAY_REDUCTION": 0.8,
        "ERROR_THRESHOLD": 2,
        "ERROR_DELAY_INCREASE": 1.5,
        "TICKER_PRIORITY": {"HIGH": 0.7, "MEDIUM": 1.0, "LOW": 1.5},
        "VIP_TICKERS": set(),
        "SLOW_TICKERS": set(),
        "ENABLE_ADAPTIVE_STRATEGY": True,
        "MONITOR_INTERVAL": 60,
        "BATCH_SIZE": 25,     # Optimized: increased from 10
        "BATCH_DELAY": 0.0,   # Optimized: no delay for maximum performance
        "MAX_CONCURRENT_CALLS": 30,  # Optimized: increased from 10
        # Connection pooling configuration
        "MAX_TOTAL_CONNECTIONS": 50,
        "MAX_CONNECTIONS_PER_HOST": 20,
        "KEEPALIVE_TIMEOUT": 60,
        "DNS_CACHE_TTL": 300,
        "SESSION_MAX_AGE": 3600,
        "API_TIMEOUT": 30,
        "MAX_RETRY_ATTEMPTS": 3,
    }
    TRADING_CRITERIA = {
        "CONFIDENCE": {
            "MIN_ANALYST_COUNT": 5,
            "MIN_PRICE_TARGETS": 5,
        },
        "BUY": {
            "MIN_UPSIDE": 20.0,
            "MIN_BUY_PERCENTAGE": 85.0,
        },
        "SELL": {
            "MAX_UPSIDE": 5.0,
            "MIN_BUY_PERCENTAGE": 65.0,
        },
    }
    PORTFOLIO_CONFIG = {
        "PORTFOLIO_VALUE": 450_000,
        "MIN_POSITION_USD": 1_000,
        "MAX_POSITION_USD": 40_000,
        "MAX_POSITION_PCT": 8.9,
        "BASE_POSITION_PCT": 0.5,
        "HIGH_CONVICTION_PCT": 2.0,
        "SMALL_CAP_THRESHOLD": 2_000_000_000,
        "MID_CAP_THRESHOLD": 10_000_000_000,
        "LARGE_CAP_THRESHOLD": 50_000_000_000,
    }
    PROVIDER_CONFIG = {
        "ENABLE_YAHOOQUERY": False,
    }
    POSITIVE_GRADES = {
        "Buy",
        "Outperform",
        "Strong Buy",
        "Overweight",
        "Accumulate",
        "Add",
        "Conviction Buy",
        "Top Pick",
        "Positive",
    }
    SPECIAL_TICKERS = {
        "US_SPECIAL_CASES": {
            "BRK.A",
            "BRK.B",  # Berkshire Hathaway
            "BF.A", 
            "BF.B",  # Brown-Forman
        },
    }
    US_SPECIAL_CASES = {}
    COLUMN_NAMES = {
        # Minimal fallback column names to prevent import errors
        "EARNINGS_DATE": "Earnings Date",
        "BUY_PERCENTAGE": "% BUY",
        "DIVIDEND_YIELD_DISPLAY": "DIV %",
        "COMPANY_NAME": "COMPANY",
        "TICKER": "TICKER",
        "MARKET_CAP": "CAP",
        "PRICE": "PRICE",
        "TARGET_PRICE": "TARGET",
        "UPSIDE": "UPSIDE",
        "ANALYST_COUNT": "# T",
        "TOTAL_RATINGS": "# A",
        "ACTION": "ACT",
        "POSITION_SIZE": "SIZE",
        "RATING_TYPE": "A",
        "EXPECTED_RETURN": "EXRET",
        "BETA": "BETA",
        "PE_TRAILING": "PET",
        "PE_FORWARD": "PEF",
        "PEG_RATIO": "PEG",
    }
    CIRCUIT_BREAKER = {
        "FAILURE_THRESHOLD": 5,
        "RECOVERY_TIMEOUT": 60,
        "HALF_OPEN_MAX_CALLS": 2,
        "STATE_FILE": "yahoofinance/data/circuit_state.json",
        "MAX_RETRY_ATTEMPTS": 3,
    }
    PAGINATION = {
        "DEFAULT_PAGE_SIZE": 50,
        "MAX_PAGE_SIZE": 1000,
        "PAGE_SIZE": 50,
        "MAX_PAGES": 100,
        "MAX_RETRIES": 3,
        "RETRY_DELAY": 1.0,
    }
    CACHE_CONFIG = {
        "ENABLE_MEMORY_CACHE": False,
        "ENABLE_DISK_CACHE": False,
        "MEMORY_ONLY_MODE": True,
        "MEMORY_CACHE_SIZE": 10000,
        "MEMORY_CACHE_TTL": 300,
        "THREAD_LOCAL_CACHE_SIZE": 100,
        "ENABLE_ULTRA_FAST_PATH": True,
        "BATCH_UPDATE_THRESHOLD": 5,
        "CACHE_ERRORS": True,
        "ERROR_CACHE_TTL": 60,
        "DISK_CACHE_SIZE_MB": 100,
        "DISK_CACHE_TTL": 3600,
        "DISK_CACHE_DIR": "yahoofinance/data/cache",
        "TICKER_INFO_MEMORY_TTL": 86400,
        "TICKER_INFO_DISK_TTL": 604800,
        "MARKET_DATA_MEMORY_TTL": 60,
        "MARKET_DATA_DISK_TTL": 180,
        "FUNDAMENTALS_MEMORY_TTL": 60,
        "FUNDAMENTALS_DISK_TTL": 180,
        "NEWS_MEMORY_TTL": 600,
        "NEWS_DISK_TTL": 1200,
        "ANALYSIS_MEMORY_TTL": 600,
        "ANALYSIS_DISK_TTL": 1200,
        "HISTORICAL_DATA_MEMORY_TTL": 86400,
        "HISTORICAL_DATA_DISK_TTL": 172800,
        "EARNINGS_DATA_MEMORY_TTL": 600,
        "EARNINGS_DATA_DISK_TTL": 1200,
        "INSIDER_TRADES_MEMORY_TTL": 86400,
        "INSIDER_TRADES_DISK_TTL": 172800,
        "DIVIDEND_DATA_MEMORY_TTL": 86400,
        "DIVIDEND_DATA_DISK_TTL": 172800,
        "TARGET_PRICE_MEMORY_TTL": 600,
        "TARGET_PRICE_DISK_TTL": 1200,
        "MISSING_DATA_MEMORY_TTL": 259200,
        "MISSING_DATA_DISK_TTL": 604800,
        "US_STOCK_TTL_MULTIPLIER": 1.0,
        "NON_US_STOCK_TTL_MULTIPLIER": 2.0,
    }
    MESSAGES = {
        "NO_DATA": "No data available",
        "ERROR": "An error occurred", 
        "SUCCESS": "Operation completed successfully",
        "PROMPT_ENTER_TICKERS": "Enter tickers (comma-separated): ",
        "PROMPT_SOURCE_SELECTION": "Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? ",
        "PROMPT_TICKER_SOURCE": "Select ticker source: ",
        "PROMPT_TICKER_SOURCE_OPTIONS": "Available ticker sources",
        "PROMPT_TICKER_SOURCE_CHOICE": "Enter your choice: ",
        "PROMPT_INVALID_CHOICE": "Invalid choice. Please try again.",
        "INFO_TICKERS_LOADED": "Loaded {count} tickers from {file_path}",
        "ERROR_LOADING_FILE": "Error loading file {file_path}: {error}",
    }
    STANDARD_DISPLAY_COLUMNS = [
        "#",
        "TICKER",
        "COMPANY",
        "CAP",
        "PRICE",
        "TARGET",
        "UPSIDE",
        "# T",
        "% BUY",
        "# A",
        "A",
        "EXRET",
        "BETA",
        "PET",
        "PEF",
        "PEG",
        "DIV %",
        "SI",
        "EG",
        "PP",
        "EARNINGS",
        "SIZE",
        "ACT",
    ]
    # Missing config constants for backward compatibility
    RISK_METRICS = {
        "RISK_FREE_RATE": 0.03,
        "TRADING_DAYS_PER_YEAR": 252,
    }
    PERFORMANCE_CONFIG = {
        "ENABLE_MEMORY_PROFILING": True,
        "RESULTS_DIR": "benchmarks",
        "BASELINE_DIR": "benchmarks",
        "BENCHMARK": {
            "BENCHMARK_DIR": "benchmarks",
            "SAMPLE_TICKERS": [
                "AAPL", "MSFT", "GOOG", "AMZN", "META",
                "NVDA", "TSLA", "JPM", "V", "JNJ",
            ],
            "BASELINE_FILE": "baseline_performance.json",
            "MEMORY_PROFILE_THRESHOLD": 1.2,
            "RESOURCE_MONITOR_INTERVAL": 0.5,
            "MAX_BENCHMARK_DURATION": 300,
            "DEFAULT_ITERATIONS": 3,
            "DEFAULT_WARMUP_ITERATIONS": 1,
        },
    }
    DISPLAY = {
        "MAX_COMPANY_NAME_LENGTH": 14,
        "DEFAULT_COLUMNS": [
            "ticker", "company", "market_cap", "price", "target_price",
            "upside", "analyst_count", "buy_percentage", "total_ratings",
            "beta", "pe_trailing", "pe_forward", "peg_ratio",
            "dividend_yield", "short_float_pct",
        ],
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
            "exret": {"precision": 1, "as_percentage": True},
        },
    }

__all__ = [
    'get_config',
    'get_setting', 
    'update_setting',
    'config',
    'BaseConfig',
    'DevelopmentConfig',
    'ProductionConfig',
    'RateLimitConfig',
    'TradingCriteriaConfig',
    'ProviderConfig',
    # Context managers
    'config_override',
    'rate_limit_override',
    'trading_criteria_override',
    'provider_override',
    'create_test_config',
    'reset_to_defaults',
    'ConfigurationError',
    # Legacy exports
    'FILE_PATHS',
    'PATHS',
    'RATE_LIMIT',
    'TRADING_CRITERIA',
    'PORTFOLIO_CONFIG',
    'PROVIDER_CONFIG',
    'POSITIVE_GRADES',
    'SPECIAL_TICKERS',
    'COLUMN_NAMES',
    'US_SPECIAL_CASES',
    'CIRCUIT_BREAKER',
    'PAGINATION',
    'CACHE_CONFIG',
    'MESSAGES',
    'STANDARD_DISPLAY_COLUMNS',
    'RISK_METRICS',
    'PERFORMANCE_CONFIG',
    'DISPLAY',
]