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
        # Minimal fallback rate limit configuration
        "WINDOW_SIZE": 60,
        "MAX_CALLS": 75,
        "BASE_DELAY": 0.3,
        "MIN_DELAY": 0.1,
        "MAX_DELAY": 30.0,
        "SUCCESS_THRESHOLD": 5,
        "SUCCESS_DELAY_REDUCTION": 0.8,
        "ERROR_THRESHOLD": 2,
        "ERROR_DELAY_INCREASE": 1.5,
        "TICKER_PRIORITY": {"HIGH": 0.7, "MEDIUM": 1.0, "LOW": 1.5},
        "VIP_TICKERS": set(),
        "SLOW_TICKERS": set(),
        "ENABLE_ADAPTIVE_STRATEGY": True,
        "MONITOR_INTERVAL": 60,
        "BATCH_SIZE": 10,
        "BATCH_DELAY": 0.5,
        "MAX_CONCURRENT_CALLS": 10,
        # Connection pooling configuration
        "MAX_TOTAL_CONNECTIONS": 50,
        "MAX_CONNECTIONS_PER_HOST": 20,
        "KEEPALIVE_TIMEOUT": 60,
        "DNS_CACHE_TTL": 300,
        "SESSION_MAX_AGE": 3600,
        "API_TIMEOUT": 30,
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
    PORTFOLIO_CONFIG = {}
    PROVIDER_CONFIG = {}
    POSITIVE_GRADES = []
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
    }
    PAGINATION = {
        "DEFAULT_PAGE_SIZE": 50,
        "MAX_PAGE_SIZE": 1000,
    }
    CACHE_CONFIG = {
        "TTL": 300,
        "MAX_SIZE": 1000,
        "ENABLED": True,
    }
    MESSAGES = {
        "NO_DATA": "No data available",
        "ERROR": "An error occurred", 
        "SUCCESS": "Operation completed successfully",
        "PROMPT_ENTER_TICKERS": "Enter tickers (comma-separated): ",
        "PROMPT_SOURCE_SELECTION": "Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? ",
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
        "EARNINGS",
        "SIZE",
        "ACT",
    ]

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
]