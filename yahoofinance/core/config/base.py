"""
Base configuration class with common settings.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Set

from .rate_limiting import RateLimitConfig
from .trading_criteria import TradingCriteriaConfig
from .providers import ProviderConfig


class BaseConfig(ABC):
    """Base configuration class with common settings."""
    
    def __init__(self):
        """Initialize base configuration."""
        # Core paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.yahoofinance_root = os.path.dirname(os.path.dirname(__file__))
        
        # Directory paths
        self.paths = {
            "INPUT_DIR": os.path.join(self.yahoofinance_root, "input"),
            "OUTPUT_DIR": os.path.join(self.yahoofinance_root, "output"),
            "LOG_DIR": os.path.join(self.project_root, "logs"),
            "DEFAULT_LOG_FILE": os.path.join(self.project_root, "logs", "yahoofinance.log"),
        }
        
        # File paths - built from directory paths
        self.file_paths = self._build_file_paths()
        
        # Component configurations
        self.rate_limit = RateLimitConfig()
        self.trading_criteria = TradingCriteriaConfig()
        self.providers = ProviderConfig()
        
        # Portfolio configuration
        self.portfolio = {
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
        
        # Special tickers and grading
        self.positive_grades = {
            "Buy", "Outperform", "Strong Buy", "Overweight", "Accumulate",
            "Add", "Conviction Buy", "Top Pick", "Positive",
        }
        
        self.special_tickers = {
            "US_SPECIAL_CASES": {"BRK.A", "BRK.B", "BF.A", "BF.B"},
        }
        
        # Logging configuration
        self.logging = self._get_logging_config()
        
        # Environment-specific settings
        self._configure_environment()
    
    def _build_file_paths(self) -> Dict[str, str]:
        """Build file paths dictionary from base paths."""
        input_dir = self.paths["INPUT_DIR"]
        output_dir = self.paths["OUTPUT_DIR"]
        
        return {
            # Input files
            "MARKET_FILE": os.path.join(input_dir, "market.csv"),
            "PORTFOLIO_FILE": os.path.join(input_dir, "portfolio.csv"),
            "ETORO_FILE": os.path.join(input_dir, "etoro.csv"),
            "YFINANCE_FILE": os.path.join(input_dir, "yfinance.csv"),
            "NOTRADE_FILE": os.path.join(input_dir, "notrade.csv"),
            "CONS_FILE": os.path.join(input_dir, "cons.csv"),
            "US_TICKERS_FILE": os.path.join(input_dir, "us_tickers.csv"),
            "EUROPE_FILE": os.path.join(input_dir, "europe.csv"),
            "CHINA_FILE": os.path.join(input_dir, "china.csv"),
            "USA_FILE": os.path.join(input_dir, "usa.csv"),
            "USINDEX_FILE": os.path.join(input_dir, "usindex.csv"),
            # Output files
            "MARKET_OUTPUT": os.path.join(output_dir, "market.csv"),
            "PORTFOLIO_OUTPUT": os.path.join(output_dir, "portfolio.csv"),
            "BUY_OUTPUT": os.path.join(output_dir, "buy.csv"),
            "SELL_OUTPUT": os.path.join(output_dir, "sell.csv"),
            "HOLD_OUTPUT": os.path.join(output_dir, "hold.csv"),
            "MANUAL_OUTPUT": os.path.join(output_dir, "manual.csv"),
            "HTML_OUTPUT": os.path.join(output_dir, "index.html"),
            "PORTFOLIO_HTML": os.path.join(output_dir, "portfolio_dashboard.html"),
            "PORTFOLIO_PERFORMANCE_JSON": os.path.join(output_dir, "portfolio_performance.json"),
            "MONTHLY_PERFORMANCE_JSON": os.path.join(output_dir, "monthly_performance.json"),
            "WEEKLY_PERFORMANCE_JSON": os.path.join(output_dir, "weekly_performance.json"),
            "CSS_OUTPUT": os.path.join(output_dir, "styles.css"),
            "JS_OUTPUT": os.path.join(output_dir, "script.js"),
        }
    
    def _get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "VERSION": 1,
            "DISABLE_EXISTING_LOGGERS": False,
            "HANDLERS": {
                "CONSOLE": {
                    "CLASS": "logging.StreamHandler",
                    "LEVEL": self.get_log_level(),
                    "FORMATTER": "DETAILED",
                    "STREAM": "ext://sys.stdout",
                },
                "FILE": {
                    "CLASS": "logging.handlers.RotatingFileHandler",
                    "LEVEL": "DEBUG",
                    "FORMATTER": "DETAILED",
                    "FILENAME": self.paths["DEFAULT_LOG_FILE"],
                    "MAX_BYTES": 10485760,  # 10MB
                    "BACKUP_COUNT": 3,
                },
            },
            "FORMATTERS": {
                "SIMPLE": {
                    "FORMAT": "%(levelname)s - %(message)s"
                },
                "DETAILED": {
                    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "LOGGERS": {
                "yahoofinance": {
                    "LEVEL": "DEBUG",
                    "HANDLERS": ["CONSOLE", "FILE"],
                    "PROPAGATE": False,
                },
            },
            "ROOT": {
                "LEVEL": "WARNING",
                "HANDLERS": ["CONSOLE"],
            },
        }
    
    @abstractmethod
    def get_log_level(self) -> str:
        """Get the log level for this environment."""
        pass
    
    @abstractmethod 
    def _configure_environment(self) -> None:
        """Configure environment-specific settings."""
        pass
    
    def get_rate_limit_dict(self) -> Dict[str, Any]:
        """Get rate limit configuration as dictionary for backward compatibility."""
        return self.rate_limit.to_dict()
    
    def get_trading_criteria_dict(self) -> Dict[str, Any]:
        """Get trading criteria configuration as dictionary for backward compatibility."""
        return self.trading_criteria.to_dict()
    
    def get_provider_config_dict(self) -> Dict[str, Any]:
        """Get provider configuration as dictionary for backward compatibility."""
        return self.providers.to_dict()