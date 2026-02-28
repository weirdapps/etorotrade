"""
Logging configuration for Yahoo Finance data access.

Provides logging setup, noise suppression for yfinance/yahooquery,
and log level management utilities.
"""

import logging
import logging.config
import os
import sys
from typing import Any, Dict, List, Optional, Union

# Lazy import to avoid circular dependency
PATHS = None

def _get_paths():
    """Lazy load PATHS to avoid circular imports."""
    global PATHS
    if PATHS is None:
        from .config import PATHS as _PATHS
        PATHS = _PATHS
    return PATHS


# Default logging formats
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
CONSOLE_FORMAT = "%(levelname)s - %(message)s"

# Type for log record attributes
LogRecordAttrs = Dict[str, Any]


class YFinanceErrorFilter(logging.Filter):
    """Filter to suppress noisy yfinance/network error messages."""
    
    def filter(self, record):
        # Suppress delisting, earnings, and HTTP error messages
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            if any(pattern in msg.lower() for pattern in [
                'possibly delisted',
                'no earnings dates found',
                'earnings date',
                'delisted',
                'no earnings',
                'http error 404',
                'http error 400',
                'http error 403',
                'http error 500',
                'connection error',
                'timeout error',
                'request failed'
            ]):
                return False
        return True


def suppress_yfinance_noise():
    """Apply filter to suppress yfinance delisting/earnings/HTTP error messages."""
    import warnings
    
    # Suppress yfinance deprecation warnings about earnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
    warnings.filterwarnings("ignore", message=".*Ticker.earnings.*")
    warnings.filterwarnings("ignore", message=".*Net Income.*")
    
    # Apply filter to multiple loggers that might generate noise
    logger_names = ['yfinance', 'urllib3', 'requests', 'yahooquery']
    
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        # Check if filter is already applied
        filter_already_applied = any(
            isinstance(filter_obj, YFinanceErrorFilter) 
            for filter_obj in logger.filters
        )
        
        if not filter_already_applied:
            # Apply the filter
            error_filter = YFinanceErrorFilter()
            logger.addFilter(error_filter)


# ===== Basic Logging Functions =====

def setup_logging(
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console: bool = True,
) -> None:
    """
    Set up basic logging configuration.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None)
        log_format: Logging format string (default: None)
        console: Whether to log to console (default: True)

    Side Effects:
        Configures logging for the application
    """
    # Convert string level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Set default format if not provided
    if log_format is None:
        log_format = DEFAULT_FORMAT

    # Configure handlers
    handlers = []
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers if handlers else None
    )

    # Create file handler if log file is specified
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Add file handler to root logger
        logging.root.addHandler(file_handler)

    # Add filter to suppress yfinance delisting errors
    yfinance_logger = logging.getLogger('yfinance')
    yfinance_filter = YFinanceErrorFilter()
    yfinance_logger.addFilter(yfinance_filter)

    # Log configuration details
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured with level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.debug(f"Log file: {log_file}")


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    console_level: Optional[Union[int, str]] = None,
    format_string: Optional[str] = None,
    debug: bool = False,
) -> None:
    """
    Configure logging for the application with advanced options.

    Args:
        level: Base logging level (default: INFO)
        log_file: Path to log file (default: None, uses default path from config)
        console: Whether to log to console (default: True)
        console_level: Console logging level (default: same as base level)
        format_string: Log format string (default: DEFAULT_FORMAT or DEBUG_FORMAT if debug=True)
        debug: Whether to enable debug mode (more verbose logging)

    Returns:
        None

    Side Effects:
        Configures the Python logging system
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    if console_level is None:
        console_level = level
    elif isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper())

    # Determine log format
    if format_string is None:
        if debug:
            format_string = DEBUG_FORMAT
        else:
            format_string = DEFAULT_FORMAT

    # Set up a basic configuration
    handlers: List[Dict[str, Any]] = []

    # Add console handler if requested
    if console:
        console_handler = {
            "level": console_level,
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": sys.stdout,
        }
        handlers.append(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = {
            "level": level,
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": log_file,
            "encoding": "utf-8",
        }
        handlers.append(file_handler)
    elif _get_paths().get("DEFAULT_LOG_FILE"):
        # Use default log file if none specified
        log_dir = os.path.dirname(_get_paths()["DEFAULT_LOG_FILE"])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = {
            "level": level,
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": _get_paths()["DEFAULT_LOG_FILE"],
            "encoding": "utf-8",
        }
        handlers.append(file_handler)

    # Create logging config dictionary
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": format_string,
            },
            "console": {
                "format": CONSOLE_FORMAT,
            },
        },
        "handlers": {},
        "root": {
            "level": level,
            "handlers": [],
        },
        "loggers": {
            "yahoofinance": {
                "level": level,
                "handlers": [],
                "propagate": False,
            },
            # Make third-party libraries less verbose
            "requests": {
                "level": logging.WARNING,
                "propagate": True,
            },
            "urllib3": {
                "level": logging.WARNING,
                "propagate": True,
            },
            "yfinance": {
                "level": logging.ERROR,  # Set to ERROR to reduce noise during progress
                "propagate": False,  # Don't propagate to prevent console output during progress
            },
            "yahooquery": {
                "level": logging.WARNING,
                "propagate": True,
            },
        },
    }

    # Add handlers to config
    for i, handler in enumerate(handlers):
        handler_name = f"handler_{i}"
        logging_config["handlers"][handler_name] = handler  # type: ignore[index]
        logging_config["root"]["handlers"].append(handler_name)  # type: ignore[index]
        logging_config["loggers"]["yahoofinance"]["handlers"].append(handler_name)  # type: ignore[index]

    # Apply configuration if any handlers are defined
    if handlers:
        logging.config.dictConfig(logging_config)

        # Add filter to suppress yfinance delisting errors
        yfinance_logger = logging.getLogger('yfinance')
        yfinance_filter = YFinanceErrorFilter()
        yfinance_logger.addFilter(yfinance_filter)

        # Log initial message
        logger = logging.getLogger(__name__)
        logger.debug(f"Logging configured with level: {logging.getLevelName(level)}")
        if log_file:
            logger.debug(f"Log file: {log_file}")
    else:
        # Basic configuration if no handlers
        logging.basicConfig(level=level, format=format_string)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    This is the recommended way to get a logger in the application.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_ticker_logger(logger: logging.Logger, ticker: str) -> logging.Logger:
    """
    Get a logger with the ticker appended to the name.

    This creates a child logger that adds the ticker to all log messages,
    making it easier to trace issues with specific tickers.

    Args:
        logger: Parent logger
        ticker: Ticker symbol

    Returns:
        Logger instance with ticker context
    """
    return logging.getLogger(f"{logger.name}.{ticker}")


def set_log_level(level: Union[int, str], logger_name: Optional[str] = None) -> None:
    """
    Set the log level for a specific logger or the root logger.

    Args:
        level: Logging level (can be string like 'INFO' or int like logging.INFO)
        logger_name: Name of logger to set level for (default: None, root logger)

    Returns:
        None
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    logger.setLevel(level)


def enable_debug_for_module(module_name: str) -> None:
    """
    Enable debug logging for a specific module.

    Args:
        module_name: Name of the module to enable debug for

    Returns:
        None
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Ensure we have at least one handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(DEBUG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def disable_logging() -> None:
    """
    Completely disable all logging.

    Use this function when you want to silence all log output.

    Returns:
        None
    """
    logging.disable(logging.CRITICAL)


def enable_logging() -> None:
    """
    Re-enable logging after it has been disabled.

    Returns:
        None
    """
    logging.disable(logging.NOTSET)