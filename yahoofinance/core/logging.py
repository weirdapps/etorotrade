"""
Consolidated logging configuration for Yahoo Finance data access.

This module provides comprehensive logging functionality including:
- Basic logging setup and configuration
- Structured JSON logging
- Context-aware logging with request correlation
- Multiple formatters and handlers
- Log level management and debug helpers
"""

import json
import logging
import logging.config
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

from .config import PATHS


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
    elif PATHS.get("DEFAULT_LOG_FILE"):
        # Use default log file if none specified
        log_dir = os.path.dirname(PATHS["DEFAULT_LOG_FILE"])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = {
            "level": level,
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": PATHS["DEFAULT_LOG_FILE"],
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
        logging_config["handlers"][handler_name] = handler
        logging_config["root"]["handlers"].append(handler_name)
        logging_config["loggers"]["yahoofinance"]["handlers"].append(handler_name)

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


# ===== Structured JSON Logging =====

class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for use with structured logging systems.

    This formatter produces JSON output that includes all the standard
    LogRecord attributes along with any additional context provided.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger_name: bool = True,
        rename_fields: Optional[Dict[str, str]] = None,
        excluded_fields: Optional[Set[str]] = None,
        additional_fields: Optional[Dict[str, Any]] = None,
        sanitize_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the JSON formatter.

        Args:
            include_timestamp: Whether to include timestamp in output
            include_level: Whether to include log level in output
            include_logger_name: Whether to include logger name in output
            rename_fields: Dictionary mapping LogRecord attributes to output names
            excluded_fields: Set of fields to exclude from output
            additional_fields: Dictionary of additional fields to include in output
            sanitize_keys: List of keys that should have their values sanitized
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger_name = include_logger_name
        self.rename_fields = rename_fields or {}
        self.excluded_fields = excluded_fields or set()
        self.additional_fields = additional_fields or {}
        self.sanitize_keys = sanitize_keys or ["password", "api_key", "secret", "token", "auth"]

        # Standard attributes that should be included from LogRecord
        self.standard_attrs = {
            "name",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "asctime",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "process",
            "processName",
            "message",
        }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format LogRecord as JSON string.

        Args:
            record: LogRecord instance to format

        Returns:
            JSON-formatted string representation of the log record
        """
        # Process the record with the standard formatter
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        # Build the dictionary of fields to include in JSON output
        log_record: Dict[str, Any] = {}

        # Add standard attributes if needed
        if self.include_timestamp:
            log_record["timestamp"] = datetime.fromtimestamp(record.created).isoformat()
        if self.include_level:
            log_record["level"] = record.levelname
        if self.include_logger_name:
            log_record["logger"] = record.name

        # Add standard LogRecord attributes, applying renames and exclusions
        for key in self.standard_attrs:
            if key in self.excluded_fields:
                continue

            # Skip fields we've already handled
            if key in ("levelname", "name") and (
                (key == "levelname" and self.include_level)
                or (key == "name" and self.include_logger_name)
            ):
                continue

            value = getattr(record, key, None)

            # Apply field renaming if specified
            output_key = self.rename_fields.get(key, key)

            log_record[output_key] = value

        # Add additional static fields
        log_record.update(self.additional_fields)

        # Include exception info if available
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra attributes from the LogRecord
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            for key, value in record.extra.items():
                log_record[key] = value

        # Add any context from the record context_dict
        if hasattr(record, "context_dict") and isinstance(record.context_dict, dict):
            # Apply sanitization to sensitive fields if needed
            sanitized_context = self._sanitize_dict(record.context_dict)
            log_record.update(sanitized_context)

        # Return the JSON string
        return json.dumps(log_record, default=self._json_default)

    def _json_default(self, obj: Any) -> Any:
        """
        Handle non-serializable objects during JSON serialization.

        Args:
            obj: Object to serialize

        Returns:
            Serializable representation of the object
        """
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return vars(obj)
        if hasattr(obj, "__str__"):
            return str(obj)
        return repr(obj)

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize sensitive information in a dictionary.

        Args:
            data: Dictionary to sanitize

        Returns:
            Sanitized dictionary
        """
        result = {}
        for key, value in data.items():
            # Check if this key should be sanitized
            if any(sensitive_key in key.lower() for sensitive_key in self.sanitize_keys):
                if isinstance(value, str) and value:
                    # Mask the value, showing just first and last character
                    if len(value) > 4:
                        result[key] = f"{value[0]}***{value[-1]}"
                    else:
                        result[key] = "****"
                else:
                    result[key] = "****"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                result[key] = self._sanitize_dict(value)
            else:
                result[key] = value
        return result


class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to all log records.

    This adapter allows adding context that will be included in all
    subsequent log messages.
    """

    def __init__(
        self,
        logger: logging.Logger,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the context adapter.

        Args:
            logger: The logger to adapt
            context: Initial context dictionary
            request_id: Optional request ID for correlation
        """
        self.context = context or {}
        if request_id:
            self.context["request_id"] = request_id
        super().__init__(logger, self.context)

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the logging message and keywords.

        Args:
            msg: The log message
            kwargs: Additional logging parameters

        Returns:
            Tuple of (message, kwargs)
        """
        # Create a copy of the context to avoid modifying the original
        context_dict = self.context.copy()

        # Add extra context from kwargs if provided
        if "extra" in kwargs:
            extra = kwargs["extra"]
            if isinstance(extra, dict):
                # Merge with context, preserving both
                kwargs["extra"].update({"context_dict": context_dict})
            else:
                # If extra is not a dict, create a new dict
                kwargs["extra"] = {"context_dict": context_dict}
        else:
            # No extra provided, add our context dict
            kwargs["extra"] = {"context_dict": context_dict}

        return msg, kwargs

    def add_context(self, **kwargs: Any) -> None:
        """
        Add additional context to the logger.

        Args:
            **kwargs: Context key-value pairs to add
        """
        self.context.update(kwargs)

    def bind(self, **kwargs: Any) -> "ContextAdapter":
        """
        Create a new logger with additional bound context.

        Args:
            **kwargs: Context key-value pairs to add

        Returns:
            New ContextAdapter with combined context
        """
        new_context = self.context.copy()
        new_context.update(kwargs)
        return ContextAdapter(self.logger, new_context)


def get_structured_logger(
    name: str, context: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None
) -> ContextAdapter:
    """
    Get a structured logger with context.

    Args:
        name: Logger name
        context: Initial context dictionary
        request_id: Optional request ID for correlation

    Returns:
        ContextAdapter instance
    """
    logger = get_logger(name)
    return ContextAdapter(logger, context, request_id)


def generate_request_id() -> str:
    """
    Generate a unique request ID.

    Returns:
        Unique request ID string
    """
    return str(uuid.uuid4())


def setup_json_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    sanitize_keys: Optional[List[str]] = None,
    additional_fields: Optional[Dict[str, Any]] = None,
    console: bool = True,
) -> None:
    """
    Set up JSON structured logging.

    Args:
        level: Logging level
        log_file: Path to log file
        sanitize_keys: List of keys to sanitize
        additional_fields: Additional fields to include in all logs
        console: Whether to log to console
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Set up additional fields
    additional_fields = additional_fields or {}
    additional_fields.update(
        {
            "app": "etorotrade",
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "pid": os.getpid(),
        }
    )

    # Create JSON formatter
    json_formatter = JSONFormatter(sanitize_keys=sanitize_keys, additional_fields=additional_fields)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Set up file handler if provided
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Log that we've configured JSON logging
    logger = get_logger(__name__)
    logger.info(
        "JSON structured logging configured",
        extra={"context_dict": {"log_level": logging.getLevelName(level)}},
    )


# Middleware for adding request context to logs
class RequestLoggingMiddleware:
    """
    Middleware for adding request context to logs.

    This middleware can be used with API requests to add request-specific
    context to all logs generated during request processing.
    """

    def __init__(
        self,
        app_name: str = "etorotrade",
        include_headers: bool = False,
        include_body: bool = False,
        sensitive_headers: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the request logging middleware.

        Args:
            app_name: Application name
            include_headers: Whether to include headers in logs
            include_body: Whether to include request body in logs
            sensitive_headers: List of sensitive headers to sanitize
        """
        self.app_name = app_name
        self.include_headers = include_headers
        self.include_body = include_body
        self.sensitive_headers = sensitive_headers or [
            "authorization",
            "cookie",
            "x-api-key",
            "api-key",
        ]

        self.logger = get_structured_logger(__name__)

    def __call__(self, get_response: Callable) -> Callable:
        """
        Middleware callable.

        Args:
            get_response: Response getter function

        Returns:
            Middleware function
        """

        def middleware(request: Any) -> Any:
            # Generate a unique request ID
            request_id = generate_request_id()

            # Add request ID to request object
            setattr(request, "request_id", request_id)

            # Create request context
            context = {
                "request_id": request_id,
                "app": self.app_name,
                "method": getattr(request, "method", "UNKNOWN"),
                "path": getattr(request, "path", "UNKNOWN"),
                "ip": self._get_client_ip(request),
                "start_time": time.time(),
            }

            # Add headers if configured
            if self.include_headers and hasattr(request, "headers"):
                context["headers"] = self._sanitize_headers(request.headers)

            # Add body if configured
            if self.include_body and hasattr(request, "body"):
                context["body"] = str(request.body)[:1000]  # Limit size

            # Log request start
            request_logger = get_structured_logger("request", context)
            request_logger.info(f"Request started: {context['method']} {context['path']}")

            # Process the request
            start_time = time.time()
            try:
                response = get_response(request)

                # Add response info to context
                status_code = getattr(response, "status_code", 0)
                context.update(
                    {
                        "status_code": status_code,
                        "duration_ms": (time.time() - start_time) * 1000,
                    }
                )

                # Log successful response
                request_logger.info(
                    f"Request completed: {context['method']} {context['path']} - {status_code}"
                )

                return response
            except Exception as e:
                # Add error info to context
                context.update(
                    {
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                        "duration_ms": (time.time() - start_time) * 1000,
                    }
                )

                # Log error
                request_logger.error(
                    f"Request failed: {context['method']} {context['path']} - {e}", exc_info=True
                )

                # Re-raise the exception
                raise

        return middleware

    def _get_client_ip(self, request: Any) -> str:
        """
        Get client IP address from request.

        Args:
            request: Request object

        Returns:
            Client IP address
        """
        # Try common header fields for proxied requests
        if hasattr(request, "headers"):
            headers = request.headers
            for header in ["X-Forwarded-For", "X-Real-IP"]:
                if header in headers:
                    return headers[header].split(",")[0].strip()

        # Fall back to REMOTE_ADDR if available
        if hasattr(request, "META") and "REMOTE_ADDR" in request.META:
            return request.META["REMOTE_ADDR"]

        # Last resort: check for remote attribute (common in some frameworks)
        if hasattr(request, "remote") and hasattr(request.remote, "host"):
            return request.remote.host

        return "unknown"

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize sensitive headers.

        Args:
            headers: Request headers

        Returns:
            Sanitized headers
        """
        result = {}
        for key, value in headers.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_headers):
                result[key] = "****"
            else:
                result[key] = value
        return result