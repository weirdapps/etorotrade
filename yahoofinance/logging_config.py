"""
Logging configuration for Yahoo Finance Client.

This module configures logging for the entire application, providing
consistent log formatting, level control, and log file management.
"""

import os
import logging
import logging.handlers
from typing import Optional, Dict, Any, Tuple


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Path to log file (if None, logs only to console)
        log_format: Custom log format string (if None, uses default format)
        enable_console: Whether to enable console logging
        max_file_size: Maximum size in bytes for log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Root logger configured with specified settings
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    
    # Default format if none provided
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Add console handler if enabled
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Use rotating file handler for log rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name, usually __name__ of the module
        
    Returns:
        Logger instance with specified name
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds contextual information to log messages.
    
    This adapter allows adding context like ticker symbol or operation
    to log messages without modifying the message itself.
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        """
        Initialize logger adapter with context.
        
        Args:
            logger: Base logger to adapt
            extra: Dictionary with contextual information
        """
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the log message by adding context.
        
        Args:
            msg: Original log message
            kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (modified message, kwargs)
        """
        context_str = ' '.join(f'{k}={v}' for k, v in self.extra.items())
        if context_str:
            return f"{msg} [{context_str}]", kwargs
        return msg, kwargs


def get_ticker_logger(logger: logging.Logger, ticker: str) -> LoggerAdapter:
    """
    Get a logger adapter with ticker context.
    
    Args:
        logger: Base logger to adapt
        ticker: Ticker symbol to add as context
        
    Returns:
        LoggerAdapter with ticker context
    """
    return LoggerAdapter(logger, {'ticker': ticker})