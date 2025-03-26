"""
Logging configuration for Yahoo Finance data access.

This module provides functions for setting up logging and creating loggers.
"""

import logging
import os
from typing import Optional


def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None, 
                  log_format: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None)
        log_format: Logging format string (default: None)
        
    Side Effects:
        Configures logging for the application
    """
    # Set default format if not provided
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
    # Configure root logger
    logging.basicConfig(level=log_level, format=log_format)
    
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
        
    # Log configuration details
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured with level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.debug(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
        
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
        Logger instance
    """
    return logging.getLogger(f"{logger.name}.{ticker}")