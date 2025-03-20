"""
Core client module for v2 implementation.

This module defines the base YFinanceClient class used by various providers
and compatibility layers. It provides a foundation for API communication
with appropriate error handling and configuration.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from .config import RATE_LIMIT
from .errors import YFinanceError, ValidationError

# Set up logging
logger = logging.getLogger(__name__)

class YFinanceClient:
    """
    Base client for Yahoo Finance data access.
    
    This class provides common functionality used by providers and
    serves as a compatibility layer for v1 code.
    """
    
    def __init__(self, 
                 max_retries: int = None,
                 timeout: int = None):
        """
        Initialize YFinance client.
        
        Args:
            max_retries: Maximum number of retry attempts for API calls
            timeout: API request timeout in seconds
        """
        self.max_retries = max_retries or RATE_LIMIT["MAX_RETRY_ATTEMPTS"]
        self.timeout = timeout or RATE_LIMIT["API_TIMEOUT"]
        
        logger.debug(f"Initialized YFinanceClient with max_retries={self.max_retries}, timeout={self.timeout}")
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate a ticker symbol format.
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If ticker format is invalid
        """
        if not ticker or not isinstance(ticker, str):
            raise ValidationError("Ticker must be a non-empty string")
            
        # Basic validation - more complex validation happens in providers
        if len(ticker) > 20:
            raise ValidationError(f"Ticker '{ticker}' exceeds maximum length of 20 characters")
            
        return True