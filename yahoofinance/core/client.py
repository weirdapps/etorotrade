"""
Core client module for v2 implementation.

This module defines the base YFinanceClient class used by various providers
and compatibility layers. It provides a foundation for API communication
with appropriate error handling and configuration.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .config import RATE_LIMIT
from .errors import YFinanceError, ValidationError

@dataclass
class StockData:
    """
    Data container for stock information.
    
    This class provides a structured way to store stock data
    and is used by both the core client and providers.
    """
    ticker: str
    name: Optional[str] = None
    price: Optional[float] = None
    price_change: Optional[float] = None
    price_change_percentage: Optional[float] = None
    market_cap: Optional[float] = None
    analyst_count: Optional[int] = None
    target_price: Optional[float] = None
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    peg_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    short_float_pct: Optional[float] = None
    last_earnings: Optional[str] = None
    insider_buy_pct: Optional[float] = None
    insider_transactions: Optional[int] = None
    total_ratings: Optional[int] = None
    hold_pct: Optional[float] = None
    buy_pct: Optional[float] = None
    upside: Optional[float] = None
    sector: Optional[str] = None
    recommendation: Optional[str] = None

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
            raise YFinanceError("An error occurred")
            
        # Basic validation - more complex validation happens in providers
        if len(ticker) > 20:
            raise YFinanceError("An error occurred")
            
        return True