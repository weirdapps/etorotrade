"""
Core client module for v2 implementation.

This module defines the base YFinanceClient class used by various providers
and compatibility layers. It provides a foundation for API communication
with appropriate error handling and configuration.
"""

import logging
from dataclasses import dataclass
from typing import Any

from .config import RATE_LIMIT
from .errors import ValidationError


@dataclass
class StockData:
    """
    Data container for stock information.

    This class provides a structured way to store stock data
    and is used by both the core client and providers.
    """

    ticker: str
    name: str | None = None
    price: float | None = None
    price_change: float | None = None
    price_change_percentage: float | None = None
    market_cap: float | None = None
    analyst_count: int | None = None
    target_price: float | None = None
    pe_trailing: float | None = None
    pe_forward: float | None = None
    peg_ratio: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    short_float_pct: float | None = None
    last_earnings: str | None = None
    insider_buy_pct: float | None = None
    insider_transactions: int | None = None
    total_ratings: int | None = None
    hold_pct: float | None = None
    buy_pct: float | None = None
    sector: str | None = None
    recommendation: str | None = None


# Set up logging
logger = logging.getLogger(__name__)


class YFinanceClient:
    """
    Base client for Yahoo Finance data access.

    This class provides common functionality used by providers and
    serves as a compatibility layer for v1 code.
    """

    def __init__(self, max_retries: int = None, timeout: int = None):
        """
        Initialize YFinance client.

        Args:
            max_retries: Maximum number of retry attempts for API calls
            timeout: API request timeout in seconds
        """
        self.max_retries = max_retries or RATE_LIMIT["MAX_RETRY_ATTEMPTS"]
        self.timeout = timeout or RATE_LIMIT["API_TIMEOUT"]

        logger.debug(
            f"Initialized YFinanceClient with max_retries={self.max_retries}, timeout={self.timeout}"
        )

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
            error_details: dict[str, Any] = {
                "ticker": ticker,
                "issue": "invalid_type_or_empty",
                "expected": "non-empty string",
                "received": type(ticker).__name__,
            }
            raise ValidationError(
                f"Invalid ticker: must be a non-empty string, got {type(ticker).__name__}",
                error_details,
            )

        # Basic validation - more complex validation happens in providers
        if len(ticker) > 20:
            length_details: dict[str, Any] = {
                "ticker": ticker,
                "issue": "invalid_length",
                "max_length": 20,
                "actual_length": len(ticker),
            }
            raise ValidationError(
                f"Invalid ticker '{ticker}': exceeds maximum length of 20 characters",
                length_details,
            )

        return True
