"""
Rate limiting utilities for API calls.

This module provides rate limiting and adaptive delay management for API requests.
"""

import time
from collections import deque
from typing import Deque, Dict

from yahoofinance.core.logging import get_logger


logger = get_logger(__name__)


class RateLimitTracker:
    """Tracks API calls and manages rate limiting with adaptive delays"""

    def __init__(self, window_size: int = 60, max_calls: int = 60):
        """
        Initialize RateLimitTracker.

        Args:
            window_size: Time window in seconds (default: 60)
            max_calls: Maximum calls allowed in window (default: 60)
        """
        self.window_size = window_size
        self.max_calls = max_calls
        self.calls: Deque[float] = deque(maxlen=1000)  # Timestamp queue
        self.errors: Deque[float] = deque(maxlen=20)  # Recent errors
        self.base_delay = 1.0  # Base delay between calls
        self.min_delay = 0.5  # Minimum delay
        self.max_delay = 30.0  # Maximum delay
        self.batch_delay = 0.0   # No delay between batches for optimal performance
        self.error_counts: Dict[str, int] = {}  # Track error counts per ticker
        self.success_streak = 0  # Track successful calls

    def add_call(self):
        """Record an API call"""
        now = time.time()
        self.calls.append(now)

        # Remove old calls outside the window
        while self.calls and self.calls[0] < now - self.window_size:
            self.calls.popleft()

        # Adjust base delay based on success
        self.success_streak += 1
        if self.success_streak >= 10 and self.base_delay > self.min_delay:
            self.base_delay = max(self.min_delay, self.base_delay * 0.9)

    def add_error(self, error: Exception, ticker: str):
        """Record an error and adjust delays"""
        now = time.time()
        self.errors.append(now)
        self.success_streak = 0

        # Track errors per ticker
        self.error_counts[ticker] = self.error_counts.get(ticker, 0) + 1

        # Exponential backoff based on recent errors
        recent_errors = sum(1 for t in self.errors if t > now - 300)  # Last 5 minutes

        if recent_errors >= 3:
            # Significant rate limiting, increase delay aggressively
            self.base_delay = min(self.base_delay * 2, self.max_delay)
            self.batch_delay = min(self.batch_delay * 1.5, self.max_delay)
            logger.warning(
                f"Rate limiting detected. Increasing delays - Base: {self.base_delay:.1f}s, Batch: {self.batch_delay:.1f}s"
            )

            # Clear old errors to allow recovery
            if recent_errors >= 10:
                self.errors.clear()

    def get_delay(self, ticker: str = None) -> float:
        """Calculate needed delay based on recent activity and ticker history"""
        now = time.time()
        recent_calls = sum(1 for t in self.calls if t > now - self.window_size)

        # Base delay calculation
        if recent_calls >= self.max_calls * 0.8:  # Near limit
            delay = self.base_delay * 2
        elif self.errors:  # Recent errors
            delay = self.base_delay * 1.5
        else:
            delay = self.base_delay

        # Adjust for ticker-specific issues
        if ticker and self.error_counts.get(ticker, 0) > 0:
            delay *= 1 + (self.error_counts[ticker] * 0.5)  # Increase delay for problematic tickers

        return min(delay, self.max_delay)

    def get_batch_delay(self) -> float:
        """Get delay between batches"""
        return self.batch_delay

    def should_skip_ticker(self, ticker: str) -> bool:
        """Determine if a ticker should be skipped due to excessive errors"""
        return self.error_counts.get(ticker, 0) >= 5  # Skip after 5 errors
