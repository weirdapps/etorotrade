"""
Asynchronous rate limiting utilities for Yahoo Finance API.

This module provides utilities for rate limiting async API requests,
allowing for efficient handling of concurrent requests while respecting
rate limits.

CANONICAL SOURCE: This module is now the canonical source for async rate limiting functionality.
"""

import logging
import time
import asyncio
from functools import wraps
from typing import Dict, Any, Optional, Callable, TypeVar, cast, Coroutine

from ...core.config import RATE_LIMIT
from ...core.errors import RateLimitError

logger = logging.getLogger(__name__)

# Define a generic type variable for the return type
T = TypeVar('T')

class AsyncRateLimiter:
    """
    Thread-safe adaptive rate limiter for async API requests.
    
    This class provides a mechanism to limit the rate of async API requests
    based on a sliding window. It adaptively adjusts the delay between
    requests based on success and failure patterns.
    
    Attributes:
        window_size: Size of the sliding window in seconds
        max_calls: Maximum number of calls allowed in the window
        base_delay: Base delay between calls in seconds
        min_delay: Minimum delay after many successful calls in seconds
        max_delay: Maximum delay after errors in seconds
        call_timestamps: List of timestamps of recent calls
        delay: Current delay between calls in seconds
        success_streak: Number of consecutive successful calls
        failure_streak: Number of consecutive failed calls
        lock: Asyncio lock for synchronization
        ticker_specific_delays: Dictionary mapping tickers to delays
    """
    
    def __init__(
        self,
        window_size: Optional[int] = None,
        max_calls: Optional[int] = None,
        base_delay: Optional[float] = None,
        min_delay: Optional[float] = None,
        max_delay: Optional[float] = None
    ):
        """
        Initialize the rate limiter.
        
        Args:
            window_size: Size of the sliding window in seconds
            max_calls: Maximum number of calls allowed in the window
            base_delay: Base delay between calls in seconds
            min_delay: Minimum delay after many successful calls in seconds
            max_delay: Maximum delay after errors in seconds
        """
        # Set default values from configuration
        self.window_size = window_size or RATE_LIMIT["WINDOW_SIZE"]
        self.max_calls = max_calls or RATE_LIMIT["MAX_CALLS"]
        self.base_delay = base_delay or RATE_LIMIT["BASE_DELAY"]
        self.min_delay = min_delay or RATE_LIMIT["MIN_DELAY"]
        self.max_delay = max_delay or RATE_LIMIT["MAX_DELAY"]
        
        # Initialize state
        self.call_timestamps: list[float] = []
        self.delay = self.base_delay
        self.success_streak = 0
        self.failure_streak = 0
        
        # Initialize lock
        self.lock = asyncio.Lock()
        
        # Initialize ticker-specific delays
        self.ticker_specific_delays: Dict[str, float] = {}
        self.slow_tickers = RATE_LIMIT["SLOW_TICKERS"]
        
        logger.debug(f"Initialized async rate limiter with window_size={self.window_size}, "
                    f"max_calls={self.max_calls}, base_delay={self.base_delay}")
    
    def get_delay_for_ticker(self, ticker: Optional[str] = None) -> float:
        """
        Get the appropriate delay for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Delay in seconds
        """
        if ticker is None:
            return self.delay
        
        # Check if the ticker has a specific delay
        if ticker in self.ticker_specific_delays:
            return self.ticker_specific_delays[ticker]
        
        # Check if the ticker is in the slow tickers set
        if ticker in self.slow_tickers:
            # Use a longer delay for slow tickers
            return min(self.delay * 2, self.max_delay)
        
        return self.delay
    
    async def _clean_old_timestamps(self) -> None:
        """
        Remove timestamps that are outside the window.
        
        This method is called internally to maintain the sliding window.
        """
        current_time = time.time()
        async with self.lock:
            # Remove timestamps outside the window
            self.call_timestamps = [
                ts for ts in self.call_timestamps
                if current_time - ts <= self.window_size
            ]
    
    async def would_exceed_rate_limit(self) -> bool:
        """
        Check if making a call now would exceed the rate limit.
        
        Returns:
            True if making a call would exceed the rate limit, False otherwise
        """
        await self._clean_old_timestamps()
        async with self.lock:
            return len(self.call_timestamps) >= self.max_calls
    
    async def wait_if_needed(self, ticker: Optional[str] = None) -> None:
        """
        Wait if needed to avoid exceeding the rate limit.
        
        This method blocks until it's safe to make a call.
        
        Args:
            ticker: Ticker symbol for ticker-specific delay
        """
        # Check if we need to wait
        if await self.would_exceed_rate_limit():
            # Calculate how long to wait
            async with self.lock:
                if self.call_timestamps:
                    # Get the oldest timestamp in the window
                    oldest_ts = min(self.call_timestamps)
                    # Calculate when that timestamp will be outside the window
                    next_available = oldest_ts + self.window_size
                    # Calculate how long to wait
                    wait_time = next_available - time.time()
                    if wait_time > 0:
                        logger.warning(f"Rate limit would be exceeded. Waiting {wait_time:.2f} seconds.")
                        await asyncio.sleep(wait_time)
        
        # Apply appropriate delay based on success/failure pattern and ticker
        delay = self.get_delay_for_ticker(ticker)
        if delay > 0:
            await asyncio.sleep(delay)
    
    async def record_call(self) -> None:
        """
        Record that a call was made.
        
        This method should be called after making an API call.
        """
        async with self.lock:
            self.call_timestamps.append(time.time())
    
    async def record_success(self, ticker: Optional[str] = None) -> None:
        """
        Record a successful call and adjust delay.
        
        Args:
            ticker: Ticker symbol
        """
        async with self.lock:
            # Record success
            self.success_streak += 1
            self.failure_streak = 0
            
            # Adjust delay
            if self.success_streak >= 10:
                # Decrease delay after many consecutive successes
                self.delay = max(self.delay * 0.9, self.min_delay)
                # Reset success streak
                self.success_streak = 0
                
                logger.debug(f"Decreased delay to {self.delay:.2f} after 10 consecutive successes")
            
            # Update ticker-specific delay if needed
            if ticker is not None and ticker in self.ticker_specific_delays:
                # Decrease ticker-specific delay
                self.ticker_specific_delays[ticker] = max(
                    self.ticker_specific_delays[ticker] * 0.9,
                    self.delay  # Don't go below the global delay
                )
    
    async def record_failure(self, ticker: Optional[str] = None, is_rate_limit: bool = False) -> None:
        """
        Record a failed call and adjust delay.
        
        Args:
            ticker: Ticker symbol
            is_rate_limit: Whether the failure was due to rate limiting
        """
        async with self.lock:
            # Record failure
            self.failure_streak += 1
            self.success_streak = 0
            
            # Adjust delay
            if is_rate_limit or self.failure_streak >= 3:
                # Increase delay significantly for rate limit errors
                # or after several consecutive failures
                multiplier = 2.0 if is_rate_limit else 1.5
                self.delay = min(self.delay * multiplier, self.max_delay)
                
                logger.warning(
                    f"Increased delay to {self.delay:.2f} due to "
                    f"{'rate limiting' if is_rate_limit else 'consecutive failures'}"
                )
                
                # Reset failure streak
                self.failure_streak = 0
            
            # Update ticker-specific delay if needed
            if ticker is not None:
                # Use a longer delay for this ticker in the future
                current_delay = self.ticker_specific_delays.get(ticker, self.delay)
                new_delay = min(current_delay * 2, self.max_delay)
                self.ticker_specific_delays[ticker] = new_delay
                
                logger.debug(f"Set ticker-specific delay for {ticker} to {new_delay:.2f}")
                
                # Add to slow tickers set if not already there
                if ticker not in self.slow_tickers:
                    self.slow_tickers.add(ticker)
    
    async def reset(self) -> None:
        """
        Reset the rate limiter to its initial state.
        
        This method clears all recorded calls and resets the delay.
        """
        async with self.lock:
            self.call_timestamps = []
            self.delay = self.base_delay
            self.success_streak = 0
            self.failure_streak = 0
            self.ticker_specific_delays = {}

# Create a global async rate limiter instance
global_async_rate_limiter = AsyncRateLimiter()

def async_rate_limited(
    func: Optional[Callable[..., Coroutine[Any, Any, T]]] = None,
    *,
    limiter: Optional[AsyncRateLimiter] = None,
    ticker_arg: Optional[str] = None
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Decorator for rate-limiting async function calls.
    
    This decorator can be used to rate-limit any async function that makes API calls.
    It automatically handles waiting between calls and adjusting delays based
    on success and failure patterns.
    
    Args:
        func: Async function to rate-limit
        limiter: Rate limiter to use (defaults to global_async_rate_limiter)
        ticker_arg: Name of the argument that contains the ticker symbol
        
    Returns:
        Decorated async function
    """
    # Set default rate limiter
    if limiter is None:
        limiter = global_async_rate_limiter
        
    def decorator(f: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(f)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get ticker from args or kwargs if ticker_arg is specified
            ticker = None
            if ticker_arg is not None:
                if ticker_arg in kwargs:
                    ticker = kwargs[ticker_arg]
                elif args and len(args) > 0:
                    # This is a simplification - it assumes the ticker is the first argument
                    # For more complex cases, inspect.signature should be used
                    ticker = args[0]
            
            # Wait if needed to avoid exceeding rate limit
            await limiter.wait_if_needed(ticker)
            
            # Record the call
            await limiter.record_call()
            
            try:
                # Call the function
                result = await f(*args, **kwargs)
                
                # Record success
                await limiter.record_success(ticker)
                
                return result
            except Exception as e:
                # Record failure
                is_rate_limit = isinstance(e, RateLimitError)
                await limiter.record_failure(ticker, is_rate_limit)
                
                # Re-raise the exception
                raise
        
        return cast(Callable[..., Coroutine[Any, Any, T]], wrapper)
    
    # Handle the case where the decorator is used without parentheses
    if func is not None:
        return decorator(func)
    
    return decorator