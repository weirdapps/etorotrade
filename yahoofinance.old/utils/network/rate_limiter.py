"""
Advanced rate limiting utilities to prevent API throttling.

This module contains tools for managing API request rates,
including adaptive delays and safe concurrent access patterns.

CANONICAL SOURCE:
This is the canonical source for synchronous rate limiting functionality. Other modules
that provide similar functionality are compatibility layers that import from 
this module. Always prefer to import directly from this module in new code:

    from yahoofinance.utils.network.rate_limiter import (
        AdaptiveRateLimiter, rate_limited, batch_process
    )

Key Components:
- AdaptiveRateLimiter: Class for adaptive rate limiting
- global_rate_limiter: Singleton instance for library-wide rate limiting
- rate_limited: Decorator for rate-limiting functions
- batch_process: Process items in batches with rate limiting

Example usage:
    # Rate-limited function
    @rate_limited(ticker_param='ticker')
    def fetch_data(ticker: str):
        # Fetch data with proper rate limiting
        
    # Process multiple items in batches
    results = batch_process(
        items=['AAPL', 'MSFT', 'GOOG'],
        process_func=fetch_data,
        batch_size=5
    )
"""

import time
import logging
import threading
from collections import deque
from typing import Dict, Optional, Callable, Any, TypeVar, List
from datetime import datetime

from ...core.config import RATE_LIMIT
from ...core.errors import RateLimitError

logger = logging.getLogger(__name__)

# Type variable for generic function wrapping
T = TypeVar('T')


class AdaptiveRateLimiter:
    """
    Advanced rate limiter with adaptive delays based on API response patterns.
    
    Features:
    - Thread-safe tracking of API calls
    - Exponential backoff for rate limit errors
    - Adaptive delays based on success/failure patterns
    - Individual ticker tracking for problematic symbols
    """
    
    def __init__(self,
                window_size: int = None,
                max_calls: int = None):
        """
        Initialize rate limiter with configurable window and limits.
        
        Args:
            window_size: Time window in seconds (default from config)
            max_calls: Maximum calls per window (default from config)
        """
        # Use config values or fallback to defaults
        self.window_size = window_size if window_size is not None else RATE_LIMIT["WINDOW_SIZE"]
        self.max_calls = max_calls if max_calls is not None else RATE_LIMIT["MAX_CALLS"]
        
        # Call tracking
        self.calls = deque(maxlen=1000)  # Timestamp queue
        self.errors = deque(maxlen=20)   # Recent errors
        
        # Delay settings
        self.base_delay = RATE_LIMIT["BASE_DELAY"]
        self.min_delay = RATE_LIMIT["MIN_DELAY"]
        self.max_delay = RATE_LIMIT["MAX_DELAY"]
        self.batch_delay = RATE_LIMIT["BATCH_DELAY"]
        
        # Performance tracking
        self.error_counts: Dict[str, int] = {}  # Track error counts per ticker
        self.success_streak = 0                 # Track successful calls
        
        # Thread safety
        self._lock = threading.RLock()
        
        # API call history (for diagnostics)
        self.api_call_history = deque(maxlen=100)
        
    def add_call(self, ticker: Optional[str] = None, endpoint: Optional[str] = None) -> None:
        """
        Record an API call with thread safety.
        
        Args:
            ticker: Optional ticker symbol associated with the call
            endpoint: Optional API endpoint identifier
        """
        with self._lock:
            now = time.time()
            self.calls.append(now)
            
            # Track call for diagnostics
            self.api_call_history.append({
                'timestamp': datetime.fromtimestamp(now).isoformat(),
                'ticker': ticker,
                'endpoint': endpoint
            })
            
            # Remove old calls outside the window
            while self.calls and self.calls[0] < now - self.window_size:
                self.calls.popleft()
                
            # Adjust base delay based on success
            self.success_streak += 1
            if self.success_streak >= 10 and self.base_delay > self.min_delay:
                self.base_delay = max(self.min_delay, self.base_delay * 0.9)
    
    def add_error(self, error: Exception, ticker: Optional[str] = None) -> None:
        """
        Record an error and adjust delays with thread safety.
        
        Args:
            error: The exception that occurred
            ticker: Optional ticker symbol associated with the error
        """
        with self._lock:
            now = time.time()
            self.errors.append(now)
            self.success_streak = 0
            
            # Track errors per ticker
            if ticker:
                self.error_counts[ticker] = self.error_counts.get(ticker, 0) + 1
            
            # Track error for diagnostics
            self.api_call_history.append({
                'timestamp': datetime.fromtimestamp(now).isoformat(),
                'ticker': ticker,
                'error': str(error),
                'error_type': error.__class__.__name__
            })
            
            # Exponential backoff based on recent errors
            recent_errors = sum(1 for t in self.errors if t > now - 300)  # Last 5 minutes
            
            if recent_errors >= 3:
                # Significant rate limiting, increase delay aggressively
                self.base_delay = min(self.base_delay * 2, self.max_delay)
                self.batch_delay = min(self.batch_delay * 1.5, self.max_delay)
                logger.warning(
                    f"Rate limiting detected. Increasing delays - Base: {self.base_delay:.1f}s, "
                    f"Batch: {self.batch_delay:.1f}s"
                )
                
                # Clear old errors to allow recovery
                if recent_errors >= 10:
                    self.errors.clear()
    
    def get_delay(self, ticker: Optional[str] = None) -> float:
        """
        Calculate needed delay based on recent activity and ticker history.
        Thread-safe implementation.
        
        Args:
            ticker: Optional ticker symbol to adjust delay for problematic symbols
            
        Returns:
            Recommended delay in seconds
        """
        with self._lock:
            now = time.time()
            recent_calls = sum(1 for t in self.calls if t > now - self.window_size)
            
            # Calculate load percentage (how close we are to the limit)
            load_percentage = recent_calls / self.max_calls if self.max_calls > 0 else 0
            
            # Base delay calculation
            if load_percentage >= 0.8:  # Near limit (80%+)
                delay = self.base_delay * 2
                logger.debug(f"High API load ({load_percentage:.1%}), increasing delay")
            elif load_percentage >= 0.5:  # Medium load (50-80%)
                delay = self.base_delay * 1.5
            elif self.errors:  # Recent errors
                delay = self.base_delay * 1.5
            else:
                delay = self.base_delay
                
            # Adjust for ticker-specific issues
            if ticker and self.error_counts.get(ticker, 0) > 0:
                # Increase delay for problematic tickers
                ticker_factor = 1 + (self.error_counts[ticker] * 0.5)
                delay *= ticker_factor
                logger.debug(
                    f"Ticker {ticker} has {self.error_counts[ticker]} errors, "
                    f"applying factor: {ticker_factor:.1f}"
                )
                
            return min(delay, self.max_delay)
    
    def get_batch_delay(self) -> float:
        """
        Get delay between batches with thread safety.
        
        Returns:
            Recommended batch delay in seconds
        """
        with self._lock:
            return self.batch_delay
    
    def should_skip_ticker(self, ticker: str) -> bool:
        """
        Determine if a ticker should be skipped due to excessive errors.
        Thread-safe implementation.
        
        Args:
            ticker: Ticker symbol to check
            
        Returns:
            True if ticker should be skipped, False otherwise
        """
        with self._lock:
            max_errors = 5  # Skip after 5 errors
            return self.error_counts.get(ticker, 0) >= max_errors
    
    def wait(self, ticker: Optional[str] = None) -> None:
        """
        Wait for the appropriate delay time.
        
        Args:
            ticker: Optional ticker symbol to adjust delay
        """
        delay = self.get_delay(ticker)
        if delay > 0:
            time.sleep(delay)
    
    def execute_with_rate_limit(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with rate limiting applied.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            Original exceptions from the function
        """
        # Extract ticker from kwargs if present
        ticker = kwargs.get('ticker')
        if ticker is None and args:
            # Try to find a string arg that might be a ticker
            for arg in args:
                if isinstance(arg, str) and len(arg) <= 5:
                    ticker = arg
                    break
        
        # Apply rate limiting delay
        self.wait(ticker)
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            # Record successful call
            self.add_call(ticker=ticker)
            return result
        except Exception as e:
            # Record error and re-raise
            self.add_error(e, ticker=ticker)
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current rate limiter status for diagnostics.
        
        Returns:
            Dictionary with current status
        """
        with self._lock:
            now = time.time()
            recent_calls = sum(1 for t in self.calls if t > now - self.window_size)
            recent_errors = sum(1 for t in self.errors if t > now - 300)  # Last 5 minutes
            
            return {
                'recent_calls': recent_calls,
                'max_calls': self.max_calls,
                'load_percentage': recent_calls / self.max_calls if self.max_calls > 0 else 0,
                'recent_errors': recent_errors,
                'problematic_tickers': {k: v for k, v in self.error_counts.items() if v > 0},
                'current_delays': {
                    'base_delay': self.base_delay,
                    'batch_delay': self.batch_delay
                },
                'success_streak': self.success_streak
            }


# Create a global rate limiter instance
global_rate_limiter = AdaptiveRateLimiter()


def rate_limited(ticker_param: Optional[str] = None):
    """
    Decorator for rate limiting functions that make API calls.
    
    Args:
        ticker_param: Name of the parameter containing ticker symbol
        
    Returns:
        Decorated function with rate limiting
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            return global_rate_limiter.execute_with_rate_limit(func, *args, **kwargs)
        
        # Preserve function metadata
        import functools
        return functools.wraps(func)(wrapper)
    
    return decorator


def batch_process(items: List[Any], processor: Callable[[Any], T], 
                  batch_size: Optional[int] = None) -> List[T]:
    """
    Process a list of items in batches with rate limiting.
    
    Args:
        items: List of items to process
        processor: Function to process each item
        batch_size: Number of items per batch (default from config)
        
    Returns:
        List of processed results
    """
    if batch_size is None:
        batch_size = RATE_LIMIT["BATCH_SIZE"]
    
    results = []
    total_batches = (len(items) - 1) // batch_size + 1
    
    for batch_num, i in enumerate(range(0, len(items), batch_size)):
        # Process current batch
        batch = items[i:i + batch_size]
        batch_results = [processor(item) for item in batch]
        results.extend([r for r in batch_results if r is not None])
        
        # Add adaptive delay between batches (except for last batch)
        if batch_num < total_batches - 1:
            success_count = sum(1 for r in batch_results if r is not None)
            success_rate = success_count / len(batch) if batch else 0
            
            # Calculate delay based on success rate
            if success_rate < 0.5:  # Poor success rate
                batch_delay = global_rate_limiter.batch_delay * 2
            elif success_rate > 0.75:  # Good success rate (threshold reduced from 0.8)
                batch_delay = max(global_rate_limiter.min_delay, 
                                  global_rate_limiter.batch_delay * 0.75)  # Faster reduction (from 0.8)
            else:
                batch_delay = global_rate_limiter.batch_delay
            
            logger.info(
                f"Batch {batch_num + 1}/{total_batches} complete "
                f"(Success rate: {success_rate:.1%}). Waiting {batch_delay:.1f} seconds..."
            )
            time.sleep(batch_delay)
    
    return results