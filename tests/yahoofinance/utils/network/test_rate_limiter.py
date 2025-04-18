"""
Comprehensive rate limiter tests.

This module provides centralized testing for rate limiting functionality across
the entire codebase, testing both legacy and new implementations.

Key features tested:
- Basic functionality of AdaptiveRateLimiter
- Thread safety and concurrency
- Rate tracking and call recording
- Integration with rate_limited decorator
- Error handling and recovery strategies
- Batch processing with rate limiting
"""

import time
import threading
import pytest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock, Mock

# Import both legacy and new implementations to ensure compatibility
from yahoofinance.utils.network.rate_limiter import (
    AdaptiveRateLimiter, rate_limited, batch_process, global_rate_limiter
)
from yahoofinance.presentation.console import RateLimitTracker
from yahoofinance.core.errors import RateLimitError, APIError
from yahoofinance.core.config import RATE_LIMIT


#
# Base AdaptiveRateLimiter tests
#
class TestAdaptiveRateLimiter:
    """Tests for the AdaptiveRateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initializes with correct parameters."""
        limiter = AdaptiveRateLimiter(
            window_size=120,
            max_calls=100
        )
        
        assert limiter.window_size == 120
        assert limiter.max_calls == 100
        assert limiter.base_delay == RATE_LIMIT["BASE_DELAY"]
        assert limiter.min_delay == RATE_LIMIT["MIN_DELAY"]
        assert limiter.max_delay == RATE_LIMIT["MAX_DELAY"]
        assert isinstance(limiter.calls, deque)
        assert limiter.success_streak == 0
    
    def test_add_call(self):
        """Test adding calls to the limiter."""
        limiter = AdaptiveRateLimiter()
        initial_streak = limiter.success_streak
        
        # Add a call
        limiter.add_call("AAPL", "info")
        
        # Should increment success streak
        assert limiter.success_streak == initial_streak + 1
        assert len(limiter.calls) == 1
        assert "AAPL" in limiter.ticker_calls
    
    def test_add_error(self):
        """Test error handling in the limiter."""
        limiter = AdaptiveRateLimiter()
        limiter.success_streak = 10
        
        # Add an error
        error = RateLimitError("Rate limit exceeded")
        limiter.add_error(error, "AAPL", "info")
        
        # Should reset success streak and record error
        assert limiter.success_streak == 0
        assert limiter.consecutive_errors > 0
        assert "AAPL" in limiter.ticker_errors
    
    def test_clean_old_calls(self):
        """Test removal of old calls outside the window."""
        limiter = AdaptiveRateLimiter(window_size=1)
        
        # Add some calls
        limiter.add_call("AAPL", "info")
        
        # Wait for window to pass
        time.sleep(1.1)
        
        # Add another call to trigger cleaning
        limiter.add_call("MSFT", "info")
        
        # Old calls should be removed
        assert len(limiter.calls) == 1
        assert limiter.calls[0][1] == "MSFT"
    
    @with_retry
    
    
def test_get_delay(self):
        """Test delay calculation based on call history."""
        limiter = AdaptiveRateLimiter(max_calls=5, window_size=5)
        
        # No calls, should use base delay
        initial_delay = limiter.get_delay()
        assert initial_delay == limiter.base_delay
        
        # Add calls to approach limit
        for _ in range(4):
            limiter.add_call()
        
        # Near limit, delay should increase
        high_load_delay = limiter.get_delay()
        assert high_load_delay > initial_delay
        
        # Add error to trigger longer delay
        limiter.add_error(RateLimitError("Rate limit exceeded"))
        error_delay = limiter.ge@with_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
def test_get_ticker_delay(rror_delay > high_load_delay
    
    def test_get_ticker_delay(self):
        """Test ticker-specific delays."""
        limiter = AdaptiveRateLimiter()
        
        # Add some calls for a specific ticker
        for _ in range(3):
            limiter.add_call("AAPL")
        
        # Add an error for a specific ticker
        limiter.add_error(RateLimitError("Rate limit"), "MSFT")
        
        # Check delays for different tickers
        aapl_delay = limiter.get_delay("AAPL")
        msft_delay = limiter.get_delay("MSFT")
        other_delay = limiter.get_delay("GOOG")
        
        # Ticker with error should have highest delay
        assert msft_delay > aapl_delay
        assert aapl_delay > other_delay


#
# Thread safety tests
#
class TestRateLimiterThreadSafety:
    """Tests for thread safety of the rate limiter."""
    
    def test_concurrent_access(self):
        """Test rate limiter with concurrent access from multiple threads."""
        limiter = AdaptiveRateLimiter(max_calls=100)
        
        def worker():
            for _ in range(10):
                limiter.add_call()
                time.sleep(0.01)
        
        # Create and start threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all calls were recorded
        assert len(limiter.calls) == 50
    
    def test_lock_during_operations(self):
        """Test that lock is properly acquired during operations."""
        limiter = AdaptiveRateLimiter()
        
        # Mock the lock to track acquire/release
        original_lock = limiter.lock
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock()
        mock_lock.__exit__ = MagicMock()
        limiter.lock = mock_lock
        
        # Call methods that should use the lock
        limiter.add_call()
        limiter.get_delay()
        limiter.add_error(Exception("Test error"))
        
        # Verify lock was used
        assert mock_lock.__enter__.call_count >= 3
        assert mock_lock.__exit__.call_count >= 3
        
        # Restore original lock
        limiter.lock = original_lock


#
# Decorator tests
#
class TestRateLimitedDecorator:
    """Tests for the rate_limited decorator."""
    
    def test_decorator_basic(self):
        """Test basic functionality of rate_limited decorator."""
        test_limiter = AdaptiveRateLimiter(max_calls=10, window_size=1)
        
        # Define a function with the decorator
        @rate_limited(limiter=test_limiter)
        def test_function(x):
            return x * 2
        
        # Call the function multiple times
        results = [test_function(i) for i in range(5)]
        
        # Verify results and rate limiting
        assert results == [0, 2, 4, 6, 8]
        assert len(test_limiter.calls) == 5
    
    def test_decorator_with_ticker(self):
        """Test rate_limited decorator with ticker parameter."""
        test_limiter = AdaptiveRateLimiter()
        
        # Define a function with ticker parameter
        @rate_limited("ticker", limiter=test_limiter)
        def get_stock_info(ticker, info_type="summary"):
            return f"{ticker}:{info_type}"
        
        # Call the function with different tickers
        get_stock_info("AAPL")
        get_stock_info("MSFT")
        get_stock_info("GOOG", info_type="detailed")
        
        # Verify ticker-specific tracking
        assert "AAPL" in test_limiter.ticker_calls
        assert "MSFT" in test_limiter.ticker_calls
        assert "GOOG" in test_limiter.ticker_calls
    
    def test_decorator_error_handling(self):
        """Test error handling in rate_limited decorator."""
        test_limiter = AdaptiveRateLimiter()
        
        # Define a function that raises an error
        @rate_limited(limiter=test_limiter)
        def error_function():
            raise RateLimitError("Test rate limit error")
        
        # Call the function and verify error handling
        with pytest.raises(RateLimitError):
            error_function()
        
        # Error should be recorded
        assert test_limiter.consecutive_errors > 0


#
# Batch processing tests
#
class TestBatchProcessing:
    """Tests for batch processing with rate limiting."""
    
    def test_batch_process(self):
        """Test batch_process function with rate limiting."""
        test_limiter = AdaptiveRateLimiter()
        
        # Define a processor function
        def process_item(item):
            return item * 2
        
        # Process a batch of items
        items = [1, 2, 3, 4, 5]
        results = batch_process(items, process_item, limiter=test_limiter)
        
        # Verify results and rate limiting
        assert results == [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]
        assert len(test_limiter.calls) == 5
    
    def test_batch_process_with_errors(self):
        """Test batch_process with error handling."""
        test_limiter = AdaptiveRateLimiter()
        
        # Define a processor that sometimes fails
        def process_item(item):
            if item == 3:
                raise ValueError("Test error")
            return item * 2
        
        # Process a batch with an error
        items = [1, 2, 3, 4, 5]
        results = batch_process(
            items, 
            process_item, 
            limiter=test_limiter,
            continue_on_error=True
        )
        
        # Check results and error handling
        assert len(results) == 5
        assert results[0] == (1, 2)
        assert results[1] == (2, 4)
        assert results[2][0] == 3
        assert results[2][1] is None  # Error resulted in None
        assert results[3] == (4, 8)
        assert results[4] == (5, 10)
        
        # Error should be recorded
        assert test_limiter.consecutive_errors > 0


#
# Legacy RateLimitTracker tests (for compatibility)
#
class TestRateLimitTracker:
    """Tests for the legacy RateLimitTracker from display.py."""
    
    def test_init(self):
        """Test initialization of RateLimitTracker."""
        limiter = RateLimitTracker(window_size=30, max_calls=50)
        assert limiter.window_size == 30
        assert limiter.max_calls == 50
        assert limiter.base_delay == pytest.approx(1.5, 0.001)
        assert limiter.min_delay == pytest.approx(0.8, 0.001)
    
    def test_add_call(self):
        """Test adding calls and delay calculation."""
        limiter = RateLimitTracker(window_size=1, max_calls=5)
        
        # Add some calls
        for _ in range(3):
            limiter.add_call()
        
        # Check delay calculation
        delay = limiter.calculate_delay()
        assert delay >= limiter.min_delay
        
        # Add more calls to approach limit
        for _ in range(2):
            limiter.add_call()
        
        # Delay should be higher with more calls
        high_load_delay = limiter.calculate_delay()
        assert high_load_delay > delay


# Global rate limiter tests
def test_global_rate_limiter():
    """Test the global rate limiter instance."""
    # The global limiter should be an instance of AdaptiveRateLimiter
    assert isinstance(global_rate_limiter, AdaptiveRateLimiter)
    
    # Basic functionality
    initial_calls = len(global_rate_limiter.calls)
    global_rate_limiter.add_call("TEST")
    assert len(global_rate_limiter.calls) == initial_calls + 1


# Import compatibility tests
def test_import_compatibility():
    """Test compatibility between different import patterns."""
    # Import from both locations and verify they refer to the same implementation
    from yahoofinance.utils.rate_limiter import AdaptiveRateLimiter as RL1
    from yahoofinance.utils.network.rate_limiter import AdaptiveRateLimiter as RL2
    
    # Create instances
    limiter1 = RL1()
    limiter2 = RL2()
    
    # They should have the same attributes
    assert dir(limiter1) == dir(limiter2)
    assert limiter1.base_delay == limiter2.base_delay
    assert limiter1.min_delay == limiter2.min_delay
    assert limiter1.max_delay == limiter2.max_delay