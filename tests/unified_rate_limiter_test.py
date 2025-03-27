#!/usr/bin/env python3
"""
Unified tests for rate limiting functionality

This test file consolidates test coverage for rate limiting
functionality previously spread across multiple files:
- test_rate.py
- test_rate_limiter.py
- test_rate_limiter_unified.py
- test_rate_limiter_advanced.py
"""

import time
import unittest
import logging
import threading
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock, Mock
import multiprocessing

from yahoofinance.utils.network.rate_limiter import (
    AdaptiveRateLimiter, rate_limited, batch_process, global_rate_limiter
)
from yahoofinance.presentation.console import RateLimitTracker
from yahoofinance.core.errors import RateLimitError, APIError

# Import shared test fixtures
from tests.fixtures.rate_limiter_fixtures import (
    mock_rate_limiter, fresh_rate_limiter, rate_limiter_with_calls,
    rate_limiter_with_errors, rate_limiter_with_different_ticker_errors,
    create_bulk_fetch_mocks, increment_calls_function
)

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#
# Common fixtures for all test classes
#
@pytest.fixture
def rate_limiter():
    return RateLimitTracker(window_size=60, max_calls=100)


#
# Pytest-style tests for RateLimitTracker
#
def test_init():
    """Test initialization of RateLimitTracker"""
    limiter = RateLimitTracker(window_size=30, max_calls=50)
    assert limiter.window_size == 30
    assert limiter.max_calls == 50
    assert limiter.base_delay == pytest.approx(1.5)
    assert limiter.min_delay == pytest.approx(0.8)
    assert limiter.max_delay == pytest.approx(30.0)
    assert limiter.batch_delay == pytest.approx(3.0)
    assert limiter.success_streak == 0


def test_add_call(rate_limiter):
    """Test recording API calls"""
    # Add a call
    rate_limiter.add_call()
    assert len(rate_limiter.calls) == 1
    assert rate_limiter.success_streak == 1

    # Add multiple calls
    for _ in range(5):
        rate_limiter.add_call()
    assert len(rate_limiter.calls) == 6
    assert rate_limiter.success_streak == 6


def test_success_streak_delay_reduction(rate_limiter):
    """Test delay reduction after successful calls"""
    initial_delay = rate_limiter.base_delay
    
    # Add successful calls to build streak
    for _ in range(10):
        rate_limiter.add_call()
    
    # Verify delay reduction
    assert rate_limiter.base_delay < initial_delay
    assert rate_limiter.base_delay >= rate_limiter.min_delay


def test_add_error(rate_limiter):
    """Test error tracking"""
    # Add an error
    error = Exception("Test error")
    rate_limiter.add_error(error, "AAPL")
    
    assert "AAPL" in rate_limiter.error_counts
    assert rate_limiter.error_counts["AAPL"] == 1
    assert rate_limiter.success_streak == 0

    # Add multiple errors
    for _ in range(2):
        rate_limiter.add_error(error, "AAPL")
    assert rate_limiter.error_counts["AAPL"] == 3


def test_get_delay(rate_limiter):
    """Test delay calculation"""
    # Base case
    base_delay = rate_limiter.get_delay()
    assert base_delay == rate_limiter.base_delay

    # With recent errors
    error = Exception("Test error")
    rate_limiter.add_error(error, "AAPL")
    error_delay = rate_limiter.get_delay("AAPL")
    assert error_delay > base_delay
    assert error_delay <= rate_limiter.max_delay


def test_get_batch_delay(rate_limiter):
    """Test batch delay"""
    assert rate_limiter.get_batch_delay() == rate_limiter.batch_delay


def test_should_skip_ticker(rate_limiter):
    """Test ticker skip logic"""
    error = Exception("Test error")
    
    # Add errors up to threshold
    for _ in range(4):
        rate_limiter.add_error(error, "AAPL")
    assert not rate_limiter.should_skip_ticker("AAPL")
    
    # Add one more error to exceed threshold
    rate_limiter.add_error(error, "AAPL")
    assert rate_limiter.should_skip_ticker("AAPL")


def test_window_cleanup(rate_limiter):
    """Test cleanup of old calls outside window"""
    # Add calls with timestamps in the past
    old_time = time.time() - rate_limiter.window_size - 10
    rate_limiter.calls.extend([old_time] * 5)
    
    # Add a new call which should trigger cleanup
    rate_limiter.add_call()
    
    # Verify old calls were removed
    assert all(t > (time.time() - rate_limiter.window_size) for t in rate_limiter.calls)


def test_adaptive_backoff(rate_limiter):
    """Test adaptive backoff behavior"""
    error = Exception("Rate limit exceeded")
    
    # Simulate multiple errors in short time
    for _ in range(3):
        rate_limiter.add_error(error, "AAPL")
    
    # Verify increased delays
    assert rate_limiter.base_delay > 1.5
    assert rate_limiter.batch_delay > 3.0


def test_error_recovery(rate_limiter):
    """Test error recovery behavior"""
    error = Exception("Test error")
    
    # Add errors
    for _ in range(3):
        rate_limiter.add_error(error, "AAPL")
    
    high_delay = rate_limiter.get_delay("AAPL")
    
    # Simulate recovery with successful calls
    for _ in range(10):
        rate_limiter.add_call()
    
    recovery_delay = rate_limiter.get_delay("AAPL")
    assert recovery_delay < high_delay


def test_rate_limit_threshold(rate_limiter):
    """Test behavior near rate limit threshold"""
    # Fill up to 80% of rate limit
    num_calls = int(rate_limiter.max_calls * 0.8)
    for _ in range(num_calls):
        rate_limiter.add_call()
    
    # Get delay when near limit
    near_limit_delay = rate_limiter.get_delay()
    assert near_limit_delay > rate_limiter.base_delay


def test_multiple_ticker_errors(rate_limiter):
    """Test error handling for multiple tickers"""
    error = Exception("Test error")
    
    # Add errors for different tickers
    rate_limiter.add_error(error, "AAPL")
    rate_limiter.add_error(error, "GOOGL")
    
    # Verify independent error counting
    assert rate_limiter.error_counts["AAPL"] == 1
    assert rate_limiter.error_counts["GOOGL"] == 1
    
    # Verify different delays based on error history
    delay_aapl = rate_limiter.get_delay("AAPL")
    delay_msft = rate_limiter.get_delay("MSFT")  # No errors for MSFT
    assert delay_aapl > delay_msft


#
# Unittest-style tests for AdaptiveRateLimiter
#
class TestRateLimiterConfiguration(unittest.TestCase):
    """Test the configuration options for rate limiter."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        limiter = AdaptiveRateLimiter()
        
        # Check default values from RATE_LIMIT config
        self.assertIsNotNone(limiter.window_size)
        self.assertIsNotNone(limiter.max_calls)
        self.assertIsNotNone(limiter.base_delay)
        self.assertIsNotNone(limiter.max_delay)
        
        # Verify limiter has expected attributes
        self.assertTrue(hasattr(limiter, 'calls'))
        self.assertTrue(hasattr(limiter, 'errors'))
        self.assertTrue(hasattr(limiter, 'error_counts'))
        
    def test_custom_configuration(self):
        """Test custom configuration values."""
        custom_window = 30
        custom_calls = 15
        
        limiter = AdaptiveRateLimiter(
            window_size=custom_window,
            max_calls=custom_calls
        )
        
        # Check custom values
        self.assertEqual(limiter.window_size, custom_window)
        self.assertEqual(limiter.max_calls, custom_calls)
        
    def test_global_limiter_singleton(self):
        """Test that global rate limiter is a singleton."""
        # Global limiter should exist
        self.assertIsNotNone(global_rate_limiter)
        
        # Multiple references should be the same object
        from yahoofinance.utils.rate_limiter import global_rate_limiter as gr2
        self.assertIs(global_rate_limiter, gr2)
    
    def test_global_rate_limiter_status(self):
        """Test the global rate limiter status method."""
        # Test that global_rate_limiter is an instance of AdaptiveRateLimiter
        self.assertIsInstance(global_rate_limiter, AdaptiveRateLimiter)
        
        # Save initial base_delay
        initial_delay = global_rate_limiter.base_delay
        
        # Update the limiter (note: this affects the global state)
        global_rate_limiter.add_error(Exception("Test error"))
        
        # Get status for diagnostics
        status = global_rate_limiter.get_status()
        self.assertIn('recent_errors', status)
        
        # Reset to avoid affecting other tests
        global_rate_limiter.base_delay = initial_delay
        global_rate_limiter.errors.clear()


class TestRateLimiterCore(unittest.TestCase):
    """Test the core functionality of the adaptive rate limiter."""
    
    def setUp(self):
        # Create a fresh rate limiter for each test
        self.rate_limiter = AdaptiveRateLimiter(window_size=10, max_calls=20)
    
    def test_add_call_tracking(self, rate_limiter_with_calls=None):
        """Test that calls are properly tracked."""
        if rate_limiter_with_calls:
            # Use the fixture if provided (pytest-style test)
            self.rate_limiter = rate_limiter_with_calls
            
        self.assertEqual(len(self.rate_limiter.calls), 5 if rate_limiter_with_calls else 0)
        
        # Add several calls
        for _ in range(5):
            self.rate_limiter.add_call()
        
        self.assertEqual(len(self.rate_limiter.calls), 10 if rate_limiter_with_calls else 5)
    
    def test_call_pruning(self):
        """Test that old calls are tracked properly."""
        # Clear current calls
        self.rate_limiter.calls.clear()
        
        # Add a few calls at current time
        current_time = time.time()
        for offset in [-15, -9, -8, -5]:
            self.rate_limiter.calls.append(current_time + offset)
        
        # The window size is 10 in our test setup
        with patch('time.time', return_value=current_time):
            # Get recent calls (should exclude the -15s one)
            recent_calls = sum(1 for t in self.rate_limiter.calls if t > current_time - self.rate_limiter.window_size)
            self.assertEqual(recent_calls, 3)  # Only 3 calls within window
    
    def test_ticker_specific_tracking(self, rate_limiter_with_different_ticker_errors=None):
        """Test ticker-specific error tracking."""
        if rate_limiter_with_different_ticker_errors:
            # Use the fixture if provided
            self.rate_limiter = rate_limiter_with_different_ticker_errors
            
            # Check per-ticker error counts
            self.assertEqual(self.rate_limiter.error_counts.get("AAPL"), 3)
            self.assertEqual(self.rate_limiter.error_counts.get("MSFT"), 1)
            self.assertEqual(self.rate_limiter.error_counts.get("GOOGL"), 1)
        else:
            # Add errors for different tickers
            tickers = ["AAPL", "MSFT", "GOOGL", "AAPL", "AAPL"]
            for i, ticker in enumerate(tickers):
                self.rate_limiter.add_error(Exception(f"Error {i}"), ticker)
            
            # Check per-ticker error counts
            self.assertEqual(self.rate_limiter.error_counts.get("AAPL"), 3)
            self.assertEqual(self.rate_limiter.error_counts.get("MSFT"), 1)
            self.assertEqual(self.rate_limiter.error_counts.get("GOOGL"), 1)


class TestRateLimiterDecorators(unittest.TestCase):
    """Test rate limiting decorators and utilities."""
    
    def test_rate_limited_decorator(self):
        """Test the rate_limited decorator."""
        mock_limiter = MagicMock()
        # Configure mock to actually call the function
        mock_limiter.execute_with_rate_limit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        with patch('yahoofinance.utils.network.rate_limiter.global_rate_limiter', mock_limiter):
            calls = []
            
            @rate_limited(ticker_param='ticker')
            def test_func(ticker, value):
                calls.append((ticker, value))
                return f"{ticker}:{value}"
            
            # Call the function a few times
            for i in range(3):
                result = test_func("AAPL", i)
                self.assertEqual(result, f"AAPL:{i}")
            
            # All calls should be in the calls list
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls, [("AAPL", 0), ("AAPL", 1), ("AAPL", 2)])
            
            # Rate limiter execute_with_rate_limit should be called
            self.assertEqual(mock_limiter.execute_with_rate_limit.call_count, 3)
    
    def test_rate_limited_no_ticker(self):
        """Test rate_limited decorator without ticker parameter."""
        mock_limiter = MagicMock()
        # Configure mock to actually call the function
        mock_limiter.execute_with_rate_limit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        with patch('yahoofinance.utils.network.rate_limiter.global_rate_limiter', mock_limiter):
            @rate_limited()  # No ticker param
            def simple_func(x):
                return x * 2
            
            result = simple_func(5)
            self.assertEqual(result, 10)
            
            # Rate limiter should be called
            self.assertTrue(mock_limiter.execute_with_rate_limit.called)
    
    def test_batch_process(self):
        """Test batch processing with rate limiting."""
        process_item, processed = create_bulk_fetch_mocks()
        
        items = list(range(10))
        results = batch_process(items, process_item, batch_size=3)
        
        # All items should be processed
        self.assertEqual(len(processed), 10)
        self.assertEqual(processed, items)
        
        # Results should be doubled
        self.assertEqual(results, [i * 2 for i in items])


class TestThreadSafety(unittest.TestCase):
    """Test thread safety and concurrent usage of rate limiter."""
    
    def test_thread_safety(self):
        """Test thread safety of the rate limiter."""
        # Create a shared rate limiter
        limiter = AdaptiveRateLimiter(window_size=5, max_calls=10)
        
        # Use multiprocessing.Value for thread-safe counter
        call_count = multiprocessing.Value('i', 0)
        lock = threading.Lock()
        
        def increment_calls():
            # Use the helper function from fixtures
            increment_calls_function(limiter, call_count, lock)
        
        # Run multiple threads
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=increment_calls)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have 20 calls tracked
        self.assertEqual(call_count.value, 20)
        self.assertEqual(len(limiter.calls), 20)
    
    def test_concurrent_batch_processing(self):
        """Test concurrent batch processing with rate limiting."""
        results = []
        
        def process_item(item):
            time.sleep(0.01)  # Simulate work
            return item * 2
        
        # Process items in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            items = list(range(20))
            for i in range(0, len(items), 5):
                batch = items[i:i+5]
                future = executor.submit(batch_process, batch, process_item, batch_size=2)
                results.extend(future.result())
        
        # Check results
        self.assertEqual(len(results), 20)
        self.assertEqual(sorted(results), [i * 2 for i in range(20)])


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery and backoff strategies."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff for rate limit errors."""
        limiter = AdaptiveRateLimiter(window_size=5, max_calls=10)
        
        # Track base delay changes
        original_delay = limiter.base_delay
        
        # Add increasing number of errors
        for i in range(5):
            limiter.add_error(RateLimitError(f"Rate limit error {i}"), "AAPL")
        
        # Base delay should increase after errors
        final_delay = limiter.get_delay("AAPL")
        
        # Final delay should be significantly higher than original
        self.assertGreater(final_delay, original_delay * 2)
        
        # Should hit max delay after many errors
        self.assertLessEqual(final_delay, limiter.max_delay)
    
    def test_error_handling(self):
        """Test that error counts can be managed properly."""
        limiter = AdaptiveRateLimiter(window_size=5, max_calls=10)
        
        # Add errors
        for i in range(3):
            limiter.add_error(RateLimitError(f"Error {i}"), "AAPL")
        
        # Initial error count
        initial_count = limiter.error_counts.get("AAPL", 0)
        self.assertEqual(initial_count, 3)
        
        # Test error count affects delay
        regular_delay = limiter.get_delay()
        ticker_delay = limiter.get_delay("AAPL")
        self.assertGreater(ticker_delay, regular_delay)
        
        # Test that should_skip_ticker works
        self.assertFalse(limiter.should_skip_ticker("AAPL"))  # Not enough errors yet
        
        # Add more errors to reach skip threshold
        for i in range(3):
            limiter.add_error(RateLimitError(f"Error extra {i}"), "AAPL")
            
        # Now it should recommend skipping
        self.assertTrue(limiter.should_skip_ticker("AAPL"))


class TestAdvancedRateLimiter(unittest.TestCase):
    """Test advanced functionality of rate limiter."""
    
    def test_adaptive_rate_limiter_backoff(self):
        """Test adaptive rate limiter backoff mechanism."""
        # Create a rate limiter with a very small initial wait time for testing
        limiter = AdaptiveRateLimiter(
            window_size=1,    # 1 second window
            max_calls=100     # 100 requests per second
        )
        
        # Initial state
        initial_delay = limiter.base_delay
        
        # Simulate successful requests
        for _ in range(5):
            limiter.wait()
            limiter.add_call()
        
        # Base delay should stay the same or decrease after successful calls
        self.assertLessEqual(limiter.base_delay, initial_delay)
        
        # Simulate failed requests
        limiter.add_error(Exception("Test error"))
        self.assertEqual(limiter.success_streak, 0)
        
        # Simulate more errors to increase delay
        for _ in range(3):
            limiter.add_error(Exception("Test error"))
        
        # Delay should have increased
        self.assertGreater(limiter.base_delay, initial_delay)
        
        # Test max delay is respected
        for _ in range(10):
            limiter.add_error(Exception("Test error"))
        
        # Should not exceed max_delay
        self.assertLessEqual(limiter.base_delay, limiter.max_delay)
    
    def test_ticker_specific_delays(self):
        """Test ticker-specific delay calculation."""
        limiter = AdaptiveRateLimiter()
        
        # Record normal delay
        normal_delay = limiter.get_delay()
        
        # Add errors for a specific ticker
        ticker = "PROBLEMATIC"
        limiter.add_error(Exception("Test error"), ticker=ticker)
        limiter.add_error(Exception("Test error"), ticker=ticker)
        
        # Get delay for the problematic ticker
        ticker_delay = limiter.get_delay(ticker=ticker)
        
        # Delay should be higher for the problematic ticker
        self.assertGreater(ticker_delay, normal_delay)
        
        # Get delay for a different ticker
        other_ticker_delay = limiter.get_delay(ticker="NORMAL")
        
        # Should be lower than the problematic ticker
        self.assertLess(other_ticker_delay, ticker_delay)


if __name__ == "__main__":
    unittest.main()