#!/usr/bin/env python3
"""
Tests for rate limiting functionality

This test file verifies core rate limiting utilities that protect against API throttling:
- AdaptiveRateLimiter functionality and configuration
- Thread safety and concurrency controls
- Error recovery and backoff strategies
"""

import time
import unittest
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

from yahoofinance.utils.network.rate_limiter import (
    AdaptiveRateLimiter, rate_limited, batch_process, global_rate_limiter
)
from yahoofinance.core.errors import RateLimitError, APIError

# Create a reusable test fixture module
from tests.fixtures import create_bulk_fetch_mocks

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


class TestRateLimiterCore(unittest.TestCase):
    """Test the core functionality of the adaptive rate limiter."""
    
    def setUp(self):
        # Create a fresh rate limiter for each test
        self.rate_limiter = AdaptiveRateLimiter(window_size=10, max_calls=20)
    
    def test_add_call_tracking(self):
        """Test that calls are properly tracked."""
        self.assertEqual(len(self.rate_limiter.calls), 0)
        
        # Add several calls
        for _ in range(5):
            self.rate_limiter.add_call()
        
        self.assertEqual(len(self.rate_limiter.calls), 5)
    
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
    
    def test_error_tracking(self):
        """Test error tracking and backoff behavior."""
        # Add several errors
        for i in range(3):
            self.rate_limiter.add_error(Exception(f"Test error {i}"), "AAPL")
        
        # Should have 3 errors recorded
        self.assertEqual(len(self.rate_limiter.errors), 3)
        
        # AAPL should be marked as problematic
        self.assertEqual(self.rate_limiter.error_counts.get("AAPL"), 3)
        
        # Base delay should have increased
        self.assertGreater(self.rate_limiter.base_delay, 2.0)
        
    def test_ticker_specific_tracking(self):
        """Test ticker-specific error tracking."""
        # Add errors for different tickers
        tickers = ["AAPL", "MSFT", "GOOGL", "AAPL", "AAPL"]
        for i, ticker in enumerate(tickers):
            self.rate_limiter.add_error(Exception(f"Error {i}"), ticker)
        
        # Check per-ticker error counts
        self.assertEqual(self.rate_limiter.error_counts.get("AAPL"), 3)
        self.assertEqual(self.rate_limiter.error_counts.get("MSFT"), 1)
        self.assertEqual(self.rate_limiter.error_counts.get("GOOGL"), 1)
    
    def test_should_skip_ticker(self):
        """Test that problematic tickers are identified for skipping."""
        # Add 5 errors for AAPL
        for i in range(5):
            self.rate_limiter.add_error(Exception(f"Test error {i}"), "AAPL")
        
        # Should recommend skipping AAPL
        self.assertTrue(self.rate_limiter.should_skip_ticker("AAPL"))
        
        # But not MSFT
        self.assertFalse(self.rate_limiter.should_skip_ticker("MSFT"))
    
    def test_get_delay(self):
        """Test delay calculation based on load and ticker history."""
        # Initial delay should be the base delay
        initial_delay = self.rate_limiter.get_delay()
        self.assertEqual(initial_delay, self.rate_limiter.base_delay)
        
        # Add many calls to increase load - just at 80% of max_calls (16 out of 20)
        for _ in range(16):
            self.rate_limiter.add_call()
        
        # Delay should increase due to higher load
        high_load_delay = self.rate_limiter.get_delay()
        self.assertGreaterEqual(high_load_delay, initial_delay)
        
        # Add errors for a specific ticker
        self.rate_limiter.add_error(Exception("Test error"), "AAPL")
        
        # Delay for problematic ticker should be higher
        ticker_delay = self.rate_limiter.get_delay("AAPL")
        self.assertGreater(ticker_delay, high_load_delay)


class TestRateLimiterDecorators(unittest.TestCase):
    """Test rate limiting decorators and utilities."""
    
    def test_rate_limited_decorator(self):
        """Test the rate_limited decorator."""
        mock_limiter = MagicMock()
        # Configure mock to actually call the function
        mock_limiter.execute_with_rate_limit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        with patch('yahoofinance.utils.network.rate_limiter.global_rate_limiter', mock_limiter):
            calls = []
            
            @rate_limited
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
        processed = []
        
        def process_item(item):
            processed.append(item)
            return item * 2
        
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
        call_count = 0
        lock = threading.Lock()
        
        def increment_calls():
            nonlocal call_count
            # Get delay (reads state) - we only care about the call, not the value
            _ = limiter.get_delay()
            # Small sleep to increase chance of race conditions
            time.sleep(0.01)
            # Add call (writes state)
            limiter.add_call()
            with lock:
                call_count += 1
        
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
        self.assertEqual(call_count, 20)
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
        self.assertEqual(results, [i * 2 for i in range(20)])


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


if __name__ == "__main__":
    unittest.main()