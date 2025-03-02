#!/usr/bin/env python3
"""
Tests for rate limiting functionality

This test file verifies the rate limiting utilities that protect against API throttling:
- AdaptiveRateLimiter for thread-safe rate limiting
- Rate limiting decorators for synchronous and async functions
- Batch processing with rate limiting
"""

import asyncio
import time
import unittest
import logging
import threading
from unittest.mock import patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from yahoofinance.utils.rate_limiter import (
    AdaptiveRateLimiter, rate_limited, batch_process, global_rate_limiter
)
from yahoofinance.errors import RateLimitError, APIError

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRateLimiter(unittest.TestCase):
    """Test the adaptive rate limiter functionality."""
    
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
        
        # Delay should increase or remain the same due to higher load
        high_load_delay = self.rate_limiter.get_delay()
        self.assertGreaterEqual(high_load_delay, initial_delay)
        
        # Add errors for a specific ticker
        self.rate_limiter.add_error(Exception("Test error"), "AAPL")
        
        # Delay for problematic ticker should be higher
        ticker_delay = self.rate_limiter.get_delay("AAPL")
        self.assertGreater(ticker_delay, high_load_delay)
    
    def test_rate_limited_decorator(self):
        """Test the rate_limited decorator."""
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
    
    def test_thread_safety(self):
        """Test thread safety of the rate limiter."""
        # Create a shared rate limiter
        limiter = AdaptiveRateLimiter(window_size=5, max_calls=10)
        call_count = 0
        lock = threading.Lock()
        
        def increment_calls():
            nonlocal call_count
            # Get delay (reads state)
            delay = limiter.get_delay()
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


if __name__ == "__main__":
    unittest.main()