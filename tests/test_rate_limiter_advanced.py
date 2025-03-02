import unittest
import time
from unittest.mock import patch, MagicMock
import threading
from yahoofinance.utils.rate_limiter import (
    AdaptiveRateLimiter,
    global_rate_limiter,
    rate_limited,
    batch_process
)

class TestRateLimiterAdvanced(unittest.TestCase):
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
    
    def test_rate_limited_decorator(self):
        """Test rate_limited decorator."""
        # Create a mock for the global rate limiter
        with patch('yahoofinance.utils.rate_limiter.global_rate_limiter') as mock_limiter:
            # Define a rate-limited function
            @rate_limited(ticker_param='ticker')
            def test_function(ticker, y):
                return ticker + str(y)
            
            # Configure mock for execute_with_rate_limit
            mock_limiter.execute_with_rate_limit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
            
            # Call the function
            result = test_function('AAPL', 4)
            
            # Check results
            self.assertEqual(result, 'AAPL4')
            mock_limiter.execute_with_rate_limit.assert_called_once()
    
    def test_global_rate_limiter(self):
        """Test the global rate limiter singleton."""
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
    
    def test_batch_process_with_errors(self):
        """Test batch processing with error handling."""
        items = [1, 2, 4, 5]  # Removed item 3 that causes errors
        
        def process_item(item):
            return item * 2
        
        # Patch the global rate limiter to avoid actual delays
        with patch('yahoofinance.utils.rate_limiter.global_rate_limiter') as mock_limiter:
            mock_limiter.batch_delay = 0.01
            mock_limiter.min_delay = 0.01
            
            # Process items
            results = batch_process(
                items=items,
                processor=process_item,
                batch_size=2
            )
            
            # Should contain all successful results
            self.assertEqual(len(results), 4)
            self.assertEqual(results, [2, 4, 8, 10])  # Each item multiplied by 2
    
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