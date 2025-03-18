"""
Unit tests for rate limiter functionality.

This module tests the rate limiting implementations in the core.rate_limiter
and utils.network.rate_limiter modules to ensure proper throttling and
adaptivity in API call frequency.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import concurrent.futures
from collections import deque

from yahoofinance.core.errors import RateLimitError
from yahoofinance.utils.network.rate_limiter import AdaptiveRateLimiter, rate_limited
from yahoofinance.core.config import RATE_LIMIT
from yahoofinance.display import RateLimitTracker

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
        assert len(limiter.api_call_history) == 1
    
    def test_add_error(self):
        """Test adding errors to the limiter."""
        limiter = AdaptiveRateLimiter()
        
        # Set success streak
        limiter.success_streak = 5
        
        # Add an error
        error = RateLimitError("Test error")
        limiter.add_error(error, ticker="AAPL")
        
        # Should reset success streak and record error
        assert limiter.success_streak == 0
        assert len(limiter.errors) == 1
        assert len(limiter.api_call_history) == 1
        assert "AAPL" in limiter.error_counts
        assert limiter.error_counts["AAPL"] == 1
    
    def test_get_delay(self):
        """Test delay calculation."""
        limiter = AdaptiveRateLimiter()
        
        # Base case - no calls, no errors
        base_delay = limiter.get_delay()
        assert base_delay == limiter.base_delay
        
        # Add some calls (near limit)
        current_time = time.time()
        for _ in range(int(limiter.max_calls * 0.9)):  # 90% of max
            limiter.calls.append(current_time)
        
        # Should increase delay due to high load
        high_load_delay = limiter.get_delay()
        assert high_load_delay > base_delay
        
        # Add errors for a specific ticker
        limiter.error_counts["AAPL"] = 2
        
        # Should increase delay for problematic ticker
        ticker_delay = limiter.get_delay("AAPL")
        assert ticker_delay > limiter.get_delay("MSFT")  # Compare to non-problematic ticker
    
    def test_batch_delay(self):
        """Test batch delay calculation."""
        limiter = AdaptiveRateLimiter()
        
        # Should return the configured batch delay
        assert limiter.get_batch_delay() == limiter.batch_delay
    
    def test_wait_method(self):
        """Test wait method with mocked sleep."""
        limiter = AdaptiveRateLimiter()
        
        with patch('time.sleep') as mock_sleep:
            limiter.wait("AAPL")
            mock_sleep.assert_called_once()
    
    def test_execute_with_rate_limit(self):
        """Test execute_with_rate_limit method."""
        limiter = AdaptiveRateLimiter()
        mock_func = Mock(return_value="test_result")
        
        with patch.object(limiter, 'wait') as mock_wait:
            result = limiter.execute_with_rate_limit(mock_func, "AAPL", arg2="test")
            
            # Should call wait, then function, then add_call
            mock_wait.assert_called_once_with("AAPL")
            mock_func.assert_called_once_with("AAPL", arg2="test")
            assert result == "test_result"
    
    def test_error_handling(self):
        """Test error handling in execute_with_rate_limit."""
        limiter = AdaptiveRateLimiter()
        mock_func = Mock(side_effect=RateLimitError("Test error"))
        
        with patch.object(limiter, 'wait'):
            with patch.object(limiter, 'add_error') as mock_add_error:
                with pytest.raises(RateLimitError):
                    limiter.execute_with_rate_limit(mock_func, "AAPL")
                
                # Should call add_error
                mock_add_error.assert_called_once()
    
    def test_get_status(self):
        """Test getting status information."""
        limiter = AdaptiveRateLimiter()
        
        # Add some calls and errors
        current_time = time.time()
        limiter.calls.append(current_time)
        limiter.errors.append(current_time)
        limiter.error_counts["AAPL"] = 2
        
        # Get status
        status = limiter.get_status()
        
        # Verify status contains expected fields
        assert "recent_calls" in status
        assert "max_calls" in status
        assert "load_percentage" in status
        assert "recent_errors" in status
        assert "problematic_tickers" in status
        assert "current_delays" in status
        assert "AAPL" in status["problematic_tickers"]


class TestRateLimitTracker:
    """Tests for the RateLimitTracker class from display.py."""
    
    def test_initialization(self):
        """Test RateLimitTracker initializes with correct parameters."""
        tracker = RateLimitTracker(
            window_size=120,
            max_calls=100
        )
        
        assert tracker.window_size == 120
        assert tracker.max_calls == 100
        assert tracker.base_delay == RATE_LIMIT["BASE_DELAY"]
        assert tracker.min_delay == RATE_LIMIT["MIN_DELAY"]
        assert tracker.max_delay == RATE_LIMIT["MAX_DELAY"]
        assert isinstance(tracker.calls, deque)
        assert tracker.error_counts == {}
    
    def test_add_call(self):
        """Test adding calls to the limiter."""
        tracker = RateLimitTracker()
        initial_streak = tracker.success_streak
        
        # Add a call
        tracker.add_call()
        
        # Should increment success streak and add to calls
        assert tracker.success_streak == initial_streak + 1
        assert len(tracker.calls) == 1
    
    def test_add_error(self):
        """Test adding errors to the limiter."""
        tracker = RateLimitTracker()
        
        # Set success streak
        tracker.success_streak = 5
        
        # Add an error
        error = RateLimitError("Test error")
        tracker.add_error(error, ticker="AAPL")
        
        # Should reset success streak and update error counts
        assert tracker.success_streak == 0
        assert len(tracker.errors) == 1
        assert "AAPL" in tracker.error_counts
        assert tracker.error_counts["AAPL"] == 1
    
    def test_get_delay(self):
        """Test delay calculation."""
        tracker = RateLimitTracker()
        
        # Base case - no calls, no errors
        base_delay = tracker.get_delay()
        assert base_delay == tracker.base_delay
        
        # Add some calls (near limit)
        current_time = time.time()
        for _ in range(int(tracker.max_calls * 0.8)):  # 80% of max
            tracker.calls.append(current_time)
        
        # Should increase delay due to high load
        high_load_delay = tracker.get_delay()
        assert high_load_delay > base_delay
        
        # Add errors for a specific ticker
        tracker.error_counts["AAPL"] = 2
        
        # Should increase delay for problematic ticker
        ticker_delay = tracker.get_delay("AAPL")
        assert ticker_delay > tracker.get_delay("MSFT")  # Compare to non-problematic ticker
    
    def test_batch_delay(self):
        """Test batch delay calculation."""
        tracker = RateLimitTracker()
        
        # Should return the configured batch delay
        assert tracker.get_batch_delay() == tracker.batch_delay
    
    def test_should_skip_ticker(self):
        """Test should_skip_ticker method."""
        tracker = RateLimitTracker()
        
        # Should not skip ticker with few errors
        tracker.error_counts["AAPL"] = 2
        assert not tracker.should_skip_ticker("AAPL")
        
        # Should skip ticker with excessive errors
        tracker.error_counts["MSFT"] = 5
        assert tracker.should_skip_ticker("MSFT")


@pytest.mark.parametrize("limiter_class", [AdaptiveRateLimiter])
class TestRateLimiterIntegration:
    """Integration tests for rate limiter implementations."""
    
    def test_adaptive_behavior(self, limiter_class):
        """Test rate limiter adapts delay based on success/failure patterns."""
        limiter = limiter_class()
        
        # Record initial state
        initial_base_delay = limiter.base_delay
        
        # Simulate some errors
        for _ in range(3):
            limiter.add_error(RateLimitError("Test error"), ticker="AAPL")
        
        # Base delay should increase
        assert limiter.base_delay > initial_base_delay
    
    def test_handling_rate_limit_errors(self, limiter_class):
        """Test handling of rate limit errors."""
        limiter = limiter_class()
        
        # Mock a function that raises RateLimitError
        mock_func = Mock(side_effect=RateLimitError("Rate limited"))
        
        with patch.object(limiter, 'add_error') as mock_add_error:
            # Apply rate limiting and call function
            with pytest.raises(RateLimitError):
                limiter.execute_with_rate_limit(mock_func, "AAPL")
                
            # Should call add_error
            mock_add_error.assert_called_once()