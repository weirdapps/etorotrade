import pytest
from unittest.mock import Mock, patch
import time
from collections import deque
from yahoofinance.display import RateLimitTracker

@pytest.fixture
def rate_limiter():
    return RateLimitTracker(window_size=60, max_calls=100)

def test_init():
    """Test initialization of RateLimitTracker"""
    limiter = RateLimitTracker(window_size=30, max_calls=50)
    assert limiter.window_size == 30
    assert limiter.max_calls == 50
    assert limiter.base_delay == pytest.approx(2.0)
    assert limiter.min_delay == pytest.approx(1.0)
    assert limiter.max_delay == pytest.approx(30.0)
    assert limiter.batch_delay == pytest.approx(5.0)
    assert limiter.success_streak == 0
    assert isinstance(limiter.calls, deque)
    assert isinstance(limiter.errors, deque)
    assert isinstance(limiter.error_counts, dict)

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
    assert rate_limiter.base_delay > 2.0  # Initial base delay
    assert rate_limiter.batch_delay > 5.0  # Initial batch delay

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