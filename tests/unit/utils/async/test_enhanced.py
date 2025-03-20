"""
Unit tests for enhanced async utilities.

This module contains tests for the enhanced async utilities, including
AsyncRateLimiter, retry_async_with_backoff, gather_with_concurrency,
process_batch_async, and the various decorators.
"""

import asyncio
import pytest
import time
from unittest.mock import patch, AsyncMock, MagicMock, call

from yahoofinance_v2.utils.async_utils.enhanced import (
    AsyncRateLimiter,
    retry_async_with_backoff,
    gather_with_concurrency,
    process_batch_async,
    async_rate_limited,
    enhanced_async_rate_limited
)
from yahoofinance_v2.utils.network.circuit_breaker import CircuitOpenError


@pytest.fixture
def async_rate_limiter():
    """Create a test async rate limiter with controlled configuration"""
    return AsyncRateLimiter(
        window_size=5,
        max_calls=10,
        base_delay=0.01,
        min_delay=0.005,
        max_delay=0.1
    )


@pytest.mark.asyncio
async def test_async_rate_limiter_wait():
    """Test AsyncRateLimiter wait function."""
    limiter = AsyncRateLimiter(
        window_size=1,
        max_calls=3,
        base_delay=0.01
    )
    
    # Wait should return the delay
    delay1 = await limiter.wait()
    assert delay1 >= 0
    
    # Check that call times are recorded
    assert len(limiter.call_times) == 1
    
    # Make multiple calls to reach the limit
    await limiter.wait()
    await limiter.wait()
    
    assert len(limiter.call_times) == 3
    
    # Next call should wait for oldest to exit window
    with patch('asyncio.sleep') as mock_sleep:
        start_time = time.time()
        # Mock time.time to simulate time passing during the async sleep
        with patch('time.time', side_effect=[
            start_time,  # Initial check
            start_time,  # After sleep when rechecking
            start_time + 0.5  # After waiting, for recording new call time
        ]):
            await limiter.wait()
        
        # Should have tried to sleep to wait for window
        mock_sleep.assert_called()


@pytest.mark.asyncio
async def test_async_rate_limiter_record_success():
    """Test AsyncRateLimiter record_success function."""
    limiter = AsyncRateLimiter(
        base_delay=0.1,
        min_delay=0.05
    )
    
    # Record multiple successes to reduce delay
    for _ in range(5):
        await limiter.record_success()
    
    assert limiter.success_streak == 5
    assert limiter.failure_streak == 0
    
    # Delay should be reduced after 5 successes
    assert limiter.current_delay < 0.1


@pytest.mark.asyncio
async def test_async_rate_limiter_record_failure():
    """Test AsyncRateLimiter record_failure function."""
    limiter = AsyncRateLimiter(
        base_delay=0.1,
        max_delay=0.5
    )
    
    # Record failures
    await limiter.record_failure(is_rate_limit=False)
    assert limiter.failure_streak == 1
    assert limiter.success_streak == 0
    # Non-rate-limit failure increases delay by factor of 1.5
    assert round(limiter.current_delay, 2) == 0.15
    
    # Record rate limit failure
    await limiter.record_failure(is_rate_limit=True)
    assert limiter.failure_streak == 2
    # Rate-limit failure increases delay by factor of 2
    assert round(limiter.current_delay, 2) == 0.30


@pytest.mark.asyncio
async def test_retry_async_with_backoff_success():
    """Test retry_async_with_backoff with successful function."""
    mock_func = AsyncMock(return_value="success")
    
    result = await retry_async_with_backoff(
        mock_func, "arg1", kwarg1="value1", max_retries=3
    )
    
    assert result == "success"
    mock_func.assert_called_once_with("arg1", kwarg1="value1")


@pytest.mark.asyncio
async def test_retry_async_with_backoff_with_retries():
    """Test retry_async_with_backoff with retries before success."""
    # Function fails twice then succeeds
    mock_func = AsyncMock(side_effect=[
        ValueError("fail1"), 
        ValueError("fail2"), 
        "success"
    ])
    
    # Mock asyncio.sleep to avoid actual waiting
    with patch('asyncio.sleep', AsyncMock()) as mock_sleep:
        result = await retry_async_with_backoff(
            mock_func, "arg1", max_retries=3, base_delay=0.1
        )
    
    assert result == "success"
    assert mock_func.call_count == 3
    # Should sleep twice between retries
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_retry_async_with_backoff_max_retries_exceeded():
    """Test retry_async_with_backoff with max retries exceeded."""
    mock_func = AsyncMock(side_effect=ValueError("always fails"))
    
    with patch('asyncio.sleep', AsyncMock()), \
         pytest.raises(ValueError, match="always fails"):
        await retry_async_with_backoff(
            mock_func, max_retries=2, base_delay=0.1
        )
    
    # Function called initial + 2 retries = 3 times
    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_retry_async_with_backoff_with_circuit_breaker():
    """Test retry_async_with_backoff with circuit breaker."""
    # Create mocks for circuit breaker
    mock_circuit = MagicMock()
    mock_circuit._should_allow_request = MagicMock(return_value=True)
    mock_circuit.state = "CLOSED"
    mock_circuit.get_metrics = MagicMock(return_value={})
    mock_circuit.record_success = MagicMock()
    mock_circuit.record_failure = MagicMock()
    
    # Create a function that returns a value
    mock_func = AsyncMock(return_value="success value")
    
    with patch('yahoofinance_v2.utils.network.circuit_breaker.get_async_circuit_breaker',
               return_value=mock_circuit):
        
        result = await retry_async_with_backoff(
            mock_func, "arg1", circuit_name="test_circuit"
        )
    
    assert result == "success value"
    # Verify check was performed using _should_allow_request
    mock_circuit._should_allow_request.assert_called()
    # Verify function was properly executed
    mock_func.assert_called_once_with("arg1")
    # Verify success was recorded
    mock_circuit.record_success.assert_called_once()


@pytest.mark.asyncio
async def test_retry_async_with_backoff_never_retries_circuit_open():
    """Test retry_async_with_backoff never retries CircuitOpenError."""
    from yahoofinance_v2.utils.network.circuit_breaker import CircuitState

    mock_func = AsyncMock()
    # Create a proper mock for circuit breaker with correct state
    mock_circuit = MagicMock()
    mock_circuit._should_allow_request = MagicMock(return_value=False)
    mock_circuit.state = CircuitState.OPEN  # Use the actual enum value
    mock_circuit.name = "test_circuit"
    mock_circuit.get_metrics = MagicMock(return_value={})
    
    # Patch both get_async_circuit_breaker and import CircuitOpenError
    with patch('yahoofinance_v2.utils.network.circuit_breaker.get_async_circuit_breaker',
               return_value=mock_circuit):
        
        # Should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await retry_async_with_backoff(
                mock_func, circuit_name="test_circuit", max_retries=3
            )
    
    # Verify _should_allow_request was called
    mock_circuit._should_allow_request.assert_called_once()
    
    # Function should not be called when circuit is open
    mock_func.assert_not_called()


@pytest.mark.asyncio
async def test_gather_with_concurrency():
    """Test gather_with_concurrency limits concurrency."""
    # Create tasks that track their start/end times
    results = []
    concurrency_tracker = set()
    max_concurrency = 0
    
    async def tracked_task(i):
        nonlocal max_concurrency
        concurrency_tracker.add(i)
        max_concurrency = max(max_concurrency, len(concurrency_tracker))
        await asyncio.sleep(0.01)  # Small delay to ensure overlap
        concurrency_tracker.remove(i)
        return i * 2
    
    # Create 10 tasks with concurrency limit of 3
    tasks = [tracked_task(i) for i in range(10)]
    results = await gather_with_concurrency(tasks, limit=3)
    
    assert results == [i * 2 for i in range(10)]
    assert max_concurrency <= 3  # Should never exceed concurrency limit


@pytest.mark.asyncio
async def test_process_batch_async():
    """Test process_batch_async processes items in batches with concurrency."""
    processed_items = []
    
    async def processor(item):
        await asyncio.sleep(0.01)  # Small delay
        processed_items.append(item)
        return item * 10
    
    # Process 10 items in batches of 3 with concurrency of 2
    items = list(range(10))
    
    with patch('asyncio.sleep', AsyncMock()) as mock_sleep:
        results = await process_batch_async(
            items,
            processor,
            batch_size=3,
            concurrency=2,
            delay_between_batches=0.05
        )
    
    # Verify all items processed
    assert set(processed_items) == set(items)
    
    # Verify results dict maps items to their results
    assert results == {i: i * 10 for i in items}
    
    # Should sleep between batches (4 batches - 1 = 3 sleeps)
    assert mock_sleep.call_count == 3
    
    # Ensure batches had the right delay
    for call_args in mock_sleep.call_args_list:
        assert call_args[0][0] == 0.05


@pytest.mark.asyncio
async def test_async_rate_limited_decorator():
    """Test async_rate_limited decorator."""
    limiter = AsyncRateLimiter(base_delay=0.01)
    
    # Patch wait and record methods to track calls
    limiter.wait = AsyncMock(return_value=0.01)
    limiter.record_success = AsyncMock()
    limiter.record_failure = AsyncMock()
    
    # Create decorated function
    @async_rate_limited(limiter)
    async def test_func(arg1, kwarg1=None):
        return f"{arg1}-{kwarg1}"
    
    # Test successful execution
    result = await test_func("test", kwarg1="value")
    
    assert result == "test-value"
    limiter.wait.assert_called_once()
    limiter.record_success.assert_called_once()
    limiter.record_failure.assert_not_called()
    
    # Test failure
    limiter.wait.reset_mock()
    limiter.record_success.reset_mock()
    
    @async_rate_limited(limiter)
    async def fail_func():
        raise ValueError("test error")
    
    with pytest.raises(ValueError, match="test error"):
        await fail_func()
    
    limiter.wait.assert_called_once()
    limiter.record_success.assert_not_called()
    limiter.record_failure.assert_called_once()
    
    # The rate limit flag should be False for non-rate-limit errors
    assert limiter.record_failure.call_args[1]["is_rate_limit"] is False
    
    # Test rate limit detection
    limiter.record_failure.reset_mock()
    
    @async_rate_limited(limiter)
    async def rate_limit_func():
        raise Exception("rate limit exceeded")
    
    with pytest.raises(Exception, match="rate limit exceeded"):
        await rate_limit_func()
    
    # The rate limit flag should be True for rate limit errors
    assert limiter.record_failure.call_args[1]["is_rate_limit"] is True


@pytest.mark.asyncio
async def test_enhanced_async_rate_limited_decorator():
    """Test enhanced_async_rate_limited decorator combining rate limiting, circuit breaking, and retries."""
    limiter = AsyncRateLimiter(base_delay=0.01)
    
    # Mock all the underlying functionality
    limiter.wait = AsyncMock(return_value=0.01)
    limiter.record_success = AsyncMock()
    
    mock_circuit = MagicMock()
    mock_circuit.execute_async = AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
    
    # Create test function
    async def test_func(arg1, kwarg1=None):
        return f"{arg1}-{kwarg1}"
    
    # Add the decorator
    decorated = enhanced_async_rate_limited(
        circuit_name="test_circuit",
        max_retries=2,
        rate_limiter=limiter
    )(test_func)
    
    # Test with circuit breaker and retries
    with patch('yahoofinance_v2.utils.async.enhanced.retry_async_with_backoff', 
               AsyncMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))), \
         patch('yahoofinance_v2.utils.network.circuit_breaker.get_async_circuit_breaker',
               return_value=mock_circuit):
        
        result = await decorated("test", kwarg1="value")
    
    assert result == "test-value"
    
    # Test without retries
    decorated_no_retry = enhanced_async_rate_limited(
        circuit_name="test_circuit",
        max_retries=0,
        rate_limiter=limiter
    )(test_func)
    
    with patch('yahoofinance_v2.utils.network.circuit_breaker.get_async_circuit_breaker',
               return_value=mock_circuit):
        
        result = await decorated_no_retry("test", kwarg1="value")
    
    assert result == "test-value"
    mock_circuit.execute_async.assert_called()
    
    # Test without circuit breaker
    decorated_no_circuit = enhanced_async_rate_limited(
        circuit_name=None,
        max_retries=0,
        rate_limiter=limiter
    )(test_func)
    
    mock_circuit.execute_async.reset_mock()
    
    result = await decorated_no_circuit("test", kwarg1="value")
    
    assert result == "test-value"
    mock_circuit.execute_async.assert_not_called()  # Should not call circuit breaker