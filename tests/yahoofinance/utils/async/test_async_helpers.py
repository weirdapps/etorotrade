"""
Comprehensive tests for asynchronous utilities.

This module provides tests for the asynchronous utilities in yahoofinance.utils.network.async
including rate limiting, gathering, batch processing, and retry mechanisms.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from yahoofinance.utils.network.async_utils import (
    AsyncRateLimiter,
    async_rate_limited,
    gather_with_rate_limit,
    process_batch_async,
    retry_async,
    global_async_limiter
)
from yahoofinance.core.errors import RateLimitError, APIError


# Helper functions for testing
async def async_identity(x):
    """Simple async function that returns its input."""
    return x

async def async_sleep_and_return(x, delay=0.1):
    """Async function that sleeps and returns its input."""
    await asyncio.sleep(delay)
    return x

async def async_error(error_type=Exception):
    """Async function that raises an error."""
    raise error_type("Test error")


# Fixtures
@pytest.fixture
def async_limiter():
    """Create a fresh async rate limiter for each test."""
    return AsyncRateLimiter(max_concurrency=5)


# Tests for AsyncRateLimiter
class TestAsyncRateLimiter:
    """Tests for the AsyncRateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initializing the async rate limiter."""
        limiter = AsyncRateLimiter(max_concurrency=3)
        assert limiter.semaphore._value == 3
    
    @pytest.mark.asyncio
    async def test_execute_success(self, async_limiter):
        """Test successful execution with rate limiting."""
        result = await async_limiter.execute(async_identity, 42)
        assert result == 42
    
    @pytest.mark.asyncio
    async def test_execute_error(self, async_limiter):
        """Test error handling during execution."""
        with pytest.raises(ValueError):
            await async_limiter.execute(
                lambda: asyncio.create_task(async_error(ValueError))
            )
    
    @pytest.mark.asyncio
    async def test_concurrency_control(self):
        """Test that concurrency is properly limited."""
        limiter = AsyncRateLimiter(max_concurrency=2)
        
        # Keep track of concurrent executions
        max_concurrent = 0
        current_concurrent = 0
        
        async def tracked_function(x):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.1)
            current_concurrent -= 1
            return x
        
        # Run multiple tasks
        tasks = [limiter.execute(tracked_function, i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify concurrency was limited
        assert max_concurrent <= 2
        assert results == list(range(5))


# Tests for async_rate_limited decorator
class TestAsyncRateLimitedDecorator:
    """Tests for the async_rate_limited decorator."""
    
    @pytest.mark.asyncio
    async def test_basic_decoration(self):
        """Test basic functionality of the decorator."""
        # Unused variable removed
        
        @async_rate_limited()
        async def test_func(x):
            return x * 2
        
        result = await test_func(5)
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_ticker_parameter(self):
        """Test with ticker parameter extraction."""
        # Unused variable removed
        
        # Define a function with the decorator that uses ticker
        # Use a different parameter name than 'ticker' to avoid conflict
        @async_rate_limited(ticker_param='symbol')
        async def get_stock_data(symbol):
            return f"Data for {symbol}"
        
        # Call with positional and keyword args
        result1 = await get_stock_data("AAPL")
        result2 = await get_stock_data(symbol="MSFT")
        
        assert result1 == "Data for AAPL"
        assert result2 == "Data for MSFT"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in decorated function."""
        @async_rate_limited()
        async def error_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await error_func()


# Tests for gather_with_rate_limit
class TestGatherWithRateLimit:
    """Tests for the gather_with_rate_limit function."""
    
    @pytest.mark.asyncio
    async def test_basic_gathering(self):
        """Test basic gathering of tasks."""
        tasks = [async_identity(i) for i in range(5)]
        results = await gather_with_rate_limit(tasks)
        assert results == list(range(5))
    
    @pytest.mark.asyncio
    async def test_concurrency_control(self):
        """Test that concurrency is properly limited."""
        # Create tasks that sleep to ensure they run concurrently
        start_time = time.time()
        tasks = [async_sleep_and_return(i, 0.2) for i in range(6)]
        
        # Gather with limited concurrency
        results = await gather_with_rate_limit(
            tasks, 
            max_concurrent=2,
            delay_between_tasks=0.1
        )
        
        # Verify results
        assert results == list(range(6))
        
        # With max_concurrent=2 and 6 tasks of 0.2s each,
        # plus 0.1s delay between tasks, this should take at least 0.7s
        # (1st: 0s, 2nd: 0.1s, 3rd: 0.3s, 4th: 0.4s, 5th: 0.6s, 6th: 0.7s)
        duration = time.time() - start_time
        assert duration >= 0.7
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during gathering."""
        # Create tasks, some of which will fail
        error_task = async_error()
        tasks = [
            async_identity(1),
            error_task,
            async_identity(3)
        ]
        
        # Without return_exceptions, should propagate error
        with pytest.raises(Exception):
            await gather_with_rate_limit(tasks)
        
        # With return_exceptions, should return exceptions as results
        # Create new tasks since the previous ones have been consumed
        error_task = async_error()  
        tasks = [
            async_identity(1),
            error_task,
            async_identity(3)
        ]
        results = await gather_with_rate_limit(tasks, return_exceptions=True)
        assert results[0] == 1
        assert isinstance(results[1], Exception)
        assert results[2] == 3


# Tests for process_batch_async
class TestProcessBatchAsync:
    """Tests for the process_batch_async function."""
    
    @pytest.mark.asyncio
    async def test_basic_batch_processing(self):
        """Test basic batch processing."""
        items = [1, 2, 3, 4, 5]
        
        async def process(x):
            return x * 2
        
        results = await process_batch_async(items, process, batch_size=3)
        
        # Convert to dict for easier verification
        result_dict = {item: result for item, result in results}
        assert result_dict == {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during batch processing."""
        items = [1, 2, 3, 4, 5]
        
        async def process(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2
        
        results = await process_batch_async(items, process)
        
        # Check results
        for item, result in results:
            if item == 3:
                assert result is None  # Error converted to None
            else:
                assert result == item * 2
    
    @pytest.mark.asyncio
    async def test_batching_behavior(self):
        """Test that items are properly batched."""
        items = list(range(10))
        processed_batches = []
        
        async def process(x):
            # Keep track of when items are processed
            processed_batches.append(x // 3)
            return x
        
        await process_batch_async(
            items,
            process,
            batch_size=3,
            max_concurrency=3
        )
        
        # Items should be processed in batches
        assert len(set(processed_batches[:3])) == 1
        assert len(set(processed_batches[3:6])) == 1
        assert len(set(processed_batches[6:9])) == 1


# Tests for retry_async
class TestRetryAsync:
    """Tests for the retry_async function."""
    
    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test retry with immediately successful call."""
        func_mock = AsyncMock(return_value=42)
        result = await retry_async(func_mock, 3, 0.1, 1.0)
        assert result == 42
        assert func_mock.call_count == 1
    
    @pytest.mark.asyncio
    async def test_eventual_success(self):
        """Test retry with eventual success after failures."""
        # Function that will succeed on the 3rd attempt
        attempt = 0
        
        async def flaky_func():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise RateLimitError("Rate limit")
            return "success"
        
        # Should retry and eventually succeed
        result = await retry_async(flaky_func, max_retries=3, base_delay=0.1)
        assert result == "success"
        assert attempt == 3
    
    @pytest.mark.asyncio
    async def test_continued_failure(self):
        """Test retry that continues to fail."""
        func_mock = AsyncMock(side_effect=RateLimitError("Rate limit"))
        
        # Should retry max_retries times then fail
        with pytest.raises(RateLimitError):
            await retry_async(func_mock, max_retries=2, base_delay=0.1)
        
        assert func_mock.call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_selective_retry(self):
        """Test retry only for specified exception types."""
        # Function that raises different errors
        async def mixed_errors():
            raise ValueError("Not a rate limit")
        
        # Should not retry for ValueError
        with pytest.raises(ValueError):
            await retry_async(
                mixed_errors,
                max_retries=3,
                base_delay=0.1,
                retry_on=[RateLimitError]
            )


# Test global instance
def test_global_async_limiter():
    """Test the global async rate limiter instance."""
    assert isinstance(global_async_limiter, AsyncRateLimiter)


# Test import compatibility
def test_import_compatibility():
    """Test compatibility between different import patterns."""
    # Import from both locations
    from yahoofinance.utils.async_helpers import AsyncRateLimiter as RL1
    from yahoofinance.utils.network.async_utils import AsyncRateLimiter as RL2
    from yahoofinance.utils.async_utils.helpers import AsyncRateLimiter as RL3
    
    # They should be the same class
    assert RL1 is RL2
    assert RL2 is RL3
    
    # Create instances
    limiter1 = RL1()
    limiter2 = RL2()
    
    # They should have the same methods
    assert dir(limiter1) == dir(limiter2)