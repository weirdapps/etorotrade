"""
Test fixtures for async testing.

This module provides fixtures and helpers for testing asynchronous functionality.
"""

import asyncio
from typing import Callable
from unittest.mock import AsyncMock

from yahoofinance.errors import RateLimitError, APIError


async def create_flaky_function(fail_count: int = 2) -> Callable[[], AsyncMock]:
    """
    Create a mock async function that fails a specified number of times.
    
    Args:
        fail_count: Number of times the function should fail before succeeding
        
    Returns:
        Mock async function that raises RateLimitError initially then succeeds
    """
    call_count = 0
    
    async def flaky_function():
        nonlocal call_count
        call_count += 1
        
        if call_count <= fail_count:
            raise RateLimitError("Rate limit exceeded")
        
        return "success"
    
    return flaky_function


def create_async_processor_mock(error_item: int = 3) -> AsyncMock:
    """
    Create a mock async processor function for batch processing tests.
    
    Args:
        error_item: Item value that should trigger an error
        
    Returns:
        AsyncMock that doubles items or raises an error for the specified item
    """
    async def mock_processor(item):
        if item == error_item:
            raise APIError("Test error")
        return item * 2
    
    return mock_processor