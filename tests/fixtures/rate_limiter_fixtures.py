"""Shared test fixtures for rate limiter tests.

This module contains fixtures and utilities used by multiple
rate limiter test modules to reduce duplication.
"""

import time
from unittest.mock import MagicMock

import pytest

from yahoofinance.utils.network.rate_limiter import RateLimiter as AdaptiveRateLimiter


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter for testing decorators."""
    mock = MagicMock()
    # Configure mock to actually call the function
    mock.execute_with_rate_limit.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
    return mock


@pytest.fixture
def fresh_rate_limiter():
    """Create a fresh rate limiter for each test."""
    return AdaptiveRateLimiter(window_size=10, max_calls=20)


@pytest.fixture
def rate_limiter_with_calls():
    """Create a rate limiter with some existing calls."""
    limiter = AdaptiveRateLimiter(window_size=10, max_calls=20)
    # Add several calls
    for _ in range(5):
        limiter.add_call()
    return limiter


@pytest.fixture
def rate_limiter_with_errors():
    """Create a rate limiter with some existing errors."""
    limiter = AdaptiveRateLimiter(window_size=10, max_calls=20)
    # Add several errors
    for i in range(3):
        limiter.add_error(Exception(f"Test error {i}"), "AAPL")
    return limiter


@pytest.fixture
def rate_limiter_with_different_ticker_errors():
    """Create a rate limiter with errors for different tickers."""
    limiter = AdaptiveRateLimiter(window_size=10, max_calls=20)
    tickers = ["AAPL", "MSFT", "GOOGL", "AAPL", "AAPL"]
    for i, ticker in enumerate(tickers):
        limiter.add_error(Exception(f"Error {i}"), ticker)
    return limiter


@pytest.fixture
def create_bulk_fetch_mocks():
    """Create mocks for batch processing tests."""
    processed = []

    def process_item(item):
        processed.append(item)
        return item * 2

    return process_item, processed


def increment_calls_function(limiter, call_count, lock):
    """Thread-safe function to increment calls for thread safety tests."""
    # Get delay (reads state) - we only care about the call, not the value
    _ = limiter.get_delay()
    # Small sleep to increase chance of race conditions
    time.sleep(0.01)
    # Add call (writes state)
    limiter.add_call()
    with lock:
        call_count.value += 1
