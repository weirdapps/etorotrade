"""
Common test fixtures and helpers for pagination and async tests.

This module helps reduce code duplication across test files by providing:
- Mock data generators
- Common test fixtures
- Helper functions for setting up test cases
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

from yahoofinance.core.errors import APIError, RateLimitError


# Pagination Test Fixtures


def create_paginated_data(num_pages: int = 3, items_per_page: int = 3) -> List[Dict[str, Any]]:
    """
    Create mock paginated data for testing.

    Args:
        num_pages: Number of pages to create
        items_per_page: Number of items per page

    Returns:
        List of page responses with items and next page tokens
    """
    pages = []
    item_counter = 1

    for page_num in range(1, num_pages + 1):
        page_items = list(range(item_counter, item_counter + items_per_page))
        item_counter += items_per_page

        # Last page has no next token
        next_token = f"page{page_num+1}" if page_num < num_pages else None

        pages.append({"items": page_items, "next_page_token": next_token})

    return pages


def create_mock_fetcher(pages: List[Dict[str, Any]]) -> Callable[[Optional[str]], Dict[str, Any]]:
    """
        Create a mock page fetcher function for pagination tests.

        Args:
            pages: List of page data to return

        Returns:
            A function that simulates@with_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
    def mock_fetcher(of results
    """
    page_index = 0

    def mock_fetcher(token=None):
        nonlocal page_index
        # Simply return pages in sequence - simpler implementation that matches
        # the original test behavior
        if page_index < len(pages):
            result = pages[page_index]
            page_index += 1
            return result
        else:
            # Out of pages
            return {"items": [], "next_page_token": None}

    return mock_fetcher


def create_bulk_fetch_mocks() -> (
    Tuple[List[int], Callable[[Any], Dict[str, Any]], Callable[[Dict[str, Any]], int]]
):
    """
        Create mock objects for bulk fetch tests.

        Returns:@with_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
    def mock_fetcher(ntaining:
                - List of items to fetch
                - Mock fetcher function
                - Mock result extractor function
    """
    items = list(range(1, 6))

    def mock_fetcher(item):
        if item == 3:
            raise YFinanceError("An error occurred")
        return {"result": item * 2}

    def mock_extractor(response):
        return response["result"]

    return items, mock_fetcher, mock_extractor


# Async Test Fixtures


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
            raise YFinanceError("An error occurred")
        return item * 2

    return mock_processor
