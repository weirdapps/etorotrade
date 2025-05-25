"""
Test fixtures for async testing.

This module provides fixtures and helpers for testing asynchronous functionality.
"""

from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

from yahoofinance.core.errors import APIError, RateLimitError, YFinanceError


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


def create_paginated_data(num_pages: int = 3, items_per_page: int = 3) -> List[Dict[str, Any]]:
    """
    Create mock paginated data for testing.

    Args:
        num_pages: Number of pages to create
        items_per_page: Number of items per page

    Returns:
        List of dictionaries representing paginated data
    """
    pages = []
    item_counter = 1

    for page_num in range(num_pages):
        items = [item_counter + i for i in range(items_per_page)]
        item_counter += items_per_page

        # Create a page with items and a next page token, except for the last page
        page = {
            "items": items,
            "next_page_token": f"page_{page_num + 1}" if page_num < num_pages - 1 else None,
        }

        pages.append(page)

    return pages


def create_mock_fetcher(pages: List[Dict[str, Any]]) -> MagicMock:
    """
    Create a mock fetcher function for paginated data.

    Args:
        pages: List of page dictionaries

    Returns:
        MagicMock that returns pages in sequence
    """
    # For regular sync tests, we'll use a regular MagicMock instead of AsyncMock
    mock = MagicMock()
    mock.side_effect = pages
    return mock


def create_bulk_fetch_mocks() -> Tuple[List[int], AsyncMock, Callable[[Dict[str, Any]], Any]]:
    """
    Create mocks for bulk fetch testing.

    Returns:
        Tuple containing items list, mock fetcher, and mock extractor
    """
    items = [1, 2, 3, 4, 5]

    async def mock_fetcher(item):
        if item == 3:
            raise APIError("API error")
        return {"result": item * 2}

    def mock_extractor(data):
        if data:
            return data.get("result")
        return None

    fetcher_mock = AsyncMock(side_effect=mock_fetcher)

    return items, fetcher_mock, mock_extractor
