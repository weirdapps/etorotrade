#!/usr/bin/env python3
"""
Tests for asynchronous utilities and pagination

This test file verifies:
- Pagination utilities for handling API results
- Async helpers with proper rate limiting
- Bulk processing of data with safety controls
"""

import asyncio
import time
import unittest
import logging
from unittest.mock import patch, MagicMock, AsyncMock

from yahoofinance.utils.pagination import PaginatedResults, paginated_request, bulk_fetch
from yahoofinance.utils.async_helpers import (
    async_rate_limited, gather_with_rate_limit, process_batch_async, retry_async
)
from yahoofinance.errors import RateLimitError, APIError

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestPagination(unittest.TestCase):
    """Test the pagination utilities."""
    
    def test_paginated_results(self):
        """Test iterating through paginated results."""
        # Mock a paginated API
        page1 = {"items": [1, 2, 3], "next_page_token": "page2"}
        page2 = {"items": [4, 5, 6], "next_page_token": "page3"}
        page3 = {"items": [7, 8, 9], "next_page_token": None}
        
        pages = [page1, page2, page3]
        page_index = 0
        
        def mock_fetcher(token=None):
            nonlocal page_index
            result = pages[page_index]
            page_index += 1
            return result
        
        # Create paginated results iterator
        paginator = PaginatedResults(
            fetcher=mock_fetcher,
            items_key="items",
            token_key="next_page_token"
        )
        
        # Collect all results
        all_items = list(paginator)
        
        # Should have all items
        self.assertEqual(all_items, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    def test_paginated_request_function(self):
        """Test the paginated_request convenience function."""
        # Mock a paginated API
        page1 = {"items": [1, 2, 3], "next_page_token": "page2"}
        page2 = {"items": [4, 5, 6], "next_page_token": None}
        
        pages = [page1, page2]
        
        def mock_fetcher(token=None):
            return pages[0] if token is None else pages[1]
        
        # Get all results
        all_items = paginated_request(
            fetcher=mock_fetcher,
            items_key="items",
            token_key="next_page_token",
            max_pages=2
        )
        
        # Should have all items
        self.assertEqual(all_items, [1, 2, 3, 4, 5, 6])
    
    def test_bulk_fetch(self):
        """Test bulk fetching with rate limiting."""
        # Mock a fetcher function
        def mock_fetcher(item):
            if item == 3:
                raise APIError("Test error")
            return {"result": item * 2}
        
        # Mock result extractor
        def mock_extractor(response):
            return response["result"]
        
        # Fetch results for multiple items
        items = [1, 2, 3, 4, 5]
        results = bulk_fetch(
            items=items,
            fetcher=mock_fetcher,
            result_extractor=mock_extractor,
            batch_size=2
        )
        
        # Check results
        expected = [
            (1, 2),   # (item, result)
            (2, 4),
            (3, None),  # Error case
            (4, 8),
            (5, 10)
        ]
        self.assertEqual(results, expected)


class TestAsyncHelpers(unittest.IsolatedAsyncioTestCase):
    """Test async helper functions."""
    
    async def test_async_rate_limited_decorator(self):
        """Test the async_rate_limited decorator."""
        calls = []
        
        @async_rate_limited(ticker_param='ticker')
        async def test_func(ticker, value):
            calls.append((ticker, value))
            return f"{ticker}:{value}"
        
        # Call the function a few times
        for i in range(3):
            result = await test_func("AAPL", i)
            self.assertEqual(result, f"AAPL:{i}")
        
        # All calls should be in the calls list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls, [("AAPL", 0), ("AAPL", 1), ("AAPL", 2)])
    
    async def test_gather_with_rate_limit(self):
        """Test gathering async tasks with rate limiting."""
        # Create test coroutines
        async def coro(i):
            # Simulate work
            await asyncio.sleep(0.1)
            return i * 2
        
        tasks = [coro(i) for i in range(5)]
        
        # Gather results with rate limiting
        results = await gather_with_rate_limit(
            tasks,
            max_concurrent=2,
            delay_between_tasks=0.1
        )
        
        # Check results
        self.assertEqual(results, [0, 2, 4, 6, 8])
    
    async def test_process_batch_async(self):
        """Test async batch processing."""
        # Mock async processor
        async def mock_processor(item):
            if item == 3:
                raise APIError("Test error")
            return item * 2
        
        # Process batch of items
        items = [1, 2, 3, 4, 5]
        results = await process_batch_async(
            items=items,
            processor=mock_processor,
            batch_size=2,
            max_concurrency=2
        )
        
        # Check results
        expected = [
            (1, 2),    # (item, result)
            (2, 4),
            (3, None),  # Error case
            (4, 8),
            (5, 10)
        ]
        self.assertEqual(results, expected)
    
    async def test_retry_async(self):
        """Test async retry with exponential backoff."""
        # Mock function that fails a few times
        call_count = 0
        
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                raise RateLimitError("Rate limit exceeded")
            
            return "success"
        
        # Retry the flaky function
        result = await retry_async(
            flaky_function,
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0
        )
        
        # Check results
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)  # Should succeed on third try


if __name__ == "__main__":
    unittest.main()