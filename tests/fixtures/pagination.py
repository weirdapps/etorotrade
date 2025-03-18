"""
Test fixtures for pagination testing.

This module provides fixtures and helpers for testing pagination functionality.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from yahoofinance.core.errors import APIError


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
        
        pages.append({
            "items": page_items,
            "next_page_token": next_token
        })
    
    return pages


def create_mock_fetcher(pages: List[Dict[str, Any]]) -> Callable[[Optional[str]], Dict[str, Any]]:
    """
    Create a mock page fetcher function for pagination tests.
    
    Args:
        pages: List of page data to return
        
    Returns:
        A function that simulates fetching a page of results
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
        
        # Out of pages
        return {"items": [], "next_page_token": None}
    
    return mock_fetcher


def create_bulk_fetch_mocks() -> Tuple[
    List[int], 
    Callable[[Any], Dict[str, Any]], 
    Callable[[Dict[str, Any]], int]
]:
    """
    Create mock objects for bulk fetch tests.
    
    Returns:
        Tuple containing:
            - List of items to fetch
            - Mock fetcher function
            - Mock result extractor function
    """
    items = list(range(1, 6))
    
    def mock_fetcher(item):
        if item == 3:
            raise APIError("Test error")
        return {"result": item * 2}
    
    def mock_extractor(response):
        return response["result"]
    
    return items, mock_fetcher, mock_extractor