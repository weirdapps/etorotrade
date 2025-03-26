"""
Pagination utilities for Yahoo Finance API.

This module provides utilities for handling paginated API results,
allowing for efficient retrieval of large result sets while respecting
rate limits.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Generator, TypeVar, Generic, Union

from .rate_limiter import RateLimiter, global_rate_limiter, rate_limited

logger = logging.getLogger(__name__)

# Define a generic type variable for the item type
T = TypeVar('T')

class PaginatedResults(Generic[T]):
    """
    Handle paginated API results.
    
    This class provides a way to handle paginated API results, with
    automatic rate limiting and error handling.
    
    Attributes:
        fetch_page_func: Function to fetch a page of results
        parse_items_func: Function to parse items from a page
        get_next_page_token_func: Function to get the next page token
        rate_limiter: Rate limiter to use
        page_size: Number of items per page
    """
    
    def __init__(
        self,
        fetch_page_func: Callable[[Optional[str]], Dict[str, Any]],
        parse_items_func: Callable[[Dict[str, Any]], List[T]],
        get_next_page_token_func: Callable[[Dict[str, Any]], Optional[str]],
        rate_limiter: RateLimiter = None,
        page_size: int = 100
    ):
        """
        Initialize the paginated results handler.
        
        Args:
            fetch_page_func: Function to fetch a page of results
            parse_items_func: Function to parse items from a page
            get_next_page_token_func: Function to get the next page token
            rate_limiter: Rate limiter to use
            page_size: Number of items per page
        """
        self.fetch_page_func = fetch_page_func
        self.parse_items_func = parse_items_func
        self.get_next_page_token_func = get_next_page_token_func
        self.rate_limiter = rate_limiter or global_rate_limiter
        self.page_size = page_size
        
        logger.debug(f"Initialized paginated results handler with page_size={page_size}")
    
    def _fetch_page_with_rate_limiting(self, page_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch a page of results with rate limiting.
        
        Args:
            page_token: Token for the page to fetch
            
        Returns:
            Page of results
            
        Raises:
            Any exception raised by the fetch_page_func
        """
        # Wait if needed to avoid exceeding rate limit
        self.rate_limiter.wait_if_needed()
        
        # Record the call
        self.rate_limiter.record_call()
        
        try:
            # Fetch the page
            page = self.fetch_page_func(page_token)
            
            # Record success
            self.rate_limiter.record_success()
            
            return page
        except Exception as e:
            # Record failure
            from ...core.errors import RateLimitError
            is_rate_limit = isinstance(e, RateLimitError)
            self.rate_limiter.record_failure(None, is_rate_limit)
            
            # Re-raise the exception
            raise
    
    def get_page(self, page_token: Optional[str] = None) -> tuple[List[T], Optional[str]]:
        """
        Get a page of results.
        
        Args:
            page_token: Token for the page to fetch
            
        Returns:
            Tuple containing:
                - List of items in the page
                - Token for the next page, or None if there are no more pages
        """
        # Fetch the page
        page = self._fetch_page_with_rate_limiting(page_token)
        
        # Parse items from the page
        items = self.parse_items_func(page)
        
        # Get the next page token
        next_page_token = self.get_next_page_token_func(page)
        
        return items, next_page_token
    
    def get_all(self, max_pages: Optional[int] = None) -> List[T]:
        """
        Get all results.
        
        Args:
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of all items
        """
        all_items: List[T] = []
        page_token = None
        page_count = 0
        
        while True:
            # Check if we've reached the maximum number of pages
            if max_pages is not None and page_count >= max_pages:
                logger.info(f"Reached maximum number of pages ({max_pages})")
                break
            
            # Get a page of results
            try:
                items, next_page_token = self.get_page(page_token)
            except Exception as e:
                logger.error(f"Error fetching page {page_count}: {str(e)}")
                break
            
            # Add items to the result list
            all_items.extend(items)
            
            # Check if there are more pages
            if not next_page_token:
                break
            
            # Update page token and increment page count
            page_token = next_page_token
            page_count += 1
            
            logger.debug(f"Fetched page {page_count} with {len(items)} items")
        
        logger.info(f"Fetched {len(all_items)} items from {page_count + 1} pages")
        return all_items
    
    def iter_pages(self, max_pages: Optional[int] = None) -> Generator[List[T], None, None]:
        """
        Iterate through pages of results.
        
        Args:
            max_pages: Maximum number of pages to fetch
            
        Yields:
            Lists of items, one page at a time
        """
        page_token = None
        page_count = 0
        
        while True:
            # Check if we've reached the maximum number of pages
            if max_pages is not None and page_count >= max_pages:
                logger.info(f"Reached maximum number of pages ({max_pages})")
                break
            
            # Get a page of results
            try:
                items, next_page_token = self.get_page(page_token)
            except Exception as e:
                logger.error(f"Error fetching page {page_count}: {str(e)}")
                break
            
            # Yield the items
            yield items
            
            # Check if there are more pages
            if not next_page_token:
                break
            
            # Update page token and increment page count
            page_token = next_page_token
            page_count += 1
            
            logger.debug(f"Fetched page {page_count} with {len(items)} items")
        
        logger.info(f"Fetched {page_count + 1} pages")
    
    def iter_items(self, max_pages: Optional[int] = None) -> Generator[T, None, None]:
        """
        Iterate through items in all pages.
        
        Args:
            max_pages: Maximum number of pages to fetch
            
        Yields:
            Items one at a time
        """
        for page in self.iter_pages(max_pages):
            for item in page:
                yield item


@rate_limited
def paginated_request(
    url: str,
    params: Dict[str, Any],
    page_param: str = 'page',
    items_key: str = 'items',
    next_page_key: Optional[str] = None,
    total_pages_key: Optional[str] = None,
    rate_limiter: RateLimiter = None,
    max_pages: Optional[int] = None,
    session = None
) -> List[Dict[str, Any]]:
    """
    Make a paginated API request.
    
    This function makes a paginated API request, fetching all pages
    and combining the results.
    
    Args:
        url: URL to request
        params: Request parameters
        page_param: Parameter name for the page number
        items_key: Key for the items in the response
        next_page_key: Key for the next page token in the response
        total_pages_key: Key for the total number of pages in the response
        rate_limiter: Rate limiter to use
        max_pages: Maximum number of pages to fetch
        session: Requests session to use
        
    Returns:
        List of all items from all pages
    """
    # Import here to avoid circular imports
    import requests
    
    # Create a session if not provided
    if session is None:
        session = requests.Session()
    
    # Define the fetch_page function
    def fetch_page(page_token: Optional[str] = None) -> Dict[str, Any]:
        # Update parameters with page token or number
        page_params = params.copy()
        if page_token is not None:
            if next_page_key is not None:
                # Token-based pagination
                page_params[page_param] = page_token
            else:
                # Page number-based pagination
                page_params[page_param] = int(page_token)
        
        # Make the request
        response = session.get(url, params=page_params)
        response.raise_for_status()
        
        # Parse the response
        return response.json()
    
    # Define the parse_items function
    def parse_items(page: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Get the items from the page
        if items_key in page:
            return page[items_key]
        return []
    
    # Define the get_next_page_token function
    def get_next_page_token(page: Dict[str, Any]) -> Optional[str]:
        if next_page_key is not None:
            # Token-based pagination
            return page.get(next_page_key)
        elif total_pages_key is not None:
            # Page number-based pagination with total pages
            current_page = int(params.get(page_param, 1))
            total_pages = int(page.get(total_pages_key, 0))
            if current_page < total_pages:
                return str(current_page + 1)
        elif items_key in page and len(page[items_key]) >= params.get('limit', 100):
            # Assume there might be more pages if items limit was reached
            current_page = int(params.get(page_param, 1))
            return str(current_page + 1)
        return None
    
    # Create paginated results handler
    paginator = PaginatedResults(
        fetch_page_func=fetch_page,
        parse_items_func=parse_items,
        get_next_page_token_func=get_next_page_token,
        rate_limiter=rate_limiter
    )
    
    # Get all results
    return paginator.get_all(max_pages)