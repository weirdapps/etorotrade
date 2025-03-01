"""
Utilities for handling paginated API results efficiently.

This module provides functionality for working with paginated API responses
while respecting rate limits and preventing excessive API calls.
"""

import time
import logging
from typing import TypeVar, Generic, List, Callable, Iterator, Dict, Any, Optional, Tuple

from .rate_limiter import global_rate_limiter
from ..errors import APIError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type of individual result items
P = TypeVar('P')  # Type of pagination token


class PaginatedResults(Generic[T, P]):
    """
    Iterator for efficiently handling paginated API results with rate limiting.
    
    This class abstracts the complexities of pagination, automatically handling:
    - Rate limiting between page requests
    - Exponential backoff on errors
    - Result buffering for efficient access
    
    Example usage:
        def fetch_page(token=None):
            # API call to fetch a page of results
            return {'items': [...], 'next_page_token': 'abc123'}
        
        results = PaginatedResults(
            fetcher=fetch_page,
            items_key='items',
            token_key='next_page_token'
        )
        
        for item in results:
            process_item(item)
    """
    
    def __init__(
            self,
            fetcher: Callable[[Optional[P]], Dict[str, Any]],
            items_key: str = 'items',
            token_key: str = 'next_page_token',
            max_pages: int = 10,
            ticker: Optional[str] = None
    ):
        """
        Initialize paginated results iterator.
        
        Args:
            fetcher: Function that takes an optional page token and returns a page of results
            items_key: Key in response dict containing the items
            token_key: Key in response dict containing the next page token
            max_pages: Maximum number of pages to retrieve
            ticker: Optional ticker symbol for rate limiting
        """
        self.fetcher = fetcher
        self.items_key = items_key
        self.token_key = token_key
        self.max_pages = max_pages
        self.ticker = ticker
        
        # State tracking
        self.current_page = 0
        self.next_token: Optional[P] = None
        self.buffer: List[T] = []
        self.buffer_position = 0
        self.has_more = True
        self.is_first_page = True
        self.error_count = 0
        self.max_errors = 3
    
    def __iter__(self) -> Iterator[T]:
        """Return self as iterator."""
        return self
    
    def __next__(self) -> T:
        """Get next item, fetching new page if needed."""
        # If buffer is exhausted, fetch next page
        if self.buffer_position >= len(self.buffer):
            if not self.has_more:
                raise StopIteration
            
            self._fetch_next_page()
            
            # If buffer is still empty after fetch, we're done
            if not self.buffer:
                raise StopIteration
        
        # Return next item from buffer
        item = self.buffer[self.buffer_position]
        self.buffer_position += 1
        return item
    
    def _fetch_next_page(self) -> None:
        """Fetch next page of results with rate limiting."""
        if self.current_page >= self.max_pages:
            logger.debug(f"Reached max pages limit ({self.max_pages})")
            self.has_more = False
            return
        
        # Apply rate limiting - more delay between pages
        delay = global_rate_limiter.get_delay(self.ticker)
        # Add extra delay for pagination to be respectful
        delay = max(delay * 1.5, 2.0)
        time.sleep(delay)
        
        try:
            # Fetch next page
            response = self.fetcher(self.next_token)
            
            # Record successful call
            global_rate_limiter.add_call(ticker=self.ticker)
            
            # Extract items and next token
            self.buffer = response.get(self.items_key, [])
            self.buffer_position = 0
            self.next_token = response.get(self.token_key)
            self.has_more = bool(self.next_token)
            self.current_page += 1
            self.is_first_page = False
            self.error_count = 0  # Reset error count on success
            
            logger.debug(
                f"Fetched page {self.current_page} with {len(self.buffer)} items. "
                f"{'Has more pages' if self.has_more else 'Last page'}"
            )
            
        except RateLimitError as e:
            # Handle rate limit error
            self.error_count += 1
            global_rate_limiter.add_error(e, ticker=self.ticker)
            
            if self.error_count > self.max_errors:
                logger.warning("Too many errors during pagination, stopping")
                self.has_more = False
                return
            
            # Apply exponential backoff
            backoff = min(30, 2 ** self.error_count)
            logger.warning(f"Rate limit during pagination, backing off for {backoff}s")
            time.sleep(backoff)
            
            # Retry (recursively)
            self._fetch_next_page()
            
        except APIError as e:
            # Handle other API errors
            global_rate_limiter.add_error(e, ticker=self.ticker)
            logger.error(f"API error during pagination: {str(e)}")
            self.has_more = False
            
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Unexpected error during pagination: {str(e)}")
            self.has_more = False
    
    def get_all(self) -> List[T]:
        """
        Fetch all results into a list.
        
        Returns:
            List of all items across all pages
        """
        return list(self)
        

def paginated_request(
        fetcher: Callable[[Optional[P]], Dict[str, Any]],
        items_key: str = 'items',
        token_key: str = 'next_page_token',
        max_pages: int = 10,
        ticker: Optional[str] = None
) -> List[T]:
    """
    Helper function to fetch all items from a paginated API with rate limiting.
    
    Args:
        fetcher: Function that takes an optional page token and returns a page of results
        items_key: Key in response dict containing the items
        token_key: Key in response dict containing the next page token
        max_pages: Maximum number of pages to retrieve
        ticker: Optional ticker symbol for rate limiting
        
    Returns:
        List of all items across all pages
    """
    paginator = PaginatedResults(
        fetcher=fetcher,
        items_key=items_key,
        token_key=token_key,
        max_pages=max_pages,
        ticker=ticker
    )
    return paginator.get_all()


def bulk_fetch(
        items: List[Any],
        fetcher: Callable[[Any], Dict[str, Any]],
        result_extractor: Callable[[Dict[str, Any]], T],
        batch_size: int = 10
) -> List[Tuple[Any, Optional[T]]]:
    """
    Fetch data for multiple items in batches with rate limiting.
    
    Args:
        items: List of identifiers to fetch data for
        fetcher: Function that takes an item ID and returns API response
        result_extractor: Function to extract result from API response
        batch_size: Number of items per batch
        
    Returns:
        List of tuples (item, result) where result may be None on error
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = []
        
        for item in batch:
            try:
                # Apply rate limiting
                delay = global_rate_limiter.get_delay()
                time.sleep(delay)
                
                # Fetch data
                response = fetcher(item)
                
                # Record successful call
                global_rate_limiter.add_call()
                
                # Extract result
                result = result_extractor(response)
                batch_results.append((item, result))
                
            except Exception as e:
                logger.warning(f"Error fetching {item}: {str(e)}")
                global_rate_limiter.add_error(e)
                batch_results.append((item, None))
        
        results.extend(batch_results)
        
        # Add delay between batches
        if i + batch_size < len(items):
            batch_delay = global_rate_limiter.get_batch_delay()
            logger.debug(f"Completed batch of {len(batch)}. Waiting {batch_delay:.1f}s before next batch.")
            time.sleep(batch_delay)
    
    return results