"""
Pagination utilities for API requests.

This module provides utilities for handling paginated API responses, including
automatic page fetching, rate limiting, and result aggregation.
"""

import logging
import time
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar, Union

from ...core.config import PAGINATION
from ...core.errors import APIError, DataError, ValidationError, YFinanceError
from ...core.logging import get_logger


# Set up logging
logger = get_logger(__name__)

# Define type variables for generic types
T = TypeVar("T")  # Return type for page fetching
R = TypeVar("R")  # Return type for result processing


class PaginatedResults:
    """
    Handler for paginated API results.

    This class provides utilities for fetching paginated results from an API,
    with automatic rate limiting and result aggregation.

    Attributes:
        fetch_page_func: Function to fetch a page of results
        process_results_func: Function to process results
        get_next_page_token_func: Function to get the next page token
        page_size: Number of items per page
        max_pages: Maximum number of pages to fetch
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
    """

    def __init__(
        self,
        fetch_page_func: Callable[[Optional[str]], Dict[str, Any]],
        process_results_func: Callable[[Dict[str, Any]], List[T]],
        get_next_page_token_func: Callable[[Dict[str, Any]], Optional[str]],
        page_size: int = None,
        max_pages: int = None,
        max_retries: int = None,
        retry_delay: float = None,
    ):
        """
        Initialize the paginated results handler.

        Args:
            fetch_page_func: Function to fetch a page of results
            process_results_func: Function to process results
            get_next_page_token_func: Function to get the next page token
            page_size: Number of items per page (default: from config)
            max_pages: Maximum number of pages to fetch (default: from config)
            max_retries: Maximum number of retry attempts (default: from config)
            retry_delay: Delay in seconds between retries (default: from config)
        """
        self.fetch_page_func = fetch_page_func
        self.process_results_func = process_results_func
        self.get_next_page_token_func = get_next_page_token_func
        self.page_size = page_size or PAGINATION["PAGE_SIZE"]
        self.max_pages = max_pages or PAGINATION["MAX_PAGES"]
        self.max_retries = max_retries or PAGINATION["MAX_RETRIES"]
        self.retry_delay = retry_delay or PAGINATION["RETRY_DELAY"]

        logger.debug(f"Initialized paginated results handler with page_size={page_size}")

    def _fetch_page_with_rate_limiting(self, page_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch a page of results with rate limiting.

        Args:
            page_token: Token for the page to fetch

        Returns:
            Page response

        Raises:
            APIError: When a page fetch fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                page_response = self.fetch_page_func(page_token)
                return page_response
            except YFinanceError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt+1}/{self.max_retries} failed to fetch page: {str(e)}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed to fetch page: {str(e)}")
                    raise APIError(
                        f"Failed to fetch page after {self.max_retries} attempts: {str(e)}"
                    )

    def fetch_all(self) -> List[T]:
        """
        Fetch all pages of results.

        Returns:
            List of processed results

        Raises:
            APIError: When a page fetch fails after retries
        """
        all_results = []
        next_page_token = None
        page_count = 0

        while page_count < self.max_pages:
            # Fetch the next page
            logger.debug(f"Fetching page {page_count + 1} with token: {next_page_token}")
            page_response = self._fetch_page_with_rate_limiting(next_page_token)

            # Process the results
            page_results = self.process_results_func(page_response)
            all_results.extend(page_results)

            # Get the next page token
            next_page_token = self.get_next_page_token_func(page_response)
            page_count += 1

            # Stop if there are no more pages
            if not next_page_token:
                logger.debug(f"No more pages to fetch after page {page_count}")
                break

            # Check if we've reached the maximum number of pages
            if page_count >= self.max_pages:
                logger.warning(f"Reached maximum number of pages ({self.max_pages})")
                break

        logger.debug(f"Fetched {page_count} pages with {len(all_results)} total results")
        return all_results

    def fetch_first_page(self) -> List[T]:
        """
        Fetch only the first page of results.

        Returns:
            List of processed results from the first page

        Raises:
            APIError: When the page fetch fails after retries
        """
        logger.debug("Fetching first page only")
        page_response = self._fetch_page_with_rate_limiting(None)
        page_results = self.process_results_func(page_response)
        logger.debug(f"Fetched first page with {len(page_results)} results")
        return page_results

    def iter_pages(self) -> Iterator[List[T]]:
        """
        Iterator for pages of results.

        Yields:
            List of processed results for each page

        Raises:
            APIError: When a page fetch fails after retries
        """
        next_page_token = None
        page_count = 0

        while page_count < self.max_pages:
            # Fetch the next page
            logger.debug(f"Fetching page {page_count + 1} with token: {next_page_token}")
            page_response = self._fetch_page_with_rate_limiting(next_page_token)

            # Process the results
            page_results = self.process_results_func(page_response)
            yield page_results

            # Get the next page token
            next_page_token = self.get_next_page_token_func(page_response)
            page_count += 1

            # Stop if there are no more pages
            if not next_page_token:
                logger.debug(f"No more pages to fetch after page {page_count}")
                break

            # Check if we've reached the maximum number of pages
            if page_count >= self.max_pages:
                logger.warning(f"Reached maximum number of pages ({self.max_pages})")
                break


def paginate(
    fetch_page_func: Callable[[Optional[str]], Dict[str, Any]],
    process_results_func: Callable[[Dict[str, Any]], List[T]],
    get_next_page_token_func: Callable[[Dict[str, Any]], Optional[str]],
    page_size: int = None,
    max_pages: int = None,
    max_retries: int = None,
    retry_delay: float = None,
    first_page_only: bool = False,
) -> List[T]:
    """
    Fetch paginated results from an API.

    This is a convenience function for fetching paginated results from an API,
    with automatic rate limiting and result aggregation.

    Args:
        fetch_page_func: Function to fetch a page of results
        process_results_func: Function to process results
        get_next_page_token_func: Function to get the next page token
        page_size: Number of items per page (default: from config)
        max_pages: Maximum number of pages to fetch (default: from config)
        max_retries: Maximum number of retry attempts (default: from config)
        retry_delay: Delay in seconds between retries (default: from config)
        first_page_only: If True, fetch only the first page

    Returns:
        List of processed results

    Raises:
        APIError: When a page fetch fails after retries
    """
    paginator = PaginatedResults(
        fetch_page_func=fetch_page_func,
        process_results_func=process_results_func,
        get_next_page_token_func=get_next_page_token_func,
        page_size=page_size,
        max_pages=max_pages,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    if first_page_only:
        return paginator.fetch_first_page()
    else:
        return paginator.fetch_all()


# Legacy alias for backwards compatibility
paginated_request = paginate
