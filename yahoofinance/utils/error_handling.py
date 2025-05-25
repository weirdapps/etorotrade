"""
Standardized error handling utilities.

This module provides utilities for consistent error handling, context enrichment,
error translation, and recovery strategies throughout the application.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

from ..core.errors import (
    APIError,
    CacheError,
    ConnectionError,
    DataError,
    RateLimitError,
    ResourceNotFoundError,
    TimeoutError,
    ValidationError,
    YFinanceError,
)
from ..core.logging import get_logger


# Create a logger for this module
logger = get_logger(__name__)

# Define generic type variables for the return types
T = TypeVar("T")
R = TypeVar("R")


def enrich_error_context(error: YFinanceError, context: Dict[str, Any]) -> YFinanceError:
    """
    Enrich an error with additional context information.

    This utility adds context information to an error's details
    dictionary to help with debugging and error reporting.

    Args:
        error: The original error
        context: Dictionary of context information to add

    Returns:
        The enriched error object

    Example:
        ```
        try:
            # Code that might raise an error
        except YFinanceError as e:
            # Add context information
            enriched_error = enrich_error_context(e, {
                'ticker': ticker,
                'operation': 'get_price_data',
                'timestamp': time.time()
            })
            # Re-raise the enriched error
            raise enriched_error
        ```
    """
    # Create a new details dictionary if one doesn't exist
    if error.details is None:
        error.details = {}

    # Add all context items to the details dictionary
    for key, value in context.items():
        # Don't overwrite existing keys
        if key not in error.details:
            error.details[key] = value

    return error


def translate_error(
    error: Exception,
    default_message: str = "An error occurred",
    context: Optional[Dict[str, Any]] = None,
) -> YFinanceError:
    """
    Translate a standard Python exception into our custom error hierarchy.

    This function maps standard Python exceptions to our custom error
    hierarchy with appropriate context information.

    Args:
        error: Standard Python exception
        default_message: Default error message if none is provided
        context: Additional context information

    Returns:
        An appropriate error from our custom hierarchy

    Example:
        ```
        try:
            # Code that might raise a standard exception
        except YFinanceError as e:
            # Translate the error to our custom hierarchy
            custom_error = translate_error(e, context={
                'ticker': ticker,
                'operation': 'get_price_data'
            })
            # Handle or re-raise the custom error
            raise custom_error
        ```
    """
    context = context or {}
    error_type = type(error)
    error_message = str(error) or default_message

    # Map standard exceptions to our custom error hierarchy
    if error_type is ValueError:
        return ValidationError(error_message, context)
    elif error_type is KeyError:
        return DataError(f"Missing key: {error_message}", context)
    elif error_type is AttributeError:
        return DataError(f"Missing attribute: {error_message}", context)
    elif error_type is TimeoutError or error_type.__name__ == "Timeout":
        return TimeoutError(f"Operation timed out: {error_message}", context)
    elif error_type is ConnectionError or error_type.__name__ == "ConnectionError":
        return ConnectionError(f"Connection error: {error_message}", context)
    elif error_type is FileNotFoundError:
        return ResourceNotFoundError(f"File not found: {error_message}", context)
    elif error_type is PermissionError:
        return ResourceNotFoundError(f"Permission denied: {error_message}", context)
    elif error_type is MemoryError:
        return ResourceNotFoundError(f"Out of memory: {error_message}", context)

    # For unknown errors, use the base YFinanceError
    return YFinanceError(f"Unexpected error: {error_message}", context)


def with_error_context(
    context_provider: Callable[..., Dict[str, Any]],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add context information to errors raised by a function.

    This decorator automatically adds context information to any
    YFinanceError raised by the decorated function, making error
    debugging and reporting more effective.

    Args:
        context_provider: Function that returns a context dictionary

    Returns:
        Decorated function

    Example:
        ```
        def get_context_for_ticker_info(ticker, **kwargs):
            return {
                'ticker': ticker,
                'operation': 'get_ticker_info',
                'timestamp': time.time()
            }

        @with_error_context(get_context_for_ticker_info)
        def get_ticker_info(ticker, **kwargs):
            # Function implementation
            # Any YFinanceError raised will automatically get context
        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except YFinanceError as e:
                # Get context information from the provider function
                context = context_provider(*args, **kwargs)
                # Enrich the error with the context
                enriched_error = enrich_error_context(e, context)
                # Re-raise the enriched error
                raise enriched_error
            except Exception as e:
                # For non-YFinanceError exceptions, translate to our hierarchy
                context = context_provider(*args, **kwargs)
                custom_error = translate_error(e, context=context)
                # Log the error with the original traceback
                logger.error(f"Translated error in {func.__name__}: {str(e)}", exc_info=True)
                # Raise the translated error
                raise custom_error

        return wrapper

    return decorator


def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_errors: Optional[Set[Type[Exception]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically retry a function on certain errors.

    This decorator implements an exponential backoff retry strategy
    for functions that might fail due to transient errors.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for the delay after each retry
        retryable_errors: Set of error types that should trigger a retry

    Returns:
        Decorated function

    Example:
        ```
        @with_retry

        def fetch_data_with_retry(url):
            # Function implementation
            # Will be retried up to 3 times if it raises ConnectionError or TimeoutError
        ```
    """
    # Set default retryable errors if not provided
    if retryable_errors is None:
        retryable_errors = {ConnectionError, TimeoutError, RateLimitError}

    # This handles the case where the decorator is called without arguments
    # like @with_retry instead of @with_retry()
    if callable(max_retries):
        func = max_retries
        default_decorator = with_retry()
        return default_decorator(func)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retry_count = 0
            delay = retry_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_errors) as e:
                    retry_count += 1

                    # If we've exceeded max retries, re-raise the error
                    if retry_count > max_retries:
                        logger.warning(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}: {str(e)}"
                        )
                        raise e

                    # For RateLimitError, use the retry_after value if provided
                    if isinstance(e, RateLimitError) and e.retry_after is not None:
                        delay = e.retry_after

                    # Log the retry
                    logger.info(
                        f"Retry {retry_count}/{max_retries} for {func.__name__} after {delay:.2f}s: {str(e)}"
                    )

                    # Wait before retrying
                    time.sleep(delay)

                    # Increase the delay for the next retry
                    delay *= backoff_factor
                except YFinanceError as e:
                    # For non-retryable errors, re-raise immediately
                    raise

        return wrapper

    return decorator


def safe_operation(
    default_value: Optional[R] = None,
    log_errors: bool = True,
    reraise: bool = False,
) -> Callable[[Callable[..., R]], Callable[..., Optional[R]]]:
    """
    Decorator to safely execute an operation with fallback to a default value.

    This decorator provides a safe way to execute operations that might fail
    but where it's acceptable to return a default value instead of raising an error.

    Args:
        default_value: Default value to return if the operation fails
        log_errors: Whether to log errors
        reraise: Whether to re-raise the error after logging

    Returns:
        Decorated function

    Example:
        ```
        @safe_operation(default_value={}, log_errors=True)
        def get_optional_data(ticker):
            # Function implementation
            # If it fails, will return {} instead of raising an error
        ```
    """

    def decorator(func: Callable[..., R]) -> Callable[..., Optional[R]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[R]:
            try:
                return func(*args, **kwargs)
            except YFinanceError as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                if reraise:
                    raise e
                return default_value

        return wrapper

    return decorator
