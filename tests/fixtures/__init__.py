"""
Test fixtures for etorotrade tests.

This package provides common test fixtures and helpers used across all test files.
"""

from .async_fixtures import create_async_processor_mock, create_flaky_function
from .pagination import create_bulk_fetch_mocks, create_mock_fetcher, create_paginated_data


__all__ = [
    "create_paginated_data",
    "create_mock_fetcher",
    "create_bulk_fetch_mocks",
    "create_flaky_function",
    "create_async_processor_mock",
]
