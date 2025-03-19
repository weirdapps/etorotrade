"""
Test fixtures for pagination testing.

This module provides fixtures and helpers for testing pagination functionality.
This is a compatibility layer that re-exports from tests.utils.test_fixtures.
"""

from ..utils.test_fixtures import (
    create_paginated_data,
    create_mock_fetcher,
    create_bulk_fetch_mocks
)

__all__ = [
    'create_paginated_data',
    'create_mock_fetcher',
    'create_bulk_fetch_mocks'
]