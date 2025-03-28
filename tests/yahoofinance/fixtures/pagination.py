"""
Test fixtures for pagination testing.

This module provides fixtures and helpers for testing pagination functionality.

DEPRECATED: This module is deprecated in favor of tests.fixtures.pagination.
Use the fixtures from that module instead, which are automatically registered in conftest.py.
"""

import warnings

warnings.warn(
    "This module (tests.yahoofinance.fixtures.pagination) is deprecated. "
    "Use tests.fixtures.pagination instead, which is automatically registered in conftest.py.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all fixtures and functions from the canonical source
from tests.fixtures.pagination import (
    create_paginated_data,
    create_mock_fetcher,
    create_bulk_fetch_mocks
)

__all__ = [
    'create_paginated_data',
    'create_mock_fetcher',
    'create_bulk_fetch_mocks'
]