"""
Test fixtures for etorotrade tests.

DEPRECATED: These fixtures are now in tests/fixtures/. 
Please import them from there instead.
"""

import warnings

warnings.warn(
    "The fixtures in tests/yahoofinance/fixtures/ are deprecated. "
    "Please import fixtures from tests/fixtures/ instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from canonical source to maintain backwards compatibility
from tests.fixtures import (
    create_paginated_data,
    create_mock_fetcher,
    create_bulk_fetch_mocks,
    create_flaky_function,
    create_async_processor_mock
)

__all__ = [
    'create_paginated_data',
    'create_mock_fetcher',
    'create_bulk_fetch_mocks',
    'create_flaky_function',
    'create_async_processor_mock'
]