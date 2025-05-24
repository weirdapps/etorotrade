"""Shared test fixtures for rate limiter tests.

This module contains fixtures and utilities used by multiple
rate limiter test modules to reduce duplication.

DEPRECATED: This module is deprecated in favor of tests.fixtures.rate_limiter_fixtures.
Use the fixtures from that module instead, which are automatically registered in conftest.py.
"""

import warnings


warnings.warn(
    "This module (tests.yahoofinance.fixtures.rate_limiter_fixtures) is deprecated. "
    "Use tests.fixtures.rate_limiter_fixtures instead, which is automatically registered in conftest.py.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all fixtures from the canonical source
from tests.fixtures.rate_limiter_fixtures import (  # Fixtures; Helper functions
    create_bulk_fetch_mocks,
    fresh_rate_limiter,
    increment_calls_function,
    mock_rate_limiter,
    rate_limiter_with_calls,
    rate_limiter_with_different_ticker_errors,
    rate_limiter_with_errors,
)
