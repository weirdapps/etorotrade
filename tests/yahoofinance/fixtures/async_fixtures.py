"""
Test fixtures for async testing.

This module provides fixtures and helpers for testing asynchronous functionality.

DEPRECATED: This module is deprecated in favor of tests.fixtures.async_fixtures.
Use the fixtures from that module instead, which are automatically registered in conftest.py.
"""

import warnings

warnings.warn(
    "This module (tests.yahoofinance.fixtures.async_fixtures) is deprecated. "
    "Use tests.fixtures.async_fixtures instead, which is automatically registered in conftest.py.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all fixtures and functions from the canonical source
from tests.fixtures.async_fixtures import (
    create_flaky_function,
    create_async_processor_mock
)