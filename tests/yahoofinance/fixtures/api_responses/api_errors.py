"""
API error response fixtures for etorotrade tests.

This module contains fixtures for simulating various API error scenarios
for unit and integration testing.

DEPRECATED: This module is deprecated in favor of tests.fixtures.api_responses.api_errors.
Use the fixtures from that module instead, which are automatically registered in conftest.py.
"""

import warnings


warnings.warn(
    "This module (tests.yahoofinance.fixtures.api_responses.api_errors) is deprecated. "
    "Use tests.fixtures.api_responses.api_errors instead, which is automatically registered in conftest.py.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all fixtures from the canonical source
from tests.fixtures.api_responses.api_errors import (  # Class; Fixtures; Constants
    NON_JSON_TEXT,
    MockResponse,
    auth_error_response,
    connection_error_response,
    data_error,
    malformed_json_response,
    mock_error_responses,
    not_found_response,
    rate_limit_response,
    server_error_response,
    timeout_response,
    validation_error,
)
