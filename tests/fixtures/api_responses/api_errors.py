"""
API error response fixtures for etorotrade tests.

This module contains fixtures for simulating various API error scenarios
for unit and integration testing.
"""

import json

import pytest
import requests

from yahoofinance.core.errors import (
    DataError,
    ValidationError,
)


# Constants for repeated strings
NON_JSON_TEXT = "This is not JSON"


class MockResponse:
    """
    Mock HTTP response for testing API error handling.
    """

    def __init__(self, status_code, json_data=None, text=None, headers=None, error=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or ""
        self.headers = headers or {}
        self.error = error
        self.reason = "Error" if status_code >= 400 else "OK"

    def json(self):
        if self.error:
            raise self.error
        return self._json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP Error: {self.status_code}")


@pytest.fixture
def rate_limit_response():
    """
    Create a rate limit error response (HTTP 429).

    Returns:
        MockResponse: A mock response with rate limit headers
    """
    headers = {
        "Retry-After": "30",
        "X-RateLimit-Limit": "100",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": "1617978000",
    }
    return MockResponse(
        status_code=429,
        text="Too Many Requests",
        headers=headers,
        json_data={"error": "rate_limit_exceeded"},
    )


@pytest.fixture
def not_found_response():
    """
    Create a not found error response (HTTP 404).

    Returns:
        MockResponse: A mock response for resource not found
    """
    return MockResponse(
        status_code=404, text="Not Found", json_data={"error": "resource_not_found"}
    )


@pytest.fixture
def server_error_response():
    """
    Create a server error response (HTTP 500).

    Returns:
        MockResponse: A mock response for server error
    """
    return MockResponse(
        status_code=500, text="Internal Server Error", json_data={"error": "internal_server_error"}
    )


@pytest.fixture
def malformed_json_response():
    """
    Create a response with malformed JSON.

    Returns:
        MockResponse: A mock response that raises an error when json() is called
    """
    return MockResponse(
        status_code=200,
        text=NON_JSON_TEXT,
        error=json.JSONDecodeError("Expecting value", NON_JSON_TEXT, 0),
    )


@pytest.fixture
def timeout_response():
    """
    Create a timeout error for requests.

    Returns:
        Exception: A requests.Timeout exception
    """
    return requests.Timeout("Request timed out")


@pytest.fixture
def connection_error_response():
    """
    Create a connection error for requests.

    Returns:
        Exception: A requests.ConnectionError exception
    """
    return requests.ConnectionError("Connection failed")


@pytest.fixture
def auth_error_response():
    """
    Create an authentication error response (HTTP 401).

    Returns:
        MockResponse: A mock response for authentication error
    """
    return MockResponse(
        status_code=401, text="Unauthorized", json_data={"error": "invalid_credentials"}
    )


@pytest.fixture
def validation_error():
    """
    Create a validation error.

    Returns:
        Exception: A ValidationError exception
    """
    return ValidationError("Invalid ticker format")


@pytest.fixture
def data_error():
    """
    Create a data error.

    Returns:
        Exception: A DataError exception
    """
    return DataError("Missing required data")


@pytest.fixture
def mock_error_responses():
    """
    Create a dictionary of all error responses.

    Returns:
        dict: A dictionary of error names to mock responses or exceptions
    """
    return {
        "rate_limit": MockResponse(
            status_code=429,
            text="Too Many Requests",
            headers={"Retry-After": "30"},
            json_data={"error": "rate_limit_exceeded"},
        ),
        "not_found": MockResponse(
            status_code=404, text="Not Found", json_data={"error": "resource_not_found"}
        ),
        "server_error": MockResponse(
            status_code=500,
            text="Internal Server Error",
            json_data={"error": "internal_server_error"},
        ),
        "malformed_json": MockResponse(
            status_code=200,
            text=NON_JSON_TEXT,
            error=json.JSONDecodeError("Expecting value", NON_JSON_TEXT, 0),
        ),
        "timeout": requests.Timeout("Request timed out"),
        "connection_error": requests.ConnectionError("Connection failed"),
        "auth_error": MockResponse(
            status_code=401, text="Unauthorized", json_data={"error": "invalid_credentials"}
        ),
        "validation_error": ValidationError("Invalid ticker format"),
        "data_error": DataError("Missing required data"),
    }
