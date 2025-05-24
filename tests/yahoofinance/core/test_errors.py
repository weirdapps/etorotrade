#!/usr/bin/env python3
"""
Tests for centralized error handling system

This test file verifies:
- Error class hierarchy and inheritance
- Error details and formatting
- API error classification
- Error handling in client operations
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

import requests

from yahoofinance.core.errors import (
    APIError,
    ConnectionError,
    DataError,
    RateLimitError,
    ResourceNotFoundError,
    TimeoutError,
    ValidationError,
    YFinanceError,
    classify_api_error,
    format_error_details,
)


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestErrorHierarchy(unittest.TestCase):
    """Test the error class hierarchy and inheritance."""

    def test_basic_inheritance(self):
        """Test basic inheritance relationships between error classes."""
        # Create test errors
        api_error = APIError("API error")
        validation_error = ValidationError("Validation error")
        rate_limit_error = RateLimitError("Rate limit error")

        # Test inheritance
        self.assertIsInstance(api_error, YFinanceError)
        self.assertIsInstance(validation_error, YFinanceError)
        self.assertIsInstance(rate_limit_error, APIError)
        self.assertIsInstance(rate_limit_error, YFinanceError)

    def test_specific_error_types(self):
        """Test specific error types and their properties."""
        # Rate limit error
        rate_error = RateLimitError("Rate limited", retry_after=30)
        self.assertEqual(rate_error.retry_after, 30)
        self.assertEqual(rate_error.message, "Rate limited")

        # Resource not found
        details = {"ticker": "AAPL"}
        not_found = ResourceNotFoundError("AAPL ticker not found", details)
        self.assertEqual(not_found.details.get("ticker"), "AAPL")

        # Connection error
        details = {"url": "https://api.example.com"}
        conn_error = ConnectionError("Failed to connect", details)
        self.assertEqual(conn_error.details.get("url"), "https://api.example.com")

        # Timeout error
        details = {"timeout": 5}
        timeout = TimeoutError("Request timed out", details)
        self.assertEqual(timeout.details.get("timeout"), 5)

        # Data error
        details = {"value": "invalid"}
        data_error = DataError("Invalid data format", details)
        self.assertEqual(data_error.details.get("value"), "invalid")


class TestErrorFormatting(unittest.TestCase):
    """Test error formatting and details."""

    def test_simple_error_formatting(self):
        """Test formatting of simple errors without details."""
        simple_error = YFinanceError("Simple error")
        formatted = format_error_details(simple_error)
        self.assertIn("Simple error", formatted)

    def test_detailed_error_formatting(self):
        """Test formatting of errors with details."""
        details = {
            "status_code": 404,
            "url": "https://api.example.com/resource",
            "ticker": "INVALID",
        }
        error = YFinanceError("Detailed error", details)

        formatted = format_error_details(error)

        # Verify error message is included
        self.assertIn("Detailed error", formatted)

        # Verify details are mentioned somehow
        self.assertTrue(any(key in formatted for key in details.keys()))


class TestAPIErrorClassification(unittest.TestCase):
    """Test classification of API errors by status code."""

    def test_not_found_classification(self):
        """Test classification of 404 errors."""
        error = classify_api_error(404, "Resource not found")
        self.assertIsInstance(error, ResourceNotFoundError)
        self.assertIn("Resource not found", error.message)
        self.assertEqual(error.details.get("status_code"), 404)

    def test_rate_limit_classification(self):
        """Test classification of 429 errors."""
        error = classify_api_error(429, "Too many requests")
        self.assertIsInstance(error, RateLimitError)
        self.assertIn("Rate limit exceeded", error.message)
        self.assertEqual(error.details.get("status_code"), 429)

    def test_server_error_classification(self):
        """Test classification of 5xx errors."""
        error = classify_api_error(500, "Internal server error")
        self.assertIsInstance(error, APIError)  # Generic API error
        self.assertIn("Server error", error.message)
        self.assertEqual(error.details.get("status_code"), 500)

    def test_unknown_error_classification(self):
        """Test classification of non-standard error codes."""
        error = classify_api_error(418, "I'm a teapot")
        self.assertIsInstance(error, APIError)  # Generic API error
        self.assertEqual(error.details.get("status_code"), 418)


if __name__ == "__main__":
    unittest.main()
