#!/usr/bin/env python3
"""
Tests for centralized error handling system

This test file verifies:
- Error class hierarchy and inheritance
- Error details and formatting
- API error classification
- Error handling in client operations
"""

from ..core.logging_config import get_logger
import requests
import unittest
from unittest.mock import patch, MagicMock

from yahoofinance.errors import (
    YFinanceError, APIError, ValidationError, 
    RateLimitError, ResourceNotFoundError, 
    ConnectionError, TimeoutError, DataError,
    format_error_details, classify_api_error
)
from yahoofinance.client import YFinanceClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)


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
        self.assertEqual(formatted, "Simple error")
    
    def test_detailed_error_formatting(self):
        """Test formatting of errors with details."""
        details = {
            "status_code": 404,
            "url": "https://api.example.com/resource",
            "ticker": "INVALID"
        }
        error = YFinanceError("Detailed error", details)
        
        formatted = format_error_details(error)
        
        # Verify all details are included
        self.assertIn("Detailed error", formatted)
        self.assertIn("status_code: 404", formatted)
        self.assertIn("url: https://api.example.com/resource", formatted)
        self.assertIn("ticker: INVALID", formatted)
    
    def test_nested_details_formatting(self):
        """Test formatting of errors with nested details."""
        nested_details = {
            "response": {
                "status": 429,
                "headers": {
                    "Retry-After": "30"
                }
            },
            "ticker": "AAPL"
        }
        error = APIError("Nested details error", nested_details)
        
        formatted = format_error_details(error)
        
        # Verify nested details are formatted correctly
        self.assertIn("Nested details error", formatted)
        self.assertIn("response:", formatted)
        self.assertIn("status", formatted)
        self.assertIn("429", formatted)
        self.assertIn("Retry-After", formatted)
        self.assertIn("30", formatted)
        self.assertIn("ticker: AAPL", formatted)


class TestAPIErrorClassification(unittest.TestCase):
    """Test classification of API errors by status code."""
    
    def test_not_found_classification(self):
        """Test classification of 404 errors."""
        error = classify_api_error(404, "Resource not found")
        self.assertIsInstance(error, ResourceNotFoundError)
        self.assertEqual(error.message, "Resource not found: Resource not found")
        self.assertEqual(error.details.get("status_code"), 404)
    
    def test_rate_limit_classification(self):
        """Test classification of 429 errors."""
        error = classify_api_error(429, "Too many requests")
        self.assertIsInstance(error, RateLimitError)
        self.assertEqual(error.message, "Rate limit exceeded: Too many requests")
        self.assertEqual(error.details.get("status_code"), 429)
    
    def test_timeout_classification(self):
        """Test classification of 504 (gateway timeout) errors."""
        error = classify_api_error(504, "Gateway timeout")
        self.assertIsInstance(error, TimeoutError)
        self.assertEqual(error.message, "Gateway timeout: Gateway timeout")
        self.assertEqual(error.details.get("status_code"), 504)
    
    def test_server_error_classification(self):
        """Test classification of 5xx errors."""
        error = classify_api_error(500, "Internal server error")
        self.assertIsInstance(error, APIError)  # Generic API error
        self.assertEqual(error.message, "Server error: Internal server error")
        self.assertEqual(error.details.get("status_code"), 500)
    
    def test_unknown_error_classification(self):
        """Test classification of non-standard error codes."""
        error = classify_api_error(418, "I'm a teapot")
        self.assertIsInstance(error, APIError)  # Generic API error
        self.assertEqual(error.message, "API error (418): I'm a teapot")
        self.assertEqual(error.details.get("status_code"), 418)


class TestClientErrorHandling(unittest.TestCase):
    """Test error handling in the client module."""
    
    @patch('yahoofinance.client.yf.Ticker')
    def test_rate_limit_handling(self, mock_ticker):
        """Test handling of rate limit errors."""
        # Setup mock to raise rate limit error
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Simulate 429 response
        response_mock = MagicMock()
        response_mock.status_code = 429
        response_mock.text = "Rate limit exceeded"
        http_error = requests.exceptions.HTTPError(response=response_mock)
        mock_ticker_instance.get_earnings_dates.side_effect = http_error
        
        # Test client handling
        client = YFinanceClient(retry_attempts=1)  # Just 1 retry for faster tests
        
        with self.assertRaises(RateLimitError):
            client.get_past_earnings_dates("AAPL")
        
        # The client should have called the method once
        mock_ticker_instance.get_earnings_dates.assert_called_once()
    
    @patch('yahoofinance.client.yf.Ticker')
    def test_not_found_handling(self, mock_ticker):
        """Test handling of not found errors."""
        # Setup mock to raise not found error
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Simulate 404 response
        response_mock = MagicMock()
        response_mock.status_code = 404
        response_mock.text = "Ticker not found"
        http_error = requests.exceptions.HTTPError(response=response_mock)
        
        # Make get_earnings_dates throw the error, not history
        mock_ticker_instance.get_earnings_dates.side_effect = http_error
        mock_ticker_instance.info = {}  # Empty info
        
        # Test client handling
        client = YFinanceClient()
        
        with self.assertRaises(ResourceNotFoundError):
            client.get_past_earnings_dates("INVALID")


if __name__ == "__main__":
    unittest.main()