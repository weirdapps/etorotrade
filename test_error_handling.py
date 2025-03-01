#!/usr/bin/env python3
"""
Test script to verify the improved error handling capabilities.
"""

import logging
import requests
import unittest
from unittest.mock import patch, MagicMock

from yahoofinance.errors import (
    YFinanceError, APIError, ValidationError,
    RateLimitError, ResourceNotFoundError,
    format_error_details, classify_api_error
)
from yahoofinance.client import YFinanceClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_error_handling")

class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling improvements."""
    
    def test_error_hierarchy(self):
        """Test the error class hierarchy relationships."""
        # Create test errors
        base_error = YFinanceError("Base error")
        api_error = APIError("API error")
        rate_limit_error = RateLimitError("Rate limit error", 30)
        
        # Test inheritance
        self.assertIsInstance(api_error, YFinanceError)
        self.assertIsInstance(rate_limit_error, APIError)
        self.assertIsInstance(rate_limit_error, YFinanceError)
        
        # Test properties
        self.assertEqual(rate_limit_error.retry_after, 30)
        self.assertEqual(base_error.message, "Base error")
        
    def test_error_formatting(self):
        """Test error formatting utility."""
        # Simple error
        simple_error = YFinanceError("Simple error")
        simple_formatted = format_error_details(simple_error)
        self.assertEqual(simple_formatted, "Simple error")
        
        # Error with details
        detailed_error = YFinanceError(
            "Detailed error",
            {"code": 123, "source": "API", "ticker": "AAPL"}
        )
        detailed_formatted = format_error_details(detailed_error)
        
        # Check that all details are included
        self.assertIn("Detailed error", detailed_formatted)
        self.assertIn("code: 123", detailed_formatted)
        self.assertIn("source: API", detailed_formatted)
        self.assertIn("ticker: AAPL", detailed_formatted)
        
    def test_api_error_classification(self):
        """Test classification of API errors by status code."""
        # Test 404 error
        not_found_error = classify_api_error(404, "Ticker not found")
        self.assertIsInstance(not_found_error, ResourceNotFoundError)
        
        # Test 429 error
        rate_limit_error = classify_api_error(429, "Too many requests")
        self.assertIsInstance(rate_limit_error, RateLimitError)
        
        # Test generic error
        generic_error = classify_api_error(418, "I'm a teapot")  # Non-standard code
        self.assertIsInstance(generic_error, APIError)
        self.assertEqual(generic_error.details.get("status_code"), 418)
    
    @patch('yahoofinance.client.yf.Ticker')
    def test_client_error_handling(self, mock_ticker):
        """Test improved error handling in the client."""
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
        client = YFinanceClient(retry_attempts=1)  # Just 1 retry for faster test
        
        with self.assertRaises(RateLimitError):
            client.get_past_earnings_dates("AAPL")
        
        # The client should have called the method once
        mock_ticker_instance.get_earnings_dates.assert_called_once()


if __name__ == "__main__":
    logger.info("Running error handling tests...")
    unittest.main()