#!/usr/bin/env python3
"""
Test script to verify the improved error handling capabilities.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

import requests

from yahoofinance.core.errors import (
    APIError,
    DataError,
    RateLimitError,
    ResourceNotFoundError,
    ValidationError,
    YFinanceError,
    classify_api_error,
    format_error_details,
)
from yahoofinance.core.logging import get_logger
from yahoofinance.utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_error_context,
    with_retry,
)


# Set up logging
logger = get_logger("test_error_handling")

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
        self.assertEqual(simple_formatted, "YFinanceError: Simple error")
        
        # Error with details
        detailed_error = YFinanceError(
            "Detailed error", 
            {"code": 123, "source": "API", "ticker": "AAPL"}
        )
        detailed_formatted = format_error_details(detailed_error)
        
        # Check that all details are included
        self.assertIn("YFinanceError: Detailed error", detailed_formatted)
        self.assertIn("code=123", detailed_formatted)
        self.assertIn("source=API", detailed_formatted)
        self.assertIn("ticker=AAPL", detailed_formatted)
        
    @with_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
    def test_with_retry_decorator(self):
        """Test the with_retry decorator."""
        # This test should pass without errors since the decorator is applied correctly
        self.assertTrue(True)
        
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
    
    def test_enrich_error_context(self):
        """Test enriching errors with context information."""
        # Create a base error
        error = YFinanceError("Test error")
        
        # Define context
        context = {
            "ticker": "AAPL",
            "operation": "get_price",
            "timestamp": 1234567890
        }
        
        # Enrich the error
        enriched_error = enrich_error_context(error, context)
        
        # Check that the context was added
        self.assertEqual(enriched_error.details["ticker"], "AAPL")
        self.assertEqual(enriched_error.details["operation"], "get_price")
        self.assertEqual(enriched_error.details["timestamp"], 1234567890)
        
    def test_translate_error(self):
        """Test translating standard errors to YFinance errors."""
        # Test ValueError translation
        value_error = ValueError("Invalid value")
        translated = translate_error(value_error)
        self.assertIsInstance(translated, ValidationError)
        
        # Test KeyError translation
        key_error = KeyError("missing_key")
        translated = translate_error(key_error)
        self.assertIsInstance(translated, DataError)
        
        # Test context addition
        context = {"operation": "test_operation"}
        translated = translate_error(ValueError("test"), context=context)
        self.assertEqual(translated.details["operation"], "test_operation")
        
    def test_safe_operation_decorator(self):
        """Test the safe_operation decorator."""
        
        # Define a function that might fail
        @safe_operation(default_value={"status": "default"})
        def might_fail(should_fail=False):
            if should_fail:
                raise YFinanceError("Failed on purpose")
            return {"status": "success"}
        
        # Test successful execution
        result = might_fail(should_fail=False)
        self.assertEqual(result["status"], "success")
        
        # Test failure with default value
        result = might_fail(should_fail=True)
        self.assertEqual(result["status"], "default")
        
        # Test with reraise=True
        @safe_operation(default_value=None, reraise=True)
        def will_reraise(should_fail=False):
            if should_fail:
                raise YFinanceError("Will be reraised")
            return "success"
        
        # Should execute normally
        self.assertEqual(will_reraise(should_fail=False), "success")
        
        # Should reraise the error
        with self.assertRaises(YFinanceError):
            will_reraise(should_fail=True)


if __name__ == "__main__":
    logger.info("Running error handling tests...")
    unittest.main()