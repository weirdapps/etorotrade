import unittest
from unittest.mock import Mock, patch

from yahoofinance.core.client import YFinanceClient
from yahoofinance.core.config import RATE_LIMIT
from yahoofinance.core.errors import ValidationError


class TestYFinanceClient(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.client = YFinanceClient()

    def test_initialization(self):
        """Test client initialization with default and custom values."""
        # Test default values
        client = YFinanceClient()
        # Use the actual values from the config rather than hardcoded values
        self.assertEqual(client.max_retries, RATE_LIMIT["MAX_RETRY_ATTEMPTS"])
        self.assertEqual(client.timeout, RATE_LIMIT["API_TIMEOUT"])

        # Test custom values
        custom_client = YFinanceClient(max_retries=5, timeout=60)
        self.assertEqual(custom_client.max_retries, 5)
        self.assertEqual(custom_client.timeout, 60)

    def test_validate_ticker_valid(self):
        """Test ticker validation with valid inputs."""
        # Test simple ticker
        self.assertTrue(self.client.validate_ticker("AAPL"))

        # Test with exchange suffix
        self.assertTrue(self.client.validate_ticker("AAPL.US"))

        # Test with hyphen
        self.assertTrue(self.client.validate_ticker("BRK-B"))

        # Test with dot
        self.assertTrue(self.client.validate_ticker("BRK.B"))

        # Test with international ticker
        self.assertTrue(self.client.validate_ticker("0700.HK"))

        # Test with crypto
        self.assertTrue(self.client.validate_ticker("BTC-USD"))

    def test_validate_ticker_invalid(self):
        """Test ticker validation with invalid inputs."""
        # Test None
        with self.assertRaises(ValidationError):
            self.client.validate_ticker(None)

        # Test empty string
        with self.assertRaises(ValidationError):
            self.client.validate_ticker("")

        # Test non-string
        with self.assertRaises(ValidationError):
            self.client.validate_ticker(123)

        # Test too long ticker
        with self.assertRaises(ValidationError):
            self.client.validate_ticker("A" * 21)
