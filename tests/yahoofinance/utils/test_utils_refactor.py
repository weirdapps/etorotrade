"""Test the refactored utils package for backward compatibility."""

import unittest


class TestUtilsRefactor(unittest.TestCase):
    """Test that the refactored utils package maintains backward compatibility."""

    def test_recursive_imports(self):
        """Test multilevel imports work correctly."""
        # Test specific imports
        from yahoofinance.utils.data.format_utils import format_number
        from yahoofinance.utils.market.ticker_utils import is_us_ticker

        # Test functionality
        self.assertTrue(is_us_ticker("AAPL"))
        self.assertEqual(format_number(123.456), "123.46")
