"""Test the refactored utils package for backward compatibility."""

import unittest
import sys
import os


class TestUtilsRefactor(unittest.TestCase):
    """Test that the refactored utils package maintains backward compatibility."""
    
    def test_backward_compatibility(self):
        """Test imports from the old module still work."""
        # Import from the old module
        from yahoofinance.utils import (
            is_us_ticker, 
            normalize_hk_ticker, 
            FormatUtils, 
            DateUtils,
            global_rate_limiter,
            rate_limited,
            batch_process
        )
        
        # Test basic functionality from market utils
        self.assertTrue(is_us_ticker("AAPL"))
        self.assertEqual(normalize_hk_ticker("00700.HK"), "0700.HK")
        
        # Test FormatUtils
        formatted = FormatUtils.format_number(123.456)
        self.assertEqual(formatted, "123.46")
        
    def test_new_imports(self):
        """Test imports from the new module structure."""
        # Import from the new module structure
        from yahoofinance.utils.market import is_us_ticker, normalize_hk_ticker
        from yahoofinance.utils.data import FormatUtils
        from yahoofinance.utils.date import DateUtils
        from yahoofinance.utils.network import global_rate_limiter, rate_limited
        
        # Test basic functionality from market utils
        self.assertTrue(is_us_ticker("AAPL"))
        self.assertEqual(normalize_hk_ticker("00700.HK"), "0700.HK")
        
        # Test FormatUtils
        formatted = FormatUtils.format_number(123.456)
        self.assertEqual(formatted, "123.46")
        
    def test_recursive_imports(self):
        """Test multilevel imports work correctly."""
        # Test specific imports
        from yahoofinance.utils.market.ticker_utils import is_us_ticker
        from yahoofinance.utils.data.format_utils import format_number
        
        # Test functionality
        self.assertTrue(is_us_ticker("AAPL"))
        self.assertEqual(format_number(123.456), "123.46")


if __name__ == "__main__":
    unittest.main()