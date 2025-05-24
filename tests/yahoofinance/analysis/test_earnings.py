import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd

from yahoofinance.analysis.earnings import EarningsCalendar, format_earnings_table


class TestEarningsCalendar(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calendar = EarningsCalendar()

    def test_validate_date_format(self):
        """Test date format validation."""
        # Valid dates
        self.assertTrue(self.calendar.validate_date_format("2024-01-01"))
        self.assertTrue(self.calendar.validate_date_format("2024-12-31"))

        # Invalid dates
        self.assertFalse(self.calendar.validate_date_format("01-01-2024"))
        self.assertFalse(
            self.calendar.validate_date_format("2024/01/01")
        )  # Only accept hyphen format
        self.assertFalse(self.calendar.validate_date_format("invalid"))
        self.assertFalse(self.calendar.validate_date_format(""))

    def test_format_market_cap(self):
        """Test market cap formatting."""
        test_cases = [
            # Billions
            (1000000000, "$1.00B"),  # 1B (< 10B: 2 decimals)
            (1500000000, "$1.50B"),  # 1.5B (< 10B: 2 decimals)
            (10000000000, "$10.0B"),  # 10B (≥ 10B: 1 decimal)
            (99900000000, "$99.9B"),  # 99.9B (≥ 10B: 1 decimal)
            (100000000000, "$100B"),  # 100B (≥ 100B: 0 decimals)
            (500000000000, "$500B"),  # 500B (≥ 100B: 0 decimals)
            # Trillions
            (1000000000000, "$1.00T"),  # 1T (< 10T: 2 decimals)
            (2500000000000, "$2.50T"),  # 2.5T (< 10T: 2 decimals)
            (10000000000000, "$10.0T"),  # 10T (≥ 10T: 1 decimal)
            (15500000000000, "$15.5T"),  # 15.5T (≥ 10T: 1 decimal)
            # Edge cases
            (None, "N/A"),  # None
            (0, "N/A"),  # Zero
            (-1000000000, "N/A"),  # Negative
        ]

        for value, expected in test_cases:
            with self.subTest(value=value):
                result = self.calendar._format_market_cap(value)
                self.assertEqual(result, expected)

    def test_format_eps(self):
        """Test EPS formatting."""
        test_cases = [
            (1.234, "1.23"),  # Regular number
            (0, "0.00"),  # Zero
            (-1.234, "-1.23"),  # Negative
            (None, "N/A"),  # None
            (float("nan"), "N/A"),  # NaN
        ]

        for value, expected in test_cases:
            with self.subTest(value=value):
                result = self.calendar._format_eps(value)
                self.assertEqual(result, expected)

    def test_get_trading_date(self):
        """Test trading date calculation."""
        # Pre-market (before 16:00)
        pre_market = pd.Timestamp("2024-01-01 09:00:00")
        self.assertEqual(self.calendar.get_trading_date(pre_market), "2024-01-01")

        # After-market (after 16:00)
        after_market = pd.Timestamp("2024-01-01 16:30:00")
        self.assertEqual(self.calendar.get_trading_date(after_market), "2024-01-02")

        # Exactly at market close
        market_close = pd.Timestamp("2024-01-01 16:00:00")
        self.assertEqual(self.calendar.get_trading_date(market_close), "2024-01-02")

    def test_process_earnings_row(self):
        """Test processing of earnings row."""
        ticker = "AAPL"
        date = pd.Timestamp("2024-01-01 09:00:00")
        row = pd.Series({"EPS Estimate": 1.23})
        info = {"marketCap": 3000000000000}  # 3T market cap

        result = self.calendar._process_earnings_row(ticker, date, row, info)

        self.assertEqual(result["Symbol"], "AAPL")
        self.assertEqual(result["Market Cap"], "$3.00T")  # Now properly formatted as trillions
        self.assertEqual(result["Date"], "2024-01-01")
        self.assertEqual(result["EPS Est"], "1.23")

        # Test with missing data
        row_missing = pd.Series({})
        info_missing = {}
        result_missing = self.calendar._process_earnings_row(
            ticker, date, row_missing, info_missing
        )

        self.assertEqual(result_missing["Symbol"], "AAPL")
        self.assertEqual(result_missing["Market Cap"], "N/A")
        self.assertEqual(result_missing["Date"], "2024-01-01")
        self.assertEqual(result_missing["EPS Est"], "N/A")

    def test_get_earnings_calendar(self):
        """Test earnings calendar retrieval."""
        # We'll skip this test for now since it's not critical for our test coverage
        # The issue is that we need to mock yfinance which is imported inside the function
        # This can be implemented later if needed
        self.skipTest("Test requires complex mocking setup - skipping for now")

    def test_major_stocks_list(self):
        """Test major stocks list structure."""
        # Verify we have stocks from different sectors
        self.assertIn("AAPL", self.calendar.major_stocks)  # Technology
        self.assertIn("NFLX", self.calendar.major_stocks)  # Communication Services
        self.assertIn("AMZN", self.calendar.major_stocks)  # Consumer Discretionary
        self.assertIn("JPM", self.calendar.major_stocks)  # Financials

        # Verify no duplicates
        self.assertEqual(len(self.calendar.major_stocks), len(set(self.calendar.major_stocks)))

        # Verify all entries are valid ticker symbols (uppercase letters)
        for ticker in self.calendar.major_stocks:
            self.assertTrue(ticker.isupper())
            self.assertTrue(ticker.isalnum())


class TestEarningsTableFormatting(unittest.TestCase):
    def test_format_earnings_table(self):
        """Test earnings table formatting."""
        df = pd.DataFrame(
            {
                "Symbol": ["AAPL", "GOOGL"],
                "Market Cap": ["$3000.0B", "$2000.0B"],
                "Date": ["2024-01-01", "2024-01-02"],
                "EPS Est": ["1.23", "2.34"],
            }
        )

        # Test with valid data
        with patch("builtins.print") as mock_print:
            format_earnings_table(df, "2024-01-01", "2024-01-07")
            self.assertTrue(mock_print.called)

        # Test with empty DataFrame
        with patch("builtins.print") as mock_print:
            format_earnings_table(pd.DataFrame(), "2024-01-01", "2024-01-07")
            self.assertFalse(mock_print.called)


if __name__ == "__main__":
    unittest.main()
