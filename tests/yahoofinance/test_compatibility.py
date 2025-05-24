import datetime
import unittest
from io import StringIO
from unittest.mock import patch

import pandas as pd

from yahoofinance import utils
from yahoofinance.utils.data import format_utils as FormatUtils
from yahoofinance.utils.date import date_utils as DateUtils


class TestCompatibilityLayer(unittest.TestCase):
    """Test the compatibility layer that maintains backward compatibility."""

    def test_date_utils_validation(self):
        """Test DateUtils validate_date_format method."""
        # Valid date formats
        self.assertTrue(DateUtils.validate_date_format("2024-01-01"))
        self.assertTrue(DateUtils.validate_date_format("2023-12-31"))

        # Invalid date formats
        self.assertFalse(DateUtils.validate_date_format("01-01-2024"))  # Wrong format
        self.assertFalse(DateUtils.validate_date_format("2024/01/01"))  # Using slashes
        self.assertFalse(DateUtils.validate_date_format("2024.01.01"))  # Using dots
        self.assertFalse(DateUtils.validate_date_format("abcdef"))  # Not a date
        self.assertFalse(DateUtils.validate_date_format(""))  # Empty string
        self.assertFalse(DateUtils.validate_date_format("2024-13-01"))  # Invalid month
        self.assertFalse(DateUtils.validate_date_format("2024-01-32"))  # Invalid day

    @patch("yahoofinance.utils.date.date_utils.input")
    def test_date_utils_functionality(self, mock_input):
        """Test date_utils functionality for getting date ranges."""
        # Since get_user_dates doesn't exist in the module anymore,
        # we'll test the actual date_range functionality that replaced it

        # Test with explicit dates
        start = "2024-02-01"
        end = "2024-02-07"

        start_date, end_date = DateUtils.get_date_range(start, end)

        self.assertEqual(start_date.strftime("%Y-%m-%d"), start)
        self.assertEqual(end_date.strftime("%Y-%m-%d"), end)

        # Test with default end date (today)
        days_ago = 7
        start_date, end_date = DateUtils.get_date_range(days=days_ago)

        today = datetime.date.today()
        expected_start = today - datetime.timedelta(days=days_ago)

        self.assertEqual(end_date, today)
        self.assertEqual(start_date, expected_start)

    def test_format_utils_number_formatting(self):
        """Test FormatUtils format_number method."""
        # Test with various inputs
        self.assertEqual(FormatUtils.format_number(1.234, precision=2), "1.23")
        self.assertEqual(FormatUtils.format_number(0, precision=2), "0.00")
        self.assertEqual(FormatUtils.format_number(-1.234, precision=2), "-1.23")
        self.assertEqual(FormatUtils.format_number(None), "N/A")
        self.assertEqual(FormatUtils.format_number(float("nan")), "N/A")

        # Test with different precision
        self.assertEqual(FormatUtils.format_number(1.234, precision=3), "1.234")
        self.assertEqual(FormatUtils.format_number(1.234, precision=0), "1")

    def test_format_utils_table_formatting(self):
        """Test format_utils table formatting methods."""
        # Create test data for the modernized format_table function
        data = [
            {"Symbol": "AAPL", "Price": 150.0, "Change": 1.5},
            {"Symbol": "MSFT", "Price": 300.0, "Change": -2.0},
        ]

        columns = ["Symbol", "Price", "Change"]

        # Test format_table function
        formatted_table = FormatUtils.format_table(data, columns)

        # Verify the table has a header row plus data rows
        self.assertEqual(len(formatted_table), 3)  # Header + 2 data rows

        # Verify the header row
        self.assertEqual(formatted_table[0], columns)

        # Verify the data rows contain the right values
        self.assertEqual(formatted_table[1][0], "AAPL")
        self.assertEqual(formatted_table[2][0], "MSFT")

        # Test with empty data
        empty_table = FormatUtils.format_table([], columns)
        self.assertEqual(empty_table, [])
