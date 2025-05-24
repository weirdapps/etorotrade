import unittest

from yahoofinance.core.config import COLUMN_NAMES


class TestColumnNameConstants(unittest.TestCase):
    def test_column_name_constants(self):
        """Test that the column name constants are properly defined."""
        self.assertEqual(COLUMN_NAMES["EARNINGS_DATE"], "Earnings Date")
        self.assertEqual(COLUMN_NAMES["BUY_PERCENTAGE"], "% BUY")


if __name__ == "__main__":
    unittest.main()
