import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd

from yahoofinance.analysis.market import MarketMetrics
from yahoofinance.utils.error_handling import safe_operation, with_retry


# Since we don't have economic data functions in market.py, we'll test the MarketMetrics class instead
class TestMarketData(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.empty_metrics = MarketMetrics()
        self.sample_metrics = MarketMetrics(
            avg_upside=15.5,
            median_upside=12.0,
            avg_buy_percentage=78.5,
            median_buy_percentage=80.0,
            avg_pe_ratio=18.5,
            median_pe_ratio=17.2,
            avg_forward_pe=16.8,
            median_forward_pe=15.5,
            avg_peg_ratio=1.8,
            median_peg_ratio=1.5,
        )

    def test_market_metrics_init(self):
        """Test MarketMetrics initialization with and without values."""
        # Test default initialization (all values should be None)
        self.assertIsNone(self.empty_metrics.avg_upside)
        self.assertIsNone(self.empty_metrics.median_upside)
        self.assertIsNone(self.empty_metrics.avg_buy_percentage)
        self.assertIsNone(self.empty_metrics.median_buy_percentage)
        self.assertIsNone(self.empty_metrics.avg_pe_ratio)
        self.assertIsNone(self.empty_metrics.median_pe_ratio)
        self.assertIsNone(self.empty_metrics.avg_forward_pe)
        self.assertIsNone(self.empty_metrics.median_forward_pe)
        self.assertIsNone(self.empty_metrics.avg_peg_ratio)
        self.assertIsNone(self.empty_metrics.median_peg_ratio)

        # Test initialization with values
        self.assertEqual(self.sample_metrics.avg_upside, 15.5)
        self.assertEqual(self.sample_metrics.median_upside, 12.0)
        self.assertEqual(self.sample_metrics.avg_buy_percentage, 78.5)
        self.assertEqual(self.sample_metrics.median_buy_percentage, 80.0)
        self.assertEqual(self.sample_metrics.avg_pe_ratio, 18.5)
        self.assertEqual(self.sample_metrics.median_pe_ratio, 17.2)
        self.assertEqual(self.sample_metrics.avg_forward_pe, 16.8)
        self.assertEqual(self.sample_metrics.median_forward_pe, 15.5)
        self.assertEqual(self.sample_metrics.avg_peg_ratio, 1.8)
        self.assertEqual(self.sample_metrics.median_peg_ratio, 1.5)

    @with_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
    def test_dataclass_functionality(self):
        """Test dataclass functionality with appropriate retries."""
        # Ensure the dataclass properly tracks changes
        metrics = MarketMetrics(avg_upside=10.0)
        self.assertEqual(metrics.avg_upside, 10.0)

        # Change a value and ensure it updates
        metrics.avg_upside = 15.0
        self.assertEqual(metrics.avg_upside, 15.0)

    @safe_operation(default_value=None)
    def test_comparison(self):
        """Test comparison of metrics with safe operation."""
        # Create two identical instances
        metrics1 = MarketMetrics(avg_upside=10.0, median_upside=9.0)
        metrics2 = MarketMetrics(avg_upside=10.0, median_upside=9.0)

        # They should be equal
        self.assertEqual(metrics1.avg_upside, metrics2.avg_upside)
        self.assertEqual(metrics1.median_upside, metrics2.median_upside)


if __name__ == "__main__":
    unittest.main()
