import unittest
from unittest.mock import Mock

import yfinance as yf

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from yahoofinance.core.types import StockData
from yahoofinance.utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)


class TestExceptions(unittest.TestCase):
    """Test suite for custom exceptions"""

    def test_exception_hierarchy(self):
        """Test exception class inheritance hierarchy"""
        self.assertTrue(issubclass(APIError, YFinanceError))
        self.assertTrue(issubclass(ValidationError, YFinanceError))
        self.assertTrue(issubclass(YFinanceError, Exception))

    def test_error_messages(self):
        """Test error message handling"""
        error_msg = "Test error message"
        yf_error = YFinanceError(error_msg)
        api_error = APIError(error_msg)
        validation_error = ValidationError(error_msg)

        self.assertEqual(str(yf_error), error_msg)
        self.assertEqual(str(api_error), error_msg)
        self.assertEqual(str(validation_error), error_msg)

    def test_exception_chaining(self):
        """Test exception chaining"""
        original_error = ValueError("Original error")
        try:
            try:
                raise original_error
            except ValueError as e:
                raise APIError("Chained error") from e
        except APIError as e:
            self.assertIs(e.__cause__, original_error)


class TestStockData(unittest.TestCase):
    """Test suite for StockData class"""

    def test_required_fields(self):
        """Test required fields initialization"""
        # Test with minimum required fields
        stock = StockData(name="Test Stock", sector="Technology", recommendation_key="buy")

        self.assertEqual(stock.name, "Test Stock")
        self.assertEqual(stock.sector, "Technology")
        self.assertEqual(stock.recommendation_key, "buy")

        # All optional fields should be None by default
        self.assertIsNone(stock.market_cap)
        self.assertIsNone(stock.current_price)
        self.assertIsNone(stock.target_price)
        self.assertIsNone(stock.price_change_percentage)
        self.assertIsNone(stock.mtd_change)
        self.assertIsNone(stock.ytd_change)
        self.assertIsNone(stock.two_year_change)
        self.assertIsNone(stock.recommendation_mean)
        self.assertIsNone(stock.analyst_count)
        self.assertIsNone(stock.pe_trailing)
        self.assertIsNone(stock.pe_forward)
        self.assertIsNone(stock.peg_ratio)
        self.assertIsNone(stock.quick_ratio)
        self.assertIsNone(stock.current_ratio)
        self.assertIsNone(stock.debt_to_equity)
        self.assertIsNone(stock.short_float_pct)
        self.assertIsNone(stock.short_ratio)
        self.assertIsNone(stock.beta)
        self.assertIsNone(stock.alpha)
        self.assertIsNone(stock.sharpe_ratio)
        self.assertIsNone(stock.sortino_ratio)
        self.assertIsNone(stock.cash_percentage)
        self.assertIsNone(stock.dividend_yield)
        self.assertIsNone(stock.last_earnings)
        self.assertIsNone(stock.previous_earnings)
        self.assertIsNone(stock.insider_buy_pct)
        self.assertIsNone(stock.insider_transactions)
        self.assertIsNone(stock.ticker_object)

    def test_optional_fields(self):
        """Test optional fields initialization"""
        stock = StockData(
            name="Test Stock",
            sector="Technology",
            recommendation_key="buy",
            market_cap=1000000000.0,
            current_price=150.0,
            target_price=180.0,
            price_change_percentage=5.0,
            mtd_change=3.0,
            ytd_change=10.0,
            two_year_change=20.0,
            recommendation_mean=2.0,
            analyst_count=10,
            pe_trailing=20.5,
            pe_forward=18.2,
            peg_ratio=1.5,
            quick_ratio=1.1,
            current_ratio=1.2,
            debt_to_equity=150.0,
            short_float_pct=2.0,
            short_ratio=1.5,
            beta=1.1,
            alpha=0.2,
            sharpe_ratio=1.8,
            sortino_ratio=2.1,
            cash_percentage=15.0,
            dividend_yield=0.025,
            last_earnings="2024-01-01",
            previous_earnings="2023-10-01",
            insider_buy_pct=75.0,
            insider_transactions=5,
        )

        # Verify all optional fields are set correctly
        self.assertEqual(stock.market_cap, 1000000000.0)
        self.assertEqual(stock.current_price, 150.0)
        self.assertEqual(stock.target_price, 180.0)
        self.assertEqual(stock.price_change_percentage, 5.0)
        self.assertEqual(stock.mtd_change, 3.0)
        self.assertEqual(stock.ytd_change, 10.0)
        self.assertEqual(stock.two_year_change, 20.0)
        self.assertEqual(stock.recommendation_mean, 2.0)
        self.assertEqual(stock.analyst_count, 10)
        self.assertEqual(stock.pe_trailing, 20.5)
        self.assertEqual(stock.pe_forward, 18.2)
        self.assertEqual(stock.peg_ratio, 1.5)
        self.assertEqual(stock.quick_ratio, 1.1)
        self.assertEqual(stock.current_ratio, 1.2)
        self.assertEqual(stock.debt_to_equity, 150.0)
        self.assertEqual(stock.short_float_pct, 2.0)
        self.assertEqual(stock.short_ratio, 1.5)
        self.assertEqual(stock.beta, 1.1)
        self.assertEqual(stock.alpha, 0.2)
        self.assertEqual(stock.sharpe_ratio, 1.8)
        self.assertEqual(stock.sortino_ratio, 2.1)
        self.assertEqual(stock.cash_percentage, 15.0)
        self.assertEqual(stock.dividend_yield, 0.025)
        self.assertEqual(stock.last_earnings, "2024-01-01")
        self.assertEqual(stock.previous_earnings, "2023-10-01")
        self.assertEqual(stock.insider_buy_pct, 75.0)
        self.assertEqual(stock.insider_transactions, 5)

    def test_ticker_object_property(self):
        """Test ticker_object property with valid ticker object"""
        mock_ticker = Mock(spec=yf.Ticker)
        stock = StockData(
            name="Test Stock",
            sector="Technology",
            recommendation_key="buy",
            ticker_object=mock_ticker,
        )

        self.assertIs(stock.ticker_object, mock_ticker)

    def test_to_dict_excludes_ticker_object(self):
        """Test that to_dict excludes ticker_object"""
        mock_ticker = Mock(spec=yf.Ticker)
        stock = StockData(
            name="Test Stock",
            sector="Technology",
            recommendation_key="buy",
            ticker_object=mock_ticker,
        )

        data_dict = stock.to_dict()
        self.assertNotIn("ticker_object", data_dict)
        self.assertIn("name", data_dict)
        self.assertEqual(data_dict["name"], "Test Stock")

    def test_from_dict_method(self):
        """Test the from_dict class method"""
        data = {
            "name": "Test Stock",
            "sector": "Technology",
            "recommendation_key": "buy",
            "market_cap": 1000000000.0,
            "current_price": 150.0,
            "target_price": 180.0,
            "unknown_field": "This should be filtered out",
        }

        stock = StockData.from_dict(data)

        # Check that valid fields are set
        self.assertEqual(stock.name, "Test Stock")
        self.assertEqual(stock.sector, "Technology")
        self.assertEqual(stock.market_cap, 1000000000.0)
        self.assertEqual(stock.current_price, 150.0)

        # Verify that unknown fields are filtered out
        self.assertFalse(hasattr(stock, "unknown_field"))

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test empty strings
        stock = StockData(name="", sector="", recommendation_key="")
        self.assertEqual(stock.name, "")
        self.assertEqual(stock.sector, "")
        self.assertEqual(stock.recommendation_key, "")

        # Test extreme numeric values
        stock = StockData(
            name="Test Stock",
            sector="Technology",
            recommendation_key="buy",
            market_cap=float("inf"),
            beta=float("-inf"),
            current_price=0.0,
        )
        self.assertEqual(stock.market_cap, float("inf"))
        self.assertEqual(stock.beta, float("-inf"))
        self.assertEqual(stock.current_price, 0.0)

        # Test with None values for optional fields
        stock = StockData(
            name="Test Stock",
            sector="Technology",
            recommendation_key="buy",
            market_cap=None,
            current_price=None,
            target_price=None,
        )
        self.assertIsNone(stock.market_cap)
        self.assertIsNone(stock.current_price)
        self.assertIsNone(stock.target_price)


if __name__ == "__main__":
    unittest.main()
