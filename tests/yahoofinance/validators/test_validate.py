#!/usr/bin/env python3
"""
Test suite for the validate module.
"""

import concurrent.futures
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import yfinance as yf

from yahoofinance.validators.validate import (
    is_valid_ticker,
    main,
    save_valid_tickers,
    validate_tickers_batch,
)


class TestValidate(unittest.TestCase):
    """Test cases for the validate module."""

    @patch("yahoofinance.validators.validate.yf.Ticker")
    def test_is_valid_ticker_valid(self, mock_ticker):
        """Test validation of a valid ticker."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {"symbol": "AAPL", "name": "Apple Inc."}
        mock_ticker_instance.history.return_value = pd.DataFrame({"Close": [150.0, 151.0]})
        mock_ticker.return_value = mock_ticker_instance

        # Test
        result = is_valid_ticker("AAPL")

        # Verify
        self.assertTrue(result)
        mock_ticker.assert_called_once_with("AAPL")
        mock_ticker_instance.history.assert_called_once()

    @patch("yahoofinance.validators.validate.yf.Ticker")
    def test_is_valid_ticker_empty_info(self, mock_ticker):
        """Test validation of a ticker with empty info."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {}
        mock_ticker_instance.history.return_value = pd.DataFrame({"Close": [150.0, 151.0]})
        mock_ticker.return_value = mock_ticker_instance

        # Test
        result = is_valid_ticker("INVALID")

        # Verify
        self.assertFalse(result)
        mock_ticker.assert_called_once_with("INVALID")

    @patch("yahoofinance.validators.validate.yf.Ticker")
    def test_is_valid_ticker_no_history(self, mock_ticker):
        """Test validation of a ticker with no price history."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {"symbol": "TEST", "name": "Test Inc."}
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_ticker_instance

        # Test
        result = is_valid_ticker("TEST")

        # Verify
        self.assertFalse(result)
        mock_ticker.assert_called_once_with("TEST")
        mock_ticker_instance.history.assert_called_once()

    @patch("yahoofinance.validators.validate.yf.Ticker")
    def test_is_valid_ticker_exception(self, mock_ticker):
        """Test validation of a ticker that raises an exception."""
        # Setup mock
        mock_ticker.side_effect = Exception("API Error")

        # Test
        result = is_valid_ticker("ERROR")

        # Verify
        self.assertFalse(result)
        mock_ticker.assert_called_once_with("ERROR")

    @patch("yahoofinance.validators.validate.concurrent.futures.ThreadPoolExecutor")
    @patch("yahoofinance.validators.validate.is_valid_ticker")
    def test_validate_tickers_batch(self, mock_is_valid, mock_executor):
        """Test batch validation of tickers."""
        # Setup mocks
        mock_future1 = MagicMock()
        mock_future1.result.return_value = True

        mock_future2 = MagicMock()
        mock_future2.result.return_value = False

        mock_future3 = MagicMock()
        mock_future3.result.return_value = True

        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value.submit.side_effect = [
            mock_future1,
            mock_future2,
            mock_future3,
        ]
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed to return futures in order
        with patch(
            "yahoofinance.validators.validate.concurrent.futures.as_completed",
            return_value=[mock_future1, mock_future2, mock_future3],
        ):

            # Test
            tickers = ["AAPL", "INVALID", "MSFT"]
            result = validate_tickers_batch(tickers, max_workers=3)

            # Verify
            self.assertEqual(result, ["AAPL", "MSFT"])
            self.assertEqual(mock_executor_instance.__enter__.return_value.submit.call_count, 3)

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    def test_save_valid_tickers(self, mock_to_csv, mock_mkdir):
        """Test saving valid tickers to CSV."""
        # Test
        valid_tickers = ["AAPL", "MSFT", "GOOGL"]
        save_valid_tickers(valid_tickers)

        # Verify
        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_to_csv.assert_called_once()

    @patch("yahoofinance.validators.validate.input", return_value="AAPL,MSFT,GOOGL")
    @patch("yahoofinance.validators.validate.validate_tickers_batch")
    @patch("yahoofinance.validators.validate.save_valid_tickers")
    def test_main_success(self, mock_save, mock_validate, mock_input):
        """Test main function with successful flow."""
        # Setup mocks
        mock_validate.return_value = ["AAPL", "MSFT"]

        # Test
        with patch("builtins.print"):  # Suppress print statements
            main()

        # Verify
        mock_input.assert_called_once()
        mock_validate.assert_called_once_with(["AAPL", "MSFT", "GOOGL"])
        mock_save.assert_called_once_with(["AAPL", "MSFT"])

    @patch("yahoofinance.validators.validate.input", return_value="")
    @patch("yahoofinance.validators.validate.validate_tickers_batch")
    @patch("yahoofinance.validators.validate.save_valid_tickers")
    def test_main_no_tickers(self, mock_save, mock_validate, mock_input):
        """Test main function with no tickers."""
        # Test
        with patch("builtins.print"):  # Suppress print statements
            main()

        # Verify
        mock_input.assert_called_once()
        mock_validate.assert_not_called()
        mock_save.assert_not_called()

    @patch("yahoofinance.validators.validate.concurrent.futures.ThreadPoolExecutor")
    def test_validate_tickers_batch_exception(self, mock_executor):
        """Test batch validation with an exception."""
        # Setup mock
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Test error")

        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed to return futures in order
        with patch(
            "yahoofinance.validators.validate.concurrent.futures.as_completed",
            return_value=[mock_future],
        ):

            # Test
            result = validate_tickers_batch(["TEST"])

            # Verify
            self.assertEqual(result, [])

    @patch("pandas.DataFrame.to_csv")
    def test_save_valid_tickers_exception(self, mock_to_csv):
        """Test saving valid tickers with an exception."""
        # Setup mock
        mock_to_csv.side_effect = Exception("Save error")

        # Test
        # Should not raise an exception
        save_valid_tickers(["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
