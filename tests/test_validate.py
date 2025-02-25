#!/usr/bin/env python3
"""
Test suite for the validate module.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from pathlib import Path
import yfinance as yf
import concurrent.futures
from yahoofinance.validate import (
    is_valid_ticker,
    validate_tickers_batch,
    load_constituents,
    save_valid_tickers,
    main
)

class TestValidate(unittest.TestCase):
    """Test cases for the validate module."""

    @patch('yahoofinance.validate.yf.Ticker')
    def test_is_valid_ticker_valid(self, mock_ticker):
        """Test validation of a valid ticker."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'symbol': 'AAPL', 'name': 'Apple Inc.'}
        mock_ticker_instance.history.return_value = pd.DataFrame({'Close': [150.0, 151.0]})
        mock_ticker.return_value = mock_ticker_instance

        # Test
        result = is_valid_ticker('AAPL')

        # Verify
        self.assertTrue(result)
        mock_ticker.assert_called_once_with('AAPL')
        mock_ticker_instance.history.assert_called_once()

    @patch('yahoofinance.validate.yf.Ticker')
    def test_is_valid_ticker_empty_info(self, mock_ticker):
        """Test validation of a ticker with empty info."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {}
        mock_ticker_instance.history.return_value = pd.DataFrame({'Close': [150.0, 151.0]})
        mock_ticker.return_value = mock_ticker_instance

        # Test
        result = is_valid_ticker('INVALID')

        # Verify
        self.assertFalse(result)
        mock_ticker.assert_called_once_with('INVALID')

    @patch('yahoofinance.validate.yf.Ticker')
    def test_is_valid_ticker_no_history(self, mock_ticker):
        """Test validation of a ticker with no price history."""
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'symbol': 'TEST', 'name': 'Test Inc.'}
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_ticker_instance

        # Test
        result = is_valid_ticker('TEST')

        # Verify
        self.assertFalse(result)
        mock_ticker.assert_called_once_with('TEST')
        mock_ticker_instance.history.assert_called_once()

    @patch('yahoofinance.validate.yf.Ticker')
    def test_is_valid_ticker_exception(self, mock_ticker):
        """Test validation of a ticker that raises an exception."""
        # Setup mock
        mock_ticker.side_effect = Exception("API Error")

        # Test
        result = is_valid_ticker('ERROR')

        # Verify
        self.assertFalse(result)
        mock_ticker.assert_called_once_with('ERROR')

    @patch('yahoofinance.validate.concurrent.futures.ThreadPoolExecutor')
    @patch('yahoofinance.validate.is_valid_ticker')
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
            mock_future1, mock_future2, mock_future3
        ]
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed to return futures in order
        with patch('yahoofinance.validate.concurrent.futures.as_completed', 
                  return_value=[mock_future1, mock_future2, mock_future3]):
            
            # Test
            tickers = ['AAPL', 'INVALID', 'MSFT']
            result = validate_tickers_batch(tickers, max_workers=3, batch_size=5)
            
            # Verify
            self.assertEqual(result, ['AAPL', 'MSFT'])
            self.assertEqual(mock_executor_instance.__enter__.return_value.submit.call_count, 3)

    @patch('yahoofinance.validate.Path')
    @patch('yahoofinance.validate.pd.read_csv')
    def test_load_constituents(self, mock_read_csv, mock_path):
        """Test loading constituents from CSV."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        mock_df = pd.DataFrame({'symbol': ['AAPL', 'MSFT', 'GOOGL']})
        mock_read_csv.return_value = mock_df
        
        # Test
        result = load_constituents()
        
        # Verify
        self.assertEqual(result, ['AAPL', 'MSFT', 'GOOGL'])
        mock_read_csv.assert_called_once()

    @patch('yahoofinance.validate.Path')
    @patch('yahoofinance.validate.pd.read_csv')
    def test_load_constituents_file_not_found(self, mock_read_csv, mock_path):
        """Test loading constituents when file doesn't exist."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        # Test
        result = load_constituents()
        
        # Verify
        self.assertEqual(result, [])

    @patch('yahoofinance.validate.Path')
    @patch('yahoofinance.validate.pd.read_csv')
    def test_load_constituents_no_symbol_column(self, mock_read_csv, mock_path):
        """Test loading constituents with no symbol column."""
        # Setup mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        # DataFrame without 'symbol' column
        mock_df = pd.DataFrame({'name': ['Apple', 'Microsoft', 'Google']})
        mock_read_csv.return_value = mock_df
        
        # Test
        result = load_constituents()
        
        # Verify
        self.assertEqual(result, [])
        mock_read_csv.assert_called_once()

    @patch('pathlib.Path.mkdir')
    @patch('pandas.DataFrame.to_csv')
    def test_save_valid_tickers(self, mock_to_csv, mock_mkdir):
        """Test saving valid tickers to CSV."""
        # Test
        valid_tickers = ['AAPL', 'MSFT', 'GOOGL']
        save_valid_tickers(valid_tickers)
        
        # Verify
        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_to_csv.assert_called_once()

    @patch('yahoofinance.validate.load_constituents')
    @patch('yahoofinance.validate.validate_tickers_batch')
    @patch('yahoofinance.validate.save_valid_tickers')
    def test_main_success(self, mock_save, mock_validate, mock_load):
        """Test main function with successful flow."""
        # Setup mocks
        mock_load.return_value = ['AAPL', 'INVALID', 'MSFT']
        mock_validate.return_value = ['AAPL', 'MSFT']
        
        # Test
        main()
        
        # Verify
        mock_load.assert_called_once()
        mock_validate.assert_called_once_with(['AAPL', 'INVALID', 'MSFT'])
        mock_save.assert_called_once_with(['AAPL', 'MSFT'])

    @patch('yahoofinance.validate.load_constituents')
    @patch('yahoofinance.validate.validate_tickers_batch')
    @patch('yahoofinance.validate.save_valid_tickers')
    def test_main_no_constituents(self, mock_save, mock_validate, mock_load):
        """Test main function with no constituents."""
        # Setup mocks
        mock_load.return_value = []
        
        # Test
        main()
        
        # Verify
        mock_load.assert_called_once()
        mock_validate.assert_not_called()
        mock_save.assert_not_called()
        
    @patch('yahoofinance.validate.concurrent.futures.ThreadPoolExecutor')
    def test_validate_tickers_batch_exception(self, mock_executor):
        """Test batch validation with an exception."""
        # Setup mock
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Test error")
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed to return futures in order
        with patch('yahoofinance.validate.concurrent.futures.as_completed', 
                  return_value=[mock_future]):
            
            # Test
            result = validate_tickers_batch(['TEST'])
            
            # Verify
            self.assertEqual(result, [])
            
    @patch('pandas.read_csv')
    def test_load_constituents_exception(self, mock_read_csv):
        """Test loading constituents with an exception."""
        # Setup mock
        mock_read_csv.side_effect = Exception("File error")
        
        # Test
        result = load_constituents()
        
        # Verify
        self.assertEqual(result, [])
        
    @patch('pandas.DataFrame.to_csv')
    def test_save_valid_tickers_exception(self, mock_to_csv):
        """Test saving valid tickers with an exception."""
        # Setup mock
        mock_to_csv.side_effect = Exception("Save error")
        
        # Test
        # Should not raise an exception
        save_valid_tickers(['AAPL', 'MSFT'])


if __name__ == '__main__':
    unittest.main()