import unittest
from unittest.mock import patch, Mock, call
import sys
from io import StringIO
from trade import main

class TestTrade(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.held_output = StringIO()
        sys.stdout = self.held_output

    def tearDown(self):
        """Clean up after each test method."""
        sys.stdout = sys.__stdout__

    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_portfolio_input_success(self, mock_display_class, mock_input):
        """Test successful portfolio input flow"""
        # Mock user input sequence
        mock_input.side_effect = ["P", "E"]
        
        # Mock MarketDisplay
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = ["AAPL", "MSFT"]
        
        # Run main function
        main()
        
        # Verify interactions
        expected_calls = [
            call("Load tickers for Portfolio (P), Market (M), Trade Analysis (T) or Manual Input (I)? "),
            call("Use existing portfolio file (E) or download new one (N)? ")
        ]
        mock_input.assert_has_calls(expected_calls)
        mock_display.load_tickers.assert_called_once_with("P")
        mock_display.display_report.assert_called_once_with(["AAPL", "MSFT"], "P")

    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_market_input_success(self, mock_display_class, mock_input):
        """Test successful market input flow"""
        # Mock user input
        mock_input.return_value = "M"
        
        # Mock MarketDisplay
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = ["SPY", "QQQ"]
        
        # Run main function
        main()
        
        # Verify interactions
        mock_input.assert_called_once_with("Load tickers for Portfolio (P), Market (M), Trade Analysis (T) or Manual Input (I)? ")
        mock_display.load_tickers.assert_called_once_with("M")
        mock_display.display_report.assert_called_once_with(["SPY", "QQQ"], "M")

    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_manual_input_success(self, mock_display_class, mock_input):
        """Test successful manual input flow"""
        # Mock user input
        mock_input.return_value = "I"
        
        # Mock MarketDisplay
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = ["GOOGL"]
        
        # Run main function
        main()
        
        # Verify interactions
        mock_input.assert_called_once_with("Load tickers for Portfolio (P), Market (M), Trade Analysis (T) or Manual Input (I)? ")
        mock_display.load_tickers.assert_called_once_with("I")
        mock_display.display_report.assert_called_once_with(["GOOGL"], None)

    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_no_tickers_found(self, mock_display_class, mock_input):
        """Test handling when no tickers are found"""
        # Mock user input sequence
        mock_input.side_effect = ["P", "E"]
        
        # Mock MarketDisplay
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = []
        
        # Run main function
        main()
        
        # Verify interactions
        expected_calls = [
            call("Load tickers for Portfolio (P), Market (M), Trade Analysis (T) or Manual Input (I)? "),
            call("Use existing portfolio file (E) or download new one (N)? ")
        ]
        mock_input.assert_has_calls(expected_calls)
        mock_display.load_tickers.assert_called_once()
        mock_display.display_report.assert_not_called()

    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_display_report_value_error(self, mock_display_class, mock_input):
        """Test handling of ValueError in display_report"""
        # Mock user input
        mock_input.return_value = "P"
        
        # Mock MarketDisplay
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = ["AAPL"]
        mock_display.display_report.side_effect = ValueError("Invalid numeric value")
        
        # Run main function
        main()
        
        # Verify error was handled
        mock_display.load_tickers.assert_called_once()
        mock_display.display_report.assert_called_once()

    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_keyboard_interrupt_handling(self, mock_display_class, mock_input):
        """Test handling of KeyboardInterrupt"""
        # Mock KeyboardInterrupt during input
        mock_input.side_effect = KeyboardInterrupt()
        
        # Run main function with exit code assertion
        with self.assertRaises(SystemExit) as cm:
            main()
        
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Operation cancelled by user", self.held_output.getvalue())

    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_unexpected_error_handling(self, mock_display_class, mock_input):
        """Test handling of unexpected errors"""
        # Mock unexpected error
        mock_input.side_effect = Exception("Unexpected error")
        
        # Run main function with exit code assertion
        with self.assertRaises(SystemExit) as cm:
            main()
        
        self.assertEqual(cm.exception.code, 1)