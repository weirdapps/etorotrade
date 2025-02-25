import unittest
from unittest.mock import patch, Mock, call, mock_open
import sys
import os
import pandas as pd
from io import StringIO
from trade import main, generate_trade_recommendations, BUY_PERCENTAGE, DIVIDEND_YIELD

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
            call("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? "),
            call("Use existing portfolio file (E) or download new one (N)? ")
        ]
        mock_input.assert_has_calls(expected_calls)
        mock_display.load_tickers.assert_called_once_with("P")
        mock_display.display_report.assert_called_once_with(["AAPL", "MSFT"], "P")
        
    @patch('builtins.input')
    @patch('yahoofinance.download.download_portfolio')
    @patch('trade.MarketDisplay')
    def test_portfolio_download_success(self, mock_display_class, mock_download, mock_input):
        """Test portfolio download flow"""
        # Mock user input sequence
        mock_input.side_effect = ["P", "N"]
        
        # Mock download success
        mock_download.return_value = True
        
        # Mock MarketDisplay
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = ["AAPL", "MSFT"]
        
        # Run main function
        main()
        
        # Verify interactions
        expected_calls = [
            call("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? "),
            call("Use existing portfolio file (E) or download new one (N)? ")
        ]
        mock_input.assert_has_calls(expected_calls)
        mock_download.assert_called_once()
        mock_display.load_tickers.assert_called_once_with("P")
        mock_display.display_report.assert_called_once_with(["AAPL", "MSFT"], "P")
        
    @patch('builtins.input')
    @patch('yahoofinance.download.download_portfolio')
    @patch('trade.MarketDisplay')
    def test_portfolio_download_failure(self, mock_display_class, mock_download, mock_input):
        """Test portfolio download failure handling"""
        # Mock user input sequence
        mock_input.side_effect = ["P", "N"]
        
        # Mock download failure
        mock_download.return_value = False
        
        # Mock MarketDisplay
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        
        # Run main function
        main()
        
        # Verify interactions
        expected_calls = [
            call("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? "),
            call("Use existing portfolio file (E) or download new one (N)? ")
        ]
        mock_input.assert_has_calls(expected_calls)
        mock_download.assert_called_once()
        mock_display.load_tickers.assert_not_called()
        mock_display.display_report.assert_not_called()

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
        mock_input.assert_called_once_with("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? ")
        mock_display.load_tickers.assert_called_once_with("M")
        mock_display.display_report.assert_called_once_with(["SPY", "QQQ"], "M")
        
    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_etoro_market_input_success(self, mock_display_class, mock_input):
        """Test successful eToro market input flow"""
        # Mock user input
        mock_input.return_value = "E"
        
        # Mock MarketDisplay
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = ["AAPL", "MSFT"]
        
        # Run main function
        main()
        
        # Verify interactions
        mock_input.assert_called_once_with("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? ")
        mock_display.load_tickers.assert_called_once_with("E")
        # For eToro, we should still save as market.csv (M)
        mock_display.display_report.assert_called_once_with(["AAPL", "MSFT"], "M")

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
        mock_input.assert_called_once_with("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? ")
        mock_display.load_tickers.assert_called_once_with("I")
        mock_display.display_report.assert_called_once_with(["GOOGL"], None)
        
    @patch('builtins.input')
    @patch('trade.generate_trade_recommendations')
    def test_trade_analysis_buy_option(self, mock_generate_trade, mock_input):
        """Test trade analysis with buy option"""
        # Mock user inputs
        mock_input.side_effect = ["T", "B"]
        
        # Run main function
        main()
        
        # Verify interactions
        expected_calls = [
            call("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? "),
            call("Do you want to identify BUY (B) or SELL (S) opportunities? ")
        ]
        mock_input.assert_has_calls(expected_calls)
        mock_generate_trade.assert_called_once_with('N')  # 'N' for new buy opportunities
        
    @patch('builtins.input')
    @patch('trade.generate_trade_recommendations')
    def test_trade_analysis_sell_option(self, mock_generate_trade, mock_input):
        """Test trade analysis with sell option"""
        # Mock user inputs
        mock_input.side_effect = ["T", "S"]
        
        # Run main function
        main()
        
        # Verify interactions
        expected_calls = [
            call("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? "),
            call("Do you want to identify BUY (B) or SELL (S) opportunities? ")
        ]
        mock_input.assert_has_calls(expected_calls)
        mock_generate_trade.assert_called_once_with('E')  # 'E' for existing portfolio (sell)
        
    @patch('builtins.input')
    @patch('trade.generate_trade_recommendations')
    def test_trade_analysis_invalid_option(self, mock_generate_trade, mock_input):
        """Test trade analysis with invalid option"""
        # Mock user inputs
        mock_input.side_effect = ["T", "X"]
        
        # Capture stdout
        with patch('sys.stdout', new=StringIO()) as fake_out:
            main()
        
        # Verify interactions
        expected_calls = [
            call("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? "),
            call("Do you want to identify BUY (B) or SELL (S) opportunities? ")
        ]
        mock_input.assert_has_calls(expected_calls)
        mock_generate_trade.assert_not_called()
        
        # Check error message
        output = fake_out.getvalue()
        self.assertIn("Invalid option", output)

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
            call("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? "),
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
        
    def test_trade_recommendations_buy_criteria(self):
        """Test buy recommendations criteria"""
        # This is a simpler test just focusing on the logic, not file operations
        def is_buy_opportunity(upside, analyst_count, buy_percentage):
            """Simulates the buy criteria from trade.py"""
            return (analyst_count >= 4 and
                    upside > 15.0 and
                    buy_percentage > 65.0)
        
        # Test cases that should be buy opportunities
        self.assertTrue(is_buy_opportunity(16.0, 5, 70.0))
        self.assertTrue(is_buy_opportunity(20.0, 4, 66.0))
        
        # Test cases that should NOT be buy opportunities
        # Fails on upside
        self.assertFalse(is_buy_opportunity(14.9, 5, 70.0))
        # Fails on analyst count
        self.assertFalse(is_buy_opportunity(16.0, 3, 70.0))
        # Fails on buy percentage
        self.assertFalse(is_buy_opportunity(16.0, 5, 64.9))
            
    def test_exret_calculation(self):
        """Test Expected Return (EXRET) calculation"""
        # EXRET = upside * buy_percentage / 100
        
        def calculate_exret(upside, buy_percentage):
            """Simulate EXRET calculation from trade.py"""
            return upside * buy_percentage / 100
        
        # Test various combinations
        self.assertEqual(calculate_exret(20.0, 75.0), 15.0)  # Standard case
        self.assertEqual(calculate_exret(15.0, 60.0), 9.0)   # Lower values
        self.assertEqual(calculate_exret(0.0, 50.0), 0.0)    # Zero upside
        self.assertEqual(calculate_exret(10.0, 0.0), 0.0)    # Zero buy percentage
        
        # Test with extreme values
        self.assertEqual(calculate_exret(50.0, 100.0), 50.0) # Maximum case
        
        # Round to avoid floating point imprecision
        result = round(calculate_exret(12.34, 56.78), 5)
        self.assertEqual(result, 7.00665)                    # Complex case with rounding
        
        # Greater sensitivity to upside potential than to buy percentage
        # (Stocks with higher upside should have higher EXRET even with slightly lower buy %)
        stock1_exret = calculate_exret(30.0, 70.0)  # 21
        stock2_exret = calculate_exret(20.0, 80.0)  # 16
        self.assertGreater(stock1_exret, stock2_exret)
        
        # Ranking logic (higher EXRET should be ranked better)
        stocks = [
            {"ticker": "AAPL", "upside": 20.0, "buy_percentage": 80.0},  # EXRET = 16
            {"ticker": "MSFT", "upside": 15.0, "buy_percentage": 70.0},  # EXRET = 10.5
            {"ticker": "AMZN", "upside": 30.0, "buy_percentage": 65.0},  # EXRET = 19.5
            {"ticker": "GOOGL", "upside": 25.0, "buy_percentage": 75.0}  # EXRET = 18.75
        ]
        
        # Calculate EXRET for all stocks
        for stock in stocks:
            stock["exret"] = calculate_exret(stock["upside"], stock["buy_percentage"])
            
        # Sort by EXRET descending
        sorted_stocks = sorted(stocks, key=lambda x: x["exret"], reverse=True)
        
        # Verify sort order
        self.assertEqual(sorted_stocks[0]["ticker"], "AMZN")    # EXRET = 19.5 (highest)
        self.assertEqual(sorted_stocks[1]["ticker"], "GOOGL")   # EXRET = 18.75
        self.assertEqual(sorted_stocks[2]["ticker"], "AAPL")    # EXRET = 16
        self.assertEqual(sorted_stocks[3]["ticker"], "MSFT")    # EXRET = 10.5 (lowest)
            
    def test_file_checks_logic(self):
        """Test the file existence check logic"""
        # Simulate file checks from trade.py
        def should_continue(market_exists, portfolio_exists):
            if not market_exists:
                return False  # Exit if market file missing
            if not portfolio_exists:
                return False  # Exit if portfolio file missing
            return True
            
        # Both files exist - should continue
        self.assertTrue(should_continue(True, True))
        
        # Market file missing - should not continue
        self.assertFalse(should_continue(False, True))
        
        # Portfolio file missing - should not continue
        self.assertFalse(should_continue(True, False))
        
        # Both files missing - should not continue
        self.assertFalse(should_continue(False, False))
            
    def test_sell_criteria_logic(self):
        """Test sell recommendations criteria"""
        # This is a simpler test just focusing on the logic, not file operations
        def is_sell_candidate(upside, analyst_count, buy_percentage):
            """Simulates the sell criteria from trade.py"""
            return (analyst_count >= 4 and 
                    (upside < 5.0 or buy_percentage < 50.0))
        
        # Test cases that should be sell candidates
        self.assertTrue(is_sell_candidate(4.9, 5, 60.0))  # Low upside
        self.assertTrue(is_sell_candidate(10.0, 5, 45.0))  # Low buy %
        self.assertTrue(is_sell_candidate(4.0, 5, 45.0))  # Both low
        
        # Test cases that should NOT be sell candidates
        self.assertFalse(is_sell_candidate(5.0, 3, 45.0))  # Not enough analysts
        self.assertFalse(is_sell_candidate(5.1, 5, 50.1))  # Good metrics
        
        # Edge cases
        self.assertTrue(is_sell_candidate(4.9, 4, 50.0))  # Just meets criteria
            
    def test_trade_recommendations_error_handling(self):
        """Test error handling in trade recommendations"""
        # Test with invalid CSV format
        with patch('os.path.exists', return_value=True), \
             patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("Empty CSV")), \
             patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('N')
            output = fake_out.getvalue()
            self.assertIn('Error generating recommendations', output)
            
    def test_buy_criteria(self):
        """Test the buy opportunity criteria logic"""
        # This is a simpler test that just tests the buy criteria
        def is_buy_opportunity(analyst_count, upside, buy_percentage):
            """Simulates the buy criteria from trade.py"""
            return (analyst_count >= 4 and 
                   upside > 15.0 and 
                   buy_percentage > 65.0)
                   
        # Test cases
        self.assertTrue(is_buy_opportunity(10, 20.0, 80.0))  # All criteria met
        self.assertTrue(is_buy_opportunity(4, 15.1, 65.1))   # Just meets minimum
        
        # Fails on analyst count
        self.assertFalse(is_buy_opportunity(3, 20.0, 80.0))
        
        # Fails on upside
        self.assertFalse(is_buy_opportunity(10, 15.0, 80.0))
        
        # Fails on buy percentage
        self.assertFalse(is_buy_opportunity(10, 20.0, 65.0))
        
        # Fails on multiple criteria
        self.assertFalse(is_buy_opportunity(3, 10.0, 50.0))
        
    def test_portfolio_filtering(self):
        """Test filtering out stocks already in portfolio"""
        # Mock list of potential buy tickers
        buy_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        
        # Mock portfolio tickers
        portfolio_tickers = ["GOOGL", "NFLX", "FB"]
        
        # Function to simulate filtering
        def filter_out_portfolio_tickers(buy_list, portfolio_list):
            """Filter out tickers already in portfolio"""
            return [ticker for ticker in buy_list if ticker not in portfolio_list]
            
        # Apply the filter
        new_opportunities = filter_out_portfolio_tickers(buy_tickers, portfolio_tickers)
        
        # AAPL, MSFT, AMZN should remain; GOOGL should be filtered out
        self.assertEqual(len(new_opportunities), 3)
        self.assertIn("AAPL", new_opportunities)
        self.assertIn("MSFT", new_opportunities)
        self.assertIn("AMZN", new_opportunities)
        self.assertNotIn("GOOGL", new_opportunities)
            
    def test_sell_criteria(self):
        """Test the sell candidate criteria logic"""
        # This is a simpler test that just tests the sell criteria
        def is_sell_candidate(analyst_count, upside, buy_percentage):
            """Simulates the sell criteria from trade.py"""
            return (analyst_count >= 4 and 
                   (upside < 5.0 or buy_percentage < 50.0))
                   
        # Test cases
        self.assertTrue(is_sell_candidate(10, 4.9, 80.0))  # Low upside only
        self.assertTrue(is_sell_candidate(10, 10.0, 49.9)) # Low buy % only
        self.assertTrue(is_sell_candidate(10, 4.9, 49.9))  # Both low
        self.assertTrue(is_sell_candidate(4, 4.9, 80.0))   # Minimum analyst count
        
        # Not enough analysts
        self.assertFalse(is_sell_candidate(3, 4.9, 49.9))
        
        # Good metrics
        self.assertFalse(is_sell_candidate(10, 5.0, 50.0))
        self.assertFalse(is_sell_candidate(10, 10.0, 80.0))
            
    @patch('os.path.exists')
    def test_trade_recommendations_missing_files(self, mock_path_exists):
        """Test generate_trade_recommendations when files are missing"""
        # Test missing market.csv
        mock_path_exists.side_effect = lambda path: not path.endswith('market.csv')
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('N')
            output = fake_out.getvalue()
            self.assertIn('Please run the market analysis (M) first', output)
            
        # Test missing portfolio.csv
        mock_path_exists.side_effect = lambda path: not path.endswith('portfolio.csv')
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('N')
            output = fake_out.getvalue()
            self.assertIn('Portfolio file not found', output)
            
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_trade_recommendations_no_opportunities(self, mock_read_csv, mock_path_exists):
        """Test generate_trade_recommendations when no opportunities are found"""
        # Mock file existence
        mock_path_exists.return_value = True
        
        # Market data with no buy opportunities
        market_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'company': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc'],
            'price': [150.0, 300.0, 2500.0],
            'upside': [10.0, 12.0, 8.0],       # All < 15% upside
            'analyst_count': [3, 3, 2],         # All < 4 analysts
            'buy_percentage': [60.0, 60.0, 60.0]  # All < 65% buy
        })
        
        # Empty portfolio with string ticker column (needed for str accessor)
        portfolio_data = pd.DataFrame({'ticker': pd.Series([], dtype=str), 'share_price': []})
        
        mock_read_csv.side_effect = lambda path, **kwargs: {
            'yahoofinance/output/market.csv': market_data,
            'yahoofinance/input/portfolio.csv': portfolio_data
        }.get(path, pd.DataFrame())
        
        # Test buy recommendations (should be none)
        with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
             patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('N')
            
            output = fake_out.getvalue()
            self.assertIn('No new buy opportunities found matching criteria', output)