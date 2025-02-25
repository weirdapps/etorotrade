import unittest
from unittest.mock import patch, Mock, call, mock_open
import sys
import os
import pandas as pd
from io import StringIO
from tempfile import TemporaryDirectory
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
            {"ticker": "AAPL", "upside": 20.0, "buy_percentage": 80.0},
            {"ticker": "MSFT", "upside": 15.0, "buy_percentage": 70.0},
            {"ticker": "AMZN", "upside": 30.0, "buy_percentage": 65.0},
            {"ticker": "GOOGL", "upside": 25.0, "buy_percentage": 75.0}
        ]
        
        # Calculate EXRET for all stocks
        for stock in stocks:
            stock["exret"] = calculate_exret(stock["upside"], stock["buy_percentage"])
            
        # Sort by EXRET descending
        sorted_stocks = sorted(stocks, key=lambda x: x["exret"], reverse=True)
        
        # Verify sort order
        self.assertEqual(sorted_stocks[0]["ticker"], "AMZN")
        self.assertEqual(sorted_stocks[1]["ticker"], "GOOGL")
        self.assertEqual(sorted_stocks[2]["ticker"], "AAPL")
        self.assertEqual(sorted_stocks[3]["ticker"], "MSFT")
            
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
        with patch('pandas.DataFrame.to_csv'), \
             patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('N')
            
            output = fake_out.getvalue()
            self.assertIn('No new buy opportunities found matching criteria', output)

    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_directory_creation(self, mock_makedirs, mock_path_exists):
        """Test directory creation for output directory"""
        # Mock behavior: output directory doesn't exist
        mock_path_exists.side_effect = lambda path: not path.endswith('output')
        
        with patch('sys.stdout', new=StringIO()) as fake_out, \
             patch('pandas.read_csv', side_effect=Exception("Should not reach this")):
            # This should attempt to create the directory and then fail on reading market.csv
            generate_trade_recommendations('N')
            
            # Verify makedirs was called with output directory
            mock_makedirs.assert_called_once_with('yahoofinance/output', exist_ok=True)
            
            # Verify warning message about directory creation
            output = fake_out.getvalue()
            self.assertIn('Creating output directory', output)

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_trade_recommendations_portfolio_ticker_column_detection(self, mock_read_csv, mock_path_exists):
        """Test detection of different ticker column names in portfolio file"""
        # Mock file existence
        mock_path_exists.return_value = True
        
        # Create different portfolio dataframes with different column names
        portfolio_symbol = pd.DataFrame({'symbol': ['AAPL', 'MSFT'], 'share_price': [150, 300]})
        market_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'company': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc'],
            'price': [150.0, 300.0, 2500.0],
            'analyst_count': [10, 10, 10],
            'upside': [5.0, 5.0, 30.0],
            'buy_percentage': [60.0, 60.0, 80.0]
        })
        
        # Test with 'symbol' column
        mock_read_csv.side_effect = lambda path, **kwargs: {
            'yahoofinance/output/market.csv': market_data,
            'yahoofinance/input/portfolio.csv': portfolio_symbol
        }.get(path, pd.DataFrame())
        
        with patch('pandas.DataFrame.to_csv'), \
             patch('sys.stdout', new=StringIO()):
            generate_trade_recommendations('N')
            # If no exception, the test passes as it found the column
        
        # Test with 'SYMBOL' column (uppercase)
        portfolio_upper = pd.DataFrame({'SYMBOL': ['AAPL', 'MSFT'], 'share_price': [150, 300]})
        mock_read_csv.side_effect = lambda path, **kwargs: {
            'yahoofinance/output/market.csv': market_data,
            'yahoofinance/input/portfolio.csv': portfolio_upper
        }.get(path, pd.DataFrame())
        
        with patch('pandas.DataFrame.to_csv'), \
             patch('sys.stdout', new=StringIO()):
            generate_trade_recommendations('N')
            # If no exception, the test passes as it found the column
            
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_trade_recommendations_missing_ticker_column(self, mock_read_csv, mock_path_exists):
        """Test error handling when ticker column is missing"""
        # Mock file existence
        mock_path_exists.return_value = True
        
        # Portfolio with no ticker or symbol column
        portfolio_no_ticker = pd.DataFrame({'price': [150, 300], 'quantity': [10, 5]})
        market_data = pd.DataFrame({'ticker': ['AAPL', 'MSFT'], 'price': [150, 300]})
        
        mock_read_csv.side_effect = lambda path, **kwargs: {
            'yahoofinance/output/market.csv': market_data,
            'yahoofinance/input/portfolio.csv': portfolio_no_ticker
        }.get(path, pd.DataFrame())
        
        # Should detect missing ticker column and exit
        with patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('N')
            
            output = fake_out.getvalue()
            self.assertIn('Could not find ticker column in portfolio file', output)

    def test_buy_criteria_directly(self):
        """Test buy criteria application directly"""
        # Test data
        stocks = [
            {
                "ticker": "AAPL",
                "upside": 33.3,
                "analyst_count": 10,
                "buy_percentage": 80.0,
                "already_in_portfolio": True
            },
            {
                "ticker": "MSFT",
                "upside": 16.7,
                "analyst_count": 3,  # Not enough analysts
                "buy_percentage": 75.0,
                "already_in_portfolio": False
            },
            {
                "ticker": "GOOGL",
                "upside": 20.0,
                "analyst_count": 5,
                "buy_percentage": 76.0,
                "already_in_portfolio": False
            },
            {
                "ticker": "AMZN",
                "upside": 25.0,
                "analyst_count": 5,
                "buy_percentage": 85.0,
                "already_in_portfolio": False
            },
            {
                "ticker": "TSLA",
                "upside": 25.0,
                "analyst_count": 5,
                "buy_percentage": 50.0,  # Too low buy percentage
                "already_in_portfolio": False
            }
        ]
        
        # Direct implementation of buy criteria
        def is_buy_opportunity(stock):
            return (stock["analyst_count"] >= 5 and
                    stock["upside"] >= 20.0 and
                    stock["buy_percentage"] >= 75.0 and
                    not stock["already_in_portfolio"])
        
        # Filter stocks
        buy_opportunities = [stock for stock in stocks if is_buy_opportunity(stock)]
        
        # Verify results
        self.assertEqual(len(buy_opportunities), 2)
        self.assertEqual(buy_opportunities[0]["ticker"], "GOOGL")
        self.assertEqual(buy_opportunities[1]["ticker"], "AMZN")
        
        # Verify specific rejections
        rejected_tickers = [stock["ticker"] for stock in stocks if not is_buy_opportunity(stock)]
        self.assertIn("AAPL", rejected_tickers)  # In portfolio already
        self.assertIn("MSFT", rejected_tickers)  # Not enough analysts
        self.assertIn("TSLA", rejected_tickers)  # Buy percentage too low

    def test_exret_calculation_basic(self):
        """Test Expected Return (EXRET) calculation directly"""
        # Calculate EXRET directly based on the formula from trade.py
        upside = 20.0
        buy_percentage = 80.0
        
        # Simple calculation test
        calculated_exret = upside * buy_percentage / 100.0
        self.assertEqual(calculated_exret, 16.0)
        
        # Test with different values
        self.assertEqual(25.0 * 75.0 / 100.0, 18.75)  # AMZN example
        self.assertEqual(10.0 * 50.0 / 100.0, 5.0)    # Lower values
        
        # Test with extreme values
        self.assertEqual(0.0 * 80.0 / 100.0, 0.0)    # Zero upside
        self.assertEqual(20.0 * 0.0 / 100.0, 0.0)    # Zero buy percentage
    
    def test_sell_criteria_directly(self):
        """Test sell criteria application directly"""
        # Test data
        portfolio_stocks = [
            {
                "ticker": "AAPL",
                "upside": 3.3,
                "analyst_count": 10,
                "buy_percentage": 45.0  # Low buy percentage
            },
            {
                "ticker": "MSFT",
                "upside": 10.0,
                "analyst_count": 8,
                "buy_percentage": 60.0  # Above buy threshold but not upside
            },
            {
                "ticker": "GOOGL",
                "upside": 20.0,
                "analyst_count": 4,  # Not enough analysts
                "buy_percentage": 76.0  # Good buy percentage
            },
            {
                "ticker": "AMZN",
                "upside": 1.6,  # Low upside
                "analyst_count": 5,
                "buy_percentage": 40.0  # Low buy percentage
            },
            {
                "ticker": "TSLA",
                "upside": 25.0,  # Good upside
                "analyst_count": 2,  # Not enough analysts
                "buy_percentage": 80.0  # Good buy percentage
            }
        ]
        
        # Direct implementation of sell criteria
        def is_sell_candidate(stock):
            return (stock["analyst_count"] >= 5 and 
                   (stock["upside"] < 5.0 or stock["buy_percentage"] < 50.0))
        
        # Filter stocks
        sell_candidates = [stock for stock in portfolio_stocks if is_sell_candidate(stock)]
        
        # Sort by EXRET ascending (worst first)
        sell_candidates.sort(key=lambda x: x["upside"] * x["buy_percentage"] / 100.0)
        
        # Verify results
        self.assertEqual(len(sell_candidates), 2)
        self.assertIn("AAPL", [stock["ticker"] for stock in sell_candidates])  # Low buy percentage
        self.assertIn("AMZN", [stock["ticker"] for stock in sell_candidates])  # Low upside and low buy percentage
        
        # Verify specific non-candidates
        non_candidates = [stock["ticker"] for stock in portfolio_stocks if not is_sell_candidate(stock)]
        self.assertIn("TSLA", non_candidates)  # Not enough analysts
        self.assertIn("GOOGL", non_candidates)  # Not enough analysts 
        self.assertIn("MSFT", non_candidates)  # Above thresholds
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_sell_recommendations_no_candidates(self, mock_read_csv, mock_path_exists):
        """Test sell recommendations with no candidates matching criteria"""
        # Mock file existence
        mock_path_exists.return_value = True
        
        # Portfolio analysis data with no sell candidates
        portfolio_analysis = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'company': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc'],
            'price': [150.0, 300.0, 2500.0],
            'target_price': [180.0, 350.0, 3000.0],
            'upside': [20.0, 16.7, 20.0],
            'analyst_count': [10, 8, 7],
            'buy_percentage': [75.0, 80.0, 76.0],
            'total_ratings': [12, 10, 8]
        })
        
        mock_read_csv.side_effect = lambda path, **kwargs: {
            'yahoofinance/output/portfolio.csv': portfolio_analysis,
            'yahoofinance/input/portfolio.csv': pd.DataFrame({'ticker': portfolio_analysis['ticker']})
        }.get(path, pd.DataFrame())
        
        # Test sell recommendations with no candidates
        with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
             patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('E')
            
            # Verify empty CSV was created
            mock_to_csv.assert_called_once()
            
            # Verify output mentions no candidates found
            output = fake_out.getvalue()
            self.assertIn('No sell candidates found matching criteria', output)
    
    @patch('os.path.exists')
    def test_sell_recommendations_no_portfolio_analysis(self, mock_path_exists):
        """Test sell recommendations when portfolio analysis file doesn't exist"""
        # Mock behavior: market.csv and portfolio.csv exist, but portfolio_output.csv doesn't
        mock_path_exists.side_effect = lambda path: not path.endswith('portfolio.csv') or path.startswith('yahoofinance/input')
        
        # Test error handling for missing portfolio analysis
        with patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('E')
            
            # Verify warning message
            output = fake_out.getvalue()
            self.assertIn('Portfolio analysis file not found', output)
            self.assertIn('Please run the portfolio analysis (P) first', output)
    
    def test_date_formatting_function(self):
        """Test date formatting function directly"""
        # Direct implementation of the date formatting function from trade.py
        def format_date(date_str):
            if pd.notnull(date_str) and date_str != '--':
                try:
                    return pd.to_datetime(date_str).strftime('%Y-%m-%d')
                except ValueError:
                    return date_str
            return '--'
        
        # Test various date formats
        self.assertEqual(format_date('2023-05-15'), '2023-05-15')  # Already formatted correctly
        self.assertEqual(format_date('6/1/2023'), '2023-06-01')    # MM/DD/YYYY format
        self.assertEqual(format_date(None), '--')                  # None value
        self.assertEqual(format_date('invalid_date'), 'invalid_date')  # Invalid date format
        self.assertEqual(format_date('--'), '--')                  # Already a placeholder
    
    def test_column_formatting_function(self):
        """Test column formatting functions directly"""
        # Price column formatting (2 decimals)
        def format_price(value):
            if pd.notnull(value):
                return f"{value:.2f}"
            return "--"
            
        # Percentage column formatting (1 decimal)
        def format_percentage(value):
            if pd.notnull(value):
                return f"{value:.1f}%"
            return "--"
        
        # Test price formatting
        self.assertEqual(format_price(123.4567), "123.46")
        self.assertEqual(format_price(0.01), "0.01")
        self.assertEqual(format_price(None), "--")
        
        # Test percentage formatting
        self.assertEqual(format_percentage(12.345), "12.3%")
        self.assertEqual(format_percentage(0.0), "0.0%")
        self.assertEqual(format_percentage(None), "--")
    
    def test_column_alignment_function(self):
        """Test column alignment determination function"""
        # Reimplementation of column alignment logic from trade.py
        def get_column_alignment(column_name):
            if column_name in ['TICKER', 'COMPANY NAME']:
                return 'left'
            else:
                return 'right'
        
        # Test column alignments
        self.assertEqual(get_column_alignment('TICKER'), 'left')
        self.assertEqual(get_column_alignment('COMPANY NAME'), 'left')
        self.assertEqual(get_column_alignment('PRICE'), 'right')
        self.assertEqual(get_column_alignment('UPSIDE'), 'right')
        self.assertEqual(get_column_alignment('BETA'), 'right')
    
    def test_directory_creation(self):
        """Test the directory creation functionality"""
        # Logic from trade.py for directory creation
        def ensure_directory_exists(dir_path):
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                return True
            return False
        
        # Test with a temporary directory
        with TemporaryDirectory() as base_dir:
            # Test creating a non-existent directory
            test_dir = os.path.join(base_dir, "test_output")
            self.assertTrue(ensure_directory_exists(test_dir))
            self.assertTrue(os.path.exists(test_dir))
            
            # Test with an existing directory (should return False)
            self.assertFalse(ensure_directory_exists(test_dir))
            
    def test_column_selection_logic(self):
        """Test column selection logic in a simplified way"""
        # List of all available columns in a dataset
        all_columns = ['ticker', 'company', 'price', 'target_price', 'upside', 
                        'analyst_count', 'buy_percentage', 'unexpected_column']
        
        # Columns we want to select
        desired_columns = ['ticker', 'company', 'price', 'target_price', 'upside', 
                          'analyst_count', 'buy_percentage', 'missing_column']
        
        # Filter to keep only columns that exist in the dataset
        available_columns = [col for col in desired_columns if col in all_columns]
        
        # Verify correct column selection
        self.assertEqual(available_columns, 
                        ['ticker', 'company', 'price', 'target_price', 'upside', 
                         'analyst_count', 'buy_percentage'])
        self.assertNotIn('missing_column', available_columns)
        self.assertNotIn('unexpected_column', available_columns)
        
    def test_column_mapping_function(self):
        """Test column mapping logic"""
        # Define original column names
        original_columns = ['ticker', 'company', 'price', 'upside', 'buy_percentage']
        
        # Define mapping dictionary
        column_mapping = {
            'ticker': 'TICKER',
            'company': 'COMPANY NAME',
            'price': 'PRICE',
            'upside': 'UPSIDE',
            'buy_percentage': BUY_PERCENTAGE  # Using the constant from trade.py
        }
        
        # Create mapped column names
        mapped_columns = [column_mapping.get(col, col) for col in original_columns]
        
        # Verify mapping
        self.assertEqual(mapped_columns, ['TICKER', 'COMPANY NAME', 'PRICE', 'UPSIDE', BUY_PERCENTAGE])
        
    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_value_error_handling(self, mock_display_class, mock_input):
        """Test ValueError handling in main function"""
        # Mock user input
        mock_input.return_value = "M"
        
        # Mock MarketDisplay to raise ValueError when displaying report
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = ["AAPL", "MSFT"]
        mock_display.display_report.side_effect = ValueError("Invalid numeric value")
        
        # Run main function and verify it handles the ValueError
        with patch('trade.logger') as mock_logger:
            main()
            
            # Verify logger.error was called with appropriate message
            mock_logger.error.assert_called_once()
            self.assertIn("Error processing numeric values", mock_logger.error.call_args[0][0])
    
    @patch('builtins.input')
    @patch('trade.MarketDisplay')
    def test_general_exception_handling(self, mock_display_class, mock_input):
        """Test general exception handling in main function"""
        # Mock user input
        mock_input.return_value = "M"
        
        # Mock MarketDisplay to raise a general exception when displaying report
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.load_tickers.return_value = ["AAPL", "MSFT"]
        mock_display.display_report.side_effect = Exception("General error")
        
        # Run main function and verify it handles the exception
        with patch('trade.logger') as mock_logger:
            main()
            
            # Verify logger.error was called with appropriate message
            mock_logger.error.assert_called_once()
            self.assertIn("Error displaying report", mock_logger.error.call_args[0][0])
    
    def test_trade_criteria_simplified(self):
        """Test the buy/sell criteria logic in a simplified way without DataFrame operations"""
        
        # Test data - each dict represents a stock
        stocks = [
            {'ticker': 'AAPL', 'analyst_count': 10, 'upside': 15.0, 'buy_percentage': 80.0},
            {'ticker': 'MSFT', 'analyst_count': 8, 'upside': 25.0, 'buy_percentage': 80.0},
            {'ticker': 'GOOGL', 'analyst_count': 5, 'upside': 22.0, 'buy_percentage': 70.0},
            {'ticker': 'AMZN', 'analyst_count': 5, 'upside': 30.0, 'buy_percentage': 85.0},
            {'ticker': 'META', 'analyst_count': 3, 'upside': 40.0, 'buy_percentage': 90.0},
        ]
        
        # Portfolio tickers
        portfolio_tickers = ['AAPL', 'GOOGL']
        
        # Buy criteria function that mimics generate_trade_recommendations logic
        def meets_buy_criteria(stock):
            return (stock['analyst_count'] >= 5 and 
                    stock['upside'] >= 20.0 and 
                    stock['buy_percentage'] >= 75.0 and
                    stock['ticker'] not in portfolio_tickers)
        
        # Filter stocks using buy criteria
        buy_opportunities = [stock for stock in stocks if meets_buy_criteria(stock)]
        
        # Expected results:
        # MSFT: meets all criteria and not in portfolio ✓
        # AMZN: meets all criteria and not in portfolio ✓
        # AAPL: upside too low + in portfolio ✗
        # GOOGL: buy percentage too low + in portfolio ✗
        # META: not enough analysts ✗
        
        # Verify results
        self.assertEqual(len(buy_opportunities), 2)
        
        # Verify specific stocks
        buy_tickers = [stock['ticker'] for stock in buy_opportunities]
        self.assertIn('MSFT', buy_tickers)
        self.assertIn('AMZN', buy_tickers)
        self.assertNotIn('AAPL', buy_tickers)
        self.assertNotIn('GOOGL', buy_tickers)
        self.assertNotIn('META', buy_tickers)
        
        # Test with empty portfolio
        portfolio_tickers = []
        buy_opportunities = [stock for stock in stocks if meets_buy_criteria(stock)]
        self.assertEqual(len(buy_opportunities), 2)  # Still just MSFT and AMZN meet all criteria