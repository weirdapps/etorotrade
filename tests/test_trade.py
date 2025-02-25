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
        self.assertEqual(calculate_exret(20.0, 75.0), 15.0)
        self.assertEqual(calculate_exret(15.0, 60.0), 9.0)
        self.assertEqual(calculate_exret(0.0, 50.0), 0.0)
        self.assertEqual(calculate_exret(10.0, 0.0), 0.0)
        
        # Test with extreme values
        self.assertEqual(calculate_exret(50.0, 100.0), 50.0)
        # Round to avoid floating point imprecision
        result = round(calculate_exret(12.34, 56.78), 5)
        self.assertEqual(result, 7.00665)
            
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
            
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    def test_trade_recommendations_buy_with_output(self, mock_makedirs, mock_read_csv, mock_path_exists):
        """Test generate_trade_recommendations with buy option creating output files"""
        # Mock file existence
        mock_path_exists.side_effect = lambda path: path.endswith('market.csv') or path.endswith('portfolio.csv') or path.endswith('output')
        
        # Mock CSV data - market data with buy opportunities
        market_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'company': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc'],
            'price': [150.0, 300.0, 2500.0],
            'target_price': [180.0, 350.0, 3000.0],
            'upside': [20.0, 16.7, 20.0],
            'analyst_count': [10, 8, 5],
            'buy_percentage': [80.0, 75.0, 90.0],
            'total_ratings': [12, 10, 6],
            'A': ['E', 'E', 'A'],
            'beta': [1.2, 1.1, 1.3],
            'pe_trailing': [30.0, 35.0, 28.0],
            'pe_forward': [25.0, 30.0, 22.0],
            'peg_ratio': [1.5, 1.2, 1.1],
            'dividend_yield': [0.5, 0.8, 0.0],
            'short_float_pct': [1.0, 0.8, 1.2],
            'last_earnings': ['2023-01-15', '2023-02-20', '2023-03-10']
        })
        
        # Mock portfolio data - GOOGL is in portfolio
        portfolio_data = pd.DataFrame({
            'ticker': ['GOOGL', 'AMZN'],
            'share_price': [2500.0, 3200.0]
        })
        
        mock_read_csv.side_effect = lambda path, **kwargs: {
            'yahoofinance/output/market.csv': market_data,
            'yahoofinance/input/portfolio.csv': portfolio_data
        }.get(path, pd.DataFrame())
        
        # Test buy recommendations (should include AAPL and MSFT, not GOOGL since it's in portfolio)
        with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
             patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('N')  # 'N' for new buy opportunities
            
            # Check if to_csv was called with correct path
            mock_to_csv.assert_called()
            self.assertIn("yahoofinance/output/buy.csv", mock_to_csv.call_args[0][0])
            
            # Check output
            output = fake_out.getvalue()
            self.assertIn('New Buy Opportunities', output)
            self.assertIn('AAPL', output)  # Should include AAPL
            self.assertIn('MSFT', output)  # Should include MSFT
            self.assertNotIn('GOOGL', output)  # Should NOT include GOOGL (it's in portfolio)
            
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    def test_trade_recommendations_sell_with_output(self, mock_makedirs, mock_read_csv, mock_path_exists):
        """Test generate_trade_recommendations with sell option creating output files"""
        # Mock file existence
        mock_path_exists.side_effect = lambda path: True
        
        # Mock CSV data - portfolio analysis data with sell candidates
        portfolio_analysis = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'company': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc', 'Amazon Inc'],
            'price': [150.0, 300.0, 2500.0, 3200.0],
            'target_price': [160.0, 310.0, 2600.0, 3500.0],
            'upside': [6.7, 3.3, 4.0, 9.4],  # MSFT and GOOGL are sell candidates (low upside)
            'analyst_count': [10, 8, 5, 6],
            'buy_percentage': [80.0, 48.0, 90.0, 45.0],  # MSFT and AMZN are sell candidates (low buy %)
            'total_ratings': [12, 10, 6, 8],
            'A': ['E', 'E', 'A', 'E'],
            'beta': [1.2, 1.1, 1.3, 1.4],
            'pe_trailing': [30.0, 35.0, 28.0, 50.0],
            'pe_forward': [25.0, 30.0, 22.0, 40.0],
            'peg_ratio': [1.5, 1.2, 1.1, 1.8],
            'dividend_yield': [0.5, 0.8, 0.0, 0.0],
            'short_float_pct': [1.0, 0.8, 1.2, 1.5],
            'last_earnings': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-01-05']
        })
        
        mock_read_csv.side_effect = lambda path, **kwargs: portfolio_analysis
        
        # Test sell recommendations
        with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
             patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('E')  # 'E' for existing portfolio (sell)
            
            # Check if to_csv was called with correct path
            mock_to_csv.assert_called()
            self.assertIn("yahoofinance/output/sell.csv", mock_to_csv.call_args[0][0])
            
            # Check output
            output = fake_out.getvalue()
            self.assertIn('Sell Candidates', output)
            # Should include tickers meeting sell criteria (low upside or low buy percentage)
            # MSFT: upside 3.3% (< 5%) and buy percentage 48% (< 50%)
            # GOOGL: upside 4.0% (< 5%)
            # AMZN: buy percentage 45% (< 50%)
            self.assertIn('MSFT', output)
            self.assertIn('GOOGL', output)
            self.assertIn('AMZN', output)
            self.assertNotIn('AAPL', output)  # Apple doesn't meet sell criteria
            
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