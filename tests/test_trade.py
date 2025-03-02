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

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def test_sell_recommendations_no_portfolio_analysis(self, mock_read_csv, mock_path_exists):
        """Test sell recommendations when portfolio analysis file doesn't exist"""
        # Set up mocks for file existence
        mock_path_exists.side_effect = lambda path: {
            'yahoofinance/output': True,
            'yahoofinance/output/market.csv': True,
            'yahoofinance/input/portfolio.csv': True,
            'yahoofinance/output/portfolio.csv': False
        }.get(path, False)
        
        # Mock the market.csv and portfolio.csv reading
        market_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'company': ['Apple', 'Microsoft'],
            'price': [150, 300]
        })
        
        portfolio_df = pd.DataFrame({
            'ticker': ['GOOGL', 'FB'],
            'company': ['Alphabet', 'Meta'],
            'price': [2500, 200]
        })
        
        # Set up the read_csv mock to return our dataframes
        mock_read_csv.side_effect = lambda path, **kwargs: {
            'yahoofinance/output/market.csv': market_df,
            'yahoofinance/input/portfolio.csv': portfolio_df
        }.get(path, pd.DataFrame())
        
        # Run the function that should check for portfolio.csv
        with patch('sys.stdout', new=StringIO()) as fake_out:
            generate_trade_recommendations('E')
            
            # Check that the correct error message was displayed
            output = fake_out.getvalue()
            self.assertIn('Portfolio analysis file not found', output)
            self.assertIn('Please run the portfolio analysis (P) first', output)

    # Additional tests would go here...
