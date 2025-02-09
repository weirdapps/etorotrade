import unittest
from unittest.mock import Mock, patch, mock_open
import pandas as pd
from io import StringIO
from yahoofinance.display import MarketDisplay
from yahoofinance.client import YFinanceClient

class TestMarketDisplay(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_client = Mock(spec=YFinanceClient)
        self.display = MarketDisplay(client=self.mock_client)

    def test_load_tickers_manual_input(self):
        """Test loading tickers from manual input."""
        with patch('builtins.input', return_value='AAPL, GOOGL, MSFT'):
            tickers = self.display.load_tickers('I')
            # Sort tickers to ensure consistent comparison
            self.assertEqual(sorted(tickers), ['AAPL', 'GOOGL', 'MSFT'])

    def test_load_tickers_portfolio(self):
        """Test loading tickers from portfolio.csv."""
        mock_csv_content = 'ticker\nAAPL\nGOOGL\nMSFT'
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))):
            tickers = self.display.load_tickers('P')
            self.assertEqual(sorted(tickers), ['AAPL', 'GOOGL', 'MSFT'])

    def test_load_tickers_market(self):
        """Test loading tickers from market.csv."""
        mock_csv_content = 'symbol\nAAPL\nGOOGL\nMSFT'
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))):
            tickers = self.display.load_tickers('M')
            self.assertEqual(sorted(tickers), ['AAPL', 'GOOGL', 'MSFT'])

    def test_load_tickers_invalid_source(self):
        """Test loading tickers with invalid source."""
        tickers = self.display.load_tickers('X')
        self.assertEqual(tickers, [])

    def test_load_tickers_file_not_found(self):
        """Test handling of missing file."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError):
            tickers = self.display.load_tickers('P')
            self.assertEqual(tickers, [])

    def test_load_tickers_empty_input(self):
        """Test handling of empty manual input."""
        with patch('builtins.input', return_value=''):
            tickers = self.display.load_tickers('I')
            self.assertEqual(tickers, [])

    def test_load_tickers_duplicate_removal(self):
        """Test that duplicate tickers are removed."""
        with patch('builtins.input', return_value='AAPL, AAPL, MSFT, MSFT'):
            tickers = self.display.load_tickers('I')
            self.assertEqual(sorted(tickers), ['AAPL', 'MSFT'])

if __name__ == '__main__':
    unittest.main()