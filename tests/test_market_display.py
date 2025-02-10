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
        self.test_input_dir = "test_input"
        self.display = MarketDisplay(
            client=self.mock_client,
            input_dir=self.test_input_dir
        )

    def test_load_tickers_from_file(self):
        """Test loading tickers from file using helper method."""
        mock_csv_content = 'ticker\nAAPL\nGOOGL\nMSFT'
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))):
            tickers = self.display._load_tickers_from_file('portfolio.csv', 'ticker')
            self.assertEqual(sorted(tickers), ['AAPL', 'GOOGL', 'MSFT'])

    def test_load_tickers_from_file_not_found(self):
        """Test file not found handling in helper method."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError), \
             self.assertRaises(FileNotFoundError):
            self.display._load_tickers_from_file('nonexistent.csv', 'ticker')

    def test_load_tickers_from_file_invalid_column(self):
        """Test invalid column handling in helper method."""
        mock_csv_content = 'wrong_column\nAAPL\nGOOGL\nMSFT'
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))), \
             self.assertRaises(KeyError):
            self.display._load_tickers_from_file('portfolio.csv', 'ticker')

    def test_load_tickers_from_input_empty(self):
        """Test empty input handling in helper method."""
        with patch('builtins.input', return_value=''):
            tickers = self.display._load_tickers_from_input()
            self.assertEqual(tickers, [])

    def test_load_tickers_from_input_duplicates(self):
        """Test duplicate removal in helper method."""
        with patch('builtins.input', return_value='AAPL, AAPL, MSFT, MSFT'):
            tickers = self.display._load_tickers_from_input()
            self.assertEqual(sorted(tickers), ['AAPL', 'MSFT'])

    def test_sort_market_data(self):
        """Test market data sorting."""
        mock_data = pd.DataFrame([
            {'_not_found': False, '_sort_exret': 10, '_sort_earnings': 5, '_ticker': 'AAPL'},
            {'_not_found': False, '_sort_exret': 5, '_sort_earnings': 10, '_ticker': 'GOOGL'},
            {'_not_found': True, '_ticker': 'INVALID'}
        ])
        sorted_df = self.display._sort_market_data(mock_data)
        self.assertEqual(sorted_df.iloc[0]['_ticker'], 'AAPL')
        self.assertEqual(sorted_df.iloc[1]['_ticker'], 'GOOGL')
        self.assertEqual(sorted_df.iloc[2]['_ticker'], 'INVALID')

    def test_format_dataframe(self):
        """Test DataFrame formatting."""
        mock_data = pd.DataFrame([
            {'_not_found': False, '_sort_exret': 10, '_sort_earnings': 5, '_ticker': 'AAPL', 'price': 150}
        ])
        formatted_df = self.display._format_dataframe(mock_data)
        self.assertIn('#', formatted_df.columns)
        self.assertNotIn('_not_found', formatted_df.columns)
        self.assertNotIn('_sort_exret', formatted_df.columns)
        self.assertEqual(formatted_df.iloc[0]['#'], 1)

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

    def test_generate_html_output(self):
        """Test HTML output generation."""
        mock_data = [
            {
                'ticker': 'AAPL',
                'price': 150.0,
                'target_price': 180.0,
                'upside': 20.0,
                'analyst_count': 10,
                'buy_percentage': 80.0,
                '_not_found': False
            }
        ]
        with patch('builtins.open', mock_open()) as mock_file:
            self.display.generate_html_output(mock_data, 'test.html')
            mock_file.assert_called_once_with('test.html', 'w', encoding='utf-8')
            
    def test_process_market_data(self):
        """Test market data processing with mock client."""
        self.mock_client.get_ticker_info.return_value = Mock(
            name='Apple Inc.',
            current_price=150.0,
            target_price=180.0,
            analyst_count=10,
            recommendation_mean=2.0
        )
        
        data = self.display.process_market_data(['AAPL'])
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['ticker'], 'AAPL')
        self.assertEqual(data[0]['price'], 150.0)
        self.assertEqual(data[0]['target_price'], 180.0)
        
    def test_process_market_data_error_handling(self):
        """Test error handling in market data processing."""
        self.mock_client.get_ticker_info.side_effect = Exception("API Error")
        
        data = self.display.process_market_data(['AAPL'])
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['ticker'], 'AAPL')
        self.assertTrue(data[0]['_not_found'])
        
    def test_display_configuration(self):
        """Test display configuration options."""
        # Test with custom configuration
        display = MarketDisplay(
            client=self.mock_client,
            use_colors=False,
            output_format='html'
        )
        self.assertFalse(display.config.use_colors)
        self.assertEqual(display.config.output_format, 'html')

if __name__ == '__main__':
    unittest.main()