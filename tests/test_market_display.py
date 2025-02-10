import unittest
from unittest.mock import Mock, patch, mock_open, PropertyMock
import pandas as pd
from io import StringIO
from yahoofinance.display import MarketDisplay
from yahoofinance.client import YFinanceClient
from yahoofinance.formatting import DisplayConfig

class TestMarketDisplay(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_client = Mock(spec=YFinanceClient)
        self.test_input_dir = "test_input"
        self.display = MarketDisplay(
            client=self.mock_client,
            input_dir=self.test_input_dir
        )
        # Mock the pricing analyzer
        self.display.pricing = Mock()
        self.display.analyst = Mock()

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

    def test_generate_market_html(self):
        """Test market HTML generation."""
        mock_csv_content = 'symbol\nAAPL\nGOOGL'
        mock_stock_data = Mock(
            current_price=150.0,
            price_change_percentage=5.0
        )
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))), \
             patch('builtins.open', mock_open()) as mock_file:
            
            self.mock_client.get_ticker_info.return_value = mock_stock_data
            self.display.generate_market_html()
            
            # Verify file was written
            mock_file.assert_called_once()
            self.assertIn('index.html', mock_file.call_args[0][0])

    def test_generate_portfolio_html(self):
        """Test portfolio HTML generation."""
        mock_csv_content = 'ticker\nAAPL\nGOOGL'
        mock_stock_data = Mock(
            current_price=150.0,
            price_change_percentage=5.0,
            mtd_change=10.0,
            ytd_change=15.0,
            two_year_change=25.0,
            beta=1.1,
            alpha=0.2,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            cash_percentage=5.0
        )
        
        with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_csv_content))), \
             patch('builtins.open', mock_open()) as mock_file:
            
            self.mock_client.get_ticker_info.return_value = mock_stock_data
            self.display.generate_portfolio_html()
            
            # Verify file was written
            mock_file.assert_called_once()
            self.assertIn('portfolio.html', mock_file.call_args[0][0])

    def test_display_configuration(self):
        """Test display configuration options."""
        config = DisplayConfig(use_colors=False)
        display = MarketDisplay(
            client=self.mock_client,
            config=config
        )
        self.assertFalse(display.formatter.config.use_colors)

    def test_generate_stock_report(self):
        """Test stock report generation."""
        mock_price_metrics = {
            'current_price': 150.0,
            'target_price': 180.0,
            'upside_potential': 20.0
        }
        mock_ratings = {
            'positive_percentage': 80.0,
            'total_ratings': 10
        }
        mock_stock_info = Mock(
            analyst_count=10,
            pe_trailing=20.5,
            pe_forward=18.2,
            peg_ratio=1.5,
            dividend_yield=2.5,
            beta=1.1,
            short_float_pct=2.0,
            last_earnings='2024-01-01',
            insider_buy_pct=75.0,
            insider_transactions=5
        )
        
        self.display.pricing.calculate_price_metrics.return_value = mock_price_metrics
        self.display.analyst.get_ratings_summary.return_value = mock_ratings
        self.mock_client.get_ticker_info.return_value = mock_stock_info
        
        report = self.display.generate_stock_report('AAPL')
        
        self.assertEqual(report['ticker'], 'AAPL')
        self.assertEqual(report['price'], 150.0)
        self.assertEqual(report['target_price'], 180.0)
        self.assertEqual(report['upside'], 20.0)
        self.assertEqual(report['analyst_count'], 10)
        self.assertEqual(report['buy_percentage'], 80.0)
        self.assertFalse(report['_not_found'])

if __name__ == '__main__':
    unittest.main()