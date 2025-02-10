import unittest
from unittest.mock import Mock, patch
import pandas as pd
from yahoofinance.client import YFinanceClient, YFinanceError, StockData

class TestYFinanceClient(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.client = YFinanceClient()

    @patch('yfinance.Ticker')
    def test_get_ticker_info_success(self, mock_yf_ticker):
        """Test successful retrieval of ticker information."""
        # Mock the yfinance Ticker object
        mock_ticker = Mock()
        mock_ticker.info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'marketCap': 2000000000000,
            'currentPrice': 150.0,
            'targetMeanPrice': 180.0,
            'recommendationMean': 2.0,
            'recommendationKey': 'buy',
            'numberOfAnalystOpinions': 10,
            'trailingPE': 20.5,
            'forwardPE': 18.2,
            'trailingPegRatio': 1.5,
            'quickRatio': 1.1,
            'currentRatio': 1.2,
            'debtToEquity': 150.0,
            'shortPercentOfFloat': 0.02,
            'shortRatio': 1.5,
            'beta': 1.1,
            'dividendYield': 0.025
        }
        mock_ticker.history.return_value = pd.DataFrame()  # Empty history
        mock_yf_ticker.return_value = mock_ticker

        # Mock get_earnings_dates
        self.client.get_earnings_dates = Mock(return_value=('2024-01-01', '2023-10-01'))
        
        # Mock insider_analyzer
        self.client.insider_analyzer.get_insider_metrics = Mock(
            return_value={'insider_buy_pct': 75.0, 'transaction_count': 5}
        )

        info = self.client.get_ticker_info('AAPL')
        
        self.assertEqual(info.name, 'Apple Inc.')
        self.assertEqual(info.sector, 'Technology')
        self.assertEqual(info.market_cap, 2000000000000)
        self.assertEqual(info.current_price, 150.0)
        self.assertEqual(info.target_price, 180.0)
        self.assertEqual(info.recommendation_mean, 2.0)
        self.assertEqual(info.recommendation_key, 'buy')
        self.assertEqual(info.analyst_count, 10)
        self.assertEqual(info.pe_trailing, 20.5)
        self.assertEqual(info.pe_forward, 18.2)
        self.assertEqual(info.peg_ratio, 1.5)
        self.assertEqual(info.quick_ratio, 1.1)
        self.assertEqual(info.current_ratio, 1.2)
        self.assertEqual(info.debt_to_equity, 150.0)
        self.assertEqual(info.short_float_pct, 2.0)  # Converted to percentage
        self.assertEqual(info.short_ratio, 1.5)
        self.assertEqual(info.beta, 1.1)
        self.assertEqual(info.dividend_yield, 0.025)
        self.assertEqual(info.last_earnings, '2024-01-01')
        self.assertEqual(info.previous_earnings, '2023-10-01')
        self.assertEqual(info.insider_buy_pct, 75.0)
        self.assertEqual(info.insider_transactions, 5)

    @patch('yfinance.Ticker')
    def test_get_ticker_info_missing_data(self, mock_yf_ticker):
        """Test handling of missing data in ticker information."""
        mock_ticker = Mock()
        mock_ticker.info = {}  # Empty info dictionary
        mock_ticker.history.return_value = pd.DataFrame()  # Empty history
        mock_yf_ticker.return_value = mock_ticker

        # Mock get_earnings_dates
        self.client.get_earnings_dates = Mock(return_value=(None, None))
        
        # Mock insider_analyzer
        self.client.insider_analyzer.get_insider_metrics = Mock(
            return_value={'insider_buy_pct': None, 'transaction_count': None}
        )

        info = self.client.get_ticker_info('AAPL')
        
        self.assertEqual(info.name, 'N/A')
        self.assertEqual(info.sector, 'N/A')
        self.assertIsNone(info.market_cap)
        self.assertIsNone(info.current_price)
        self.assertIsNone(info.target_price)
        self.assertIsNone(info.recommendation_mean)
        self.assertEqual(info.recommendation_key, 'N/A')
        self.assertIsNone(info.analyst_count)
        self.assertIsNone(info.pe_trailing)
        self.assertIsNone(info.pe_forward)
        self.assertIsNone(info.peg_ratio)
        self.assertIsNone(info.quick_ratio)
        self.assertIsNone(info.current_ratio)
        self.assertIsNone(info.debt_to_equity)
        self.assertIsNone(info.short_float_pct)
        self.assertIsNone(info.short_ratio)
        self.assertIsNone(info.beta)
        self.assertIsNone(info.dividend_yield)
        self.assertIsNone(info.last_earnings)
        self.assertIsNone(info.previous_earnings)
        self.assertIsNone(info.insider_buy_pct)
        self.assertIsNone(info.insider_transactions)

    @patch('yfinance.Ticker')
    def test_get_ticker_info_api_error(self, mock_yf_ticker):
        """Test handling of API errors."""
        mock_yf_ticker.side_effect = Exception("API Error")

        with self.assertRaises(YFinanceError):
            self.client.get_ticker_info('AAPL')

    def test_get_backoff_time(self):
        """Test exponential backoff time calculation."""
        # Test default parameters
        self.assertEqual(self.client._get_backoff_time(1), 1.0)  # First attempt
        self.assertEqual(self.client._get_backoff_time(2), 2.0)  # Second attempt
        self.assertEqual(self.client._get_backoff_time(3), 4.0)  # Third attempt
        self.assertEqual(self.client._get_backoff_time(4), 8.0)  # Fourth attempt
        self.assertEqual(self.client._get_backoff_time(5), 10.0)  # Max time reached
        
        # Test custom parameters
        self.assertEqual(self.client._get_backoff_time(1, base=0.5, max_time=5.0), 0.5)
        self.assertEqual(self.client._get_backoff_time(2, base=0.5, max_time=5.0), 1.0)
        self.assertEqual(self.client._get_backoff_time(5, base=0.5, max_time=5.0), 5.0)

    @patch('time.sleep')
    @patch('yfinance.Ticker')
    def test_retry_with_backoff(self, mock_yf_ticker, mock_sleep):
        """Test retry mechanism with exponential backoff."""
        # Make API call fail twice then succeed
        mock_yf_ticker.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            Mock(info={}, history=lambda period: pd.DataFrame())  # Success on third try
        ]
        
        # Mock other dependencies
        self.client.get_earnings_dates = Mock(return_value=(None, None))
        self.client.insider_analyzer.get_insider_metrics = Mock(
            return_value={'insider_buy_pct': None, 'transaction_count': None}
        )
        
        # Call method that uses retry mechanism
        self.client.get_ticker_info('AAPL')
        
        # Verify backoff times
        mock_sleep.assert_has_calls([
            unittest.mock.call(1.0),  # First retry
            unittest.mock.call(2.0)   # Second retry
        ])

    def test_stock_property_error(self):
        """Test StockData _stock property error handling."""
        stock_data = StockData(
            name="Test Stock",
            sector="Technology",
            market_cap=None,
            current_price=None,
            target_price=None,
            price_change_percentage=None,
            mtd_change=None,
            ytd_change=None,
            two_year_change=None,
            recommendation_mean=None,
            recommendation_key="N/A",
            analyst_count=None,
            pe_trailing=None,
            pe_forward=None,
            peg_ratio=None,
            quick_ratio=None,
            current_ratio=None,
            debt_to_equity=None,
            short_float_pct=None,
            short_ratio=None,
            beta=None,
            alpha=None,
            sharpe_ratio=None,
            sortino_ratio=None,
            cash_percentage=None,
            dividend_yield=None,
            last_earnings=None,
            previous_earnings=None,
            insider_buy_pct=None,
            insider_transactions=None
        )
        
        with self.assertRaises(AttributeError) as context:
            _ = stock_data._stock
        self.assertIn("No ticker object available", str(context.exception))

if __name__ == '__main__':
    unittest.main()