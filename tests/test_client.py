import unittest
from unittest.mock import Mock, patch
from yahoofinance.client import YFinanceClient, YFinanceError

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

if __name__ == '__main__':
    unittest.main()