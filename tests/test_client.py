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
        # Create mock ticker with empty info
        mock_ticker = Mock()
        mock_ticker.info = None  # Test handling of None info
        mock_ticker.history.return_value = pd.DataFrame()  # Empty history
        mock_yf_ticker.return_value = mock_ticker

        # Mock get_earnings_dates
        self.client.get_earnings_dates = Mock(return_value=(None, None))
        
        # Mock insider_analyzer
        self.client.insider_analyzer.get_insider_metrics = Mock(
            return_value={'insider_buy_pct': None, 'transaction_count': None}
        )

        # Get ticker info and verify all fields
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
        # Create a successful mock ticker
        mock_success = Mock()
        mock_success.info = {}
        mock_success.history.return_value = pd.DataFrame()

        # Make API call fail twice then succeed
        mock_yf_ticker.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            mock_success  # Success on third try
        ]
        
        # Mock other dependencies
        self.client.get_earnings_dates = Mock(return_value=(None, None))
        self.client.insider_analyzer.get_insider_metrics = Mock(
            return_value={'insider_buy_pct': None, 'transaction_count': None}
        )
        
        # Call method that uses retry mechanism
        result = self.client.get_ticker_info('AAPL')
        
        # Verify the mock was called exactly 3 times
        self.assertEqual(mock_yf_ticker.call_count, 3)
        
        # Verify backoff times
        mock_sleep.assert_has_calls([
            unittest.mock.call(1.0),  # First retry
            unittest.mock.call(2.0)   # Second retry
        ])
        
        # Verify we got a result
        self.assertIsInstance(result, StockData)

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

    def test_calculate_price_changes(self):
        """Test price change calculations with various scenarios"""
        client = YFinanceClient()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        changes = client._calculate_price_changes(empty_df)
        self.assertEqual(changes, (None, None, None, None))
        
        # Test with single day data
        single_day = pd.DataFrame({
            'Close': [100.0]
        })
        changes = client._calculate_price_changes(single_day)
        self.assertIsNone(changes[0])  # price_change
        self.assertIsNone(changes[1])  # mtd_change
        self.assertEqual(changes[2], 0.0)   # ytd_change (first day is reference)
        self.assertEqual(changes[3], 0.0)   # two_year_change (same as ytd for now)
        
        # Test with two days data
        two_days = pd.DataFrame({
            'Close': [100.0, 110.0]
        })
        price_change, mtd, ytd, two_year = client._calculate_price_changes(two_days)
        self.assertAlmostEqual(price_change, 10.0)  # (110-100)/100 * 100
        self.assertIsNone(mtd)  # Not enough data for MTD
        self.assertAlmostEqual(ytd, 10.0)  # Based on first day
        self.assertAlmostEqual(two_year, 10.0)  # Same as YTD
        
        # Test with full month data
        month_data = pd.DataFrame({
            'Close': [100.0] * 20 + [100.0, 110.0]  # 22 trading days
        })
        price_change, mtd, ytd, two_year = client._calculate_price_changes(month_data)
        self.assertAlmostEqual(price_change, 10.0)  # Last two values: 110.0 vs 100.0
        self.assertAlmostEqual(mtd, 10.0)  # First vs last: 110.0 vs 100.0
        self.assertAlmostEqual(ytd, 10.0)  # First vs last: 110.0 vs 100.0
        self.assertAlmostEqual(two_year, 10.0)  # Same as YTD

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculations with various scenarios"""
        client = YFinanceClient()
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        metrics = client._calculate_risk_metrics(empty_df)
        self.assertEqual(metrics, (None, None, None))
        
        # Test with single day data
        single_day = pd.DataFrame({
            'Close': [100.0]
        })
        metrics = client._calculate_risk_metrics(single_day)
        self.assertEqual(metrics, (None, None, None))
        
        # Test with stable prices (no volatility)
        stable_prices = pd.DataFrame({
            'Close': [100.0] * 10
        })
        alpha, sharpe, sortino = client._calculate_risk_metrics(stable_prices)
        self.assertIsNotNone(alpha)
        self.assertAlmostEqual(alpha, -0.05)  # Annual risk-free rate effect
        self.assertIsNone(sharpe)  # Division by zero (no volatility)
        self.assertIsNone(sortino)  # No downside volatility
        
        # Test with volatile prices
        volatile_prices = pd.DataFrame({
            'Close': [100.0, 110.0, 90.0, 105.0, 95.0]
        })
        alpha, sharpe, sortino = client._calculate_risk_metrics(volatile_prices)
        self.assertIsNotNone(alpha)
        self.assertIsNotNone(sharpe)
        self.assertIsNotNone(sortino)
        
        # Test with all negative returns
        negative_returns = pd.DataFrame({
            'Close': [100.0, 90.0, 80.0, 70.0, 60.0]
        })
        alpha, sharpe, sortino = client._calculate_risk_metrics(negative_returns)
        self.assertLess(alpha, 0)  # Negative excess returns
        self.assertLess(sharpe, 0)  # Negative Sharpe ratio
        self.assertLess(sortino, 0)  # Negative Sortino ratio

    def test_cache_management(self):
        """Test cache management functions"""
        client = YFinanceClient()
        
        # Test initial cache state
        cache_info = client.get_cache_info()
        initial_hits = cache_info['hits']
        initial_misses = cache_info['misses']
        self.assertEqual(cache_info['maxsize'], 100)  # Updated from 50 to match new config
        self.assertEqual(cache_info['currsize'], 0)
        
        # Test cache hit/miss
        with patch('yfinance.Ticker') as mock_yf_ticker:
            # Set up mock for both calls
            mock_ticker = Mock()
            mock_ticker.info = {}
            mock_ticker.history.return_value = pd.DataFrame()
            mock_yf_ticker.return_value = mock_ticker
            
            # Mock get_earnings_dates and insider_metrics for both calls
            client.get_earnings_dates = Mock(return_value=(None, None))
            client.insider_analyzer.get_insider_metrics = Mock(
                return_value={'insider_buy_pct': None, 'transaction_count': None}
            )
            
            # First call - should miss
            client.get_ticker_info('AAPL')
            cache_info = client.get_cache_info()
            self.assertEqual(cache_info['misses'], initial_misses + 1)
            
            # Second call - should hit
            client.get_ticker_info('AAPL')
            cache_info = client.get_cache_info()
            self.assertEqual(cache_info['hits'], initial_hits + 1)
        
        # Test cache clear
        client.clear_cache()
        cache_info = client.get_cache_info()
        self.assertEqual(cache_info['currsize'], 0)
        self.assertEqual(cache_info['ticker_cache_size'], 0)  # Check ticker cache was cleared too
        
    def test_validate_ticker(self):
        """Test ticker validation with various formats"""
        client = YFinanceClient()
        
        # Test valid tickers
        try:
            client._validate_ticker("AAPL")              # Standard US ticker
            client._validate_ticker("BRK.B")             # US ticker with class
            client._validate_ticker("0700.HK")           # Hong Kong ticker
            client._validate_ticker("MAERSK-A.CO")       # Longer ticker with exchange suffix
            client._validate_ticker("BP.L")              # London ticker
        except Exception as e:
            self.fail(f"Validation raised exception for valid ticker: {e}")
            
        # Test invalid tickers
        from yahoofinance.types import ValidationError
        with self.assertRaises(ValidationError):
            client._validate_ticker("")
        with self.assertRaises(ValidationError):
            client._validate_ticker(None)
        with self.assertRaises(ValidationError):
            client._validate_ticker("123")
        with self.assertRaises(ValidationError):
            client._validate_ticker("THISISAVERYLONGTICKER")
        with self.assertRaises(ValidationError):
            client._validate_ticker("THISISAVERYLONGTICKER.WITHSUFFIX.TOOLONG")
            
    def test_is_us_ticker(self):
        """Test US ticker detection function"""
        client = YFinanceClient()
        
        # US tickers
        self.assertTrue(client._is_us_ticker("AAPL"))
        self.assertTrue(client._is_us_ticker("MSFT"))
        self.assertTrue(client._is_us_ticker("BRK.B"))  # Special case that should still be treated as US
        self.assertTrue(client._is_us_ticker("AMZN.US"))
        
        # Non-US tickers
        self.assertFalse(client._is_us_ticker("0700.HK"))
        self.assertFalse(client._is_us_ticker("BP.L"))
        self.assertFalse(client._is_us_ticker("MAERSK-A.CO"))
        self.assertFalse(client._is_us_ticker("TSLA.MI"))

if __name__ == '__main__':
    unittest.main()