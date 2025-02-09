import unittest
from yahoofinance.formatting import DisplayFormatter, DisplayConfig

class TestDisplayFormatter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = DisplayConfig(use_colors=False)  # Disable colors for testing
        self.formatter = DisplayFormatter(self.config)

    def test_format_stock_row_complete_data(self):
        """Test formatting a stock row with complete data."""
        stock_data = {
            'ticker': 'AAPL',
            'price': 150.0,
            'target_price': 180.0,
            'upside': 20.0,
            'analyst_count': 10,
            'buy_percentage': 80.0,
            'total_ratings': 15,
            'pe_trailing': 20.5,
            'pe_forward': 18.2,
            'peg_ratio': 1.5,
            'dividend_yield': 2.5,
            'beta': 1.1,
            'short_float_pct': 2.0,
            'last_earnings': '2024-01-01',
            'insider_buy_pct': 75.0,
            'insider_transactions': 5,
            '_not_found': False
        }

        formatted = self.formatter.format_stock_row(stock_data)

        self.assertEqual(formatted['TICKER'], 'AAPL')
        self.assertEqual(formatted['PRICE'], '150.00')
        self.assertEqual(formatted['TARGET'], '180.0')
        self.assertEqual(formatted['UPSIDE'], '20.0%')
        self.assertEqual(formatted['# T'], '10')
        self.assertEqual(formatted['% BUY'], '80.0%')
        self.assertEqual(formatted['PET'], '20.5')
        self.assertEqual(formatted['PEF'], '18.2')
        self.assertEqual(formatted['PEG'], '1.5')
        self.assertEqual(formatted['DIV %'], '2.50%')
        self.assertEqual(formatted['BETA'], '1.1')
        self.assertEqual(formatted['SI'], '2.0%')

    def test_format_stock_row_missing_data(self):
        """Test formatting a stock row with missing data."""
        stock_data = {
            'ticker': 'AAPL',
            'price': None,
            'target_price': None,
            'upside': None,
            'analyst_count': None,
            'buy_percentage': None,
            'total_ratings': None,
            'pe_trailing': None,
            'pe_forward': None,
            'peg_ratio': None,
            'dividend_yield': None,
            'beta': None,
            'short_float_pct': None,
            'last_earnings': None,
            'insider_buy_pct': None,
            'insider_transactions': None,
            '_not_found': True
        }

        formatted = self.formatter.format_stock_row(stock_data)

        self.assertEqual(formatted['TICKER'], 'AAPL')
        self.assertEqual(formatted['PRICE'], '--')
        self.assertEqual(formatted['TARGET'], '--')
        self.assertEqual(formatted['UPSIDE'], '--')
        self.assertEqual(formatted['# T'], '--')
        self.assertEqual(formatted['% BUY'], '--')
        self.assertEqual(formatted['PET'], '--')
        self.assertEqual(formatted['PEF'], '--')
        self.assertEqual(formatted['PEG'], '--')
        self.assertEqual(formatted['DIV %'], '--')
        self.assertEqual(formatted['BETA'], '--')
        self.assertEqual(formatted['SI'], '--')

    def test_format_stock_row_negative_values(self):
        """Test formatting a stock row with negative values."""
        stock_data = {
            'ticker': 'AAPL',
            'price': 150.0,
            'target_price': 120.0,
            'upside': -20.0,
            'analyst_count': 10,
            'buy_percentage': 30.0,
            'total_ratings': 15,
            'pe_trailing': -15.5,
            'pe_forward': -12.2,
            'peg_ratio': -0.5,
            'dividend_yield': 0,
            'beta': -0.8,
            'short_float_pct': 15.0,
            'last_earnings': '2024-01-01',
            'insider_buy_pct': 25.0,
            'insider_transactions': -3,
            '_not_found': False
        }

        formatted = self.formatter.format_stock_row(stock_data)

        self.assertEqual(formatted['TICKER'], 'AAPL')
        self.assertEqual(formatted['PRICE'], '150.00')
        self.assertEqual(formatted['TARGET'], '120.0')
        self.assertEqual(formatted['UPSIDE'], '-20.0%')
        self.assertEqual(formatted['# T'], '10')
        self.assertEqual(formatted['% BUY'], '30.0%')
        self.assertEqual(formatted['PET'], '-15.5')
        self.assertEqual(formatted['PEF'], '-12.2')
        self.assertEqual(formatted['PEG'], '-0.5')
        self.assertEqual(formatted['DIV %'], '0.00%')
        self.assertEqual(formatted['BETA'], '-0.8')
        self.assertEqual(formatted['SI'], '15.0%')

    def test_format_stock_row_zero_values(self):
        """Test formatting a stock row with zero values."""
        stock_data = {
            'ticker': 'AAPL',
            'price': 0,
            'target_price': 0,
            'upside': 0,
            'analyst_count': 0,
            'buy_percentage': 0,
            'total_ratings': 0,
            'pe_trailing': 0,
            'pe_forward': 0,
            'peg_ratio': 0,
            'dividend_yield': 0,
            'beta': 0,
            'short_float_pct': 0,
            'last_earnings': None,
            'insider_buy_pct': 0,
            'insider_transactions': 0,
            '_not_found': False
        }

        formatted = self.formatter.format_stock_row(stock_data)

        self.assertEqual(formatted['TICKER'], 'AAPL')
        self.assertEqual(formatted['PRICE'], '0.00')
        self.assertEqual(formatted['TARGET'], '0.0')
        self.assertEqual(formatted['UPSIDE'], '0.0%')
        self.assertEqual(formatted['# T'], '0')
        self.assertEqual(formatted['% BUY'], '0.0%')
        self.assertEqual(formatted['PET'], '0.0')
        self.assertEqual(formatted['PEF'], '0.0')
        self.assertEqual(formatted['PEG'], '0.0')
        self.assertEqual(formatted['DIV %'], '0.00%')
        self.assertEqual(formatted['BETA'], '0.0')
        self.assertEqual(formatted['SI'], '0.0%')

if __name__ == '__main__':
    unittest.main()