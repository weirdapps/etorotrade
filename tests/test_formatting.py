import unittest
from yahoofinance.formatting import DisplayFormatter, DisplayConfig, Color, ColorCode

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

    def test_convert_numeric(self):
        """Test numeric value conversion."""
        self.assertEqual(self.formatter._convert_numeric(10), 10.0)
        self.assertEqual(self.formatter._convert_numeric("10.5"), 10.5)
        self.assertEqual(self.formatter._convert_numeric(None), 0.0)
        self.assertEqual(self.formatter._convert_numeric("N/A"), 0.0)
        self.assertEqual(self.formatter._convert_numeric("--"), 0.0)
        self.assertEqual(self.formatter._convert_numeric("invalid"), 0.0)
        self.assertEqual(self.formatter._convert_numeric(None, default=1.0), 1.0)

    def test_validate_dataframe_input(self):
        """Test DataFrame input validation."""
        valid_rows = [{
            '_sort_exret': 10,
            '_sort_earnings': '2024-01-01'
        }]
        
        # Should not raise any exception
        self.formatter._validate_dataframe_input(valid_rows)
        
        # Test empty input
        with self.assertRaises(ValueError) as context:
            self.formatter._validate_dataframe_input([])
        self.assertIn("No rows provided", str(context.exception))
        
        # Test missing required columns
        invalid_rows = [{'other_column': 'value'}]
        with self.assertRaises(ValueError) as context:
            self.formatter._validate_dataframe_input(invalid_rows)
        self.assertIn("missing required sort columns", str(context.exception))

    def test_create_sortable_dataframe(self):
        """Test creating sortable DataFrame."""
        rows = [
            {
                'ticker': 'AAPL',
                '_sort_exret': 10,
                '_sort_earnings': '2024-01-01'
            },
            {
                'ticker': 'GOOGL',
                '_sort_exret': 20,
                '_sort_earnings': '2024-01-02'
            }
        ]
        
        df = self.formatter.create_sortable_dataframe(rows)
        
        # Check sorting (GOOGL should be first due to higher _sort_exret)
        self.assertEqual(df.iloc[0]['ticker'], 'GOOGL')
        self.assertEqual(df.iloc[1]['ticker'], 'AAPL')
        
        # Check ranking column
        self.assertEqual(df.iloc[0]['#'], 1)
        self.assertEqual(df.iloc[1]['#'], 2)
        
        # Check sort columns were removed
        self.assertNotIn('_sort_exret', df.columns)
        self.assertNotIn('_sort_earnings', df.columns)

    def test_create_sortable_dataframe_invalid_input(self):
        """Test creating DataFrame with invalid input."""
        with self.assertRaises(ValueError):
            self.formatter.create_sortable_dataframe([])
            
        with self.assertRaises(ValueError):
            self.formatter.create_sortable_dataframe([{'invalid': 'data'}])

    def test_color_enum_values(self):
        """Test Color enum values and behavior."""
        # Test color values
        self.assertEqual(Color.BUY.value, ColorCode.GREEN)
        self.assertEqual(Color.LOW_CONFIDENCE.value, ColorCode.YELLOW)
        self.assertEqual(Color.SELL.value, ColorCode.RED)
        self.assertEqual(Color.NEUTRAL.value, ColorCode.DEFAULT)
        
        # Test color application with config
        formatter_with_colors = DisplayFormatter(DisplayConfig(use_colors=True))
        colored_text = formatter_with_colors.colorize("test", Color.BUY)
        self.assertIn(ColorCode.GREEN, colored_text)
        
        # Test color disabled
        formatter_no_colors = DisplayFormatter(DisplayConfig(use_colors=False))
        plain_text = formatter_no_colors.colorize("test", Color.BUY)
        self.assertEqual(plain_text, "test")

if __name__ == '__main__':
    unittest.main()