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
        self.assertEqual(formatted['PRICE'], '150.0')
        self.assertEqual(formatted['TARGET'], '180.0')
        self.assertEqual(formatted['UPSIDE'], '20.0%')
        self.assertEqual(formatted['# T'], '10')
        self.assertEqual(formatted['% BUY'], '80%')
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
        self.assertEqual(formatted['PRICE'], '150.0')
        self.assertEqual(formatted['TARGET'], '120.0')
        self.assertEqual(formatted['UPSIDE'], '-20.0%')
        self.assertEqual(formatted['# T'], '10')
        self.assertEqual(formatted['% BUY'], '30%')
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
        self.assertEqual(formatted['PRICE'], '0.0')
        self.assertEqual(formatted['TARGET'], '0.0')
        self.assertEqual(formatted['UPSIDE'], '0.0%')
        self.assertEqual(formatted['# T'], '0')
        self.assertEqual(formatted['% BUY'], '0%')
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
        
    def test_color_coding_logic(self):
        """Test the new sophisticated color coding logic with various scenarios."""
        formatter = DisplayFormatter(DisplayConfig(use_colors=True))
        
        # Test case 1: Low confidence (Yellow) - Insufficient analyst coverage
        low_confidence_data = {
            'ticker': 'TEST1',
            'analyst_count': 3,  # Below threshold of 5
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 120.0,  # 20% upside
            'buy_percentage': 90.0,
            'beta': 1.0,
            'pe_trailing': 15.0,
            'pe_forward': 12.0,
            'peg_ratio': 1.0,
            'short_float_pct': 2.0
        }
        low_confidence_metrics = {'upside': 20.0, 'ex_ret': 18.0}
        color = formatter._get_color_code(low_confidence_data, low_confidence_metrics)
        self.assertEqual(color, Color.LOW_CONFIDENCE)
        
        # Test case 2: Sell signal (Red) - Multiple sell conditions
        # 2a: Low buy percentage
        sell_data_1 = {
            'ticker': 'TEST2a',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 110.0,  # 10% upside
            'buy_percentage': 60.0,  # Below 65% threshold
            'beta': 1.3,
            'pe_trailing': 15.0,
            'pe_forward': 14.0,
            'peg_ratio': 1.5,
            'short_float_pct': 3.0
        }
        sell_metrics_1 = {'upside': 10.0, 'ex_ret': 6.0}
        color = formatter._get_color_code(sell_data_1, sell_metrics_1)
        self.assertEqual(color, Color.SELL)
        
        # 2b: PEF > PET (deteriorating earnings)
        sell_data_2 = {
            'ticker': 'TEST2b',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 130.0,  # 30% upside
            'buy_percentage': 90.0,
            'beta': 1.0,
            'pe_trailing': 15.0,
            'pe_forward': 17.0,  # Higher than trailing
            'peg_ratio': 1.0,
            'short_float_pct': 2.0
        }
        sell_metrics_2 = {'upside': 30.0, 'ex_ret': 27.0}
        color = formatter._get_color_code(sell_data_2, sell_metrics_2)
        self.assertEqual(color, Color.SELL)
        
        # 2c: High PEG ratio
        sell_data_3 = {
            'ticker': 'TEST2c',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 130.0,
            'buy_percentage': 90.0,
            'beta': 1.0,
            'pe_trailing': 15.0,
            'pe_forward': 14.0,
            'peg_ratio': 3.2,  # Above 3.0 threshold
            'short_float_pct': 2.0
        }
        sell_metrics_3 = {'upside': 30.0, 'ex_ret': 27.0}
        color = formatter._get_color_code(sell_data_3, sell_metrics_3)
        self.assertEqual(color, Color.SELL)
        
        # 2d: High short interest
        sell_data_3 = {
            'ticker': 'TEST2c',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 130.0,
            'buy_percentage': 90.0,
            'beta': 1.0,
            'pe_trailing': 15.0,
            'pe_forward': 14.0,
            'peg_ratio': 1.0,
            'short_float_pct': 6.0  # Above 5% threshold
        }
        sell_metrics_3 = {'upside': 30.0, 'ex_ret': 27.0}
        color = formatter._get_color_code(sell_data_3, sell_metrics_3)
        self.assertEqual(color, Color.SELL)
        
        # Test case 3: Buy signal (Green) - Meets all buy criteria
        buy_data = {
            'ticker': 'TEST3',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 130.0,  # 30% upside
            'buy_percentage': 90.0,  # Above 85% threshold
            'beta': 1.0,   # Below 1.25 threshold
            'pe_trailing': 15.0,
            'pe_forward': 12.0,  # PEF < PET
            'peg_ratio': 2.2,    # Below 2.5 threshold but above original 1.25
            'short_float_pct': 2.0  # Below 3% threshold
        }
        buy_metrics = {'upside': 30.0, 'ex_ret': 27.0}
        color = formatter._get_color_code(buy_data, buy_metrics)
        self.assertEqual(color, Color.BUY)
        
        # Test case 3b: Not a Buy signal - Missing PEG (should not be green)
        missing_peg_data = {
            'ticker': 'TEST3b',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 130.0,  # 30% upside
            'buy_percentage': 90.0,  # Above 85% threshold
            'beta': 1.0,   # Below 1.25 threshold
            'pe_trailing': 15.0,
            'pe_forward': 12.0,  # PEF < PET
            'peg_ratio': None,   # Missing PEG
            'short_float_pct': 2.0  # Below 3% threshold
        }
        missing_peg_metrics = {'upside': 30.0, 'ex_ret': 27.0}
        color = formatter._get_color_code(missing_peg_data, missing_peg_metrics)
        self.assertNotEqual(color, Color.BUY, "Stocks with missing PEG should not be marked as BUY")
        
        # Test case 3c: Not a Buy signal - PEG as '--' (should not be green)
        missing_peg_data2 = {
            'ticker': 'TEST3c',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 130.0,  # 30% upside
            'buy_percentage': 90.0,  # Above 85% threshold
            'beta': 1.0,   # Below 1.25 threshold
            'pe_trailing': 15.0,
            'pe_forward': 12.0,  # PEF < PET
            'peg_ratio': '--',   # Missing PEG as string
            'short_float_pct': 2.0  # Below 3% threshold
        }
        missing_peg_metrics2 = {'upside': 30.0, 'ex_ret': 27.0}
        color = formatter._get_color_code(missing_peg_data2, missing_peg_metrics2)
        self.assertNotEqual(color, Color.BUY, "Stocks with PEG='--' should not be marked as BUY")
        
        # Test case 3d: Not a Buy signal - PEF < MIN_PE_FORWARD (PEF < 0.5)
        low_pef_data = {
            'ticker': 'TEST3d',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 130.0,  # 30% upside
            'buy_percentage': 90.0,  # Above threshold
            'beta': 1.0,   # Below threshold
            'pe_trailing': 15.0,
            'pe_forward': 0.3,    # Below minimum PEF threshold of 0.5
            'peg_ratio': 1.0,
            'short_float_pct': 2.0
        }
        low_pef_metrics = {'upside': 30.0, 'ex_ret': 27.0}
        color = formatter._get_color_code(low_pef_data, low_pef_metrics)
        self.assertNotEqual(color, Color.BUY, "Stocks with PEF < 0.5 should not be marked as BUY")
        
        # Test case 4: Hold signal (Neutral) - Passes confidence but not buy or sell
        hold_data = {
            'ticker': 'TEST4',
            'analyst_count': 10,
            'total_ratings': 10,
            'price': 100.0,
            'target_price': 110.0,  # 10% upside (between 5-20%)
            'buy_percentage': 70.0,  # Between 65-85%
            'beta': 1.5,   # Above buy threshold
            'pe_trailing': 15.0,
            'pe_forward': 14.0,  # PEF < PET but other criteria not met
            'peg_ratio': 1.5,    # Between 1.25-2
            'short_float_pct': 4.0  # Between 3-5%
        }
        hold_metrics = {'upside': 10.0, 'ex_ret': 7.0}
        color = formatter._get_color_code(hold_data, hold_metrics)
        self.assertEqual(color, Color.NEUTRAL)

if __name__ == '__main__':
    unittest.main()