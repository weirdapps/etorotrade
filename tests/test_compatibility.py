import unittest
import pandas as pd
import datetime
from unittest.mock import patch
from io import StringIO
from yahoofinance.utils import DateUtils, FormatUtils
from yahoofinance import utils

class TestCompatibilityLayer(unittest.TestCase):
    """Test the compatibility layer that maintains backward compatibility."""
    
    def test_date_utils_validation(self):
        """Test DateUtils validate_date_format method."""
        # Valid date formats
        self.assertTrue(DateUtils.validate_date_format('2024-01-01'))
        self.assertTrue(DateUtils.validate_date_format('2023-12-31'))
        
        # Invalid date formats
        self.assertFalse(DateUtils.validate_date_format('01-01-2024'))  # Wrong format
        self.assertFalse(DateUtils.validate_date_format('2024/01/01'))  # Using slashes
        self.assertFalse(DateUtils.validate_date_format('2024.01.01'))  # Using dots
        self.assertFalse(DateUtils.validate_date_format('abcdef'))      # Not a date
        self.assertFalse(DateUtils.validate_date_format(''))            # Empty string
        self.assertFalse(DateUtils.validate_date_format('2024-13-01'))  # Invalid month
        self.assertFalse(DateUtils.validate_date_format('2024-01-32'))  # Invalid day
    
    @patch('builtins.input')
    def test_date_utils_get_user_dates(self, mock_input):
        """Test DateUtils get_user_dates method."""
        # Test with user input
        mock_input.side_effect = ['2024-02-01', '2024-02-07']
        start_date, end_date = DateUtils.get_user_dates()
        self.assertEqual(start_date, '2024-02-01')
        self.assertEqual(end_date, '2024-02-07')
        
        # Test with empty inputs (using defaults)
        today = datetime.datetime.now()
        default_start = today.strftime('%Y-%m-%d')
        default_end = (today + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        
        mock_input.side_effect = ['', '']
        start_date, end_date = DateUtils.get_user_dates()
        self.assertEqual(start_date, default_start)
        self.assertEqual(end_date, default_end)
    
    def test_format_utils_number_formatting(self):
        """Test FormatUtils format_number method."""
        # Test with various inputs
        self.assertEqual(FormatUtils.format_number(1.234, precision=2), '1.23')
        self.assertEqual(FormatUtils.format_number(0, precision=2), '0.00')
        self.assertEqual(FormatUtils.format_number(-1.234, precision=2), '-1.23')
        self.assertEqual(FormatUtils.format_number(None), 'N/A')
        self.assertEqual(FormatUtils.format_number(float('nan')), 'N/A')
        
        # Test with different precision
        self.assertEqual(FormatUtils.format_number(1.234, precision=3), '1.234')
        self.assertEqual(FormatUtils.format_number(1.234, precision=0), '1')
    
    def test_format_utils_table_formatting(self):
        """Test FormatUtils format_table method."""
        # Create a test DataFrame
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT'],
            'Price': [150.0, 300.0],
            'Change': [1.5, -2.0]
        })
        
        # Mock print function to capture output
        with patch('builtins.print') as mock_print:
            FormatUtils.format_table(df, title="Test Table")
            # Check that print was called with appropriate arguments
            self.assertTrue(mock_print.called)
            
            # Mock print for version with headers and alignments
            mock_print.reset_mock()
            headers = ['Symbol', 'Price', 'Change']
            alignments = ('left', 'right', 'right')
            FormatUtils.format_table(
                df, title="Test Table", 
                start_date="2024-01-01", end_date="2024-01-07",
                headers=headers, alignments=alignments
            )
            # Check that print was called with title and period
            self.assertTrue(mock_print.called)
            
            # Test with empty DataFrame
            mock_print.reset_mock()
            FormatUtils.format_table(pd.DataFrame(), title="Empty Table")
            # Should not print the table if DataFrame is empty
            self.assertFalse(mock_print.called)