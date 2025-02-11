import unittest
from unittest.mock import patch, Mock
from datetime import datetime
import pandas as pd
from yahoofinance.utils import DateUtils, FormatUtils

class TestDateUtils(unittest.TestCase):
    def test_clean_date_string(self):
        """Test cleaning date strings."""
        test_cases = [
            ('2024-01-01', '2024-01-01'),  # Already clean
            ('2024/01/01', '2024-01-01'),  # Convert slashes to hyphens
            ('2024.01.01', '2024-01-01'),  # Convert dots to hyphens
            ('2024 01 01', '2024-01-01'),  # Convert spaces to hyphens
            ('abc2024-01-01xyz', '2024-01-01'),  # Remove non-numeric/non-hyphen
            ('', ''),  # Empty string
            ('abc', '')  # No numbers
        ]
        
        for input_str, expected in test_cases:
            with self.subTest(input_str=input_str):
                result = DateUtils.clean_date_string(input_str)
                self.assertEqual(result, expected)

    def test_parse_date(self):
        """Test parsing date strings."""
        # Valid dates
        self.assertEqual(
            DateUtils.parse_date('2024-01-01'),
            datetime(2024, 1, 1)
        )
        self.assertEqual(
            DateUtils.parse_date('2024/01/01'),
            datetime(2024, 1, 1)
        )
        
        # Invalid dates
        self.assertIsNone(DateUtils.parse_date('2024-13-01'))  # Invalid month
        self.assertIsNone(DateUtils.parse_date('2024-01-32'))  # Invalid day
        self.assertIsNone(DateUtils.parse_date('invalid'))     # Not a date
        self.assertIsNone(DateUtils.parse_date(''))           # Empty string

    def test_validate_date_format(self):
        """Test date format validation."""
        # Valid formats
        self.assertTrue(DateUtils.validate_date_format('2024-01-01'))
        self.assertTrue(DateUtils.validate_date_format('2024/01/01'))
        
        # Invalid formats
        self.assertFalse(DateUtils.validate_date_format('01-01-2024'))
        self.assertFalse(DateUtils.validate_date_format('2024-1-1'))
        self.assertFalse(DateUtils.validate_date_format('2024-13-01'))
        self.assertFalse(DateUtils.validate_date_format('invalid'))
        self.assertFalse(DateUtils.validate_date_format(''))

    @patch('builtins.input')
    def test_get_user_dates(self, mock_input):
        """Test getting user dates with various scenarios."""
        today = datetime.now().strftime(DateUtils.DATE_FORMAT)
        default_end = (datetime.strptime(today, DateUtils.DATE_FORMAT) + 
                      pd.Timedelta(days=7)).strftime(DateUtils.DATE_FORMAT)
        
        # Test with default dates (empty inputs)
        mock_input.side_effect = ['', '']
        start_date, end_date = DateUtils.get_user_dates()
        self.assertEqual(start_date, today)
        self.assertEqual(end_date, default_end)
        
        # Test with custom valid dates
        mock_input.side_effect = ['2024-01-01', '2024-01-07']
        start_date, end_date = DateUtils.get_user_dates()
        self.assertEqual(start_date, '2024-01-01')
        self.assertEqual(end_date, '2024-01-07')
        
        # Test with invalid then valid dates
        mock_input.side_effect = [
            'invalid', '2024-01-01',  # Invalid then valid start date
            '2023-12-31', '2024-01-07'  # Invalid (before start) then valid end date
        ]
        start_date, end_date = DateUtils.get_user_dates()
        self.assertEqual(start_date, '2024-01-01')
        self.assertEqual(end_date, '2024-01-07')

class TestFormatUtils(unittest.TestCase):
    def test_format_number(self):
        """Test number formatting with K/M suffixes."""
        test_cases = [
            (1234567, '1.2M'),      # Millions
            (1234, '1.2K'),         # Thousands
            (123.456, '123.5'),     # Regular number
            (0, '0.0'),             # Zero
            (-1234567, '-1.2M'),    # Negative millions
            (-1234, '-1.2K'),       # Negative thousands
            (None, 'N/A'),          # None value
            ('invalid', 'N/A')      # Invalid input
        ]
        
        for value, expected in test_cases:
            with self.subTest(value=value):
                result = FormatUtils.format_number(value)
                self.assertEqual(result, expected)

    def test_format_percentage(self):
        """Test percentage formatting."""
        test_cases = [
            (0.1234, '+12.34%'),     # Positive with sign
            (-0.1234, '-12.34%'),    # Negative
            (0, '0.00%'),            # Zero
            (None, 'N/A'),           # None value
            (float('nan'), 'N/A'),   # NaN
            ('invalid', 'N/A')       # Invalid input
        ]
        
        for value, expected in test_cases:
            with self.subTest(value=value):
                result = FormatUtils.format_percentage(value)
                self.assertEqual(result, expected)
                
        # Test without sign
        self.assertEqual(FormatUtils.format_percentage(0.1234, include_sign=False), '12.34%')

    def test_format_market_metrics(self):
        """Test market metrics formatting."""
        metrics = {
            'metric1': {'value': 0.1234, 'label': 'Metric 1', 'is_percentage': True},
            'metric2': {'value': 1234567, 'label': 'Metric 2', 'is_percentage': False},
            'metric3': {'value': 'text', 'label': 'Metric 3'},
            'metric4': {'value': None, 'label': 'Metric 4'}
        }
        
        formatted = FormatUtils.format_market_metrics(metrics)
        
        self.assertEqual(len(formatted), 4)
        self.assertEqual(formatted[0]['value'], '+12.34%')
        self.assertEqual(formatted[1]['value'], '1.2M')
        self.assertEqual(formatted[2]['value'], 'text')
        self.assertIsNone(formatted[3]['value'])

    def test_format_table(self):
        """Test table formatting."""
        df = pd.DataFrame({
            'A': [1, 2],
            'B': ['x', 'y']
        })
        headers = ['Col A', 'Col B']
        alignments = ('right', 'left')
        
        # Test with valid data
        with patch('builtins.print') as mock_print:
            FormatUtils.format_table(df, 'Test Title', '2024-01-01', '2024-01-07',
                                   headers, alignments)
            self.assertTrue(mock_print.called)
        
        # Test with empty DataFrame
        with patch('builtins.print') as mock_print:
            FormatUtils.format_table(pd.DataFrame(), 'Test Title', '2024-01-01',
                                   '2024-01-07', headers, alignments)
            self.assertFalse(mock_print.called)

    def test_generate_market_html(self):
        """Test HTML generation for market display."""
        sections = [{
            'title': 'Test Section',
            'metrics': [
                {'id': 'metric1', 'value': '10%', 'label': 'Metric 1'},
                {'id': 'metric2', 'value': '1.2M', 'label': 'Metric 2'}
            ],
            'columns': 2,
            'width': '500px'
        }]
        
        html = FormatUtils.generate_market_html('Test Title', sections)
        
        self.assertIsInstance(html, str)
        self.assertIn('Test Title', html)
        self.assertIn('Test Section', html)
        self.assertIn('Metric 1', html)
        self.assertIn('10%', html)
        self.assertIn('500px', html)

if __name__ == '__main__':
    unittest.main()