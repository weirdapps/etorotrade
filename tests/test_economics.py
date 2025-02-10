import unittest
from unittest.mock import Mock, patch, mock_open
import pandas as pd
from datetime import datetime
import os
from yahoofinance.economics import EconomicCalendar, format_economic_table

class TestEconomicCalendar(unittest.TestCase):
    @patch.dict(os.environ, {'FRED_API_KEY': 'test_api_key'})
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calendar = EconomicCalendar()

    def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        with patch.dict(os.environ, clear=True):
            with self.assertRaises(ValueError) as context:
                EconomicCalendar()
            self.assertIn('FRED_API_KEY not found', str(context.exception))

    def test_validate_date_format(self):
        """Test date format validation."""
        # Valid dates
        self.assertTrue(self.calendar.validate_date_format('2024-01-01'))
        self.assertTrue(self.calendar.validate_date_format('2024-12-31'))
        
        # Invalid dates
        self.assertFalse(self.calendar.validate_date_format('01-01-2024'))
        self.assertFalse(self.calendar.validate_date_format('2024/01/01'))  # Only accept hyphen format
        self.assertFalse(self.calendar.validate_date_format('invalid'))
        self.assertFalse(self.calendar.validate_date_format(''))

    @patch('requests.get')
    def test_get_releases(self, mock_get):
        """Test getting releases from FRED API."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'release_dates': [
                {'release_id': '1', 'date': '2024-01-01'},
                {'release_id': '2', 'date': '2024-01-02'}
            ]
        }
        mock_get.return_value = mock_response
        
        releases = self.calendar._get_releases('2024-01-01', '2024-01-07')
        self.assertEqual(len(releases), 2)
        
        # Verify API parameters
        mock_get.assert_called_once()
        params = mock_get.call_args[1]['params']
        self.assertEqual(params['api_key'], 'test_api_key')
        self.assertEqual(params['realtime_start'], '2024-01-01')
        self.assertEqual(params['realtime_end'], '2024-01-07')
        
        # Test API error
        mock_response.status_code = 404
        releases = self.calendar._get_releases('2024-01-01', '2024-01-07')
        self.assertEqual(releases, [])

    @patch('requests.get')
    def test_get_release_series(self, mock_get):
        """Test getting series for a specific release."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'seriess': [
                {'id': 'GDP', 'title': 'Gross Domestic Product'},
                {'id': 'UNRATE', 'title': 'Unemployment Rate'}
            ]
        }
        mock_get.return_value = mock_response
        
        series = self.calendar._get_release_series('123')
        self.assertEqual(len(series), 2)
        
        # Verify API parameters
        mock_get.assert_called_once()
        params = mock_get.call_args[1]['params']
        self.assertEqual(params['api_key'], 'test_api_key')
        self.assertEqual(params['release_id'], '123')
        
        # Test API error
        mock_response.status_code = 404
        series = self.calendar._get_release_series('123')
        self.assertEqual(series, [])

    @patch('requests.get')
    def test_get_latest_value(self, mock_get):
        """Test getting latest value for a series."""
        # Mock successful response with numeric value
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'observations': [{'value': '123.45'}]
        }
        mock_get.return_value = mock_response
        
        value = self.calendar._get_latest_value('GDP')
        self.assertEqual(value, '123.5')
        
        # Test non-numeric value
        mock_response.json.return_value = {
            'observations': [{'value': '.'}]
        }
        value = self.calendar._get_latest_value('GDP')
        self.assertEqual(value, '.')
        
        # Test empty observations
        mock_response.json.return_value = {'observations': []}
        value = self.calendar._get_latest_value('GDP')
        self.assertEqual(value, 'N/A')
        
        # Test API error
        mock_response.status_code = 404
        value = self.calendar._get_latest_value('GDP')
        self.assertEqual(value, 'N/A')

    @patch('yahoofinance.economics.EconomicCalendar._get_latest_value')
    @patch('yahoofinance.economics.EconomicCalendar._get_release_series')
    @patch('yahoofinance.economics.EconomicCalendar._get_releases')
    def test_get_economic_calendar(self, mock_releases, mock_series, mock_value):
        """Test getting economic calendar."""
        # Mock dependencies
        mock_releases.return_value = [
            {'release_id': '1', 'date': '2024-01-01'}
        ]
        mock_series.return_value = [
            {'id': 'GDP', 'title': 'Gross Domestic Product'}
        ]
        mock_value.return_value = '123.45'
        
        # Test successful retrieval
        df = self.calendar.get_economic_calendar('2024-01-01', '2024-01-07')
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        
        # Test invalid date format
        df_invalid = self.calendar.get_economic_calendar('invalid', '2024-01-07')
        self.assertIsNone(df_invalid)
        
        # Test no events found
        mock_releases.return_value = []
        df_empty = self.calendar.get_economic_calendar('2024-01-01', '2024-01-07')
        self.assertIsNone(df_empty)
        
        # Test API error
        mock_releases.side_effect = Exception("API Error")
        df_error = self.calendar.get_economic_calendar('2024-01-01', '2024-01-07')
        self.assertIsNone(df_error)

    def test_indicators_structure(self):
        """Test economic indicators structure."""
        # Verify we have indicators from different categories
        self.assertIn('GDP', self.calendar.indicators)           # Growth
        self.assertIn('CPI', self.calendar.indicators)          # Inflation
        self.assertIn('Nonfarm Payrolls', self.calendar.indicators)  # Employment
        
        # Verify indicator structure
        for name, info in self.calendar.indicators.items():
            self.assertIn('id', info)
            self.assertIn('impact', info)
            self.assertIn('description', info)
            self.assertIn(info['impact'], ['High', 'Medium'])

class TestEconomicTableFormatting(unittest.TestCase):
    def test_format_economic_table(self):
        """Test economic table formatting."""
        df = pd.DataFrame({
            'Event': ['GDP', 'CPI'],
            'Impact': ['High', 'High'],
            'Date': ['2024-01-01', '2024-01-02'],
            'Actual': ['123.45', 'N/A'],
            'Previous': ['122.34', '2.5']
        })
        
        # Test with valid data
        with patch('builtins.print') as mock_print:
            format_economic_table(df, '2024-01-01', '2024-01-07')
            self.assertTrue(mock_print.called)
        
        # Test with empty DataFrame
        with patch('builtins.print') as mock_print:
            format_economic_table(pd.DataFrame(), '2024-01-01', '2024-01-07')
            self.assertFalse(mock_print.called)

if __name__ == '__main__':
    unittest.main()