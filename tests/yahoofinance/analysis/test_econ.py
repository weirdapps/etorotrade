import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta
from yahoofinance.analysis.market import (
    get_fred_api_key,
    get_date_input,
    get_default_dates,
    fetch_fred_data,
    format_value,
    calculate_change,
    fetch_economic_data,
    INDICATORS
)

class TestEconomicData(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api_key = "test_api_key"
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"

    def test_get_fred_api_key(self):
        """Test FRED API key retrieval."""
        with patch('os.getenv', return_value='test_key'):
            key = get_fred_api_key()
            self.assertEqual(key, 'test_key')

        with patch('os.getenv', return_value=None):
            with self.assertRaises(SystemExit):
                get_fred_api_key()

    def test_get_date_input(self):
        """Test date input handling."""
        default_date = "2024-01-01"
        
        # Test with valid input
        with patch('builtins.input', return_value="2024-02-01"):
            result = get_date_input("Enter date:", default_date)
            self.assertEqual(result, "2024-02-01")
        
        # Test with empty input (use default)
        with patch('builtins.input', return_value=""):
            result = get_date_input("Enter date:", default_date)
            self.assertEqual(result, default_date)
        
        # Test with invalid input
        with patch('builtins.input', return_value="invalid"):
            result = get_date_input("Enter date:", default_date)
            self.assertEqual(result, default_date)

    def test_get_default_dates(self):
        """Test default date range calculation."""
        mock_now = datetime(2024, 2, 1)
        with patch('yahoofinance.analysis.market.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.strftime = datetime.strftime
            mock_datetime.strptime = datetime.strptime
            mock_datetime.timedelta = timedelta
            
            start_date, end_date = get_default_dates()
            expected_start = (mock_now - timedelta(days=30)).strftime("%Y-%m-%d")
            expected_end = mock_now.strftime("%Y-%m-%d")
            
            self.assertEqual(start_date, expected_start)
            self.assertEqual(end_date, expected_end)

    @patch('requests.get')
    def test_fetch_fred_data(self, mock_get):
        """Test FRED API data fetching."""
        # Mock successful response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'observations': [
                {'date': '2024-01-01', 'value': '100.0'},
                {'date': '2024-01-02', 'value': '101.0'}
            ]
        }
        
        result = fetch_fred_data(
            self.api_key, 'GDP', self.start_date, self.end_date, 'quarterly'
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['value'], '100.0')
        
        # Test error handling
        mock_get.side_effect = Exception("API Error")
        result = fetch_fred_data(
            self.api_key, 'GDP', self.start_date, self.end_date, 'quarterly'
        )
        self.assertEqual(result, [])

    def test_format_value(self):
        """Test value formatting."""
        test_cases = [
            # (value, indicator, expected)
            ("1000", "Initial Claims (K)", "1000K"),  # Value already scaled in INDICATORS
            ("1.5", "Nonfarm Payrolls (M)", "1.5M"),  # Value already scaled in INDICATORS
            ("0.035", "GDP Growth (%)", "3.5%"),      # Value already scaled
            ("-50.5", "Trade Balance ($B)", "$-50.5B"),
            ("103.5", "Industrial Production", "103.5"),
            ("750", "Housing Starts (K)", "750K"),    # Value already in thousands
            ("", "Any Indicator", "N/A"),
        ]
        
        for value, indicator, expected in test_cases:
            result = format_value(value, indicator)
            self.assertEqual(result, expected)

    def test_calculate_change(self):
        """Test percentage change calculation."""
        test_cases = [
            # (current, previous, expected)
            ("105", "100", "+5.0%"),
            ("95", "100", "-5.0%"),
            ("100", "100", "0.0%"),
            ("invalid", "100", ""),
            ("100", "invalid", ""),
            ("", "", ""),
        ]
        
        for current, previous, expected in test_cases:
            result = calculate_change(current, previous)
            self.assertEqual(result, expected)

    @patch('yahoofinance.analysis.market.fetch_fred_data')
    def test_fetch_economic_data(self, mock_fetch):
        """Test economic data fetching."""
        # Mock successful data fetch
        mock_fetch.return_value = [
            {'date': '2024-01-01', 'value': '100.0'},
            {'date': '2024-01-02', 'value': '101.0'}
        ]
        
        result = fetch_economic_data(
            self.api_key, self.start_date, self.end_date
        )
        self.assertGreater(len(result), 0)
        self.assertIn('Date', result[0])
        self.assertIn('Indicator', result[0])
        self.assertIn('Value', result[0])
        self.assertIn('Change', result[0])
        
        # Test with empty data
        mock_fetch.return_value = []
        result = fetch_economic_data(
            self.api_key, self.start_date, self.end_date
        )
        self.assertEqual(result, [])

    def test_indicators_structure(self):
        """Test economic indicators structure."""
        required_fields = ['id', 'freq', 'scale']
        required_indicators = [
            'GDP Growth (%)',
            'Unemployment (%)',
            'CPI MoM (%)',
            'Fed Funds Rate (%)',
            'Industrial Production',
            'Retail Sales MoM (%)',
            'Housing Starts (K)',
            'Nonfarm Payrolls (M)',
            'Trade Balance ($B)',
            'Initial Claims (K)'
        ]
        
        # Verify all required indicators are present
        for indicator in required_indicators:
            self.assertIn(indicator, INDICATORS)
        
        # Verify indicator structure
        for name, details in INDICATORS.items():
            for field in required_fields:
                self.assertIn(field, details)
            self.assertIn(details['freq'], ['weekly', 'monthly', 'quarterly'])
            self.assertTrue(callable(details['scale']))

if __name__ == '__main__':
    unittest.main()