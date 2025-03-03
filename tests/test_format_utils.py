import unittest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from yahoofinance.utils.format_utils import FormatUtils

class TestFormatUtils(unittest.TestCase):
    """Test formatting utilities for market data and HTML output."""
    
    def test_format_number(self):
        """Test numeric value formatting with various inputs."""
        # Test with normal floats
        self.assertEqual(FormatUtils.format_number(123.456, precision=2), '123.46')
        self.assertEqual(FormatUtils.format_number(123.456, precision=0), '123')
        self.assertEqual(FormatUtils.format_number(123.456, precision=4), '123.4560')
        
        # Test with negative numbers
        self.assertEqual(FormatUtils.format_number(-123.456, precision=2), '-123.46')
        
        # Test with zero
        self.assertEqual(FormatUtils.format_number(0, precision=2), '0.00')
        
        # Test with None and NaN
        self.assertEqual(FormatUtils.format_number(None), 'N/A')
        import numpy as np
        self.assertEqual(FormatUtils.format_number(np.nan), 'N/A')
        self.assertEqual(FormatUtils.format_number(float('nan')), 'N/A')
        
        # Test with strings
        self.assertEqual(FormatUtils.format_number('abc'), 'abc')
        
        # Test with very large and very small numbers
        self.assertEqual(FormatUtils.format_number(1e10, precision=2), '10000000000.00')
        self.assertEqual(FormatUtils.format_number(1e-5, precision=6), '0.000010')
    
    def test_format_market_metrics(self):
        """Test formatting of market metrics dictionary."""
        metrics = {
            'price': {
                'value': 150.25,
                'label': 'Price',
                'is_percentage': False
            },
            'change': {
                'value': 2.5,
                'label': 'Change',
                'is_percentage': True
            },
            'pe_ratio': {
                'value': 25.75,
                'label': 'P/E Ratio',
                'is_percentage': False
            },
            'negative': {
                'value': -3.2,
                'label': 'Negative Metric',
                'is_percentage': True
            }
        }
        
        formatted = FormatUtils.format_market_metrics(metrics)
        
        # Check the structure of the result
        self.assertIsInstance(formatted, list)
        self.assertEqual(len(formatted), 4)
        
        # Verify each metric was formatted correctly
        for metric in formatted:
            self.assertIn('key', metric)
            self.assertIn('label', metric)
            self.assertIn('value', metric)
            self.assertIn('formatted_value', metric)
            self.assertIn('color', metric)
            self.assertIn('is_percentage', metric)
        
        # Find metrics by key to verify specific formatting
        price_metric = next(m for m in formatted if m['key'] == 'price')
        change_metric = next(m for m in formatted if m['key'] == 'change')
        negative_metric = next(m for m in formatted if m['key'] == 'negative')
        
        # Check values and formatting
        self.assertEqual(price_metric['formatted_value'], '150.25')
        self.assertEqual(price_metric['color'], 'normal')
        
        self.assertEqual(change_metric['formatted_value'], '2.5%')
        self.assertEqual(change_metric['color'], 'positive')
        
        self.assertEqual(negative_metric['formatted_value'], '-3.2%')
        self.assertEqual(negative_metric['color'], 'negative')
        
        # Test with invalid inputs
        invalid_metrics = {
            'invalid1': 'not a dict',
            'invalid2': {'no_value': 'missing value'},
            'valid': {'value': 100, 'label': 'Valid'}
        }
        formatted_invalid = FormatUtils.format_market_metrics(invalid_metrics)
        self.assertEqual(len(formatted_invalid), 1)  # Only the valid entry should be included
    
    def test_format_for_csv(self):
        """Test formatting of metrics for CSV output."""
        # Test with various metric types
        metrics = {
            'small_number': 0.0005,    # Very small number, should be rounded to 0
            'large_number': 12345.678, # Large number, should be rounded to 2 decimals
            'huge_number': 1000000,    # Very large number, should be rounded to 0 decimals
            'string_value': 'test',    # String should be preserved
            'none_value': None,        # None should be preserved
            'boolean': True,           # Boolean should be preserved
        }
        
        formatted = FormatUtils.format_for_csv(metrics)
        
        # Verify formatting rules
        self.assertEqual(formatted['small_number'], 0.0)
        self.assertEqual(formatted['large_number'], 12345.68)
        self.assertEqual(formatted['huge_number'], 1000000.0)
        self.assertEqual(formatted['string_value'], 'test')
        self.assertIsNone(formatted['none_value'])
        self.assertTrue(formatted['boolean'])
    
    def test_generate_market_html(self):
        """Test generation of HTML content for market metrics."""
        # Set up test input
        title = "Test Market Dashboard"
        sections = [
            {
                'title': 'Market Overview',
                'metrics': [
                    {'key': 'market_cap', 'label': 'Market Cap', 'formatted_value': '$2.5T', 'color': 'normal'},
                    {'key': 'pe_ratio', 'label': 'P/E Ratio', 'formatted_value': '25.5', 'color': 'normal'},
                    {'key': 'change', 'label': 'Change', 'formatted_value': '2.3%', 'color': 'positive'},
                    {'key': 'loss', 'label': 'Loss', 'formatted_value': '-1.2%', 'color': 'negative'}
                ],
                'columns': 2,
                'width': '70%'
            }
        ]
        
        # Generate HTML
        html = FormatUtils.generate_market_html(title, sections)
        
        # Verify basic structure
        self.assertIsInstance(html, str)
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<title>Test Market Dashboard</title>', html)
        
        # Verify sections are included
        self.assertIn('Market Overview', html)
        self.assertIn('Market Cap', html)
        self.assertIn('$2.5T', html)
        
        # Verify CSS classes for colors
        self.assertIn('class="metric-value positive"', html)
        self.assertIn('class="metric-value negative"', html)
        self.assertIn('class="metric-value normal"', html)
        
        # Test error handling
        with patch('logging.getLogger') as mock_logger:
            mock_logger.return_value = MagicMock()
            # Test with an exception during HTML generation
            with patch.object(FormatUtils, '_format_section', side_effect=Exception("Test error")):
                error_html = FormatUtils.generate_market_html(title, sections)
                self.assertIn("<h1>Error</h1>", error_html)