#!/usr/bin/env python3
"""
Comprehensive test coverage for analysis_engine.py
Focuses on improving coverage from 9% to target 90%+
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from trade_modules.analysis_engine import (
    calculate_action_vectorized,
    process_buy_opportunities,
    _filter_notrade_tickers
)
from tests.fixtures.mock_api_responses import MockYahooFinanceResponses, patch_yahoo_finance_api


class TestAnalysisEngineComprehensive(unittest.TestCase):
    """Comprehensive test coverage for analysis engine functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_df = MockYahooFinanceResponses.create_mock_dataframe(50)
        self.test_portfolio = ['STOCK0001', 'STOCK0002', 'STOCK0003']
        
    def test_calculate_action_vectorized_all_scenarios(self):
        """Test action calculation with all possible scenarios."""
        # Test with valid data
        result = calculate_action_vectorized(self.mock_df, "market")
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(all(action in ['B', 'S', 'H', 'I'] for action in result))
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result_empty = calculate_action_vectorized(empty_df, "market")
        self.assertEqual(len(result_empty), 0)
        
        # Test with single row
        single_row = self.mock_df.iloc[:1].copy()
        result_single = calculate_action_vectorized(single_row, "market")
        self.assertEqual(len(result_single), 1)
        
        # Test with all NaN values
        nan_df = self.mock_df.copy()
        for col in ['pe_forward', 'pe_trailing', 'peg_ratio', 'upside', 'buy_percentage']:
            if col in nan_df.columns:
                nan_df[col] = np.nan
        result_nan = calculate_action_vectorized(nan_df, "market")
        self.assertEqual(len(result_nan), len(nan_df))
        
        
        
    def test_filter_notrade_tickers(self):
        """Test no-trade ticker filtering functionality."""
        # Create test DataFrames
        opportunities = self.mock_df.copy()
        # Set index to symbol names for filtering
        opportunities.index = [f'STOCK{i:04d}' for i in range(len(opportunities))]
        
        # Create a temporary notrade CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('symbol\n')
            f.write('STOCK0001\n')
            f.write('STOCK0003\n')
            notrade_path = f.name
        
        try:
            filtered = _filter_notrade_tickers(opportunities, notrade_path)
            
            # Verify no-trade tickers are filtered out
            remaining_indices = filtered.index.tolist()
            self.assertNotIn('STOCK0001', remaining_indices)
            self.assertNotIn('STOCK0003', remaining_indices)
            
            # Verify some tickers remain
            self.assertIn('STOCK0000', remaining_indices)
            self.assertIn('STOCK0002', remaining_indices)
            
            # Test with non-existent file (should return original)
            filtered_no_file = _filter_notrade_tickers(opportunities, 'nonexistent.csv')
            self.assertEqual(len(filtered_no_file), len(opportunities))
        finally:
            # Clean up temp file
            os.unlink(notrade_path)
        
    @patch('pandas.read_csv')
    @patch('trade_modules.analysis_engine._filter_notrade_tickers')
    def test_process_buy_opportunities(self, mock_filter, mock_read_csv):
        """Test buy opportunities processing."""
        mock_read_csv.return_value = pd.DataFrame({'ticker': ['NO_TRADE_1']})
        mock_filter.return_value = self.mock_df.copy()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with valid inputs
            result = process_buy_opportunities(
                market_df=self.mock_df,
                portfolio_tickers=self.test_portfolio,
                output_dir=temp_dir,
                notrade_path="mock_notrade.csv",
                provider=Mock()
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            
            # Test with non-existent notrade file
            result_no_notrade = process_buy_opportunities(
                market_df=self.mock_df,
                portfolio_tickers=self.test_portfolio,
                output_dir=temp_dir,
                notrade_path="non_existent.csv",
                provider=Mock()
            )
            
            self.assertIsInstance(result_no_notrade, pd.DataFrame)
            
            
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with malformed data
        bad_df = pd.DataFrame({
            'ticker': ['TEST'],
            'invalid_column': ['invalid_data']
        })
        
        # Should handle missing columns gracefully
        result = calculate_action_vectorized(bad_df, "market")
        self.assertEqual(len(result), len(bad_df))
        
        # Test with mixed data types
        mixed_df = self.mock_df.copy()
        mixed_df['pe_forward'] = mixed_df['pe_forward'].astype(str)
        
        result_mixed = calculate_action_vectorized(mixed_df, "market")
        self.assertEqual(len(result_mixed), len(mixed_df))
        
    def test_vectorization_performance(self):
        """Test that vectorized operations handle large datasets efficiently."""
        large_df = MockYahooFinanceResponses.create_mock_dataframe(1000)
        
        import time
        start_time = time.time()
        result = calculate_action_vectorized(large_df, "market")
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second for 1000 rows)
        self.assertLess(end_time - start_time, 1.0)
        self.assertEqual(len(result), 1000)
        
    def test_data_type_conversions(self):
        """Test proper data type handling and conversions."""
        # Test with string numbers
        string_df = self.mock_df.copy()
        string_df['upside'] = string_df['upside'].astype(str)
        string_df['buy_percentage'] = string_df['buy_percentage'].astype(str)
        
        result = calculate_action_vectorized(string_df, "market")
        self.assertEqual(len(result), len(string_df))
        
        # Test with percentage strings
        perc_df = self.mock_df.copy()
        perc_df['short_percent'] = perc_df['short_percent'].apply(lambda x: f"{x}%" if pd.notna(x) else x)
        
        result_perc = calculate_action_vectorized(perc_df, "market")
        self.assertEqual(len(result_perc), len(perc_df))


if __name__ == '__main__':
    unittest.main()