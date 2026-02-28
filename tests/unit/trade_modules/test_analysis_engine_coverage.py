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
        # Note: calculate_action_vectorized now returns (actions, buy_scores) tuple
        result_tuple = calculate_action_vectorized(self.mock_df, "market")
        self.assertIsInstance(result_tuple, tuple)
        result, buy_scores = result_tuple
        self.assertIsInstance(result, pd.Series)
        self.assertIsInstance(buy_scores, pd.Series)
        self.assertTrue(all(action in ['B', 'S', 'H', 'I'] for action in result))

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result_empty, _ = calculate_action_vectorized(empty_df, "market")
        self.assertEqual(len(result_empty), 0)

        # Test with single row
        single_row = self.mock_df.iloc[:1].copy()
        result_single, _ = calculate_action_vectorized(single_row, "market")
        self.assertEqual(len(result_single), 1)

        # Test with all NaN values
        nan_df = self.mock_df.copy()
        for col in ['pe_forward', 'pe_trailing', 'peg_ratio', 'upside', 'buy_percentage']:
            if col in nan_df.columns:
                nan_df[col] = np.nan
        result_nan, _ = calculate_action_vectorized(nan_df, "market")
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
        result, _ = calculate_action_vectorized(bad_df, "market")
        self.assertEqual(len(result), len(bad_df))

        # Test with mixed data types
        mixed_df = self.mock_df.copy()
        mixed_df['pe_forward'] = mixed_df['pe_forward'].astype(str)

        result_mixed, _ = calculate_action_vectorized(mixed_df, "market")
        self.assertEqual(len(result_mixed), len(mixed_df))
        
    @unittest.skipIf(
        os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true',
        "Performance test skipped in CI - runner performance is too variable"
    )
    def test_vectorization_performance(self):
        """Test that vectorized operations handle large datasets efficiently."""
        large_df = MockYahooFinanceResponses.create_mock_dataframe(1000)

        import time
        start_time = time.time()
        result, _ = calculate_action_vectorized(large_df, "market")
        end_time = time.time()

        # Should complete in reasonable time (< 30 seconds for 1000 rows)
        # CI environments and full test suite runs can be slower than isolated runs
        # Local runs typically complete in <1 second
        self.assertLess(end_time - start_time, 30.0)
        self.assertEqual(len(result), 1000)
        
    def test_data_type_conversions(self):
        """Test proper data type handling and conversions."""
        # Test with string numbers
        string_df = self.mock_df.copy()
        string_df['upside'] = string_df['upside'].astype(str)
        string_df['buy_percentage'] = string_df['buy_percentage'].astype(str)

        result, _ = calculate_action_vectorized(string_df, "market")
        self.assertEqual(len(result), len(string_df))

        # Test with percentage strings
        perc_df = self.mock_df.copy()
        perc_df['short_percent'] = perc_df['short_percent'].apply(lambda x: f"{x}%" if pd.notna(x) else x)

        result_perc, _ = calculate_action_vectorized(perc_df, "market")
        self.assertEqual(len(result_perc), len(perc_df))

    def test_negative_upside_never_buy_safety_check(self):
        """
        CRITICAL TEST: Stocks with negative upside should NEVER be marked as BUY.

        This is a safety check added after review identified that stocks with
        negative upside were incorrectly appearing in buy.csv output.
        """
        # Create test data with negative upside stocks that have all other metrics favorable
        test_data = pd.DataFrame({
            'ticker': ['NEG_HIGH_CONSENSUS', 'NEG_LOW_CONSENSUS', 'ZERO_UPSIDE', 'POS_UPSIDE'],
            'upside': [-20.0, -5.0, 0.0, 25.0],  # Various negative and zero upside
            'buy_percentage': [95.0, 90.0, 85.0, 85.0],  # High buy consensus
            'analyst_count': [15, 10, 10, 10],  # Sufficient coverage
            'total_ratings': [15, 10, 10, 10],
            'market_cap': [100e9, 100e9, 100e9, 100e9],  # LARGE tier
            'pe_forward': [18.0, 18.0, 18.0, 18.0],  # Reasonable PE
            'pe_trailing': [20.0, 20.0, 20.0, 20.0],
            'beta': [1.0, 1.0, 1.0, 1.0],
            'pct_from_52w_high': [90.0, 90.0, 90.0, 90.0],  # Near highs
            'return_on_equity': [15.0, 15.0, 15.0, 15.0],  # Good ROE
            'debt_to_equity': [50.0, 50.0, 50.0, 50.0],  # Low debt
        })
        test_data.index = test_data['ticker']

        # Calculate actions - now returns tuple (actions, buy_scores)
        actions, _ = calculate_action_vectorized(test_data, "market")

        # CRITICAL ASSERTIONS: Negative upside MUST NOT be BUY
        self.assertNotEqual(actions['NEG_HIGH_CONSENSUS'], 'B',
            "SAFETY FAILURE: -20% upside marked as BUY despite high consensus")
        self.assertNotEqual(actions['NEG_LOW_CONSENSUS'], 'B',
            "SAFETY FAILURE: -5% upside marked as BUY")

        # Zero upside should also not be BUY (no profit potential)
        self.assertNotEqual(actions['ZERO_UPSIDE'], 'B',
            "Zero upside marked as BUY - no profit potential")

        # Positive upside with good metrics SHOULD be able to be BUY
        # (depending on thresholds, may or may not pass all criteria)
        # This is not a failure if it's H or S, just verify no error
        self.assertIn(actions['POS_UPSIDE'], ['B', 'S', 'H', 'I'])


if __name__ == '__main__':
    unittest.main()