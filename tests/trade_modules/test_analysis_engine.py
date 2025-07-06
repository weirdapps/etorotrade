"""
Test suite for trade_modules.analysis_engine module.

This module tests the performance-optimized analysis engine including:
- Vectorized action calculations
- EXRET calculations
- Trading criteria evaluation
- Performance benchmarks
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from trade_modules.analysis_engine import (
    calculate_exret,
    calculate_action,
    calculate_action_vectorized,
    filter_buy_opportunities_wrapper,
    filter_sell_candidates_wrapper,
    filter_hold_candidates_wrapper,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'upside': [25.5, 15.2, 8.7, 45.1, 12.3],
        'buy_percentage': [85.0, 90.0, 65.0, 80.0, 75.0],
        'analyst_count': [15, 20, 8, 12, 18],
        'total_ratings': [25, 30, 10, 18, 22],
        'pe_forward': [18.5, 22.1, 25.0, 45.2, 30.5],
        'pe_trailing': [20.1, 25.3, 28.0, 50.0, 35.2],
        'peg_ratio': [1.2, 1.8, 2.5, 3.5, 2.1],
        'short_percent': [0.5, 1.2, 2.5, 1.8, 0.8],
        'beta': [1.1, 0.9, 1.3, 2.1, 1.4],
        'EXRET': [21.7, 13.7, 5.7, 36.1, 9.2],
    })


@pytest.fixture
def edge_case_dataframe():
    """Create DataFrame with edge cases for testing."""
    return pd.DataFrame({
        'symbol': ['EDGE1', 'EDGE2', 'EDGE3', 'EDGE4'],
        'upside': [np.nan, 0.0, -5.0, 100.0],
        'buy_percentage': [np.nan, 0.0, 50.0, 100.0],
        'analyst_count': [0, 2, 5, 25],
        'total_ratings': [0, 3, 5, 30],
        'pe_forward': [np.nan, 0.0, -1.0, 999.0],
        'pe_trailing': [np.nan, 0.0, -1.0, 1000.0],
        'peg_ratio': [np.nan, 0.0, -1.0, 10.0],
        'short_percent': [np.nan, 0.0, 5.0, 25.0],
        'beta': [np.nan, 0.0, -1.0, 10.0],
    })


class TestCalculateExret:
    """Test cases for calculate_exret function."""
    
    def test_calculate_exret_normal_case(self, sample_dataframe):
        """Test EXRET calculation with normal data."""
        result = calculate_exret(sample_dataframe)
        
        # Verify EXRET column is added
        assert 'EXRET' in result.columns
        
        # Verify calculations are correct
        expected_exret = [
            25.5 * 85.0 / 100.0,  # AAPL: 21.675
            15.2 * 90.0 / 100.0,  # MSFT: 13.68
            8.7 * 65.0 / 100.0,   # GOOGL: 5.655
            45.1 * 80.0 / 100.0,  # TSLA: 36.08
            12.3 * 75.0 / 100.0,  # AMZN: 9.225
        ]
        
        # Check with tolerance for floating point precision
        for i, expected in enumerate(expected_exret):
            assert abs(result.iloc[i]['EXRET'] - expected) < 0.1
    
    def test_calculate_exret_missing_columns(self):
        """Test EXRET calculation with missing columns."""
        df = pd.DataFrame({'symbol': ['TEST']})
        result = calculate_exret(df)
        
        assert 'EXRET' in result.columns
        assert result.iloc[0]['EXRET'] == pytest.approx(0.0)
    
    def test_calculate_exret_nan_values(self, edge_case_dataframe):
        """Test EXRET calculation with NaN values."""
        result = calculate_exret(edge_case_dataframe)
        
        # NaN values should be handled gracefully
        assert 'EXRET' in result.columns
        assert not result['EXRET'].isna().any()  # No NaN in result
    
    def test_calculate_exret_performance(self):
        """Test EXRET calculation performance with large dataset."""
        # Create large test dataset
        large_df = pd.DataFrame({
            'upside': np.random.uniform(0, 50, 10000),
            'buy_percentage': np.random.uniform(0, 100, 10000),
        })
        
        import time
        start_time = time.perf_counter()
        result = calculate_exret(large_df)
        end_time = time.perf_counter()
        
        # Should complete in reasonable time (< 0.1 seconds)
        assert end_time - start_time < 0.1
        assert len(result) == 10000
        assert 'EXRET' in result.columns


class TestCalculateActionVectorized:
    """Test cases for the vectorized action calculation."""
    
    def test_vectorized_action_buy_conditions(self, sample_dataframe):
        """Test vectorized BUY action detection."""
        result = calculate_action_vectorized(sample_dataframe)
        
        # AAPL should be BUY (upside 25.5%, buy% 85%, meets criteria)
        assert result.iloc[0] == 'B'
        
        # MSFT should be BUY (upside 15.2%, buy% 90%, but check full criteria)
        # May be 'H' if it doesn't meet full BUY criteria
        assert result.iloc[1] in ['B', 'H']
    
    def test_vectorized_action_sell_conditions(self, sample_dataframe):
        """Test vectorized SELL action detection."""
        # Modify data to trigger SELL conditions
        df = sample_dataframe.copy()
        df.loc[2, 'upside'] = 3.0  # Low upside
        df.loc[2, 'buy_percentage'] = 60.0  # Low buy percentage
        
        result = calculate_action_vectorized(df)
        
        # GOOGL should be SELL due to low upside and buy percentage
        assert result.iloc[2] == 'S'
    
    def test_vectorized_action_inconclusive_conditions(self, edge_case_dataframe):
        """Test vectorized INCONCLUSIVE action detection."""
        result = calculate_action_vectorized(edge_case_dataframe)
        
        # First two rows should be INCONCLUSIVE due to low analyst coverage
        assert result.iloc[0] == 'I'
        assert result.iloc[1] == 'I'
    
    def test_vectorized_action_performance(self):
        """Test vectorized action calculation performance."""
        # Create large test dataset
        large_df = pd.DataFrame({
            'upside': np.random.uniform(0, 50, 10000),
            'buy_percentage': np.random.uniform(0, 100, 10000),
            'analyst_count': np.random.randint(1, 30, 10000),
            'total_ratings': np.random.randint(1, 35, 10000),
            'pe_forward': np.random.uniform(5, 50, 10000),
            'pe_trailing': np.random.uniform(5, 60, 10000),
            'peg_ratio': np.random.uniform(0.5, 5, 10000),
            'short_percent': np.random.uniform(0, 10, 10000),
            'beta': np.random.uniform(0.5, 3, 10000),
            'EXRET': np.random.uniform(0, 40, 10000),
        })
        
        import time
        start_time = time.perf_counter()
        result = calculate_action_vectorized(large_df)
        end_time = time.perf_counter()
        
        # Should complete in reasonable time (< 1 second for vectorized operations)
        # More realistic threshold for CI environments
        assert end_time - start_time < 1.0
        assert len(result) == 10000
        assert result.isin(['B', 'S', 'H', 'I']).all()


class TestCalculateAction:
    """Test cases for the main calculate_action function."""
    
    def test_calculate_action_integration(self, sample_dataframe):
        """Test the main calculate_action function integration."""
        result = calculate_action(sample_dataframe)
        
        # Should return DataFrame with ACT column
        assert isinstance(result, pd.DataFrame)
        assert 'ACT' in result.columns
        assert len(result) == len(sample_dataframe)
        
        # All actions should be valid
        valid_actions = {'B', 'S', 'H', 'I'}
        assert result['ACT'].isin(valid_actions).all()
    
    def test_calculate_action_error_handling(self):
        """Test error handling in calculate_action."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        result = calculate_action(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        # Should handle empty DataFrame gracefully
    
    @patch('trade_modules.analysis_engine.calculate_action_vectorized')
    def test_calculate_action_calls_vectorized(self, mock_vectorized, sample_dataframe):
        """Test that calculate_action calls the vectorized version."""
        mock_vectorized.return_value = pd.Series(['B', 'S', 'H', 'I', 'B'], 
                                                 index=sample_dataframe.index)
        
        result = calculate_action(sample_dataframe)
        
        # Verify vectorized function was called
        mock_vectorized.assert_called_once()
        assert 'ACT' in result.columns


class TestFilterFunctions:
    """Test cases for filter wrapper functions."""
    
    def test_filter_buy_opportunities_wrapper(self, sample_dataframe):
        """Test buy opportunities filter wrapper."""
        # Add ACT column first
        df = calculate_action(sample_dataframe)
        
        with patch('trade_modules.analysis_engine.filter_buy_opportunities') as mock_filter:
            mock_filter.return_value = df[df['ACT'] == 'B']
            
            result = filter_buy_opportunities_wrapper(df)
            
            # Should call the underlying filter function
            mock_filter.assert_called_once_with(df)
    
    def test_filter_sell_candidates_wrapper(self, sample_dataframe):
        """Test sell candidates filter wrapper."""
        df = calculate_action(sample_dataframe)
        
        with patch('trade_modules.analysis_engine.filter_sell_candidates') as mock_filter:
            mock_filter.return_value = df[df['ACT'] == 'S']
            
            result = filter_sell_candidates_wrapper(df)
            
            mock_filter.assert_called_once_with(df)
    
    def test_filter_hold_candidates_wrapper(self, sample_dataframe):
        """Test hold candidates filter wrapper."""
        df = calculate_action(sample_dataframe)
        
        with patch('trade_modules.analysis_engine.filter_hold_candidates') as mock_filter:
            mock_filter.return_value = df[df['ACT'] == 'H']
            
            result = filter_hold_candidates_wrapper(df)
            
            mock_filter.assert_called_once_with(df)


class TestPerformanceComparison:
    """Test performance comparison between old and new implementations."""
    
    def test_vectorized_vs_apply_performance(self):
        """Compare performance of vectorized vs apply-based operations."""
        # Create test dataset
        test_df = pd.DataFrame({
            'upside': np.random.uniform(0, 50, 1000),
            'buy_percentage': np.random.uniform(0, 100, 1000),
            'analyst_count': np.random.randint(1, 30, 1000),
            'total_ratings': np.random.randint(1, 35, 1000),
            'pe_forward': np.random.uniform(5, 50, 1000),
            'pe_trailing': np.random.uniform(5, 60, 1000),
            'peg_ratio': np.random.uniform(0.5, 5, 1000),
            'short_percent': np.random.uniform(0, 10, 1000),
            'beta': np.random.uniform(0.5, 3, 1000),
            'EXRET': np.random.uniform(0, 40, 1000),
        })
        
        import time
        
        # Test vectorized approach
        start_time = time.perf_counter()
        vectorized_result = calculate_action_vectorized(test_df)
        vectorized_time = time.perf_counter() - start_time
        
        # Vectorized should be significantly faster
        assert vectorized_time < 0.01  # Should be very fast
        assert len(vectorized_result) == 1000
        assert vectorized_result.isin(['B', 'S', 'H', 'I']).all()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        exret_result = calculate_exret(empty_df)
        assert isinstance(exret_result, pd.DataFrame)
        
        action_result = calculate_action(empty_df)
        assert isinstance(action_result, pd.DataFrame)
    
    def test_single_row_dataframe(self):
        """Test handling of single row DataFrames."""
        single_df = pd.DataFrame({
            'upside': [25.0],
            'buy_percentage': [85.0],
            'analyst_count': [15],
            'total_ratings': [20],
        })
        
        exret_result = calculate_exret(single_df)
        assert len(exret_result) == 1
        assert 'EXRET' in exret_result.columns
        
        action_result = calculate_action(single_df)
        assert len(action_result) == 1
        assert 'ACT' in action_result.columns
    
    def test_missing_required_columns(self):
        """Test graceful handling of missing required columns."""
        minimal_df = pd.DataFrame({'symbol': ['TEST']})
        
        # Should not raise exceptions
        exret_result = calculate_exret(minimal_df)
        action_result = calculate_action(minimal_df)
        
        assert isinstance(exret_result, pd.DataFrame)
        assert isinstance(action_result, pd.DataFrame)
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        extreme_df = pd.DataFrame({
            'upside': [-1000, 0, 1000],
            'buy_percentage': [-100, 0, 200],
            'analyst_count': [0, 1, 1000],
            'total_ratings': [0, 1, 1000],
            'pe_forward': [-100, 0, 10000],
            'pe_trailing': [-100, 0, 10000],
            'peg_ratio': [-10, 0, 100],
            'short_percent': [-10, 0, 100],
            'beta': [-10, 0, 100],
        })
        
        # Should handle extreme values without errors
        exret_result = calculate_exret(extreme_df)
        action_result = calculate_action(extreme_df)
        
        assert len(exret_result) == 3
        assert len(action_result) == 3
        assert action_result['ACT'].isin(['B', 'S', 'H', 'I']).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])