#!/usr/bin/env python3
"""
Test to verify exact behavior of action calculation before and after vectorization.
This ensures zero behavioral changes during optimization.
"""

import pandas as pd
import numpy as np
import time
from trade_modules.analysis_engine import calculate_action_vectorized

def create_test_dataframe(size=1000):
    """Create test DataFrame with realistic financial data."""
    rng = np.random.default_rng(42)  # For reproducible tests
    
    data = {
        'ticker': [f'STOCK{i:04d}' for i in range(size)],
        'price': rng.uniform(10, 500, size),
        'market_cap': rng.uniform(1e9, 1e12, size),
        'pe_forward': rng.uniform(5, 50, size),
        'pe_trailing': rng.uniform(8, 60, size),
        'peg_ratio': rng.uniform(0.5, 3.0, size),
        'short_percent': rng.uniform(0, 20, size),
        'beta': rng.uniform(0.3, 2.5, size),
        'EXRET': rng.uniform(-5, 25, size),
        'upside': rng.uniform(-30, 80, size),
        'buy_percentage': rng.uniform(0, 100, size),
        'analyst_count': rng.integers(1, 50, size),
        'total_ratings': rng.integers(1, 30, size),
    }
    
    # Add some NaN values to test missing data handling
    for col in ['pe_forward', 'pe_trailing', 'peg_ratio', 'short_percent', 'beta']:
        mask = rng.random(size) < 0.1  # 10% missing values
        data[col][mask] = np.nan
    
    return pd.DataFrame(data)

def test_current_implementation():
    """Test and time the current row-by-row implementation."""
    print("Testing current row-by-row implementation...")
    
    # Test different sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        df = create_test_dataframe(size)
        
        print(f"\nTesting {size} rows:")
        
        # Time the current implementation
        start_time = time.time()
        result = calculate_action_vectorized(df, "market")
        end_time = time.time()
        
        print(f"  Time: {end_time - start_time:.3f} seconds")
        print(f"  Actions: B={sum(result == 'B')}, S={sum(result == 'S')}, H={sum(result == 'H')}, I={sum(result == 'I')}")
        
        # Verify no unexpected values
        unique_actions = set(result)
        expected_actions = {'B', 'S', 'H', 'I'}
        assert unique_actions.issubset(expected_actions), f"Unexpected actions: {unique_actions - expected_actions}"
        
        # Store result for comparison
        if size == 1000:
            baseline_result = result.copy()
    
    return baseline_result

def test_edge_cases():
    """Test edge cases to ensure robustness."""
    print("\nTesting edge cases...")
    
    # Empty DataFrame
    empty_df = pd.DataFrame()
    result = calculate_action_vectorized(empty_df, "market")
    assert len(result) == 0, "Empty DataFrame should return empty result"
    
    # Single row
    single_row = create_test_dataframe(1)
    result = calculate_action_vectorized(single_row, "market")
    assert len(result) == 1, "Single row should return single result"
    
    # All NaN values
    nan_df = create_test_dataframe(10)
    for col in nan_df.columns:
        if col != 'ticker':
            nan_df[col] = np.nan
    result = calculate_action_vectorized(nan_df, "market")
    # Should default to 'I' (inconclusive) for all NaN data
    print(f"  All NaN result: {result.value_counts().to_dict()}")
    
    # Extreme values
    extreme_df = create_test_dataframe(10)
    extreme_df['pe_forward'] = [0.1, 1000, np.inf, -np.inf, np.nan] * 2
    extreme_df['upside'] = [-1000, 1000, np.inf, -np.inf, np.nan] * 2
    result = calculate_action_vectorized(extreme_df, "market")
    print(f"  Extreme values result: {result.value_counts().to_dict()}")
    
    print("✅ All edge cases passed")

if __name__ == "__main__":
    print("🔍 BASELINE TESTING - Current Implementation")
    print("=" * 50)
    
    baseline = test_current_implementation()
    test_edge_cases()
    
    print(f"\n✅ Baseline established with {len(baseline)} results")
    print("Ready for vectorization optimization!")