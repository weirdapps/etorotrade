#!/usr/bin/env python3
"""
Performance Benchmark Tool for etorotrade Trading Analysis

This script measures the performance impact of optimizations including:
- Rate limiting improvements
- Vectorized DataFrame operations
- Batch processing optimizations
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yahoofinance.core.config import RATE_LIMIT
from trade_modules.analysis_engine import calculate_action, calculate_exret


def generate_test_data(num_rows: int = 1000) -> pd.DataFrame:
    """Generate realistic test data for performance benchmarking."""
    np.random.seed(42)  # For reproducible results
    
    data = {
        'symbol': [f'TEST{i:04d}' for i in range(num_rows)],
        'upside': np.random.normal(15.0, 10.0, num_rows).clip(0, 100),
        'buy_percentage': np.random.normal(70.0, 15.0, num_rows).clip(0, 100),
        'analyst_count': np.random.randint(1, 20, num_rows),
        'total_ratings': np.random.randint(1, 25, num_rows),
        'pe_forward': np.random.exponential(15.0, num_rows).clip(0.1, 100),
        'pe_trailing': np.random.exponential(18.0, num_rows).clip(0.1, 120),
        'peg_ratio': np.random.exponential(1.5, num_rows).clip(0.1, 10),
        'short_percent': np.random.exponential(2.0, num_rows).clip(0, 20),
        'beta': np.random.normal(1.0, 0.5, num_rows).clip(0.1, 5),
        'EXRET': np.random.normal(10.0, 8.0, num_rows).clip(0, 50),
    }
    
    return pd.DataFrame(data)


def benchmark_calculate_action(df: pd.DataFrame, iterations: int = 5) -> Dict[str, float]:
    """Benchmark the calculate_action function."""
    times = []
    
    for i in range(iterations):
        start_time = time.perf_counter()
        result_df = calculate_action(df)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'std_time': np.std(times),
        'rows_per_second': len(df) / np.mean(times)
    }


def benchmark_calculate_exret(df: pd.DataFrame, iterations: int = 5) -> Dict[str, float]:
    """Benchmark the calculate_exret function."""
    times = []
    
    for i in range(iterations):
        start_time = time.perf_counter()
        result_df = calculate_exret(df)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'std_time': np.std(times),
        'rows_per_second': len(df) / np.mean(times)
    }


def benchmark_rate_limiting_config() -> Dict[str, any]:
    """Analyze current rate limiting configuration for performance."""
    config = RATE_LIMIT
    print(f"DEBUG: Current RATE_LIMIT config: {config}")  # Debug print
    
    # Calculate theoretical throughput
    batch_size = config['BATCH_SIZE']
    base_delay = config['BASE_DELAY']
    batch_delay = config['BATCH_DELAY']
    
    # Time for 100 tickers
    num_batches = 100 / batch_size
    total_api_delay = 100 * base_delay  # Individual API delays
    total_batch_delay = num_batches * batch_delay  # Between-batch delays
    total_time = total_api_delay + total_batch_delay
    
    return {
        'batch_size': batch_size,
        'base_delay': base_delay,
        'batch_delay': batch_delay,
        'estimated_time_100_tickers': total_time,
        'estimated_throughput_per_minute': 60 / (total_time / 100),
        'config': config
    }


def run_comprehensive_benchmark():
    """Run comprehensive performance benchmarks."""
    print("🚀 etorotrade Performance Benchmark")
    print("=" * 50)
    
    # Test different data sizes
    test_sizes = [100, 500, 1000, 2000]
    
    print("\n📊 DataFrame Operations Benchmark")
    print("-" * 40)
    
    for size in test_sizes:
        print(f"\nTesting with {size:,} rows:")
        df = generate_test_data(size)
        
        # Benchmark EXRET calculation
        exret_stats = benchmark_calculate_exret(df)
        print(f"  EXRET calculation: {exret_stats['mean_time']:.4f}s ({exret_stats['rows_per_second']:,.0f} rows/sec)")
        
        # Benchmark action calculation
        action_stats = benchmark_calculate_action(df)
        print(f"  Action calculation: {action_stats['mean_time']:.4f}s ({action_stats['rows_per_second']:,.0f} rows/sec)")
        
        # Memory usage estimate
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  Memory usage: {memory_mb:.2f} MB")
    
    print("\n⚡ Rate Limiting Analysis")
    print("-" * 30)
    
    rate_config = benchmark_rate_limiting_config()
    print(f"Batch size: {rate_config['batch_size']}")
    print(f"Base delay: {rate_config['base_delay']}s")
    print(f"Batch delay: {rate_config['batch_delay']}s")
    print(f"Estimated time for 100 tickers: {rate_config['estimated_time_100_tickers']:.1f}s")
    print(f"Estimated throughput: {rate_config['estimated_throughput_per_minute']:.1f} tickers/minute")
    
    print("\n🎯 Performance Recommendations")
    print("-" * 35)
    
    # Analyze and provide recommendations
    if rate_config['batch_size'] < 20:
        print("⚠️  Consider increasing batch size for better throughput")
    
    if rate_config['base_delay'] > 0.2:
        print("⚠️  Base delay could be reduced for faster API calls")
    
    if rate_config['estimated_time_100_tickers'] > 20:
        print("⚠️  Current config may be too conservative for large datasets")
    
    # Test with realistic portfolio size
    realistic_size = 50  # Typical portfolio size
    df_realistic = generate_test_data(realistic_size)
    
    total_start = time.perf_counter()
    df_realistic = calculate_exret(df_realistic)
    df_realistic = calculate_action(df_realistic)
    total_end = time.perf_counter()
    
    total_processing_time = total_end - total_start
    print(f"\n✅ Realistic Portfolio Test ({realistic_size} tickers):")
    print(f"   Total processing time: {total_processing_time:.4f}s")
    print(f"   Processing rate: {realistic_size/total_processing_time:.0f} tickers/sec")
    
    return {
        'processing_benchmarks': {size: {'exret': benchmark_calculate_exret(generate_test_data(size)), 
                                        'action': benchmark_calculate_action(generate_test_data(size))} 
                                 for size in test_sizes},
        'rate_limiting': rate_config,
        'realistic_test': {
            'size': realistic_size,
            'processing_time': total_processing_time,
            'rate': realistic_size/total_processing_time
        }
    }


if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    
    print(f"\n🎉 Benchmark Complete!")
    print(f"📈 Key Metrics:")
    print(f"   - Optimized batch size: {results['rate_limiting']['batch_size']}")
    print(f"   - Reduced API delays: {results['rate_limiting']['base_delay']}s")
    print(f"   - Vectorized operations: Enabled")
    print(f"   - Portfolio processing: {results['realistic_test']['rate']:.0f} tickers/sec")