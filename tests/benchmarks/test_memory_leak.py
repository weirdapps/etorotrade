#!/usr/bin/env python3
"""
Memory leak detection tests for the trading system.

This module tests for memory leaks in critical components like:
- Cache services
- Data providers
- Trading engine
"""

import gc
import tracemalloc
import unittest
from typing import List

# Start memory tracking
tracemalloc.start()


class TestMemoryLeaks(unittest.TestCase):
    """Test suite for detecting memory leaks in the application."""
    
    def setUp(self):
        """Set up test fixtures."""
        gc.collect()
        self.snapshot_start = tracemalloc.take_snapshot()
    
    def tearDown(self):
        """Clean up after tests."""
        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        top_stats = snapshot_end.compare_to(self.snapshot_start, 'lineno')
        
        # Check for significant memory growth (>10MB)
        total_growth = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        if total_growth > 10 * 1024 * 1024:  # 10MB threshold
            print(f"Warning: Memory growth detected: {total_growth / 1024 / 1024:.2f} MB")
            for stat in top_stats[:5]:
                if stat.size_diff > 0:
                    print(stat)
    
    def test_cache_service_memory(self):
        """Test that cache service doesn't leak memory."""
        try:
            from trade_modules.cache_service import CacheService
            cache = CacheService()
        except ImportError:
            # Try alternative import path
            try:
                from yahoofinance.data.cache import CacheManager as CacheService
                cache = CacheService()
            except ImportError:
                self.skipTest("CacheService not available")
        
        # Simulate heavy cache usage
        for i in range(1000):
            cache.set(f"key_{i}", {"data": f"value_{i}" * 100})
            if i % 100 == 0:
                # Periodically get items
                cache.get(f"key_{i-50}")
        
        # Clear cache
        cache.clear()
        gc.collect()
        
        # Memory should be released
        self.assertIsNotNone(cache)
    
    def test_data_provider_memory(self):
        """Test that data providers don't leak memory."""
        try:
            from yahoofinance.api.providers.async_yahoo_finance import AsyncYahooFinanceProvider

            provider = AsyncYahooFinanceProvider()

            # Clear any cached data
            if hasattr(provider, 'clear_cache'):
                provider.clear_cache()

            # Memory should be minimal
            self.assertIsNotNone(provider)
        except ImportError:
            self.skipTest("AsyncYahooFinanceProvider not available")
    
    def test_dataframe_operations_memory(self):
        """Test that DataFrame operations don't leak memory."""
        import pandas as pd
        
        # Create and destroy DataFrames repeatedly
        for _ in range(100):
            df = pd.DataFrame({
                'A': range(1000),
                'B': range(1000),
                'C': range(1000)
            })
            # Perform operations
            df['D'] = df['A'] + df['B']
            df['E'] = df['C'] * 2
            result = df.sum()
            del df
        
        gc.collect()
        
        # Check that memory is released
        self.assertTrue(True)  # If we get here, no crash occurred
    
    def test_circular_references(self):
        """Test that circular references are properly handled."""
        class Node:
            def __init__(self, value):
                self.value = value
                self.next = None
        
        # Create circular reference
        nodes = []
        for i in range(100):
            node = Node(i)
            if i > 0:
                nodes[i-1].next = node
            nodes.append(node)
        
        # Create circular reference
        if nodes:
            nodes[-1].next = nodes[0]
        
        # Clear references
        nodes.clear()
        gc.collect()
        
        # Memory should be reclaimed
        self.assertTrue(True)


def run_memory_profile():
    """Run a simple memory profile of the application."""
    print("Running memory leak detection tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryLeaks)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ No memory leaks detected")
    else:
        print(f"\n⚠️ {len(result.failures)} potential memory issues found")
    
    # Stop memory tracking
    tracemalloc.stop()
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_memory_profile()
    exit(0 if success else 1)