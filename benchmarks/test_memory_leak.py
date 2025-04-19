"""
Simple standalone test script for memory leak detection.

This script demonstrates the memory leak detection functionality
without relying on the full codebase. It includes examples of both
leaking and non-leaking functions for testing.
"""

import asyncio
import gc
import os
import time
import tracemalloc
from typing import Dict, Any, List, Tuple


async def find_memory_leaks_async(func, *args, iterations: int = 10, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """
    Async version of find_memory_leaks that properly handles async functions.
    
    Args:
        func: Async function to check for memory leaks
        *args: Arguments to pass to the function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (leak_detected, metrics)
    """
    # Stop tracing if already running to ensure a clean state
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    
    # Start tracing with 10 frames to get detailed traceback
    tracemalloc.start(10)
    
    memory_usage = []
    gc_stats = []
    
    # Force garbage collection before starting
    gc.collect()
    
    # Take initial snapshot
    snapshot_start = tracemalloc.take_snapshot()
    start_memory = tracemalloc.get_traced_memory()[0]
    peak_memory = tracemalloc.get_traced_memory()[1]
    
    print(f"Starting async memory leak check, running {iterations} iterations")
    
    # Run function multiple times
    for i in range(iterations):
        print(f"Memory leak check iteration {i+1}/{iterations}")
        
        # Collect memory before iteration
        before_iter_memory = tracemalloc.get_traced_memory()[0]
        
        # Run function with error handling
        try:
            # Directly await the function since we're already in an async context
            result = await func(*args, **kwargs)
        except Exception as e:
            print(f"Error during iteration {i+1}: {str(e)}")
        
        # Record GC stats before collection
        gc_count_before = (gc.get_count()[0], gc.get_count()[1], gc.get_count()[2])
        
        # Force garbage collection
        gc_collected = gc.collect(2)  # Full collection (all generations)
        
        # Record memory after iteration and GC
        current_memory = tracemalloc.get_traced_memory()[0]
        current_peak = tracemalloc.get_traced_memory()[1]
        peak_memory = max(peak_memory, current_peak)
        
        # Record memory growth in this iteration
        memory_growth_in_iteration = current_memory - before_iter_memory
        
        # Record memory usage and GC stats
        memory_usage.append({
            "iteration": i + 1,
            "memory": current_memory,
            "memory_mb": current_memory / (1024 * 1024),
            "growth_in_iteration": memory_growth_in_iteration,
            "growth_in_iteration_mb": memory_growth_in_iteration / (1024 * 1024),
        })
        
        gc_stats.append({
            "iteration": i + 1,
            "gc_count_before": gc_count_before,
            "objects_collected": gc_collected,
        })
        
        # Give the event loop a chance to clean up resources
        await asyncio.sleep(0.1)
    
    # Take final snapshot
    snapshot_end = tracemalloc.take_snapshot()
    end_memory = tracemalloc.get_traced_memory()[0]
    
    # Compare snapshots
    top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    
    # Process top memory consumers
    memory_growth = []
    for stat in top_stats[:20]:  # Capture more information for detailed analysis
        # Only include significant allocations (>100 bytes)
        if stat.size_diff > 100:
            frames = []
            for frame in stat.traceback[:3]:  # Include up to 3 frames
                frames.append({
                    "filename": os.path.basename(frame.filename),
                    "line": frame.lineno,
                })
            
            memory_growth.append({
                "size": stat.size_diff,
                "size_kb": stat.size_diff / 1024,
                "count_diff": stat.count_diff,
                "frames": frames,
                "file": os.path.basename(stat.traceback[0].filename) if stat.traceback else "unknown",
                "line": stat.traceback[0].lineno if stat.traceback else 0,
            })
    
    # Calculate metrics
    memory_diff = end_memory - start_memory
    memory_growth_per_iteration = memory_diff / iterations if iterations > 0 else 0
    
    # Analyze memory growth pattern
    # Check if memory is consistently growing across iterations
    memory_values = [entry["memory"] for entry in memory_usage]
    
    # Check for steady growth (not just fluctuating)
    # We look for a clear upward trend, not just random increases
    consistent_growth = False
    if len(memory_values) >= 3:  # Need at least 3 points for a trend
        # Calculate a simple linear regression slope
        n = len(memory_values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(memory_values) / n
        
        numerator = sum((x[i] - x_mean) * (memory_values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Consider it a leak if slope is positive and memory grew by more than 10%
        growth_percent = ((end_memory / start_memory) - 1) * 100 if start_memory > 0 else 0
        consistent_growth = slope > 0 and growth_percent > 10
    
    # Apply heuristics to detect potential leaks
    threshold_mb = 1.2  # Default 1.2MB
    absolute_growth_significant = memory_diff > (threshold_mb * 1024 * 1024)  # Convert MB to bytes
    
    # Consider it a leak if:
    # 1. Memory consistently grows across iterations, OR
    # 2. Total memory growth is significant by absolute standards
    is_leaking = consistent_growth or absolute_growth_significant
    
    detailed_metrics = {
        "start_memory": start_memory,
        "start_memory_mb": start_memory / (1024 * 1024),
        "end_memory": end_memory,
        "end_memory_mb": end_memory / (1024 * 1024),
        "peak_memory": peak_memory,
        "peak_memory_mb": peak_memory / (1024 * 1024),
        "memory_diff": memory_diff,
        "memory_diff_mb": memory_diff / (1024 * 1024),
        "memory_growth_per_iteration": memory_growth_per_iteration,
        "memory_growth_per_iteration_mb": memory_growth_per_iteration / (1024 * 1024),
        "memory_growth_percent": ((end_memory / start_memory) - 1) * 100 if start_memory > 0 else 0,
        "iterations": iterations,
        "memory_usage": memory_usage,
        "gc_stats": gc_stats,
        "top_consumers": memory_growth,
        "is_leaking": is_leaking,
        "consistent_growth": consistent_growth,
        "absolute_growth_significant": absolute_growth_significant,
    }
    
    # Stop tracing
    tracemalloc.stop()
    
    return is_leaking, detailed_metrics


# Sample functions to test memory leak detection

# Non-leaking function
async def non_leaking_function():
    """Function that doesn't leak memory."""
    result = [1, 2, 3, 4, 5]
    await asyncio.sleep(0.01)
    return result


# Leaking function
_cache = []  # Global cache that will accumulate items

async def leaking_function():
    """Function that leaks memory by appending to a global cache."""
    global _cache
    _cache.append([1, 2, 3, 4, 5] * 100000)  # Add a large list to the cache
    await asyncio.sleep(0.01)
    return len(_cache)


async def main():
    """Main test function."""
    # Test non-leaking function
    print("\n=== Testing Non-Leaking Function ===")
    is_leaking, metrics = await find_memory_leaks_async(non_leaking_function, iterations=5)
    
    # Print detailed report
    if is_leaking:
        print("\n⚠️ FALSE POSITIVE: MEMORY LEAK DETECTED ⚠️")
        print(f"Memory increased by {metrics['memory_diff_mb']:.2f} MB over 5 iterations")
    else:
        print("\n✅ Correctly identified as non-leaking")
        print(f"Total memory change: {metrics['memory_diff_mb']:.2f} MB")
    
    # Test leaking function
    print("\n=== Testing Leaking Function ===")
    is_leaking, metrics = await find_memory_leaks_async(leaking_function, iterations=5)
    
    # Print detailed report
    if is_leaking:
        print("\n✅ Correctly identified memory leak")
        print(f"Memory increased by {metrics['memory_diff_mb']:.2f} MB over 5 iterations")
        print(f"Memory growth per iteration: {metrics['memory_growth_per_iteration_mb']:.2f} MB")
        print(f"Growth percentage: {metrics['memory_growth_percent']:.1f}%")
        
        # Show top memory consumers
        print("\nTop memory consumers:")
        for i, item in enumerate(metrics['top_consumers'][:3], 1):
            print(f"#{i}: {item['file']}:{item['line']} - {item['size_kb']:.1f} KB ({item['count_diff']} objects)")
    else:
        print("\n⚠️ FALSE NEGATIVE: Failed to detect memory leak ⚠️")
        print(f"Total memory change: {metrics['memory_diff_mb']:.2f} MB")


if __name__ == "__main__":
    asyncio.run(main())