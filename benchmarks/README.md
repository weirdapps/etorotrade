# Performance Benchmarking and Memory Management

This directory contains benchmark results and test scripts for the performance benchmarking and memory management tools added to the yahoofinance package.

## Implemented Features

### 1. Enhanced Benchmarking Tools
- Comprehensive `PerformanceBenchmark` class for measuring execution time and resource usage
- Batch size and concurrency level optimization tools
- Benchmark comparison against baselines to detect performance regressions

### 2. Advanced Memory Management
- Memory leak detection with `find_memory_leaks` and `find_memory_leaks_async` functions
- Memory profiling with `MemoryProfiler` class and `profile_memory` decorator
- Resource monitoring with `ResourceMonitor` class
- Regression testing to prevent performance degradation

### 3. Priority-Based Rate Limiting
- `PriorityAsyncRateLimiter` implementation for multi-tier rate limiting
- Different quotas for HIGH, MEDIUM, and LOW priority operations
- Token bucket algorithm for smoother rate limiting
- Region-aware delays based on ticker origin (US, Europe, Asia)
- Adaptive delay adjustment based on success/failure rates

## Test Scripts

- `test_priority_limiter.py`: Standalone test for the priority-based rate limiter
- `test_memory_leak.py`: Demonstration of memory leak detection capabilities

## Usage Examples

### Running Performance Benchmarks

```bash
# Run all benchmarks
python -m yahoofinance.analysis.benchmarking --all

# Test specific provider
python -m yahoofinance.analysis.benchmarking --provider hybrid

# Test batch sizes
python -m yahoofinance.analysis.benchmarking --batch-benchmark

# Test concurrency levels
python -m yahoofinance.analysis.benchmarking --concurrency-benchmark

# Test memory leak detection
python -m yahoofinance.analysis.benchmarking --memory-check

# Test priority rate limiting
python -m yahoofinance.analysis.benchmarking --priority-test
```

### Using Memory Leak Detection in Code

```python
from yahoofinance.analysis.benchmarking import find_memory_leaks_async

async def check_for_leaks():
    provider = get_provider(async_api=True)
    is_leaking, stats = await find_memory_leaks_async(
        provider.get_ticker_info, "AAPL", iterations=10
    )
    if is_leaking:
        print(f"Memory leak detected: {stats['memory_diff_mb']:.2f} MB")
```

### Using Priority-Based Batch Processing

```python
from yahoofinance.analysis.benchmarking import prioritized_batch_process
from yahoofinance.utils.market.ticker_utils import is_us_ticker

async def process_with_priority():
    provider = get_provider(async_api=True)
    
    def get_ticker_priority(ticker):
        if ticker in ("AAPL", "MSFT"):
            return "HIGH"
        elif is_us_ticker(ticker):
            return "MEDIUM"
        else:
            return "LOW"
    
    results = await prioritized_batch_process(
        items=["AAPL", "MSFT", "GOOG", "9988.HK", "BMW.DE"],
        processor=provider.get_ticker_info,
        priority_func=get_ticker_priority,
        concurrency=5
    )
```

## Performance Benefits

The new tools provide significant performance improvements:

1. **Faster API Access**: Priority-based rate limiting ensures critical operations get faster access
2. **Optimized Resource Usage**: Memory monitoring prevents resource leaks and excessive memory usage
3. **Improved Concurrency**: Adaptive concurrency adjusts based on real-time performance metrics
4. **Better Error Recovery**: Smart error handling with exponential back-off for failed operations

These improvements ensure the application can handle more concurrent operations while maintaining stability and preventing resource exhaustion.