"""
Tests for performance benchmarking module.

This module tests the benchmarking utilities, including:
- Performance benchmarking
- Resource monitoring
- Memory profiling
- Priority-based rate limiting
"""

import pytest
import asyncio
import time
import inspect
import gc
import os
from unittest.mock import patch, AsyncMock, MagicMock

from yahoofinance.analysis.benchmarking import (
    BenchmarkResult,
    PerformanceBenchmark,
    ResourceMonitor,
    MemoryProfiler,
    PriorityAsyncRateLimiter,
    find_memory_leaks, 
    find_memory_leaks_async,
    profile,
    profile_memory,
    adaptive_fetch,
    prioritized_batch_process
)


@pytest.fixture
def benchmark_result():
    """Create a test benchmark result."""
    return BenchmarkResult(
        operation="test_operation",
        iterations=3,
        start_time=time.time() - 10,
        end_time=time.time(),
        duration=10.0,
        min_time=2.0,
        max_time=5.0,
        avg_time=3.0,
        median_time=3.0,
        std_dev=1.0,
        success_rate=0.9,
        error_count=1,
        memory_before=1024 * 1024 * 100,
        memory_after=1024 * 1024 * 110,
        memory_peak=1024 * 1024 * 120,
        cpu_usage_avg=50.0,
        thread_count_avg=5,
        parameters={"param1": "value1", "param2": 42}
    )


@pytest.mark.asyncio
async def test_performance_benchmark():
    """Test the PerformanceBenchmark class."""
    # Create a test async function to benchmark
    async def test_function(a, b):
        await asyncio.sleep(0.01)  # Small delay
        return a + b
    
    # Create benchmark and run it
    benchmark = PerformanceBenchmark(
        name="test_benchmark",
        iterations=2,
        warmup_iterations=1
    )
    benchmark.set_parameters(a=1, b=2)
    
    # Run the benchmark
    result = await benchmark.run(test_function, 1, 2)
    
    # Verify result properties
    assert result.operation == "test_benchmark"
    assert result.iterations == 2
    assert result.duration > 0
    assert result.min_time > 0
    assert result.max_time > 0
    assert result.avg_time > 0
    assert result.parameters == {"a": 1, "b": 2}
    assert result.success_rate == 1.0
    assert result.error_count == 0


@pytest.mark.asyncio
async def test_resource_monitor():
    """Test the ResourceMonitor class."""
    # Create and start the monitor
    monitor = ResourceMonitor(interval=0.1)
    await monitor.start()
    
    # Do some work to generate resource usage
    for _ in range(3):
        # Allocate some memory
        data = [0] * 1000000
        await asyncio.sleep(0.1)
    
    # Stop the monitor and get stats
    stats = await monitor.stop()
    
    # Verify stats structure
    assert "memory_initial" in stats
    assert "memory_final" in stats
    assert "memory_peak" in stats
    assert "memory_avg" in stats
    assert "cpu_avg" in stats
    assert "thread_count_avg" in stats
    assert "sample_count" in stats
    
    # Basic sanity checks
    assert stats["memory_initial"] > 0
    assert stats["memory_final"] > 0
    assert stats["memory_peak"] >= stats["memory_initial"]
    assert stats["memory_peak"] >= stats["memory_final"]
    assert stats["sample_count"] >= 3  # Should have at least 3 samples


@pytest.mark.asyncio
async def test_memory_profiler():
    """Test the MemoryProfiler class."""
    profiler = MemoryProfiler()
    
    # Start profiling
    profiler.start()
    
    # Allocate some memory that should be tracked
    data = [0] * 1000000
    more_data = [0] * 500000
    
    # Get profiling results
    stats = profiler.stop()
    
    # Verify stats structure and basic sanity checks
    assert "total_diff_bytes" in stats
    assert "total_diff_kb" in stats
    assert "total_diff_mb" in stats
    assert "top_consumers" in stats
    
    # The total difference should be positive
    assert stats["total_diff_bytes"] >= 0
    assert stats["total_diff_kb"] >= 0
    assert stats["total_diff_mb"] >= 0


def test_benchmark_result_as_dict(benchmark_result):
    """Test BenchmarkResult.as_dict method."""
    result_dict = benchmark_result.as_dict()
    
    # Verify result dictionary structure
    assert result_dict["operation"] == "test_operation"
    assert result_dict["iterations"] == 3
    assert result_dict["duration_seconds"] == 10.0
    assert result_dict["min_time"] == 2.0
    assert result_dict["max_time"] == 5.0
    assert result_dict["avg_time"] == 3.0
    assert result_dict["median_time"] == 3.0
    assert result_dict["std_dev"] == 1.0
    assert result_dict["success_rate"] == 0.9
    assert result_dict["error_count"] == 1
    assert result_dict["memory_before_mb"] == 100.0
    assert result_dict["memory_after_mb"] == 110.0
    assert result_dict["memory_peak_mb"] == 120.0
    assert result_dict["memory_change_mb"] == 10.0
    assert result_dict["memory_change_percent"] == 10.0
    assert result_dict["cpu_usage_avg_percent"] == 50.0
    assert result_dict["thread_count_avg"] == 5
    assert result_dict["parameters"] == {"param1": "value1", "param2": 42}


@pytest.fixture
def priority_limiter():
    """Create a test PriorityAsyncRateLimiter."""
    limiter = PriorityAsyncRateLimiter(
        window_size=1,
        max_calls=10,
        base_delay=0.01,
        min_delay=0.005,
        max_delay=0.1
    )
    return limiter


@pytest.mark.asyncio
async def test_priority_limiter_initialization(priority_limiter):
    """Test PriorityAsyncRateLimiter initialization."""
    # Verify properties
    assert priority_limiter.window_size == 1
    assert priority_limiter.max_calls == 10
    assert priority_limiter.base_delay == 0.01
    assert priority_limiter.min_delay == 0.005
    assert priority_limiter.max_delay == 0.1
    
    # Verify priority quotas
    assert priority_limiter.high_priority_quota == 5  # 50% of max_calls
    assert priority_limiter.medium_priority_quota == 3  # 30% of max_calls
    assert priority_limiter.low_priority_quota == 2  # 20% of max_calls
    
    # Verify token buckets
    assert priority_limiter.tokens["HIGH"] == priority_limiter.high_priority_quota
    assert priority_limiter.tokens["MEDIUM"] == priority_limiter.medium_priority_quota
    assert priority_limiter.tokens["LOW"] == priority_limiter.low_priority_quota


@pytest.mark.asyncio
async def test_priority_limiter_wait(priority_limiter):
    """Test PriorityAsyncRateLimiter wait function."""
    # Patch asyncio.sleep to avoid actual waiting
    with patch('asyncio.sleep', AsyncMock()) as mock_sleep:
        # Wait with different priorities
        await priority_limiter.wait(priority="HIGH")
        await priority_limiter.wait(priority="MEDIUM")
        await priority_limiter.wait(priority="LOW")
        
        # Verify that sleep was called for each priority
        assert mock_sleep.call_count == 3
        
        # Verify call times are tracked
        assert len(priority_limiter.priority_call_times["HIGH"]) == 1
        assert len(priority_limiter.priority_call_times["MEDIUM"]) == 1
        assert len(priority_limiter.priority_call_times["LOW"]) == 1


@pytest.mark.asyncio
async def test_priority_limiter_update_delay(priority_limiter):
    """Test PriorityAsyncRateLimiter update_delay function."""
    # Record initial delays
    high_initial = priority_limiter.current_delays["HIGH"]
    medium_initial = priority_limiter.current_delays["MEDIUM"]
    low_initial = priority_limiter.current_delays["LOW"]
    
    # Record multiple successes to reduce delay
    for _ in range(5):
        await priority_limiter.update_delay("HIGH", True)
    
    # High priority delay should decrease
    assert priority_limiter.current_delays["HIGH"] < high_initial
    
    # Record failure to increase delay
    await priority_limiter.update_delay("MEDIUM", False)
    await priority_limiter.update_delay("MEDIUM", False)  # Second error triggers increase
    
    # Medium priority delay should increase
    assert priority_limiter.current_delays["MEDIUM"] > medium_initial
    
    # Test with rate limit error
    with patch.object(priority_limiter, 'error_counts', {"LOW": 1}):
        await priority_limiter.update_delay("LOW", False)
    
    # Low priority delay should increase more after error
    assert priority_limiter.current_delays["LOW"] > low_initial


@pytest.mark.asyncio
async def test_priority_limiter_consume_token(priority_limiter):
    """Test PriorityAsyncRateLimiter consume_token function."""
    # Set limited tokens
    priority_limiter.tokens = {
        "HIGH": 2,
        "MEDIUM": 1,
        "LOW": 0
    }
    
    # Consume tokens
    high_result = await priority_limiter.consume_token("HIGH")
    assert high_result is True
    assert priority_limiter.tokens["HIGH"] == 1
    
    # Fallback to medium if high is empty
    with patch.object(priority_limiter, 'tokens', {"HIGH": 0, "MEDIUM": 1, "LOW": 0}):
        high_result = await priority_limiter.consume_token("HIGH")
        assert high_result is True
        # Should use medium token
        assert priority_limiter.tokens["MEDIUM"] == 0
    
    # Return False if no tokens available
    with patch.object(priority_limiter, 'tokens', {"HIGH": 0, "MEDIUM": 0, "LOW": 0}):
        low_result = await priority_limiter.consume_token("LOW")
        assert low_result is False


@pytest.mark.asyncio
async def test_adaptive_fetch():
    """Test adaptive_fetch function."""
    items = ["item1", "item2", "item3", "item4", "item5"]
    
    # Create a test processor function
    async def processor(item):
        await asyncio.sleep(0.01)
        if item == "item3":
            # Simulate an error for one item
            raise ValueError("Test error")
        return f"processed_{item}"
    
    # Mock gather_with_concurrency
    with patch('yahoofinance.analysis.benchmarking.gather_with_concurrency', 
                AsyncMock(side_effect=lambda limit, *coros: asyncio.gather(*coros))):
        
        # Test adaptive fetch
        results = await adaptive_fetch(
            items=items,
            fetch_func=processor,
            initial_concurrency=2,
            max_concurrency=5,
            priority_items=["item1", "item5"]
        )
    
    # Check results
    assert len(results) == 4  # 5 items - 1 error
    assert results["item1"] == "processed_item1"
    assert results["item2"] == "processed_item2"
    assert "item3" not in results  # Error item should be skipped
    assert results["item4"] == "processed_item4"
    assert results["item5"] == "processed_item5"


@pytest.mark.asyncio
async def test_prioritized_batch_process():
    """Test prioritized_batch_process function."""
    items = ["item1", "item2", "item3", "item4", "item5"]
    
    # Create a test processor function
    async def processor(item):
        await asyncio.sleep(0.01)
        if item == "item3":
            # Simulate an error for one item
            raise ValueError("Test error")
        return f"processed_{item}"
    
    # Create a priority function
    def priority_func(item):
        if item in ["item1", "item5"]:
            return "HIGH"
        elif item in ["item2"]:
            return "MEDIUM"
        else:
            return "LOW"
    
    # Mock rate limiter
    mock_limiter = MagicMock(spec=PriorityAsyncRateLimiter)
    mock_limiter.wait = AsyncMock(return_value=0.01)
    mock_limiter.update_delay = AsyncMock()
    
    # Mock gather_with_concurrency
    with patch('yahoofinance.utils.async_utils.enhanced.gather_with_concurrency', 
                AsyncMock(side_effect=lambda limit, *coros: asyncio.gather(*coros, return_exceptions=True))):
        
        # Test prioritized batch process
        results = await prioritized_batch_process(
            items=items,
            processor=processor,
            priority_func=priority_func,
            concurrency=2,
            rate_limiter=mock_limiter
        )
    
    # Check results
    assert len(results) == 5
    assert isinstance(results["item1"], str)  # Success
    assert isinstance(results["item2"], str)  # Success
    assert isinstance(results["item3"], Exception)  # Error
    assert isinstance(results["item4"], str)  # Success
    assert isinstance(results["item5"], str)  # Success


@pytest.mark.asyncio
async def test_find_memory_leaks():
    """Test find_memory_leaks function."""
    # Test function that doesn't leak memory
    def non_leaking_func():
        return [1, 2, 3]
    
    # Test find_memory_leaks with non-leaking function
    is_leaking, metrics = find_memory_leaks(non_leaking_func, iterations=3)
    
    # Verify metrics structure
    assert isinstance(is_leaking, bool)
    assert "memory_diff_mb" in metrics
    assert "total_diff_mb" in metrics
    assert "iterations" in metrics
    assert metrics["iterations"] == 3


@pytest.mark.asyncio
async def test_find_memory_leaks_async():
    """Test find_memory_leaks_async function."""
    # Test async function that doesn't leak memory
    async def non_leaking_async_func():
        await asyncio.sleep(0.01)
        return [1, 2, 3]
    
    # Test find_memory_leaks_async with non-leaking function
    is_leaking, metrics = await find_memory_leaks_async(non_leaking_async_func, iterations=3)
    
    # Verify metrics structure
    assert isinstance(is_leaking, bool)
    assert "memory_diff_mb" in metrics
    assert "memory_growth_per_iteration_mb" in metrics
    assert "iterations" in metrics
    assert metrics["iterations"] == 3


def test_profile_decorator():
    """Test profile decorator."""
    # Create a test function
    @profile
    def test_func(a, b):
        return a + b
    
    # Patch io.StringIO and pstats.Stats
    with patch('io.StringIO', MagicMock()), \
         patch('pstats.Stats', MagicMock()):
        
        # Call the decorated function
        result = test_func(1, 2)
        
        # Verify result
        assert result == 3


def test_profile_memory_decorator():
    """Test profile_memory decorator."""
    # Create a test function
    @profile_memory
    def test_func(a, b):
        # Allocate some memory
        data = [0] * 100000
        return a + b
    
    # Patch MemoryProfiler methods
    with patch.object(MemoryProfiler, 'start', MagicMock()), \
         patch.object(MemoryProfiler, 'stop', MagicMock(return_value={
             "total_diff_mb": 1.5,
             "top_consumers": [
                 {"file": "test_file.py", "line": 42, "size_diff_kb": 1500, "count_diff": 1}
             ]
         })):
        
        # Call the decorated function
        result = test_func(1, 2)
        
        # Verify result
        assert result == 3


def test_performance_benchmark_compare_with_baseline(tmp_path):
    """Test PerformanceBenchmark.compare_with_baseline method."""
    # Create a baseline file
    baseline_path = tmp_path / "baseline.json"
    baseline_data = [
        {
            "operation": "test_operation",
            "timestamp": "2023-01-01T00:00:00",
            "avg_time": 2.0,
            "median_time": 1.8,
            "success_rate": 0.95,
            "memory_after_mb": 100.0
        }
    ]
    
    import json
    with open(baseline_path, 'w') as f:
        json.dump(baseline_data, f)
    
    # Create a benchmark result
    result = BenchmarkResult(
        operation="test_operation",
        iterations=3,
        start_time=time.time(),
        end_time=time.time() + 10,
        duration=10.0,
        avg_time=3.0,
        median_time=2.7,
        success_rate=0.90,
        memory_before=1024 * 1024 * 90,
        memory_after=1024 * 1024 * 110
    )
    
    # Patch benchmark settings to use our temp dir
    with patch('yahoofinance.analysis.benchmarking.BENCHMARK_SETTINGS', {
        "BENCHMARK_DIR": str(tmp_path),
        "BASELINE_FILE": "baseline.json"
    }):
        # Compare with baseline
        comparison = PerformanceBenchmark.compare_with_baseline(result, "test_operation")
    
    # Verify comparison
    assert comparison["avg_time_change"] == pytest.approx(50.0, 0.1)  # 50% increase
    assert comparison["median_time_change"] == pytest.approx(50.0, 0.1)  # 50% increase
    assert comparison["success_rate_change"] == pytest.approx(-0.05, 0.01)  # 5% decrease
    assert comparison["memory_change"] == pytest.approx(10.0, 0.1)  # 10% increase


def test_performance_benchmark_create_baseline(tmp_path):
    """Test PerformanceBenchmark.create_baseline method."""
    # Create sample results
    results = [
        {
            "operation": "test_operation1",
            "avg_time": 2.0,
            "median_time": 1.8,
            "success_rate": 0.95
        },
        {
            "operation": "test_operation2",
            "avg_time": 3.0,
            "median_time": 2.7,
            "success_rate": 0.90
        }
    ]
    
    # Patch benchmark settings to use our temp dir
    with patch('yahoofinance.analysis.benchmarking.BENCHMARK_SETTINGS', {
        "BENCHMARK_DIR": str(tmp_path),
        "BASELINE_FILE": "baseline.json"
    }):
        # Create baseline
        PerformanceBenchmark.create_baseline(results)
    
    # Verify baseline file was created
    baseline_path = tmp_path / "baseline.json"
    assert baseline_path.exists()
    
    # Verify file contents
    import json
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    
    assert len(baseline_data) == 2
    assert baseline_data[0]["operation"] == "test_operation1"
    assert baseline_data[1]["operation"] == "test_operation2"